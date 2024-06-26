import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init

from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.config import configurable
from detectron2.utils.registry import Registry
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.layers import ASPP, Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from detectron2.modeling.postprocessing import sem_seg_postprocess
from .post_processing import get_panoptic_segmentation
from .loss import DeepLabCE
from .sigais_decoder import build_decoder

import cv2
from random import uniform
import os
import csv

from scipy.spatial.transform import Rotation as R

__all__ = ["Sigais",]

@META_ARCH_REGISTRY.register()
class Sigais(nn.Module):
    """
    Main class for "Map guided generalizable net for intersection instance segmentation.
    """
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.decoder = build_decoder(cfg)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.stuff_area = cfg.MODEL.SIGAIS.STUFF_AREA
        self.threshold = cfg.MODEL.SIGAIS.CENTER_THRESHOLD
        self.nms_kernel = cfg.MODEL.SIGAIS.NMS_KERNEL
        self.top_k = cfg.MODEL.SIGAIS.TOP_K_INSTANCE
        self.predict_instances = True
        self.use_depthwise_separable_conv = cfg.MODEL.SIGAIS.USE_DEPTHWISE_SEPARABLE_CONV
        self.size_divisibility = -1
        self.benchmark_network_speed = cfg.MODEL.SIGAIS.BENCHMARK_NETWORK_SPEED


    @property
    def device(self):
        return self.pixel_mean.device
        
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * "center": center points heatmap ground truth
                   * "offset": pixel offsets to center points ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        maps = [x["map"].to(self.device) for x in batched_inputs]
        maps = [(x - self.pixel_mean) / self.pixel_std for x in maps]

        # To avoid error in ASPP layer when input has different size.
        size_divisibility = 0
        images = ImageList.from_tensors(images, size_divisibility)
        maps = ImageList.from_tensors(maps, size_divisibility)

        if "ft_mask" in batched_inputs[0]:
            ft_targets = [x['ft_mask'].to(self.device) for x in batched_inputs]
            ft_targets = ImageList.from_tensors(ft_targets, size_divisibility, 255).tensor
        else:
            ft_targets = None

        features = self.backbone(images.tensor, maps.tensor)

        losses = {}
        if "sem_seg" in batched_inputs[0]:
            seg_targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            seg_targets = ImageList.from_tensors(seg_targets, size_divisibility, 255).tensor
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                seg_weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                seg_weights = ImageList.from_tensors(seg_weights, size_divisibility).tensor
            else:
                seg_weights = None
        else:
            seg_targets = None
            seg_weights = None

        if "center" in batched_inputs[0] and "offset" in batched_inputs[0]:
            center_targets = [x["center"].to(self.device) for x in batched_inputs]
            center_targets = ImageList.from_tensors(center_targets, size_divisibility).tensor.unsqueeze(1)
            center_weights = [x["center_weights"].to(self.device) for x in batched_inputs]
            center_weights = ImageList.from_tensors(center_weights, size_divisibility).tensor

            offset_targets = [x["offset"].to(self.device) for x in batched_inputs]
            offset_targets = ImageList.from_tensors(offset_targets, size_divisibility).tensor
            offset_weights = [x["offset_weights"].to(self.device) for x in batched_inputs]
            offset_weights = ImageList.from_tensors(offset_weights, size_divisibility).tensor

        else:
            center_targets = None
            center_weights = None
            offset_targets = None
            offset_weights = None

        ft_results, sem_seg_results, center_results, offset_results, ft_losses, sem_seg_losses, center_losses, offset_losses = \
            self.decoder(features, ft_targets, seg_targets, seg_weights, center_targets, center_weights, offset_targets, offset_weights)

        losses.update(ft_losses)
        # second training process
        # if self.two_step_train == 0 or self.two_step_train == 2:
        losses.update(sem_seg_losses)
        losses.update(center_losses)
        losses.update(offset_losses)

        if self.training:
            return losses

        if self.benchmark_network_speed:
            return []

        processed_results = []
        for sem_result_img, center_result_img, offset_result_img, input_per_image, image_size in zip(
            sem_seg_results, center_results, offset_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(sem_result_img, image_size, height, width)
            c = sem_seg_postprocess(center_result_img, image_size, height, width)
            o = sem_seg_postprocess(offset_result_img, image_size, height, width)

            # Post-processing to get panoptic segmentation.
            panoptic_image, _ = get_panoptic_segmentation(
                r.argmax(dim=0,keepdim=True),
                c,
                o,
                thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
                label_divisor=self.meta.label_divisor,
                stuff_area=self.stuff_area,
                void_label=-1,
                threshold=self.threshold,
                nms_kernel=self.nms_kernel,
                top_k=self.top_k,
            )

            # For semantic segmentation evaluation.
            processed_results.append({"sem_seg": r})
            panoptic_image = panoptic_image.squeeze(0)
            semantic_prob = F.softmax(r, dim=0)
            # For panoptic segmentation evaluation.
            processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
            # For instance segmentation evaluation.
            if self.predict_instances:
                instances = []
                panoptic_image_cpu = panoptic_image.cpu().numpy()
                for panoptic_label in np.unique(panoptic_image_cpu):
                    if panoptic_label == -1:
                        continue
                    pred_class = panoptic_label // self.meta.label_divisor
                    isthing = pred_class in list(
                        self.meta.thing_dataset_id_to_contiguous_id.values()
                    )
                    # Get instance segmentation results.
                    if isthing:
                        instance = Instances((height, width))
                        # Evaluation code takes continuous id starting from 0
                        instance.pred_classes = torch.tensor(
                            [pred_class], device=panoptic_image.device
                        )
                        mask = panoptic_image == panoptic_label
                        instance.pred_masks = mask.unsqueeze(0)
                        # Average semantic probability
                        sem_scores = semantic_prob[pred_class, ...]
                        sem_scores = torch.mean(sem_scores[mask])
                        # Center point probability
                        mask_indices = torch.nonzero(mask).float()
                        center_y, center_x = (
                            torch.mean(mask_indices[:, 0]),
                            torch.mean(mask_indices[:, 1]),
                        )
                        center_scores = c[0, int(center_y.item()), int(center_x.item())]
                        # Confidence score is semantic prob * center prob.
                        instance.scores = torch.tensor(
                            [sem_scores * center_scores], device=panoptic_image.device
                        )
                        # Get bounding boxes
                        instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
                        instances.append(instance)
                if len(instances) > 0:
                    processed_results[-1]["instances"] = Instances.cat(instances)

        return processed_results

