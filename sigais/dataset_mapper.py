# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Callable, List, Union, Optional
import torch

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .target_generator import SigaisTargetGenerator
from fvcore.transforms.transform import Transform

__all__ = ["SigaisDatasetMapper"]


class SigaisDatasetMapper:
    """
    The callable currently does the following:

    1. Read the image from "file_name" and label from "pan_seg_file_name"
    2. Applies random scale, crop and flip transforms to image and label
    3. Prepare data to Tensor and generate training targets from label
    """

    @configurable
    def __init__(
        self,
        *,
        cfg,
        augmentations,
        image_format: str,
        panoptic_target_generator: Callable,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            panoptic_target_generator: a callable that takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
        """
        # fmt: off
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

        self.panoptic_target_generator = panoptic_target_generator

    @classmethod
    def from_config(cls, cfg):
        augs = []

        if cfg.INPUT.CROP.ENABLED:
            augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        panoptic_target_generator = SigaisTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )

        ret = {
            "cfg": cfg,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "panoptic_target_generator": panoptic_target_generator,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # Load image.
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # Load map.
        static_map = utils.read_image(dataset_dict["map_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, static_map)

        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), 'I')

        # Reuses semantic transform for panoptic labels.
        aug_input = SigaisAugInput(image, static_map, sem_seg=pan_seg_gt)
        _ = self.augmentations(aug_input)
        image, static_map, pan_seg_gt = aug_input.image, aug_input.static_map, aug_input.sem_seg

        # data augument for photos (diff in img and bkg)
        aug_input = T.AugInput(image)
        augumentation = []
        augumentation.append(T.Random_HSV_SV_Shifting(sgain=0.5, vgain=0.5))
        augumentation.append(T.RandomContrast(0.8,1.2, prob=0.5))
        augmentations = T.AugmentationList(augumentation)
        _ = augmentations(aug_input)
        image = aug_input.image

        aug_input = T.AugInput(static_map)
        augumentation = []
        augumentation.append(T.Random_HSV_SV_Shifting(sgain=0.5, vgain=0.5))
        augumentation.append(T.RandomContrast(0.8,1.2, prob=0.5))
        augmentations = T.AugmentationList(augumentation)
        _ = augmentations(aug_input)
        static_map = aug_input.image

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["map"] = torch.as_tensor(np.ascontiguousarray(static_map.transpose(2, 0, 1)))

        # Generates training targets for Sigais.
        targets = self.panoptic_target_generator(pan_seg_gt, dataset_dict["segments_info"])
        dataset_dict.update(targets)

        return dataset_dict


class SigaisAugInput(T.AugInput):
    def __init__(
        self,
        image: np.ndarray,
        static_map: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None
    ):
        super().__init__(image, boxes=boxes, sem_seg=sem_seg)
        self.static_map = static_map

    def transform(self, tfm: Transform) -> None:
        self.image = tfm.apply_image(self.image)
        self.static_map = tfm.apply_image(self.static_map)

        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)


