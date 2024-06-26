# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN


def add_sigais_config(cfg):
    """
    Add config for sigais.
    """

    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Used for `poly` learning rate schedule.
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    
    # Target generation parameters.
    cfg.INPUT.GAUSSIAN_SIGMA = 10
    cfg.INPUT.IGNORE_STUFF_IN_OFFSET = True
    cfg.INPUT.SMALL_INSTANCE_AREA = 4096
    cfg.INPUT.SMALL_INSTANCE_WEIGHT = 3
    cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC = False
    # Optimizer type.
    cfg.SOLVER.OPTIMIZER = "ADAM"

    # Sigais setting
    cfg.MODEL.SIGAIS = CN()
    # Only evaluates network speed (ignores post-processing).
    cfg.MODEL.SIGAIS.BENCHMARK_NETWORK_SPEED = False
    cfg.MODEL.SIGAIS.STUFF_AREA = 2048
    cfg.MODEL.SIGAIS.CENTER_THRESHOLD = 0.1
    cfg.MODEL.SIGAIS.NMS_KERNEL = 7
    cfg.MODEL.SIGAIS.TOP_K_INSTANCE = 200
    cfg.MODEL.SIGAIS.PREDICT_INSTANCES = True
    cfg.MODEL.SIGAIS.USE_DEPTHWISE_SEPARABLE_CONV = True
    cfg.MODEL.SIGAIS.SIZE_DIVISIBILITY = -1
    cfg.MODEL.SIGAIS.BENCHMARK_NETWORK_SPEED = False
    cfg.MODEL.DECODER = CN()
    cfg.MODEL.DECODER.NAME = "build_sigais_decoder"
    cfg.MODEL.DECODER.NORM = "BN"
