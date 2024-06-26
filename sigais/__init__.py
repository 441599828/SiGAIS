# Copyright (c) Facebook, Inc. and its affiliates.
from .config import add_sigais_config
from .dataset_mapper import SigaisDatasetMapper

from .loss import DeepLabCE
from .build_solver import WarmupPolyLR
from .sigais_backbone import build_sigais_backbone
from .sigais_decoder import build_decoder

from .sigais import Sigais