import torch
import torch.nn as nn
from pathlib import Path
from cached_path import cached_path

# Utility files from examples
import os
from copy import deepcopy

import numpy as np
import skimage.io as io
import torch

import torchvision
from silk.backbones.silk.silk import SiLKVGG
from silk.backbones.superpoint.vgg import ParametricVGG

from silk.config.model import load_model_from_checkpoint
from silk.models.silk import matcher

# NO change
Matcher = matcher

# CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "../../assets/models/silk/analysis/alpha/pvgg-4.ckpt")
# CHECKPOINT_PATH = os.path.join(
#     os.path.dirname(__file__), "../../assets/models/silk/coco-rgb-aug.ckpt"
# )
# DEVICE = "cuda:0"

_SILK_NMS = 0  # NMS radius, 0 = disabled
_SILK_BORDER = 0  # remove detection on border, 0 = disabled
_SILK_THRESHOLD = 1.0  # keypoint score thresholding, if # of keypoints is less than provided top-k, then will add keypoints to reach top-k value, 1.0 = disabled
_SILK_TOP_K = 10000  # minimum number of best keypoints to output, could be higher if threshold specified above has low value
_SILK_DEFAULT_OUTPUT = (  # outputs required when running the model
    "dense_positions",
    "normalized_descriptors",
    "probability",
)
_SILK_SCALE_FACTOR = 1.41  # scaling of descriptor output, do not change
_SILK_BACKBONE = ParametricVGG(
    use_max_pooling=False,
    padding=0,
    normalization_fn=[torch.nn.BatchNorm2d(i) for i in (64, 64, 128, 128)],
)
_DEVICE = "cpu"
# _SILK_MATCHER = matcher(postprocessing="ratio-test", threshold=0.6)
# SILK_MATCHER = matcher(postprocessing="double-softmax", threshold=0.6, temperature=0.1)
# SILK_MATCHER = matcher(postprocessing="none")


def load_images(*paths, as_gray=True, device=torch.device("cpu")):
    images = np.stack([io.imread(path, as_gray=as_gray) for path in paths])
    images = torch.tensor(images, dtype=torch.float32).to(device)
    if not as_gray:
        images = images.permute(0, 3, 1, 2)
        images = images / 255.0
    else:
        images = images.unsqueeze(1)  # add channel dimension
    return images



class SiLK:
    def __init__(self, default_outputs=_SILK_DEFAULT_OUTPUT,device=_DEVICE, *args, **kwargs) -> None:
        # super().__init__(*args, **kwargs)
        # print("Here")
        self.checkpoint_path = cached_path("https://dl.fbaipublicfiles.com/silk/assets/models/silk/coco-rgb-aug.ckpt")
        # print("Here2")
        # load model
        model = SiLKVGG(
            in_channels=1,
            backbone=deepcopy(_SILK_BACKBONE),
            detection_threshold=_SILK_THRESHOLD,
            detection_top_k=_SILK_TOP_K,
            nms_dist=_SILK_NMS,
            border_dist=_SILK_BORDER,
            default_outputs= default_outputs,
            descriptor_scale_factor=_SILK_SCALE_FACTOR,
            padding=0,
        )
        model = load_model_from_checkpoint(
            model,
            checkpoint_path=self.checkpoint_path,
            state_dict_fn=lambda x: {k[len("_mods.model.") :]: v for k, v in x.items()},
            device=device,
            freeze=True,
            eval=True,
        )
        self.model = model

    def to(self, device, *args, **kwargs):
        self.model = self.model.to(device, *args, **kwargs)
        return self
    
    def __call__(self, x):
        return self.model(x)
