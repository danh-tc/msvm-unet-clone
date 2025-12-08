"""Copied from https://github.com/HuCaoFighting/Swin-Unet.
"""

import copy
import os

from loguru import logger
import torch
import torch.nn as nn

from ..utils import register_model
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


@register_model("swin_unet")
class SwinUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, img_size=224, zero_head=False):
        super(SwinUNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        assert in_channels in [1, 3], "Input channels must be 1 or 3"
        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            num_classes=self.num_classes,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=False,
            drop_rate=0.0,
            drop_path_rate=0.2,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
        )
        self.load_from()

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self):
        pretrained_path = os.path.join(os.getenv("PRETRAIN_HOME", "."), "swin_tiny_patch4_window7_224.pth")
        if os.path.exists(pretrained_path):
            logger.info(f"Loaded pretrained path: {pretrained_path}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                logger.info("Start load pretrained model by splitting")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        logger.info("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                return None

            pretrained_dict = pretrained_dict["model"]
            logger.info("start load pretrained model of swin encoder")
            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})

            for k in list(full_dict.keys()):
                if k in model_dict:
                    if (s := full_dict[k].shape) != (t := model_dict[k].shape):
                        logger.info(f"delete key: {k}; shape pretrain: {s}; shape model: {t}")
                        del full_dict[k]
            self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            logger.info("none pretrain")
