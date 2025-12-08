"""Copied from https://github.com/JCruan519/VM-UNet.
"""

import os.path as osp

from loguru import logger
import torch
from torch import nn

from ..utils import register_model
from .vmamba import VSSM


@register_model("vm_unet")
class VMUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        depths=None,
        depths_decoder=None,
        drop_path_rate=0.2,
    ):
        super().__init__()
        if depths_decoder is None:
            depths_decoder = [2, 2, 2, 1]
        if depths is None:
            depths = [2, 2, 2, 2]
        self.load_ckpt_path = osp.expandvars(osp.join("$PRETRAIN_HOME", "vmamba_small_e238_ema.pth"))
        self.num_classes = num_classes

        self.vmunet = VSSM(
            in_chans=in_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
        )

        self.load_from()
        logger.info(f"Loaded checkpoint from {self.load_ckpt_path}")

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        logits = self.vmunet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint["model"]
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            logger.info(
                "Total model_dict: {}, Total pretrained_dict: {}, update: {}".format(
                    len(model_dict), len(pretrained_dict), len(new_dict)
                )
            )
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            logger.info("Not loaded keys:", not_loaded_keys)
            logger.info("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint["model"]
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if "layers.0" in k:
                    new_k = k.replace("layers.0", "layers_up.3")
                    pretrained_dict[new_k] = v
                elif "layers.1" in k:
                    new_k = k.replace("layers.1", "layers_up.2")
                    pretrained_dict[new_k] = v
                elif "layers.2" in k:
                    new_k = k.replace("layers.2", "layers_up.1")
                    pretrained_dict[new_k] = v
                elif "layers.3" in k:
                    new_k = k.replace("layers.3", "layers_up.0")
                    pretrained_dict[new_k] = v
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            logger.info(
                "Total model_dict: {}, Total pretrained_dict: {}, update: {}".format(
                    len(model_dict), len(pretrained_dict), len(new_dict)
                )
            )
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            logger.info("Not loaded keys:", not_loaded_keys)
            logger.info("decoder loaded finished!")
