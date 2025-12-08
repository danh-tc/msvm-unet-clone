from loguru import logger
import numpy as np
from torch import nn

from ..utils import register_model
from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg, VisionTransformer as ViT_seg


@register_model("trans_unet")
class TransUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        vit_name: str = "R50-ViT-B_16",
        **kwargs,
    ):
        super().__init__()
        img_size = kwargs.get("img_size", 224)
        vit_patches_size = kwargs.get("vit_patches_size", 16)
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.in_channels = in_channels
        config_vit.n_classes = num_classes
        config_vit.n_skip = kwargs.pop("n_skip", 3)
        if vit_name.find("R50") != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size),
                int(img_size / vit_patches_size),
            )
        self.net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
        self.net.load_from(weights=np.load(config_vit.pretrained_path))
        logger.info(f"Loaded pretrained weights from {config_vit.pretrained_path}")

    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)
