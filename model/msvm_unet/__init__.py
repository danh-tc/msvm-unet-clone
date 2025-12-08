from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn

from ..utils import register_model
from .encoder import Encoder
from .decoder import DECODERS
from .utils import parse_cfg


@register_model("msvm_unet")
class MSVMUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        encoder: Union[str, Tuple[str, Dict]] = ("tiny", {"pretrained": True}),
        decoder: Union[str, Tuple[str, Dict]] = "default",
    ) -> None:
        super(MSVMUNet, self).__init__()
        enc_name, enc_cfg = parse_cfg(encoder)
        self.encoder = Encoder(enc_name, in_channels=in_channels, **enc_cfg)
        self.dims = self.encoder.dims
        dec_name, dec_cfg = parse_cfg(decoder)
        assert dec_name in DECODERS, f"encoder {dec_name} not found, supports {tuple(DECODERS.keys())}"
        self.decoder = DECODERS[dec_name](dims=self.dims[::-1], num_classes=num_classes, **dec_cfg)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.decoder(self.encoder(x)[::-1])

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_params()
