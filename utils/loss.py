from typing import List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def one_hot(input: Tensor, num_classes: int) -> Tensor:
    """
    input tensor shape: [b, h, w]
    output tensor shape: [b, num_classes, h, w]
    """
    tensors = []
    for i in range(num_classes):
        # noinspection PyUnresolvedReferences
        tensors.append((input == i).unsqueeze(1))
    input = torch.cat(tensors, dim=1)
    return input.float()


def binary_dice_loss(input: Tensor, target: Tensor) -> Tensor:
    """
    input tensor shape:
        input: [b, h, w]; target: [b, h, w]
    output tensor shape: [0]
    """
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(input * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(input * input)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return 1 - loss


def multiclass_dice_loss(
    input: Tensor, target: Tensor, weight: Optional[Tensor] = None, softmax: bool = True
) -> Tensor:
    """
    input tensor shape:
        input: [b, c, h, w]; target: [b, h, w]; weights: [c]
    output tensor shape: [0]
    """
    num_classes = input.shape[1]
    if softmax:
        input = torch.softmax(input, dim=1)
    target = one_hot(target, num_classes)
    if weight is None:
        weight = [1.0] * num_classes
    assert input.size() == target.size(), "predict {} & target {} shape do not match".format(
        input.size(), target.size()
    )

    loss = 0.0
    for i in range(0, num_classes):
        dice = binary_dice_loss(input[:, i], target[:, i])
        loss += dice * weight[i]
    # noinspection PyTypeChecker
    return loss / num_classes


class DiceLoss(nn.Module):
    def __init__(self, weight: Optional[Tensor] = None, softmax: bool = True) -> None:
        """
        input tensor shape:
            weights: [c]
        """
        super(DiceLoss, self).__init__()
        self.dice = lambda x, y: multiclass_dice_loss(x, y, softmax=softmax, weight=weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input tensor shape:
            input: [b, c, h, w]; target: [b, h, w]
        output tensor shape: [0]
        """
        return self.dice(input, target)


class DiceCELoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        dc_weight: float = 1.0,
        softmax: bool = True,
        ce_class_weights: Optional[Tensor] = None,
        dc_class_weights: Optional[Tensor] = None,
    ) -> None:
        super(DiceCELoss, self).__init__()
        self.ce_weight = ce_weight
        self.dc_weight = dc_weight
        self.ce = nn.CrossEntropyLoss(weight=ce_class_weights) if softmax else nn.NLLLoss(weight=ce_class_weights)
        self.dc = DiceLoss(softmax=softmax, weight=dc_class_weights)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input tensor shape:
            input: [b, c, h, w]; target: [b, 1, h, w]
        output tensor shape: [0]
        """
        target = target[:, 0, :, :]  # remove the channel dimension
        return self.ce(input, target[:].long()) * self.ce_weight + self.dc(input, target) * self.dc_weight


class DeepDiceCELoss(nn.Module):
    def __init__(self, factors: List[float] = None, **kwargs):
        super(DeepDiceCELoss, self).__init__()
        self.factors = factors
        self.loss = DiceCELoss(**kwargs)

    def forward(self, inputs: List[Tensor], target: Tensor) -> Tensor:
        """
        input tensor shape:
            inputs: list of [b, c, h, w]; target: [b, 1, h, w]
        output tensor shape: [0]
        """
        assert isinstance(inputs, (list, tuple)), "inputs must be a list of tensors"
        factors = self.factors if self.factors else [1.0] * len(inputs)
        total_loss = 0.0
        for i, input in enumerate(inputs):
            if input.shape[-2:] != target.shape[-2:]:
                target_resized = F.interpolate(target.float(), size=input.shape[-2:], mode="nearest").long()
                loss = self.loss(input, target_resized)
            else:
                loss = self.loss(input, target)
            total_loss += factors[i] * loss
        return total_loss


LOSSES = {
    "DiceCELoss": DiceCELoss,
    "DeepDiceCELoss": DeepDiceCELoss,
}
