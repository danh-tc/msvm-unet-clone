from collections import defaultdict
from typing import Dict

import numpy as np
import torch

from data import CLS2COLOR_MAPPING

np.bool = bool


def dc(result: torch.Tensor, reference: torch.Tensor) -> float:
    result = torch.atleast_1d(result.type(torch.bool))
    reference = torch.atleast_1d(reference.type(torch.bool))

    intersection = torch.count_nonzero(result & reference)

    size_i1 = torch.count_nonzero(result)
    size_i2 = torch.count_nonzero(reference)

    try:
        dc = (2.0 * intersection / float(size_i1 + size_i2)).item()
    except ZeroDivisionError:
        dc = 0.0

    return dc


def calc_dice_gpu(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    input tensor shape:
        pred: [[d,] h, w]; gt: [[d,] h, w]
    """
    if pred.sum() > 0 and gt.sum() > 0:
        return dc(pred, gt)
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1.0
    return 0.0


class SegMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.metric = None
        self.reset()

    def reset(self) -> None:
        self.metric = {
            "dice": (defaultdict(list), calc_dice_gpu),
        }

    def __call__(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        """
        input tensor shape:
            input: [b, 1, h, w]; target: [b, 1, h, w]
        """
        for batch_idx in range(pred.shape[0]):
            y_hat, y = pred[batch_idx], label[batch_idx]
            for class_name, (i, _) in CLS2COLOR_MAPPING[self.num_classes].items():
                for _, (v, f) in self.metric.items():
                    v[class_name].append(
                        f(
                            torch.asarray(y_hat == i, dtype=torch.int),
                            torch.asarray(y == i, dtype=torch.int),
                        )
                    )

    def get_metric(self) -> Dict[str, Dict[str, list]]:
        """
        output tensor shape:
            {
                "metric name": {
                    "class name": [val1, val2, ...], ...
                }, ...
            }
        """
        result = {}
        for metric_name, (v, _) in self.metric.items():
            result[metric_name] = v
        return result
