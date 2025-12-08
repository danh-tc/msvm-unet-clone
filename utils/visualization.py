import os
from typing import List

import cv2
import numpy as np
from numpy import ndarray
import torch

COLORMAP = (
    [0, 0, 0],
    [0, 128, 0],
    [0, 255, 0],
    [255, 0, 0],
    [0, 0, 255],
    [51, 153, 255],
    [255, 105, 180],
    [127, 0, 255],
    [255, 127, 0],
    [165, 42, 42],
    [160, 82, 45],
    [255, 0, 255],
    [0, 255, 255],
    [255, 255, 0],
)


def is_grayscale(image: ndarray) -> bool:
    """
    Check if the image is grayscale.
    """
    assert len(image.shape) in (2, 3), f"invalid image shape {image.shape}"
    return not (len(image.shape) == 3 and image.shape[2] > 1)


def norm_image(image: torch.Tensor) -> np.ndarray:
    """
    Normalize an image tensor to a numpy array with values in the range [0, 255].
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if len(image.shape) == 3:
        if image.shape[0] == 1:
            image = image[0]
        else:
            image = np.transpose(image, (1, 2, 0))

    if image.ndim == 3 and image.min() < -0.5:
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    if image.ndim == 2 and image.min() < -0.5:
        image = (image + 1) / 2  # from [-1, 1] to [0, 1]
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def norm_mask(mask: torch.Tensor) -> np.ndarray:
    """
    Normalize a mask tensor to a numpy array with integer values.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if len(mask.shape) == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            raise ValueError(f"Invalid mask shape {mask.shape}, expected [h, w] or [h, w, c]")
    return mask.astype(np.uint8)


def plot_contours(image: ndarray, mask: ndarray, colormap: list = COLORMAP, thickness: float = 1) -> ndarray:
    """
    Plot contours on the image based on the provided mask.

    Args:
        image: [h, w] or [h, w, c]
        mask: [h, w] with integer values representing class indices
        colormap: list of RGB tuples for each class
        thickness: thickness of the contour lines

    Returns: A new image with contours drawn on it.
    """
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR if is_grayscale(image) else cv2.COLOR_RGB2BGR)
    for i in np.unique(mask):
        if i == 0:  # Skip background
            continue
        t = np.array(mask == i).astype(np.uint8)
        contours, _ = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, colormap[i], thickness=thickness)
    return image


def plot_image_mask_groups(groups: List[List], output: str) -> None:
    """
    Plot a grid of images and their corresponding masks.

    Args:
        groups: List of lists, where each inner list contains an image and its corresponding masks.
                Each inner list should have at least one image and can have multiple masks.
        output: Path to save the output image.
    """
    rows = []
    max_colnum = max([len(g) for g in groups])
    for group in groups:
        cols = []
        img = norm_image(group[0])
        for i in range(1, max_colnum):
            if i < len(group) and group[i] is not None:
                pred = norm_mask(group[i])
                cols.append(plot_contours(img, pred))
            else:
                # If there is no prediction for this column, append a blank image
                cols.append(np.zeros_like(cols[-1], dtype=np.uint8))
        rows.append(np.hstack(cols))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    cv2.imwrite(output, np.vstack(rows))
