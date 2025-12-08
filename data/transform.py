from typing import Tuple

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from numpy import ndarray
from scipy.ndimage.interpolation import zoom
from torchvision.transforms import transforms


def mask_to_onehot(mask: ndarray, num_classes: int) -> ndarray:
    """Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask, -1)
    for colour in range(num_classes):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map


def augment_seg(img_aug: iaa.Augmenter, img: ndarray, seg: ndarray, num_classes: int) -> Tuple[ndarray, ndarray]:
    seg = mask_to_onehot(seg, num_classes)
    aug_det = img_aug.to_deterministic()
    image_aug = aug_det.augment_image(img)

    seg_map = ia.SegmentationMapsOnImage(seg, shape=img.shape)
    seg_map_aug = aug_det.augment_segmentation_maps(seg_map)
    seg_map_aug = seg_map_aug.get_arr()
    seg_map_aug = np.argmax(seg_map_aug, axis=-1).astype(np.float32)
    return image_aug, seg_map_aug


class OursTransform:
    def __init__(self, output_size: Tuple[int, int], num_classes: int) -> None:
        self.img_size = output_size[0]
        self.num_classes = num_classes
        self.img_aug = iaa.SomeOf(
            (0, 4),
            [
                iaa.Flipud(0.5, name="Flipud"),
                iaa.Fliplr(0.5, name="Fliplr"),
                iaa.AdditiveGaussianNoise(scale=0.05),
                iaa.GaussianBlur(sigma=1.0),
                iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
                iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
                iaa.Affine(rotate=(-40, 40)),
                iaa.Affine(shear=(-16, 16)),
                iaa.PiecewiseAffine(scale=(0.008, 0.03)),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            ],
            random_order=True,
        )
        self.norm_x_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]  # to [-1, 1]
        )
        self.norm_y_transform = transforms.ToTensor()

    def __call__(self, sample: dict) -> dict:
        image, label = sample["image"], sample["label"]
        image, label = image.astype(np.float32), label.astype(np.float32)
        image, label = augment_seg(self.img_aug, image, label, self.num_classes)
        x, y = image.shape
        if x != self.img_size or y != self.img_size:
            image = zoom(image, (self.img_size / x, self.img_size / y), order=3)
            label = zoom(label, (self.img_size / x, self.img_size / y), order=0)
        if self.norm_x_transform is not None:
            image = self.norm_x_transform(image.copy())
        if self.norm_y_transform is not None:
            label = self.norm_y_transform(label.copy())
        sample["image"] = image
        sample["label"] = label
        return sample


class NoOpsTransform:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, sample: dict) -> dict:
        return sample


TRANSFORMS = {
    "ours": OursTransform,
    "noops": NoOpsTransform,
}
