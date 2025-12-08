import os.path as osp
from typing import Callable, Literal

import h5py
import numpy as np
from torch.utils.data import Dataset

__all__ = ["BaseDataset"]


class BaseDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        list_dir: str,
        split: Literal["train", "val", "test"],
        transform: Callable = None,
        image_key: str = "image",
        label_key: str = "label",
    ) -> None:
        self.data_dir = base_dir
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        assert split in (
            "train",
            "val",
            "test",
        ), "split must be 'train', 'val' or 'test'"
        self.split = split
        with open(osp.join(list_dir, self.split + ".txt")) as fp:
            self.sample_list = fp.readlines()

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> dict:
        """
        output tensor shape:
            {
                "case_name": str,
                "image": [1, height, width] | [depth, height, width],
                "label": [1, height, width] | [depth, height, width]
            }
        """
        fname = self.sample_list[idx].strip("\n")
        data_path = osp.join(self.data_dir, self.split, fname)
        sample = self.load_data(data_path)
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = self.sample_list[idx].strip("\n")
        return sample

    def load_data(self, fname: str) -> dict:
        suffix = osp.splitext(fname)[1]
        if suffix == ".h5":
            data = h5py.File(fname, "r")
            return {"image": data[self.image_key][:], "label": data[self.label_key][:]}
        elif suffix in (".npy", ".npz"):
            data = np.load(fname)
            return {"image": data[self.image_key], "label": data[self.label_key]}
        else:
            raise ValueError(f"Unsupported file format: {fname}")
