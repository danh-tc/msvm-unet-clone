import argparse
from collections import OrderedDict
from glob import glob
import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional

from loguru import logger
from medpy import metric
import numpy as np
from scipy.ndimage import zoom
from sklearn.metrics import matthews_corrcoef

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm

from config import parse_cfg
from data import CLS2COLOR_MAPPING, DATALOADERS, DATASETS, TRANSFORMS, BaseDataset
from model import build_model

METRICS = ("dc", "hd95", "jc", "asd", "se", "sp", "pr")


def try_gpu() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pretty_number(number: float, metric_name: str) -> float:
    assert metric_name in METRICS
    if any([e in metric_name for e in ("hd", "asd")]):
        return round(number, 2)
    return round(float(number * 100), 2)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=4)


def calc_metric_per_class(pred: np.ndarray, gt: np.ndarray) -> List[float]:
    """
    input ndarray shape:
        pred: [depth, height, width]; gt: [depth, height, width]

    output float: (dice, hd95, jaccard, asd, se, sp, pr)
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = metric.binary.assd(pred, gt)
        se = metric.binary.sensitivity(pred, gt)
        sp = metric.binary.specificity(pred, gt)
        pr = metric.binary.precision(pred, gt)
        return dice, hd95, jaccard, asd, se, sp, pr
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 1, 0, 1, 1, 1
    else:
        return 0, 0, 0, 0, 1, 1, 1


def test_single_volume(
    model: nn.Module,
    volume: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    patch_size: List[int],
    **kwargs: Any,
) -> List[Any]:
    """
    input tensor shape:
        image: [1, depth, height, width]; label: [1, depth, height, width]

    output list: ([C-1, K], [0])
    """
    assert volume.shape[0] == 1, f"volume batch size must be 1, not {volume.shape[0]}"
    assert len(volume.shape[1:]) == 3, f"volume shape must be 3D, not {volume.shape[1:]}"
    assert num_classes > 1, "only support multi-classes evaluation"

    volume = volume.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    assert volume.shape == label.shape, f"volume {volume.shape} and label {label.shape} shape mismatch"

    device = try_gpu()
    prediction = np.zeros_like(label)
    for depth in tqdm(range(volume.shape[0])):
        image_slice = volume[depth, :, :]
        h, w = image_slice.shape
        if h != patch_size[0] or w != patch_size[1]:
            image_slice = zoom(image_slice, (patch_size[0] / h, patch_size[1] / w), order=3)

        if kwargs.get("norm_x_transform", None) is not None:
            input = kwargs.get("norm_x_transform")(image_slice)
        else:
            input = torch.from_numpy(image_slice).unsqueeze(0)
        input = input.unsqueeze(0).float().to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input)
            assert isinstance(outputs, torch.Tensor), "Multiple outputs detected"
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            if h != patch_size[0] or w != patch_size[1]:
                pred = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
            else:
                pred = out
            prediction[depth] = pred

    metrics = []  # [C-1, K]
    for c in tqdm(range(1, num_classes)):
        metrics.append(calc_metric_per_class(prediction == c, label == c))

    no_bg = label > 0
    mcc = matthews_corrcoef(
        y_true=label[no_bg].reshape(-1),
        y_pred=prediction[no_bg].reshape(-1,),
    )  # [0]
    return metrics, mcc


def inference(
    model: nn.Module,
    dataloader: data.DataLoader,
    num_classes: int,
    patch_size: List[int],
    **kwargs: Any,
) -> Dict:
    eval_metrics = {"per_case": {}, "mean_case": None, "mean_metric": {}}
    metric_list = 0.0
    mcc_list = 0.0
    for sample in tqdm(dataloader):
        image, label = sample["image"], sample["label"]
        case_name = sample["case_name"][0]

        metric_overall, mcc = test_single_volume(
            model=model,
            volume=image,
            label=label,
            num_classes=num_classes,
            patch_size=patch_size,
            **kwargs,
        )  # [C-1, K], [0], [C, C]
        metric_list += np.array(metric_overall)
        mcc_list += mcc

        # per class
        metric_avg_c = np.mean(metric_overall, axis=0)  # [K]
        eval_metrics["per_case"][case_name] = {
            "overall": metric_overall,
            "mcc": mcc,
            "avg_c": metric_avg_c.tolist(),
        }

    # mean case
    metric_list = metric_list / len(dataloader)  # [C, K]
    eval_metrics["mean_case"] = metric_list.tolist()
    for class_name, (i, _) in CLS2COLOR_MAPPING[num_classes].items():
        t = f"#class_name: {class_name}\n"
        for j, name in enumerate(METRICS):
            t += f"{name}: {pretty_number(metric_list[i - 1][j], name)}\n"
        logger.info(t)

    # mean metric
    mean_metric = np.mean(metric_list, axis=0)  # [K]
    mean_mcc = mcc_list / len(dataloader)
    mean_metric_dict = {"mcc": mean_mcc}
    t = f"Performance: \nmcc: {pretty_number(mean_mcc, 'dc')}\n"
    for i, name in enumerate(METRICS):
        t += f"{name}: {pretty_number(mean_metric[i], name)}\n"
        mean_metric_dict[name] = mean_metric[i]
    logger.info(t)
    eval_metrics["mean_metric"] = mean_metric_dict

    return eval_metrics


def load_training_cfg(model: str, dataset: str) -> Optional[Dict]:
    from importlib import import_module

    for cfg in glob(osp.join(".", "config", f"{model}*.py")):
        if dataset in cfg:
            mod = import_module(f"config.{osp.splitext(osp.basename(cfg))[0]}")
            logger.info(f"Loaded training cfg from {mod.__name__}")
            return getattr(mod, "CONFIG", None)
    return None


def load_checkpoint(model: nn.Module, log_dir: str) -> nn.Module:
    try:
        ckpt_names = glob(osp.join(osp.join(log_dir, "checkpoints"), "epoch*.ckpt"))
        assert len(ckpt_names) == 1, "Multiple or none checkpoints found"

        ckpt = torch.load(ckpt_names[0], map_location="cpu")
        state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            state_dict[k.replace("_model.", "", 1)] = v

        model = model.to("cpu")
        model.load_state_dict(state_dict)
        model = model.to(try_gpu())
        logger.info(f"Loaded checkpoint from {ckpt_names[0]}")
        return model
    except Exception as e:
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("-d", "--dataset", required=True, type=str)
    parser.add_argument("-o", "--output", default="./results", type=str)
    args = parser.parse_args()

    root = osp.join(args.output, args.dataset)
    os.makedirs(root, exist_ok=True)
    logger.add(osp.join(root, f"test-{args.model}.log"))

    # Loading config file
    assert args.dataset in DATASETS, f"dataset {args.dataset} not found"
    dataset_cfg = DATASETS[args.dataset]
    train_cfg = load_training_cfg(args.model, args.dataset)
    assert train_cfg is not None, f"train_cfg {args.model}_{args.dataset} not found"

    # Loading dataset
    base_dir = osp.expandvars(osp.join("$DATASET_HOME", dataset_cfg["root_suffix"]))
    tf_name, tf_cfg = parse_cfg(train_cfg, "test_transform")
    test_transform = TRANSFORMS[tf_name](**tf_cfg) if tf_name else None
    test_dataset = BaseDataset(
        base_dir=base_dir,
        split="test",
        list_dir=dataset_cfg["list_dir"],
        transform=test_transform,
    )
    loader_name, loader_cfg = parse_cfg(train_cfg, "val_dataloader")
    test_dataloader = DATALOADERS[loader_name](test_dataset, **loader_cfg)

    # Loading checkpoints
    log_root_dir = "./log"
    log_names = glob(osp.join(log_root_dir, f"{args.model}-{args.dataset}-r*"))
    assert len(log_names) == 3, "Logs is not enough"

    norm_x_transform = None
    tf_name, tf_cfg = parse_cfg(train_cfg, "train_transform")
    if tf_name == "ours":
        tf = TRANSFORMS[tf_name](**tf_cfg)
        norm_x_transform = getattr(tf, "norm_x_transform")

    results = {}
    for i, log_dir in enumerate(log_names):
        model_name, model_cfg = parse_cfg(train_cfg, "model")
        assert model_name == args.model
        model = build_model(
            name=args.model,
            in_channels=train_cfg.get("in_channels", 3),
            num_classes=dataset_cfg["num_classes"],
            **model_cfg,
        )
        model = load_checkpoint(model, log_dir)
        results[i] = inference(
            model=model,
            dataloader=test_dataloader,
            num_classes=dataset_cfg["num_classes"],
            patch_size=train_cfg["img_size"],
            norm_x_transform=norm_x_transform,
        )

    save_json(results, osp.join(root, f"{args.model}.json"))
