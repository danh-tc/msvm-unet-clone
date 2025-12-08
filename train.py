import argparse
from collections import defaultdict
import os
import os.path as osp
from pathlib import Path
import random
from typing import Any, Dict

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
import monai
from monai.metrics import CumulativeAverage
import numpy as np
import torch
from torch import Tensor

from config import get_config, parse_cfg, update_config
from data import DATALOADERS, DATASETS, TRANSFORMS, BaseDataset
from model import build_model
from utils import pretty_object_str
from utils.eval import eval_single_volume
from utils.loss import LOSSES
from utils.optimization import LR_SCHEDULERS, OPTIMIZERS
from utils.visualization import plot_image_mask_groups


class Trainer(L.LightningModule):
    def __init__(self, log_dir: str, cfg: dict) -> None:
        super(Trainer, self).__init__()
        self.log_dir = log_dir
        self.cfg = cfg
        self.log_pred_dir = Path(self.log_dir) / "train_step_images"
        self.log_pred_dir.mkdir(parents=True, exist_ok=True)

        assert args.dataset in DATASETS, f"Dataset {args.dataset} not supported"
        self.dataset_cfg = DATASETS[args.dataset]
        self.train_dataset = None
        self.val_dataset = None

        model_name, model_cfg = parse_cfg(self.cfg, "model")
        self._model = build_model(
            name=model_name,
            in_channels=self.cfg["in_channels"],
            num_classes=self.dataset_cfg["num_classes"],
            **model_cfg,
        ).to(device)

        loss_name, loss_cfg = parse_cfg(self.cfg, "loss")
        assert loss_name in LOSSES, f"Loss {loss_name} not supported"
        self.criterion = LOSSES[loss_name](**loss_cfg)

        self.tl_metric = CumulativeAverage()
        self.vs_metric = defaultdict(lambda: defaultdict(list))

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def prepare_data(self) -> None:
        root = osp.expandvars(osp.join("$DATASET_HOME", self.dataset_cfg["root_suffix"]))
        tt_name, tt_cfg = parse_cfg(self.cfg, "train_transform")
        train_transform = TRANSFORMS[tt_name](**tt_cfg) if tt_name else None
        self.train_dataset = BaseDataset(
            base_dir=root,
            split="train",
            list_dir=self.dataset_cfg["list_dir"],
            transform=train_transform,
        )

        vt_name, vt_cfg = parse_cfg(self.cfg, "test_transform")
        test_transform = TRANSFORMS[vt_name](**vt_cfg) if vt_name else None
        self.val_dataset = BaseDataset(
            base_dir=root,
            split="test",
            list_dir=self.dataset_cfg["list_dir"],
            transform=test_transform,
        )

    def train_dataloader(self) -> Any:
        def worker_init_fn(worker_id: int) -> None:
            random.seed(cfg["seed"] + worker_id)

        loader_name, loader_cfg = parse_cfg(self.cfg, "train_dataloader")
        if "worker_init_fn" not in loader_cfg:
            loader_cfg["worker_init_fn"] = worker_init_fn
        assert loader_name is not None, "train dataloader is not configured"
        return DATALOADERS[loader_name](self.train_dataset, **loader_cfg)

    def val_dataloader(self) -> Any:
        loader_name, loader_cfg = parse_cfg(self.cfg, "val_dataloader")
        assert loader_name is not None, "valid dataloader is not configured"
        return DATALOADERS[loader_name](self.val_dataset, **loader_cfg)

    def configure_optimizers(self) -> dict:
        optim_name, optim_cfg = parse_cfg(self.cfg, "optimizer")
        optimizer = OPTIMIZERS[optim_name](self._model.parameters(), **optim_cfg)

        lrsch_name, lrsch_cfg = parse_cfg(self.cfg, "lr_scheduler")
        if lrsch_name is not None:
            interval = lrsch_cfg.pop("lightning_interval", "epoch")
            lrsch_cfg = update_config(lrsch_cfg, trainer=self)
            scheduler = LR_SCHEDULERS[lrsch_name](optimizer, **lrsch_cfg)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": interval},
            }
        return {"optimizer": optimizer}

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == 0:
            # Solve: Early stopping conditioned on metric `mean_train_loss` which is not available
            self.log_and_logger("mean_train_loss", 0.0)

        freeze_encoder_epochs = self.cfg.get("freeze_encoder_epochs", 0)
        if freeze_encoder_epochs > 0:
            assert hasattr(
                self._model, "freeze_encoder"
            ), f"Model {self.cfg['model'][0]} does not support freezing encoder"
            if self.current_epoch < freeze_encoder_epochs:
                self._model.freeze_encoder()
            else:
                self._model.unfreeze_encoder()
        super().on_train_epoch_start()

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        logits = self.forward(image)
        loss = self.criterion(logits, label)

        self.log("loss", loss.item(), prog_bar=True)
        self.tl_metric.append(loss.item())
        # noinspection PyUnresolvedReferences
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        # record training image
        if self.current_epoch == 0 or self.current_epoch % 5 == 0:
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred = torch.argmax(logits, dim=1)
            plot_image_mask_groups(
                [[image[0], label[0], pred[0]]],
                self.log_pred_dir / f"{self.current_epoch}.png",
            )

        return loss

    def on_train_epoch_end(self) -> None:
        tl = self.tl_metric.aggregate().item()
        self.log_and_logger("mean_train_loss", tl)
        self.tl_metric.reset()

    def validation_step(self, batch: Dict[str, Tensor], *args: Any) -> None:
        volume, label = batch["image"], batch["label"]
        metric = eval_single_volume(
            model=self._model,
            volume=volume,
            label=label,
            num_classes=self.dataset_cfg["num_classes"],
            output=osp.join(self.log_dir, str(self.current_epoch)),
            patch_size=self.cfg["img_size"],
            device=device,
            norm_x_transform=getattr(self.train_dataset.transform, "norm_x_transform", None),
        )

        for metric_name, class_metric in metric.items():
            for class_name, value in class_metric.items():
                self.vs_metric[metric_name][class_name].append(np.mean(value))

    def on_validation_epoch_end(self) -> None:
        for metric_name, class_metric in self.vs_metric.items():
            avg_metric = []
            for class_name, value in class_metric.items():
                t = np.mean(value)
                self.log(f"val_{metric_name}_{class_name}", t)
                avg_metric.append(t)
            self.log_and_logger(f"val_mean_{metric_name}", np.mean(avg_metric))
        self.vs_metric = defaultdict(lambda: defaultdict(list))

    def log_and_logger(self, name: str, value: Any, **kwargs: Any) -> None:
        self.log(name, value, **kwargs)
        logger.info(f"{name}: {value}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset name")
    parser.add_argument("-c", "--config", type=str, required=True, help="config name")
    parser.add_argument("-r", "--round", type=int, default=0, help="round to run the experiment")
    parser.add_argument("--seed", type=int, default=42, help="seed to experiment")
    args = parser.parse_args()

    cfg = get_config(args.config)
    cfg["seed"] = args.seed + int(args.round)
    log_dir = osp.join("log", f"{cfg['model'][0]}-{args.dataset}-r{args.round}")
    os.makedirs(log_dir, exist_ok=True)

    logger.add(osp.join(log_dir, "training.log"))
    logger.info(f"Config: {pretty_object_str(cfg)}")

    L.seed_everything(cfg["seed"])
    monai.utils.set_determinism(cfg["seed"])

    callbacks = [
        ModelCheckpoint(
            dirpath=osp.join(log_dir, "checkpoints"),
            monitor="val_mean_dice",
            mode="max",
            filename="{epoch:02d}-{val_mean_dice:.4f}",
            save_last=True,
        )
    ]
    if cfg.get("early_stop", True):
        early_stop_callback = EarlyStopping(monitor="mean_train_loss", mode="min", min_delta=0.00, patience=15)
        callbacks.append(early_stop_callback)

    model = Trainer(log_dir, cfg)
    trainer = L.Trainer(
        precision=32,
        accelerator=device,
        devices="auto",
        max_epochs=cfg["max_epochs"],
        check_val_every_n_epoch=cfg.get("check_val_every_n_epoch", 20),
        gradient_clip_val=None,
        default_root_dir=log_dir,
        callbacks=callbacks,
        enable_checkpointing=True,
    )
    trainer.fit(model, ckpt_path=None)
