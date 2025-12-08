max_epochs = 300
img_size = 224

CONFIG = {
    "model": ("msvm_unet", {}),
    "max_epochs": max_epochs,
    "deep_supervision": False,
    "in_channels": 3,
    "freeze_encoder_epochs": 10,
    "img_size": (img_size, img_size),
    "train_transform": (
        "ours",
        {"output_size": (img_size, img_size), "num_classes": 4},
    ),
    "test_transform": ("noops", {}),
    "train_dataloader": (
        "default",
        {
            "batch_size": 32,
            "num_workers": 6,
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True,
        },
    ),
    "val_dataloader": (
        "default",
        {
            "batch_size": 1,
            "shuffle": False,
            "pin_memory": True,
            "num_workers": 1,
            "persistent_workers": True,
        },
    ),
    "loss": (
        "DiceCELoss",
        {
            "ce_weight": 0.4,
            "dc_weight": 0.6,
        },
    ),
    "optimizer": (
        "AdamW",
        {
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "eps": 1e-8,
            "amsgrad": False,
            "betas": (0.9, 0.999),
        },
    ),
    "lr_scheduler": ("CosineAnnealingLR", {"T_max": max_epochs, "eta_min": 1e-6}),
}
