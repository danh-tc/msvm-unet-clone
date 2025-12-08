from torch import optim
from torch.optim import lr_scheduler

OPTIMIZERS = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
    "AdamW": optim.AdamW,
}

LR_SCHEDULERS = {
    "PolynomialLR": lr_scheduler.PolynomialLR,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
}
