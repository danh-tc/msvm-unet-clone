from glob import glob
import importlib
import inspect
import os.path as osp
from typing import Any, Optional, Tuple

__all__ = ["get_config", "update_config", "parse_cfg"]


# Importing config from current package
def _bootstrap():
    package = osp.basename(osp.dirname(__file__))
    for fname in glob(osp.join(package, "*.py")):
        bname = osp.basename(fname)
        if osp.isfile(fname) and "__init__.py" != bname:
            mod = importlib.import_module(f".{osp.splitext(bname)[0]}", package)
            globals()[mod.__name__] = mod


_bootstrap()
del _bootstrap


def get_config(config_name: str) -> dict:
    assert config_name in globals(), f"Config {config_name} not found"
    return globals()[config_name].CONFIG


def update_config(base_config: dict, trainer: Any = None) -> dict:
    config = dict(base_config)
    if trainer is not None:
        for key, value in base_config.items():
            if callable(value):
                sig = inspect.signature(value)
                params = list(sig.parameters.values())
                if len(params) == 1 and params[0].name == "trainer":
                    config[key] = value(trainer)
    return config


def parse_cfg(cfg: dict, key: str, default: Any = (None, {})) -> Tuple[Optional[str], dict]:
    return cfg.get(key, (None, {})) or default
