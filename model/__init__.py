from glob import glob
import importlib
import os.path as osp

from .utils import find_candidates

__all__ = ["build_model"]


# Importing model definitions
def _bootstrap():
    def real_package(d):
        return f"{d['mod']}.{d['obj']}"

    _MODELS = {}
    par_package = osp.basename(osp.dirname(__file__))
    for fname in glob(osp.join(par_package, "*", "__init__.py")):
        sub_package = osp.basename(osp.dirname(fname))
        for model_def in find_candidates(fname):
            mod_name = model_def.get("name")
            name = model_def.get("alias") or mod_name
            import_def = {"mod": f"{par_package}.{sub_package}", "obj": mod_name}
            if name in _MODELS:
                raise ValueError(
                    f"duplicate model name {name} " f"for {real_package(_MODELS[name])} and {real_package(import_def)}"
                )
            _MODELS[name] = import_def
    globals()["_MODELS"] = _MODELS


_bootstrap()
del _bootstrap


def build_model(name: str, in_channels: int, num_classes: int, **kwargs):
    """Build model by name."""
    assert name in globals()["_MODELS"], f"Model {name} not found"
    model = globals().get(name, None)
    if not callable(model):
        model_def = globals()["_MODELS"][name]
        module = importlib.import_module(model_def["mod"])
        model = getattr(module, model_def["obj"])
        globals()[name] = model
    return model(in_channels=in_channels, num_classes=num_classes, **kwargs)
