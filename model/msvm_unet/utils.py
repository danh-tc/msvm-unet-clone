from typing import Any, Dict, Optional, Tuple, Union

CfgType = Optional[Union[str, Tuple[str, Dict]]]


def parse_cfg(cfg: Any) -> Tuple[str, Dict]:
    if cfg is None:
        return None, {}
    if isinstance(cfg, str):
        return cfg, {}
    if isinstance(cfg, tuple):
        assert isinstance(cfg[0], str), f"first item in cfg must be a str, not {type(cfg[0])}"
        assert isinstance(cfg[1], dict), f"second item in cfg must be a dict, not {type(cfg[1])}"
        return cfg
    raise ValueError("cfg must be a str or a tuple")
