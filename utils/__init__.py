from typing import List

from calflops import calculate_flops
import frozendict
from objprint import objstr
import torch


def print_flops_params(
    model: torch.nn.Module,
    input_shape: List[int] = (1, 3, 224, 224),
    output_as_string: bool = True,
    output_precision: int = 4,
    verbose: bool = True,
) -> None:
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=output_as_string,
        output_precision=output_precision,
        print_results=verbose,
        print_detailed=verbose,
    )
    print(f"FLOPs: {flops}, MACs: {macs}, Params: {params}")


def pretty_object_str(obj: object) -> str:
    if isinstance(obj, frozendict.frozendict):
        obj = dict(obj)
    return objstr(obj, indent=2)
