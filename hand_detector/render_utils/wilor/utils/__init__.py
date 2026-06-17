"""WiLoR utility helpers."""
import torch
from typing import Any


def recursive_to(x: Any, target: torch.device):
    """Recursively ``.to(target)`` every tensor in nested dict / list / tuple."""
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(target)
    if isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    return x
