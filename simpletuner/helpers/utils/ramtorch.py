import logging
from fnmatch import fnmatch
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _ramtorch_imports():
    """
    Import RamTorch lazily to avoid hard dependencies when the feature is unused.
    """
    import ramtorch  # noqa: F401
    from ramtorch.helpers import replace_linear_with_ramtorch
    from ramtorch.modules.linear import Linear as RamTorchLinear
    from ramtorch.zero1 import broadcast_zero_params, create_zero_param_groups
    from ramtorch.zero2 import setup_grad_sharding_hooks

    return {
        "Linear": RamTorchLinear,
        "replace_all": replace_linear_with_ramtorch,
        "broadcast_zero_params": broadcast_zero_params,
        "create_zero_param_groups": create_zero_param_groups,
        "setup_grad_sharding_hooks": setup_grad_sharding_hooks,
    }


def is_available() -> bool:
    try:
        _ramtorch_imports()
        return True
    except Exception:
        return False


def ensure_available() -> dict:
    try:
        return _ramtorch_imports()
    except ImportError as exc:
        raise ImportError(
            "RamTorch is required for --ramtorch but is not installed. Install from the RamTorch checkout (e.g. "
            "`pip install -e ~/src/ramtorch`) or disable --ramtorch."
        ) from exc
    except Exception as exc:
        raise ImportError(f"RamTorch import failed: {exc}") from exc


def normalize_patterns(patterns: Optional[Sequence[str]]) -> Optional[list[str]]:
    if patterns is None:
        return None
    if isinstance(patterns, str):
        patterns = [patterns]
    normalized = []
    for entry in patterns:
        text = str(entry).strip()
        if text:
            normalized.extend([segment.strip() for segment in text.split(",") if segment.strip()])
    return normalized or None


def _normalize_device(device: object) -> object:
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cuda") if torch.cuda.is_available() else "cuda"
    return device


def _matches_pattern(name: str, module: nn.Module, patterns: Iterable[str]) -> bool:
    class_name = module.__class__.__name__
    for pattern in patterns:
        candidates = [name]
        if "." in name:
            candidates.append(name.split(".", 1)[1])
        if any(fnmatch(candidate, pattern) for candidate in candidates) or fnmatch(class_name, pattern):
            return True
    return False


def replace_linear_layers_with_ramtorch(
    module: nn.Module,
    *,
    device: object,
    target_patterns: Optional[Sequence[str]] = None,
    name_prefix: str = "",
) -> int:
    """
    Replace Linear layers within a module with RamTorch equivalents.

    Args:
        module: Root module to process.
        device: Target CUDA/ROCm device identifier (torch.device or string).
        target_patterns: Optional list of glob patterns to filter which modules are replaced.
        name_prefix: Optional prefix applied to the qualified module names used for matching.

    Returns:
        Number of Linear modules replaced.
    """

    imports = ensure_available()
    ramtorch_linear = imports["Linear"]
    replace_all = imports["replace_all"]
    patterns = normalize_patterns(target_patterns)
    resolved_device = _normalize_device(device)

    if patterns is None:
        # Replace every Linear using RamTorch's helper.
        num_linear = sum(1 for _, child in module.named_modules() if isinstance(child, nn.Linear))
        replace_all(module, device=resolved_device)
        return num_linear

    replaced = 0

    def _recurse(current: nn.Module, prefix: str = ""):
        nonlocal replaced
        for child_name, child in current.named_children():
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, ramtorch_linear):
                continue

            if isinstance(child, nn.Linear) and _matches_pattern(qualified_name, child, patterns):
                new_layer = ramtorch_linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device=resolved_device,
                    dtype=child.weight.dtype,
                    skip_init=True,
                )
                new_layer.train(child.training)

                with torch.no_grad():
                    new_layer.weight.copy_(child.weight.detach().to("cpu"))
                    if child.bias is not None and new_layer.bias is not None:
                        new_layer.bias.copy_(child.bias.detach().to("cpu"))

                new_layer.weight.requires_grad = child.weight.requires_grad
                if new_layer.bias is not None and child.bias is not None:
                    new_layer.bias.requires_grad = child.bias.requires_grad

                setattr(current, child_name, new_layer)
                replaced += 1
                continue

            _recurse(child, qualified_name)

    _recurse(module, name_prefix)
    return replaced


def ramtorch_zero_utils():
    imports = ensure_available()
    return imports["broadcast_zero_params"], imports["create_zero_param_groups"], imports["setup_grad_sharding_hooks"]
