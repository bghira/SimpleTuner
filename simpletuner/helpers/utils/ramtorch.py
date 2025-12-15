import importlib
import logging
from fnmatch import fnmatch
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
_peft_lora_config_patched = False
_peft_lora_model_patched = False


@lru_cache(maxsize=1)
def _ramtorch_imports():
    """
    Import RamTorch lazily to avoid hard dependencies when the feature is unused.
    """
    importlib.import_module("ramtorch")  # noqa: F401

    from simpletuner.helpers import ramtorch_workarounds

    ramtorch_workarounds.apply_ramtorch_workarounds()
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
        imports = _ramtorch_imports()
        _maybe_patch_peft_lora_config()
        _maybe_patch_peft_lora_model()
        return imports
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
        for _, child in module.named_modules():
            if isinstance(child, ramtorch_linear):
                for param in child.parameters(recurse=False):
                    setattr(param, "is_ramtorch", True)
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
                setattr(new_layer.weight, "is_ramtorch", True)
                if new_layer.bias is not None:
                    setattr(new_layer.bias, "is_ramtorch", True)

                setattr(current, child_name, new_layer)
                replaced += 1
                continue

            _recurse(child, qualified_name)

    _recurse(module, name_prefix)
    return replaced


def register_lora_custom_module(lora_config) -> bool:
    """
    Add RamTorch Linear to PEFT's custom module map so LoRA can wrap CPUBouncingLinear layers.
    """
    if lora_config is None:
        return False

    try:
        import torch
        from peft.tuners.lora.layer import Linear as PeftLinear
        from ramtorch.modules.linear import Linear as RamTorchLinear
    except Exception:
        return False

    class RamTorchPeftLinear(PeftLinear):  # type: ignore[misc]
        """
        PEFT Linear wrapper that ensures LoRA weights live on the input device for RamTorch layers.
        """

        def _ensure_lora_on_device(self, device):
            if device is None:
                return
            try:
                active = self.active_adapter
            except Exception:
                active = None

            def _active_adapters(candidate):
                if candidate is None:
                    return []
                if isinstance(candidate, (list, tuple, set)):
                    return list(candidate)
                return [candidate]

            def _move_layer(layer):
                try:
                    if hasattr(layer, "to") and layer.weight.device != device:
                        layer.to(device)
                except Exception:
                    pass

            actives = _active_adapters(active)
            keys_to_move = actives or list(self.lora_A.keys())

            for name in keys_to_move:
                try:
                    key = name
                    if key in self.lora_A:
                        _move_layer(self.lora_A[key])
                    if key in self.lora_B:
                        _move_layer(self.lora_B[key])
                    if key in getattr(self.lora_embedding_A, "_parameters", {}):
                        emb_a = self.lora_embedding_A[key]
                        emb_b = self.lora_embedding_B.get(key)
                        try:
                            if hasattr(emb_a, "to") and emb_a.device != device:
                                emb_a.data = emb_a.data.to(device)
                        except Exception:
                            pass
                        try:
                            if emb_b is not None and hasattr(emb_b, "to") and emb_b.device != device:
                                emb_b.data = emb_b.data.to(device)
                        except Exception:
                            pass
                except Exception:
                    continue

        def forward(self, x: torch.Tensor, *args, **kwargs):  # type: ignore[override]
            self._ensure_lora_on_device(x.device if torch.is_tensor(x) else None)
            return super().forward(x, *args, **kwargs)

    custom_modules = getattr(lora_config, "_custom_modules", None)
    if custom_modules is None:
        custom_modules = {}
        lora_config._custom_modules = custom_modules
    if not isinstance(custom_modules, dict):
        return False
    if RamTorchLinear in custom_modules:
        return False

    custom_modules[RamTorchLinear] = RamTorchPeftLinear
    logger.debug("Registered RamTorch Linear for PEFT LoRA custom module dispatch.")
    return True


def _maybe_patch_peft_lora_config() -> bool:
    """
    Ensure diffusers' LoRA config creation function attaches the RamTorch custom module mapping.
    """

    global _peft_lora_config_patched
    if _peft_lora_config_patched:
        return False

    def _wrap_create_lora_config(module, attribute) -> bool:
        create_fn = getattr(module, attribute, None)
        if create_fn is None:
            return False
        if getattr(create_fn, "_ramtorch_wrapped_custom_patch", False):
            return False

        def _wrapped_create_lora_config(*args, **kwargs):
            config = create_fn(*args, **kwargs)
            try:
                register_lora_custom_module(config)
                from peft.tuners.lora.layer import Linear as PeftLinear
                from ramtorch.modules.linear import Linear as RamTorchLinear

                custom_modules = getattr(config, "_custom_modules", None)
                if not isinstance(custom_modules, dict):
                    custom_modules = {}
                    config._custom_modules = custom_modules
                custom_modules.setdefault(RamTorchLinear, PeftLinear)
            except Exception as exc:
                logger.debug("RamTorch LoRA custom module patch failed for %s.%s: %s", module.__name__, attribute, exc)
            return config

        _wrapped_create_lora_config._ramtorch_wrapped = True
        _wrapped_create_lora_config._ramtorch_wrapped_custom_patch = True
        setattr(module, attribute, _wrapped_create_lora_config)
        return True

    patched = False
    try:
        import diffusers.loaders.peft as diffusers_peft  # type: ignore

        patched |= _wrap_create_lora_config(diffusers_peft, "_create_lora_config")
    except Exception:
        pass

    try:
        import diffusers.utils.peft_utils as diffusers_peft_utils  # type: ignore

        patched |= _wrap_create_lora_config(diffusers_peft_utils, "_create_lora_config")
    except Exception:
        pass

    _peft_lora_config_patched = patched
    return patched


def _maybe_patch_peft_lora_model() -> bool:
    """
    Patch PEFT's LoRA dispatcher so CPUBouncingLinear is supported even if a config lacks custom module metadata.
    """

    global _peft_lora_model_patched
    if _peft_lora_model_patched:
        return False

    try:
        import torch
        from peft.tuners.lora.model import LoraModel
        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.tuners.lora.layer import Linear as PeftLinear
        from ramtorch.modules.linear import Linear as RamTorchLinear
    except Exception:
        return False

    class RamTorchPeftLinear(PeftLinear):  # type: ignore[misc]
        def _ensure_lora_on_device(self, device):
            if device is None:
                return
            try:
                active = self.active_adapter
            except Exception:
                active = None

            def _active_adapters(candidate):
                if candidate is None:
                    return []
                if isinstance(candidate, (list, tuple, set)):
                    return list(candidate)
                return [candidate]

            def _move_layer(layer):
                try:
                    if hasattr(layer, "to") and layer.weight.device != device:
                        layer.to(device)
                except Exception:
                    pass

            actives = _active_adapters(active)
            keys_to_move = actives or list(self.lora_A.keys())

            for name in keys_to_move:
                try:
                    key = name
                    if key in self.lora_A:
                        _move_layer(self.lora_A[key])
                    if key in self.lora_B:
                        _move_layer(self.lora_B[key])
                except Exception:
                    continue

        def forward(self, x: torch.Tensor, *args, **kwargs):  # type: ignore[override]
            self._ensure_lora_on_device(x.device if torch.is_tensor(x) else None)
            return super().forward(x, *args, **kwargs)

    orig = LoraModel._create_new_module
    if getattr(orig, "_ramtorch_wrapped", False):
        _peft_lora_model_patched = True
        return False

    @staticmethod
    def _wrapped_create_new_module(lora_config, adapter_name, target, **kwargs):
        try:
            target_base = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target
            if isinstance(target_base, RamTorchLinear):
                return RamTorchPeftLinear(target, adapter_name, **kwargs)
        except Exception as exc:
            logger.debug("RamTorch LoRA dispatcher fallback failed: %s", exc)
        return orig(lora_config, adapter_name, target, **kwargs)

    _wrapped_create_new_module._ramtorch_wrapped = True
    LoraModel._create_new_module = staticmethod(_wrapped_create_new_module)
    _peft_lora_model_patched = True
    return True


def ramtorch_zero_utils():
    imports = ensure_available()
    return imports["broadcast_zero_params"], imports["create_zero_param_groups"], imports["setup_grad_sharding_hooks"]


def mark_ddp_ignore_params(module: nn.Module) -> int:
    """
    Mark RamTorch parameters on a module to be ignored by DistributedDataParallel.

    Returns:
        Number of parameters added to the ignore list.
    """
    ramtorch_names = [name for name, param in module.named_parameters() if getattr(param, "is_ramtorch", False)]
    if not ramtorch_names:
        device_types = {param.device.type for _, param in module.named_parameters() if param.device is not None}
        if "cuda" in device_types and "cpu" in device_types:
            ramtorch_names = [name for name, param in module.named_parameters() if param.device.type == "cpu"]
    if not ramtorch_names:
        return 0

    existing = getattr(module, "_ddp_params_and_buffers_to_ignore", set())
    module._ddp_params_and_buffers_to_ignore = set(existing) | set(ramtorch_names)
    return len(ramtorch_names)
