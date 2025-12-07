import inspect
from typing import Iterable, List, Optional, Sequence, Union


def freeze_adapter_parameters(module, adapter_name: str) -> None:
    """
    Ensure all parameters belonging to an adapter remain frozen.
    """
    if module is None or adapter_name is None:
        return
    for name, param in module.named_parameters():
        if param is None:
            continue
        # Adapter parameters include the adapter name in their identifier.
        if f".{adapter_name}" in name and "lora_" in name:
            param.requires_grad_(False)


def _maybe_call_with_kwargs(fn, *args, **kwargs):
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kwargs)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **filtered_kwargs)


def load_assistant_adapter(
    *,
    transformer,
    pipeline_cls,
    lora_path: str,
    adapter_name: str = "assistant",
    low_cpu_mem_usage: bool = False,
    weight_name: Optional[str] = None,
) -> bool:
    """
    Load a LoRA adapter into a transformer using the pipeline's loader utilities.

    Returns True if an adapter was loaded, False otherwise.
    """
    if not lora_path or transformer is None or pipeline_cls is None:
        return False

    lora_state_fn = getattr(pipeline_cls, "lora_state_dict", None)
    load_fn = getattr(pipeline_cls, "load_lora_into_transformer", None)
    if lora_state_fn is None or load_fn is None:
        return False

    state_kwargs = {}
    if "return_alphas" in inspect.signature(lora_state_fn).parameters:
        state_kwargs["return_alphas"] = True

    if weight_name is not None:
        state_kwargs["weight_name"] = weight_name

    lora_state = lora_state_fn(lora_path, **state_kwargs)
    network_alphas = None
    if isinstance(lora_state, tuple) and len(lora_state) == 2:
        state_dict, network_alphas = lora_state
    else:
        state_dict = lora_state
        network_alphas = None

    # Diffusers treats a non-None alpha map specially; only forward it when present.
    if not network_alphas:
        network_alphas = None
    load_kwargs = {
        "transformer": transformer,
        "adapter_name": adapter_name,
        "_pipeline": None,
        "network_alphas": network_alphas,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    _maybe_call_with_kwargs(load_fn, state_dict, **load_kwargs)
    freeze_adapter_parameters(transformer, adapter_name)
    return True


def set_adapter_stack(
    module,
    adapter_names: Union[str, Sequence[str]],
    weights: Optional[Union[float, List[float], List[dict]]] = None,
    freeze_names: Optional[Iterable[str]] = None,
) -> None:
    """
    Configure active adapters (and optional weights) on a PEFT-enabled module.
    """
    if module is None or adapter_names is None:
        return

    names: Union[str, List[str]]
    if isinstance(adapter_names, str):
        names = adapter_names
    else:
        names = list(adapter_names)

    if isinstance(names, list) and len(names) == 0:
        return

    has_stack = hasattr(module, "set_adapters") and callable(getattr(module, "set_adapters"))
    if has_stack:
        module.set_adapters(names, weights)
    elif hasattr(module, "set_adapter") and callable(getattr(module, "set_adapter")):
        # Fallback to single adapter activation.
        if isinstance(names, list) and len(names) > 0:
            module.set_adapter(names[0])
        elif isinstance(names, str):
            module.set_adapter(names)

    if freeze_names:
        for name in freeze_names:
            freeze_adapter_parameters(module, name)


def build_adapter_stack(
    *,
    peft_config: Optional[dict],
    assistant_adapter_name: str,
    assistant_weight: Optional[float],
    include_default: bool = True,
    default_weight: float = 1.0,
    require_default: bool = False,
) -> tuple[list[str], Union[float, list[float]], list[str]]:
    """
    Build adapter names and weights for stacking, returning:
      (adapter_names, weight_argument, freeze_names)
    """
    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    freeze_names: list[str] = []
    has_default = isinstance(peft_config, dict) and "default" in peft_config

    if assistant_weight is not None and assistant_weight != 0:
        adapter_names.append(assistant_adapter_name)
        adapter_weights.append(float(assistant_weight))
        freeze_names.append(assistant_adapter_name)

    if include_default:
        if not has_default:
            message = "Expected trainable 'default' adapter to be present on the PEFT module."
            if require_default:
                raise ValueError(message)
            return [], [], []
        adapter_names.append("default")
        adapter_weights.append(default_weight)

    if not adapter_names:
        return [], [], []

    weight_arg: Union[float, list[float]]
    if len(adapter_weights) == 1:
        weight_arg = adapter_weights[0]
    else:
        weight_arg = adapter_weights

    return adapter_names, weight_arg, freeze_names
