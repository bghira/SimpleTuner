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
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **filtered_kwargs)


def load_assistant_adapter(
    *,
    transformer,
    pipeline_cls,
    lora_path: str,
    adapter_name: str = "assistant",
    low_cpu_mem_usage: bool = False,
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

    lora_state = lora_state_fn(lora_path, **state_kwargs)
    network_alphas = None
    if isinstance(lora_state, tuple) and len(lora_state) == 2:
        state_dict, network_alphas = lora_state
    else:
        state_dict = lora_state
        network_alphas = None

    if network_alphas is None:
        network_alphas = {}
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

    if hasattr(module, "set_adapters"):
        module.set_adapters(names, weights)
    elif hasattr(module, "set_adapter"):
        # Fallback to single adapter activation.
        if isinstance(names, list) and len(names) > 0:
            module.set_adapter(names[0])
        elif isinstance(names, str):
            module.set_adapter(names)

    if freeze_names:
        for name in freeze_names:
            freeze_adapter_parameters(module, name)
