from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from .modules.linear import Linear


def add_custom_hooks(tensor: torch.Tensor, hook_name: str = "_custom_hooks"):
    """
    Add a custom hook dictionary to a tensor, similar to _post_accumulate_grad_hooks

    Args:
        tensor: The tensor to add hooks to
        hook_name: Name of the hook attribute (default: "_custom_hooks")

    Returns:
        The tensor with the hook attribute added
    """
    if not hasattr(tensor, hook_name):
        setattr(tensor, hook_name, OrderedDict())
        setattr(tensor, f"{hook_name}_counter", 0)
    return tensor


def register_ramtorch_hook(tensor: torch.Tensor, hook: Callable, hook_name: str) -> int:
    """
    Register a hook to the tensor

    Args:
        tensor: The tensor to register the hook on
        hook: Callable to register
        hook_name: Name of the hook attribute

    Returns:
        hook_id: Integer ID to remove the hook later
    """
    # Ensure hook dict exists
    if not hasattr(tensor, hook_name):
        add_custom_hooks(tensor, hook_name)

    hooks = getattr(tensor, hook_name)
    counter_name = f"{hook_name}_counter"
    counter = getattr(tensor, counter_name)

    # Add hook with unique ID
    hook_id = counter
    hooks[hook_id] = hook
    setattr(tensor, counter_name, counter + 1)

    return hook_id


def register_ramtorch_grad_hook(module, hook_fn, param_names=None):
    """
    Register backward hooks on module parameters.

    Args:
        module: PyTorch module to register hooks on
        hook_fn: Hook function that takes gradient tensor and optionally returns modified gradient
        param_names: Optional list of parameter names to register hooks on. If None, registers on all parameters.

    Returns:
        List of hook handles that can be used to remove hooks later

    Example:
    ```python
        def my_hook(grad):
            print(f"Gradient norm: {grad.norm()}")
            return grad * 0.5  # Scale gradient

        handles = register_ramtorch_grad_hook(model, my_hook)
        # Later: [h.remove() for h in handles]
    ```
    """
    handles = []

    for name, param in module.named_parameters():
        if param.requires_grad:
            # Filter by parameter names if specified
            if param_names is None or name in param_names:
                if hasattr(param, "is_ramtorch") and param.is_ramtorch:
                    handle = register_ramtorch_hook(param, hook_fn, "_ramtorch_backward_hooks")
                else:
                    handle = param.register_hook(hook_fn)
                # TODO this works but if it not ramtorch then i need to add handles
                handles.append(handle)

    return handles


def register_ramtorch_post_accumulate_grad_hook(module, hook_fn, param_names=None):
    """
    Register post-accumulate gradient hooks on module parameters.

    IMPORTANT: Post-accumulate hooks work differently for ramtorch tensors:

    For ramtorch tensors (CPU-bouncing parameters):
        - Hook receives the tensor itself as argument: hook_fn(tensor)
        - Access gradients via tensor.ramtorch_grad (NOT tensor.grad)
        - Gradients are on GPU when hook executes
        - Modify in-place: tensor.ramtorch_grad.add_(value)
        - Hook should NOT return anything

    For regular tensors:
        - Hook receives the tensor itself as argument: hook_fn(tensor)
        - Access gradients via tensor.grad
        - Gradients are on their native device
        - Modify in-place: tensor.grad.add_(value)
        - Hook should NOT return anything

    Example usage:
    ```python
        def post_accum_fn(tensor):
            if hasattr(tensor, "is_ramtorch") and tensor.is_ramtorch:
                tensor.ramtorch_grad.add_(60)  # Modify GPU gradient
            else:
                tensor.grad.add_(60)  # Modify regular gradient

        register_ramtorch_post_accumulate_grad_hook(model, post_accum_fn)
    ```
    Args:
        module: PyTorch module to register hooks on
        hook_fn: Callable that takes (tensor) and modifies gradients in-place
        param_names: Optional list of parameter names to filter (None = all params)

    Returns:
        List of hook handles
    """
    handles = []

    for name, param in module.named_parameters():
        if param.requires_grad:
            # Filter by parameter names if specified
            if param_names is None or name in param_names:
                if hasattr(param, "is_ramtorch") and param.is_ramtorch:
                    handle = register_ramtorch_hook(param, hook_fn, "_ramtorch_post_accumulate_grad_hooks")
                else:
                    handle = param.register_post_accumulate_grad_hook(hook_fn)
                # TODO this works but if it not ramtorch then i need to add handles
                handles.append(handle)

    return handles


def move_model_to_device(model: nn.Module, device: Optional[torch.device] = None):
    """
    Moves model parameters and buffers to the specified device,
    but skips any parameter or buffer that has `is_ramtorch = True`.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            raise RuntimeError("RamTorch move_model_to_device requires a CUDA device")
    else:
        device = torch.device(device)

    for name, param in model.named_parameters(recurse=True):
        if getattr(param, "is_ramtorch", False):
            # Skip moving this param
            continue
        # Move only if not already on the target device
        if param.device != device:
            with torch.no_grad():
                new_param = param.to(device)
            param.data = new_param
            if param._grad is not None:
                param._grad = param._grad.to(device)

    for full_name, buf in model.named_buffers(recurse=True):
        if getattr(buf, "is_ramtorch", False):
            continue
        if buf.device == device:
            continue

        with torch.no_grad():
            new_buf = buf.to(device)

        # Traverse to the owning module
        module = model
        *parents, attr = full_name.split(".")
        for p in parents:
            module = getattr(module, p)

        module._buffers[attr] = new_buf

    return model


def replace_linear_with_ramtorch(module: nn.Module, device: str = "cuda"):
    """
    Recursively replaces all nn.Linear layers in a model with CPUBouncingLinear.

    Args:
        module (nn.Module): The input model or submodule.
        device (str): Target device for computation (used by CPUBouncingLinear).

    Returns:
        nn.Module: The modified model with replacements applied in-place.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            # Create a replacement
            new_layer = Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=device,
                dtype=child.weight.dtype,
                skip_init=True,
            )
            new_layer.train(child.training)

            # Copy into the RamTorch layer's pinned CPU storage. Rebinding
            # .data to the original tensor would discard the pinned allocation.
            with torch.no_grad():
                new_layer.weight.copy_(child.weight.detach().to("cpu"))
                new_layer.weight.is_ramtorch = True
                if child.bias is not None:
                    new_layer.bias.copy_(child.bias.detach().to("cpu"))
                    new_layer.bias.is_ramtorch = True
            new_layer.weight.requires_grad = child.weight.requires_grad
            if new_layer.bias is not None and child.bias is not None:
                new_layer.bias.requires_grad = child.bias.requires_grad

            # Replace the module in-place
            setattr(module, name, new_layer)

        else:
            # Recurse into children
            replace_linear_with_ramtorch(child, device=device)

    return module


def register_forward_prefetch_hooks(module: nn.Module):
    """
    Register hooks that prefetch the next RamTorch Linear layer's forward weights.

    Hooks are registered in module traversal order. This is intended for mostly
    sequential models, where the next RamTorch layer in ``named_modules`` is a
    good approximation of the next layer executed.
    """
    layers = [child for child in module.modules() if isinstance(child, Linear)]
    hooks = []
    for current, next_layer in zip(layers, layers[1:]):

        def _hook(_mod, _inp, _out, target=next_layer):
            target.prefetch_forward()
            return None

        hooks.append(current.register_forward_hook(_hook))
    return hooks


def reattach_is_ramtorch_flags(module: nn.Module):
    """
    Recursively traverse the module hierarchy and reattach `is_ramtorch = True`
    flags to all parameters and buffers inside any module that declares
    `is_ramtorch = True`.

    This is useful after model deserialization, replacement, or rebuilds where
    the attribute may have been lost.

    Args:
        module (nn.Module): Root module to process.
    """
    # If the current module itself is marked as a RAMTorch module,
    # mark all its parameter and buffer tensors.
    if getattr(module, "is_ramtorch", False):
        for name, param in module.named_parameters(recurse=False):
            if isinstance(param, torch.Tensor):
                param.is_ramtorch = True
        for name, buffer in module.named_buffers(recurse=False):
            if isinstance(buffer, torch.Tensor):
                buffer.is_ramtorch = True

    # Recurse into children
    for child in module.children():
        reattach_is_ramtorch_flags(child)


def _ramtorch_named_parameters(module: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    return [
        (name, param)
        for name, param in module.named_parameters()
        if isinstance(param, nn.Parameter) and getattr(param, "is_ramtorch", False)
    ]


def _ramtorch_shared_metadata(
    param: nn.Parameter,
) -> Tuple[Tuple[bytes, bytes, int], Tuple[int, ...], Tuple[int, ...], torch.dtype]:
    if param.device.type != "cpu":
        raise ValueError(f"RamTorch parameter must live on CPU for sharing, got {param.device}")
    storage = param.detach().untyped_storage()
    handle = storage._share_filename_cpu_()
    return handle, tuple(param.size()), tuple(param.stride()), param.dtype


def _attach_shared_tensor(
    param: nn.Parameter,
    handle: Tuple[bytes, bytes, int],
    shape: Tuple[int, ...],
    stride: Tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    manager, name, size = handle
    shared_storage = torch.UntypedStorage._new_shared_filename_cpu(manager, name, size)
    shared_tensor = torch.empty(0, dtype=dtype)
    shared_tensor.set_(shared_storage, 0, shape, stride)
    param.data = shared_tensor
    param.is_ramtorch = True


def attach_shared_ramtorch_parameters(module: nn.Module, process_group: Optional[dist.ProcessGroup] = None) -> int:
    """
    Rebind RamTorch parameters to shared CPU storage across independently launched processes.

    Call this after torch.distributed has been initialized (e.g., in torchrun/Accelerate
    entrypoints) so ramtorch parameters no longer rely on fork-based sharing.

    Returns:
        Number of RamTorch parameters that were attached to shared storage.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0

    group = process_group if process_group is not None else dist.group.WORLD
    rank = dist.get_rank(group)
    ramtorch_params = _ramtorch_named_parameters(module)
    if not ramtorch_params:
        return 0

    metadata: List[
        Optional[
            Tuple[
                str,
                Tuple[bytes, bytes, int],
                Tuple[int, ...],
                Tuple[int, ...],
                torch.dtype,
            ]
        ]
    ] = [None for _ in ramtorch_params]

    if rank == 0:
        for idx, (name, param) in enumerate(ramtorch_params):
            handle, shape, stride, dtype = _ramtorch_shared_metadata(param)
            metadata[idx] = (name, handle, shape, stride, dtype)

    dist.broadcast_object_list(metadata, src=0, group=group)

    attached = 0
    for (name, param), entry in zip(ramtorch_params, metadata):
        if entry is None:
            continue
        _, handle, shape, stride, dtype = entry

        if rank != 0:
            _attach_shared_tensor(param, handle, shape, stride, dtype)
        else:
            storage = param.detach().untyped_storage()
            if not storage.is_shared():
                param.data.share_memory_()

        param.is_ramtorch = True
        attached += 1

    dist.barrier(group)
    return attached
