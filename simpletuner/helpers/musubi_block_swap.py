import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import torch
import torch.nn as nn

__all__ = [
    "MusubiBlockSwapManager",
    "apply_musubi_pretrained_defaults",
    "prepare_musubi_model_for_ddp",
]


def _module_on_device(module: nn.Module, device: torch.device) -> bool:
    target = torch.device(device)
    for tensor in module.parameters():
        if not _tensor_on_device(tensor, target):
            return False
    for tensor in module.buffers():
        if not _tensor_on_device(tensor, target):
            return False
    return True


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    if actual == expected:
        return True
    if actual.type != expected.type:
        return False
    return expected.index is None and actual.index in (None, 0)


def _is_quanto_tensor(tensor) -> bool:
    module_name = type(tensor).__module__
    return module_name.startswith("optimum.quanto.") and hasattr(tensor, "_data")


def _is_sdnq_tensor(tensor) -> bool:
    return (
        type(tensor).__module__.startswith("sdnq.")
        or type(tensor).__name__ == "SDNQTensor"
        or (hasattr(tensor, "sdnq_dequantizer") and hasattr(tensor, "weight") and hasattr(tensor, "scale"))
    )


def _is_sdnq_module(module: nn.Module) -> bool:
    return type(module).__module__.startswith("sdnq.") or type(module).__name__.startswith("SDNQ")


def _tensor_on_device(tensor, device: torch.device) -> bool:
    if not _same_device(tensor.device, device):
        return False
    if not _is_quanto_tensor(tensor):
        if not _is_sdnq_tensor(tensor):
            return True
        for attr in ("weight", "scale", "zero_point", "svd_up", "svd_down"):
            value = getattr(tensor, attr, None)
            if value is not None and hasattr(value, "device") and not _same_device(value.device, device):
                return False
        return True
    for attr in ("_data", "_scale", "_shift", "_scale_shift"):
        value = getattr(tensor, attr, None)
        if value is None:
            continue
        if _is_quanto_tensor(value):
            if not _tensor_on_device(value, device):
                return False
            continue
        if hasattr(value, "device") and not _same_device(value.device, device):
            return False
    return True


def _module_has_quanto_tensor(module: nn.Module) -> bool:
    return any(_is_quanto_tensor(tensor) for tensor in module.parameters()) or any(
        _is_quanto_tensor(tensor) for tensor in module.buffers()
    )


def _module_has_sdnq_payload(module: nn.Module) -> bool:
    return (
        any(_is_sdnq_module(child) for child in module.modules())
        or any(_is_sdnq_tensor(tensor) for tensor in module.parameters())
        or any(_is_sdnq_tensor(tensor) for tensor in module.buffers())
    )


def _module_has_trainable_local_state(module: nn.Module) -> bool:
    return any(param is not None and param.requires_grad for param in module._parameters.values())


def _module_has_local_quantized_payload(module: nn.Module) -> bool:
    return (
        any(
            param is not None and (_is_quanto_tensor(param) or _is_sdnq_tensor(param))
            for param in module._parameters.values()
        )
        or any(
            buffer is not None and (_is_quanto_tensor(buffer) or _is_sdnq_tensor(buffer))
            for buffer in module._buffers.values()
        )
        or _is_sdnq_module(module)
    )


def _module_has_cpu_h2d_payload(module: nn.Module) -> bool:
    has_frozen_parameter = any(
        parameter.device.type == "cpu"
        and (not parameter.requires_grad or _is_quanto_tensor(parameter) or _is_sdnq_tensor(parameter))
        for parameter in module.parameters()
    )
    has_quantized_buffer = any(
        buffer.device.type == "cpu" and (_is_quanto_tensor(buffer) or _is_sdnq_tensor(buffer)) for buffer in module.buffers()
    )
    return has_frozen_parameter or has_quantized_buffer


def _move_quanto_tensor_to_device(tensor, device: torch.device):
    if not _same_device(tensor.device, device):
        tensor.data = tensor.data.to(device, non_blocking=True)
    for attr in ("_data", "_scale", "_shift", "_scale_shift"):
        value = getattr(tensor, attr, None)
        if value is None:
            continue
        if _is_quanto_tensor(value):
            _move_quanto_tensor_to_device(value, device)
            continue
        if hasattr(value, "device") and not _same_device(value.device, device):
            setattr(tensor, attr, value.to(device, non_blocking=True))


def _move_sdnq_tensor_to_device(tensor, device: torch.device):
    moved = tensor.to(device, non_blocking=True)
    for attr in ("weight", "scale", "zero_point", "svd_up", "svd_down"):
        value = getattr(moved, attr, None)
        if value is None:
            value = getattr(tensor, attr, None)
            if value is not None and hasattr(value, "device") and not _same_device(value.device, device):
                value = value.to(device, non_blocking=True)
        if value is not None and hasattr(value, "device"):
            setattr(tensor, attr, value)
    if not _same_device(tensor.device, device):
        tensor.data = moved.data


def _move_module_without_swapping_quantized_params(module: nn.Module, device: torch.device):
    for child in module.children():
        _move_module_without_swapping_quantized_params(child, device)

    keep_local_trainable_state = (
        device.type == "cpu"
        and not _module_has_local_quantized_payload(module)
        and any(param is not None and param.requires_grad for param in module._parameters.values())
    )

    for key, param in module._parameters.items():
        if param is None:
            continue
        if keep_local_trainable_state and param.requires_grad:
            continue
        if _is_quanto_tensor(param):
            _move_quanto_tensor_to_device(param, device)
        elif _is_sdnq_tensor(param):
            _move_sdnq_tensor_to_device(param, device)
        elif not _same_device(param.device, device):
            param.data = param.data.to(device, non_blocking=True)
        if param.grad is not None and not _same_device(param.grad.device, device):
            param.grad = param.grad.to(device, non_blocking=True)

    for key, buffer in module._buffers.items():
        if buffer is None:
            continue
        if keep_local_trainable_state:
            continue
        if _is_quanto_tensor(buffer):
            _move_quanto_tensor_to_device(buffer, device)
        elif _is_sdnq_tensor(buffer):
            _move_sdnq_tensor_to_device(buffer, device)
        elif not _same_device(buffer.device, device):
            module._buffers[key] = buffer.to(device, non_blocking=True)


def prepare_musubi_model_for_ddp(module: nn.Module, device: torch.device) -> tuple[int, int]:
    """Move trainable params to the rank device and exclude frozen params plus buffers from DDP."""
    target_device = torch.device(device)
    moved_trainable = 0
    for param in module.parameters():
        is_quantized_payload = _is_quanto_tensor(param) or _is_sdnq_tensor(param)
        if not param.requires_grad or is_quantized_payload or _same_device(param.device, target_device):
            continue
        param.data = param.data.to(target_device, non_blocking=True)
        if param.grad is not None:
            param.grad = param.grad.to(target_device, non_blocking=True)
        moved_trainable += 1

    ignored_names = {
        name
        for name, param in module.named_parameters()
        if not param.requires_grad or _is_quanto_tensor(param) or _is_sdnq_tensor(param)
    }
    ignored_names.update(name for name, _buffer in module.named_buffers())
    existing = set(getattr(module, "_ddp_params_and_buffers_to_ignore", set()))
    newly_ignored = ignored_names - existing
    module._ddp_params_and_buffers_to_ignore = existing | ignored_names
    return moved_trainable, len(newly_ignored)


@dataclass
class _TensorTree:
    tensor_type: type
    metadata: Any
    outer_size: tuple[int, ...]
    outer_stride: tuple[int, ...]
    children: tuple[tuple[str, "_TensorTree"], ...]
    leaf_index: Optional[int] = None

    def rebuild(self, leaves: List[torch.Tensor]) -> torch.Tensor:
        if self.leaf_index is not None:
            return leaves[self.leaf_index]
        inner_tensors = {name: child.rebuild(leaves) for name, child in self.children}
        return self.tensor_type.__tensor_unflatten__(
            inner_tensors,
            self.metadata,
            self.outer_size,
            self.outer_stride,
        )


def _flatten_tensor_tree(tensor: torch.Tensor, leaves: List[torch.Tensor]) -> _TensorTree:
    if type(tensor) is torch.Tensor:
        leaf_index = len(leaves)
        leaves.append(tensor)
        return _TensorTree(
            tensor_type=torch.Tensor,
            metadata=None,
            outer_size=tuple(tensor.size()),
            outer_stride=tuple(tensor.stride()),
            children=(),
            leaf_index=leaf_index,
        )

    flatten = getattr(tensor, "__tensor_flatten__", None)
    if flatten is None:
        raise TypeError(
            f"Musubi H2D block swap cannot stream tensor type "
            f"{type(tensor).__module__}.{type(tensor).__name__}: __tensor_flatten__ is unavailable."
        )
    child_names, metadata = flatten()
    children = []
    for child_name in child_names:
        child = getattr(tensor, child_name)
        if not isinstance(child, torch.Tensor):
            raise TypeError(
                f"Musubi H2D block swap expected tensor field {child_name!r} on "
                f"{type(tensor).__module__}.{type(tensor).__name__}."
            )
        children.append((child_name, _flatten_tensor_tree(child, leaves)))
    return _TensorTree(
        tensor_type=type(tensor),
        metadata=metadata,
        outer_size=tuple(tensor.size()),
        outer_stride=tuple(tensor.stride()),
        children=tuple(children),
    )


@dataclass
class _FrozenTensorBinding:
    owner: nn.Module
    name: str
    parameter: Optional[nn.Parameter]
    tree: _TensorTree
    replace_parameter: bool = False

    def bind(self, leaves: List[torch.Tensor]) -> None:
        tensor = self.tree.rebuild(leaves)
        if self.parameter is not None:
            if self.replace_parameter:
                self.owner._parameters[self.name] = tensor
            else:
                self.parameter.data = tensor
        else:
            self.owner._buffers[self.name] = tensor


@dataclass
class _H2DRingSlot:
    flat: torch.Tensor
    leaves: List[torch.Tensor]
    owner_block_id: Optional[int] = None
    reusable_event: Optional[torch.cuda.Event] = None
    load_event: Optional[torch.cuda.Event] = None


@dataclass
class _H2DBlockState:
    block: nn.Module
    cpu_flat: torch.Tensor
    cpu_leaves: List[torch.Tensor]
    bindings: List[_FrozenTensorBinding]
    signature: tuple
    slot: Optional[_H2DRingSlot] = None

    def bind(self, leaves: List[torch.Tensor]) -> None:
        for binding in self.bindings:
            binding.bind(leaves)


def _required_storage_elements(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    if not shape or any(size == 0 for size in shape):
        return 0 if shape else 1
    if any(value < 0 for value in stride):
        raise ValueError("Musubi H2D block swap does not support negative-stride frozen tensors.")
    return 1 + sum((size - 1) * step for size, step in zip(shape, stride))


def _flat_tensor_layout(leaves: List[torch.Tensor]) -> tuple[tuple, int]:
    alignment = 256
    offset = 0
    layout = []
    for leaf in leaves:
        offset = (offset + alignment - 1) // alignment * alignment
        shape = tuple(leaf.size())
        stride = tuple(leaf.stride())
        storage_elements = _required_storage_elements(shape, stride)
        storage_bytes = storage_elements * leaf.element_size()
        layout.append((offset, shape, stride, leaf.dtype, storage_bytes))
        offset += storage_bytes
    return tuple(layout), offset


def _flat_tensor_views(flat: torch.Tensor, layout: tuple) -> List[torch.Tensor]:
    views = []
    for offset, shape, stride, dtype, storage_bytes in layout:
        storage = flat[offset : offset + storage_bytes].view(dtype)
        views.append(storage.as_strided(shape, stride))
    return views


class _StagedH2DCopier:
    """Copies pageable block masters through a small pinned pool on a worker thread."""

    def __init__(self, device: torch.device, staging_count: int):
        self.device = device
        self.device_index = device.index if device.index is not None else torch.cuda.current_device()
        self.staging_count = max(1, staging_count)
        self.copy_stream = torch.cuda.Stream(device=device)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="musubi-h2d")
        self._futures: dict[int, Future] = {}
        self._staging: dict[int, List[torch.Tensor]] = {}
        self._staging_free: dict[int, List[Optional[torch.cuda.Event]]] = {}
        self._next_staging: dict[int, int] = {}

    def _ensure_staging(self, nbytes: int) -> None:
        if nbytes in self._staging:
            return
        self._staging[nbytes] = [
            torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True) for _ in range(self.staging_count)
        ]
        self._staging_free[nbytes] = [None] * self.staging_count
        self._next_staging[nbytes] = 0

    def _copy(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        reusable_event: Optional[torch.cuda.Event],
    ) -> torch.cuda.Event:
        torch.cuda.set_device(self.device_index)
        nbytes = source.numel()
        index = self._next_staging[nbytes]
        self._next_staging[nbytes] = (index + 1) % self.staging_count
        staging = self._staging[nbytes][index]
        staging_free = self._staging_free[nbytes][index]
        if staging_free is not None:
            staging_free.synchronize()

        staging.copy_(source)
        with torch.cuda.stream(self.copy_stream):
            if reusable_event is not None:
                self.copy_stream.wait_event(reusable_event)
            destination.copy_(staging, non_blocking=True)
            complete = self.copy_stream.record_event()
        self._staging_free[nbytes][index] = complete
        return complete

    def submit(
        self,
        key: int,
        destination: torch.Tensor,
        source: torch.Tensor,
        reusable_event: Optional[torch.cuda.Event],
    ) -> None:
        self._ensure_staging(source.numel())
        self._futures[key] = self._worker.submit(self._copy, destination, source, reusable_event)

    def wait(self, key: int) -> torch.cuda.Event:
        return self._futures.pop(key).result()

    def close(self) -> None:
        for future in list(self._futures.values()):
            future.result()
        self._futures.clear()
        self.copy_stream.synchronize()
        self._worker.shutdown(wait=True)


class MusubiBlockSwapManager:
    """
    Streams a subset of transformer blocks between devices to reduce VRAM usage.
    """

    def __init__(self, block_indices: List[int], offload_device: torch.device, logger: logging.Logger):
        self.block_indices = set(block_indices)
        self.offload_device = offload_device
        self._warned_grad = False
        self._warned_device = False
        self._logger = logger
        self._forward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._hooked_block_ids: tuple[int, ...] = ()
        self._backward_hook_device: Optional[torch.device] = None
        self._h2d_block_modes: dict[int, bool] = {}
        self._h2d_block_states: dict[int, _H2DBlockState] = {}
        self._h2d_ring_pools: dict[tuple, List[_H2DRingSlot]] = {}
        self._h2d_copiers: dict[torch.device, _StagedH2DCopier] = {}
        self._h2d_ring_size = 2

    def __del__(self):
        for copier in getattr(self, "_h2d_copiers", {}).values():
            try:
                copier.close()
            except Exception:
                continue

    @classmethod
    def build(
        cls,
        depth: int,
        blocks_to_swap: int,
        swap_device: str,
        logger: logging.Logger,
    ) -> Optional["MusubiBlockSwapManager"]:
        if blocks_to_swap is None or blocks_to_swap == 0:
            return None
        if blocks_to_swap < 0:
            raise ValueError(f"musubi_blocks_to_swap must be non-negative, got {blocks_to_swap}")

        max_swappable_blocks = max(depth - 1, 0)
        if max_swappable_blocks == 0:
            return None

        if blocks_to_swap > max_swappable_blocks:
            logger.warning(
                "Requested musubi_blocks_to_swap=%s but maximum swappable blocks is %s; clamping to %s.",
                blocks_to_swap,
                max_swappable_blocks,
                max_swappable_blocks,
            )
            blocks_to_swap = max_swappable_blocks

        block_indices = list(range(depth - blocks_to_swap, depth))
        try:
            offload_device = torch.device(swap_device)
        except Exception as exc:
            logger.warning("Failed to initialize Musubi block offload; continuing without offload: %s", exc)
            return None

        return cls(block_indices, offload_device, logger)

    def activate(self, blocks: Iterable[nn.Module], compute_device: torch.device, grad_enabled: bool) -> bool:
        if compute_device == self.offload_device:
            return False

        blocks_list = list(blocks)
        self._ensure_lifecycle_hooks(blocks_list, compute_device, grad_enabled)

        self.mark_blocks_for_offload(blocks_list)
        return True

    def is_managed_block(self, index: int) -> bool:
        return index in self.block_indices

    def stream_in(self, block: nn.Module, device: torch.device, checkpointed: Optional[bool] = None):
        block_id = id(block)
        if checkpointed is not None:
            self._h2d_block_modes[block_id] = bool(
                checkpointed
                and device.type == "cuda"
                and self.offload_device.type == "cpu"
                and _module_has_cpu_h2d_payload(block)
            )
        if self._h2d_block_modes.get(block_id, False):
            state = self._h2d_state(block, device)
            self._load_h2d_state(state, device)
            return

        self.move_module(block, device)
        # Verify the move succeeded
        if not _module_on_device(block, device):
            self._logger.error(
                "stream_in failed: block not fully on %s after move. " "Some parameters may still be on wrong device.",
                device,
            )

    def stream_out(self, block: nn.Module):
        state = self._h2d_block_states.get(id(block))
        if self._h2d_block_modes.get(id(block), False) and state is not None:
            self._release_h2d_state(state)
            return
        self.move_module(block, self.offload_device)

    def move_module(self, module: nn.Module, device: torch.device):
        self._move_module(module, device)

    def mark_blocks_for_offload(self, blocks: List[nn.Module]):
        for idx in self.block_indices:
            if idx < 0 or idx >= len(blocks):
                continue
            self._move_module(blocks[idx], self.offload_device)

    def _h2d_state(self, block: nn.Module, device: torch.device) -> _H2DBlockState:
        block_id = id(block)
        existing = self._h2d_block_states.get(block_id)
        if existing is not None:
            return existing

        cpu_leaves: List[torch.Tensor] = []
        bindings: List[_FrozenTensorBinding] = []
        for owner in block.modules():
            for name, parameter in owner._parameters.items():
                is_quantized_payload = parameter is not None and (_is_quanto_tensor(parameter) or _is_sdnq_tensor(parameter))
                if parameter is None or (parameter.requires_grad and not is_quantized_payload):
                    continue
                tree = _flatten_tensor_tree(parameter.detach(), cpu_leaves)
                bindings.append(
                    _FrozenTensorBinding(
                        owner=owner,
                        name=name,
                        parameter=parameter,
                        tree=tree,
                        replace_parameter=is_quantized_payload,
                    )
                )
            for name, buffer in owner._buffers.items():
                if buffer is None or buffer.device.type != "cpu":
                    continue
                tree = _flatten_tensor_tree(buffer, cpu_leaves)
                bindings.append(
                    _FrozenTensorBinding(
                        owner=owner,
                        name=name,
                        parameter=None,
                        tree=tree,
                    )
                )

        if not cpu_leaves:
            raise ValueError("Musubi H2D block swap found no frozen tensor payloads in a checkpointed block.")
        for leaf in cpu_leaves:
            if leaf.device.type != "cpu":
                raise ValueError(f"Musubi H2D CPU master must be on CPU before ring initialization, got {leaf.device}.")

        layout, total_bytes = _flat_tensor_layout(cpu_leaves)
        cpu_flat = torch.empty(total_bytes, dtype=torch.uint8, device="cpu")
        packed_cpu_leaves = _flat_tensor_views(cpu_flat, layout)
        for destination, source in zip(packed_cpu_leaves, cpu_leaves):
            destination.copy_(source)

        signature = (layout, total_bytes)
        state = _H2DBlockState(
            block=block,
            cpu_flat=cpu_flat,
            cpu_leaves=packed_cpu_leaves,
            bindings=bindings,
            signature=signature,
        )
        state.bind(state.cpu_leaves)
        self._h2d_block_states[block_id] = state
        pool_key = (device, signature)
        if pool_key not in self._h2d_ring_pools:
            slots = []
            for _ in range(self._h2d_ring_size):
                flat = torch.empty(total_bytes, dtype=torch.uint8, device=device)
                slots.append(_H2DRingSlot(flat=flat, leaves=_flat_tensor_views(flat, layout)))
            self._h2d_ring_pools[pool_key] = slots
        return state

    def _h2d_copier(self, device: torch.device) -> _StagedH2DCopier:
        copier = self._h2d_copiers.get(device)
        if copier is None:
            copier = _StagedH2DCopier(device, staging_count=self._h2d_ring_size)
            self._h2d_copiers[device] = copier
        return copier

    def _load_h2d_state(self, state: _H2DBlockState, device: torch.device) -> None:
        if state.slot is not None:
            if state.slot.load_event is not None:
                torch.cuda.current_stream(device).wait_event(state.slot.load_event)
            state.bind(state.slot.leaves)
            return

        slots = self._h2d_ring_pools[(device, state.signature)]
        slot = next((candidate for candidate in slots if candidate.owner_block_id is None), None)
        if slot is None:
            raise RuntimeError(
                "Musubi H2D ring has no reusable slot. A checkpointed block retained a slot past its execution."
            )

        copier = self._h2d_copier(device)
        copier.submit(id(state.block), slot.flat, state.cpu_flat, slot.reusable_event)
        slot.load_event = copier.wait(id(state.block))

        slot.owner_block_id = id(state.block)
        state.slot = slot
        state.bind(slot.leaves)
        torch.cuda.current_stream(device).wait_event(slot.load_event)

    def _release_h2d_state(self, state: _H2DBlockState) -> None:
        slot = state.slot
        if slot is None:
            return
        device = slot.leaves[0].device
        slot.reusable_event = torch.cuda.current_stream(device).record_event()
        state.bind(state.cpu_leaves)
        state.slot = None
        slot.owner_block_id = None

    def _clear_lifecycle_hooks(self):
        for handle in self._forward_hooks + self._backward_hooks:
            try:
                handle.remove()
            except Exception:
                continue
        self._forward_hooks.clear()
        self._backward_hooks.clear()
        self._hooked_block_ids = ()
        self._backward_hook_device = None

    def _ensure_lifecycle_hooks(self, blocks: List[nn.Module], compute_device: torch.device, grad_enabled: bool) -> None:
        managed_blocks = [block for idx, block in enumerate(blocks) if self.is_managed_block(idx)]
        managed_block_ids = tuple(id(block) for block in managed_blocks)
        hooks_ready = self._forward_hooks and (not grad_enabled or self._backward_hooks)
        if self._backward_hook_device == compute_device and self._hooked_block_ids == managed_block_ids and hooks_ready:
            return

        self._clear_lifecycle_hooks()
        self._hooked_block_ids = managed_block_ids
        self._backward_hook_device = compute_device

        for block in managed_blocks:

            def _make_forward_pre_hook(owner_block):
                def _pre_hook(_module, _args):
                    self.stream_in(owner_block, compute_device)

                return _pre_hook

            def _make_forward_hook(owner_block):
                def _forward_hook(_module, _args, output):
                    if not self._h2d_block_modes.get(id(owner_block), False):
                        self.stream_out(owner_block)
                    return output

                return _forward_hook

            self._forward_hooks.append(block.register_forward_pre_hook(_make_forward_pre_hook(block)))
            self._forward_hooks.append(block.register_forward_hook(_make_forward_hook(block), always_call=True))

            if not grad_enabled:
                continue

            def _make_pre_hook(owner_block):
                def _pre_hook(_module, _grad_output):
                    self.stream_in(owner_block, compute_device)
                    return None

                return _pre_hook

            def _make_post_hook(owner_block):
                def _post_hook(_module, _grad_input, _grad_output):
                    self.stream_out(owner_block)

                return _post_hook

            # Module-level hooks on the block itself can fire too late for saved
            # tensors inside child ops, so every descendant streams the owning
            # block back in before its own backward work begins.
            for hook_module in block.modules():
                self._backward_hooks.append(hook_module.register_full_backward_pre_hook(_make_pre_hook(block)))
            self._backward_hooks.append(block.register_full_backward_hook(_make_post_hook(block)))

    def _move_module(self, module: nn.Module, device: torch.device):
        if _module_on_device(module, device):
            return
        with torch.no_grad():
            if _module_has_quanto_tensor(module) or _module_has_sdnq_payload(module):
                _move_module_without_swapping_quantized_params(module, device)
            elif device.type == "cpu" and any(_module_has_trainable_local_state(child) for child in module.modules()):
                _move_module_without_swapping_quantized_params(module, device)
            else:
                module.to(device)


def apply_musubi_pretrained_defaults(config, pretrained_load_args: dict) -> dict:
    """
    Inject musubi block swap defaults into pretrained load kwargs for any model
    that supports the Musubi block swapping path.
    """
    args = dict(pretrained_load_args or {})
    blocks = getattr(config, "musubi_blocks_to_swap", 0)
    device = getattr(config, "musubi_block_swap_device", "cpu")
    args.setdefault("musubi_blocks_to_swap", blocks)
    args.setdefault("musubi_block_swap_device", device)
    return args
