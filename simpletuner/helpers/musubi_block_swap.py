import logging
from typing import Iterable, List, Optional

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
        if not param.requires_grad or _same_device(param.device, target_device):
            continue
        param.data = param.data.to(target_device, non_blocking=True)
        if param.grad is not None:
            param.grad = param.grad.to(target_device, non_blocking=True)
        moved_trainable += 1

    ignored_names = {name for name, param in module.named_parameters() if not param.requires_grad}
    ignored_names.update(name for name, _buffer in module.named_buffers())
    existing = set(getattr(module, "_ddp_params_and_buffers_to_ignore", set()))
    newly_ignored = ignored_names - existing
    module._ddp_params_and_buffers_to_ignore = existing | ignored_names
    return moved_trainable, len(newly_ignored)


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
        self._backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_hook_device: Optional[torch.device] = None

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
        self._ensure_backward_hooks(blocks_list, compute_device, grad_enabled)

        self.mark_blocks_for_offload(blocks_list)
        return True

    def is_managed_block(self, index: int) -> bool:
        return index in self.block_indices

    def stream_in(self, block: nn.Module, device: torch.device):
        self.move_module(block, device)
        # Verify the move succeeded
        if not _module_on_device(block, device):
            self._logger.error(
                "stream_in failed: block not fully on %s after move. " "Some parameters may still be on wrong device.",
                device,
            )

    def stream_out(self, block: nn.Module):
        self.move_module(block, self.offload_device)

    def move_module(self, module: nn.Module, device: torch.device):
        self._move_module(module, device)

    def mark_blocks_for_offload(self, blocks: List[nn.Module]):
        for idx in self.block_indices:
            if idx < 0 or idx >= len(blocks):
                continue
            self._move_module(blocks[idx], self.offload_device)

    def _clear_backward_hooks(self):
        for handle in self._backward_hooks:
            try:
                handle.remove()
            except Exception:
                continue
        self._backward_hooks.clear()
        self._backward_hook_device = None

    def _ensure_backward_hooks(self, blocks: List[nn.Module], compute_device: torch.device, grad_enabled: bool) -> None:
        if not grad_enabled:
            return

        if self._backward_hook_device == compute_device and self._backward_hooks:
            return

        self._clear_backward_hooks()

        for idx, block in enumerate(blocks):
            if not self.is_managed_block(idx):
                continue

            def _make_pre_hook(owner_block):
                def _pre_hook(_module, _grad_output):
                    self.stream_in(owner_block, compute_device)
                    return None

                return _pre_hook

            # Module-level hooks on the block itself can fire too late for saved
            # tensors inside child ops, so every descendant streams the owning
            # block back in before its own backward work begins.
            for hook_module in block.modules():
                self._backward_hooks.append(hook_module.register_full_backward_pre_hook(_make_pre_hook(block)))

        self._backward_hook_device = compute_device

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
