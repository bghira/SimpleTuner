import logging
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

__all__ = ["MusubiBlockSwapManager", "apply_musubi_pretrained_defaults"]


def _module_on_device(module: nn.Module, device: torch.device) -> bool:
    target = torch.device(device)
    for tensor in module.parameters():
        if tensor.device != target:
            return False
    for tensor in module.buffers():
        if tensor.device != target:
            return False
    return True


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
        self._move_module(block, device)
        # Verify the move succeeded
        if not _module_on_device(block, device):
            self._logger.error(
                "stream_in failed: block not fully on %s after move. " "Some parameters may still be on wrong device.",
                device,
            )

    def stream_out(self, block: nn.Module):
        self._move_module(block, self.offload_device)

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

            def _make_pre_hook():
                def _pre_hook(_module, _grad_output):
                    self.stream_in(_module, compute_device)
                    return None

                return _pre_hook

            def _make_post_hook():
                def _post_hook(_module, _grad_input, _grad_output):
                    self.stream_out(_module)
                    return None

                return _post_hook

            self._backward_hooks.append(block.register_full_backward_pre_hook(_make_pre_hook()))
            self._backward_hooks.append(block.register_full_backward_hook(_make_post_hook()))

        self._backward_hook_device = compute_device

    def _move_module(self, module: nn.Module, device: torch.device):
        if _module_on_device(module, device):
            return
        with torch.no_grad():
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
