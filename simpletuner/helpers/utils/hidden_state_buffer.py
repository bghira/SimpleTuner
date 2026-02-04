from typing import Optional

import torch


class HiddenStateBuffer(dict):
    """
    Lightweight per-forward buffer for capturing intermediate hidden states.

    Acts like a dict while providing a tiny bit of structure for code clarity.
    """

    def pop_layer(self, layer_idx: int):
        """Convenience helper for the common layer_{idx} key naming."""
        return self.pop(f"layer_{int(layer_idx)}", None)

    def get_layer(self, layer_idx: int):
        """Return the stored tensor for the requested layer, or None."""
        return self.get(f"layer_{int(layer_idx)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()


class UNetMidBlockCapture:
    """
    Utility for capturing mid-block hidden states from UNet2DConditionModel.

    UNet models produce convolutional features (B, C, H, W) at the mid-block,
    which is the natural alignment point for U-REPA.

    Usage:
        capture = UNetMidBlockCapture(unet)
        capture.enable()
        output = unet(...)
        mid_features = capture.get_captured()  # (B, C, H, W)
        capture.disable()
    """

    def __init__(self, unet: torch.nn.Module):
        self.unet = unet
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._captured: Optional[torch.Tensor] = None

    def enable(self):
        """Register forward hook on mid_block to capture its output."""
        if self._hook_handle is not None:
            return  # Already enabled

        if not hasattr(self.unet, "mid_block") or self.unet.mid_block is None:
            raise ValueError("UNet does not have a mid_block to capture from")

        def hook_fn(module, input, output):
            self._captured = output.detach()

        self._hook_handle = self.unet.mid_block.register_forward_hook(hook_fn)

    def disable(self):
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._captured = None

    def get_captured(self) -> Optional[torch.Tensor]:
        """Get the captured mid-block features and clear the buffer."""
        captured = self._captured
        self._captured = None
        return captured

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()
