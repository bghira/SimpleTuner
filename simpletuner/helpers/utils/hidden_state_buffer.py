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
