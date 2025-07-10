import torch
import torch.nn as nn
from diffusers.models.attention import Attention
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def fuse_projections_with_deletion(self, fuse=True):
    """
    Fuse QKV projections and DELETE the original layers to save memory
    and ensure nothing tries to use them.
    """
    if self.fused_projections:
        return  # Already fused

    device = self.to_q.weight.data.device
    dtype = self.to_q.weight.data.dtype

    if not self.is_cross_attention:
        # Fuse Q, K, V for self-attention
        concatenated_weights = torch.cat(
            [self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data]
        )
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]

        # Create fused layer
        self.to_qkv = nn.Linear(
            in_features, out_features, bias=self.use_bias, device=device, dtype=dtype
        )
        self.to_qkv.weight.copy_(concatenated_weights)

        if self.use_bias:
            concatenated_bias = torch.cat(
                [self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data]
            )
            self.to_qkv.bias.copy_(concatenated_bias)

        # DELETE the original layers
        del self.to_q
        del self.to_k
        del self.to_v

        # Remove from _modules to ensure they're not accessible
        if "to_q" in self._modules:
            del self._modules["to_q"]
        if "to_k" in self._modules:
            del self._modules["to_k"]
        if "to_v" in self._modules:
            del self._modules["to_v"]

    else:
        # For cross-attention, keep to_q separate, only fuse k,v
        concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]

        self.to_kv = nn.Linear(
            in_features, out_features, bias=self.use_bias, device=device, dtype=dtype
        )
        self.to_kv.weight.copy_(concatenated_weights)

        if self.use_bias:
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            self.to_kv.bias.copy_(concatenated_bias)

        # DELETE the original k,v layers
        del self.to_k
        del self.to_v

        if "to_k" in self._modules:
            del self._modules["to_k"]
        if "to_v" in self._modules:
            del self._modules["to_v"]

    # Handle added projections for SD3 and others
    if (
        getattr(self, "add_q_proj", None) is not None
        and getattr(self, "add_k_proj", None) is not None
        and getattr(self, "add_v_proj", None) is not None
    ):
        concatenated_weights = torch.cat(
            [
                self.add_q_proj.weight.data,
                self.add_k_proj.weight.data,
                self.add_v_proj.weight.data,
            ]
        )
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]

        self.to_added_qkv = nn.Linear(
            in_features,
            out_features,
            bias=self.added_proj_bias,
            device=device,
            dtype=dtype,
        )
        self.to_added_qkv.weight.copy_(concatenated_weights)

        if self.added_proj_bias:
            concatenated_bias = torch.cat(
                [
                    self.add_q_proj.bias.data,
                    self.add_k_proj.bias.data,
                    self.add_v_proj.bias.data,
                ]
            )
            self.to_added_qkv.bias.copy_(concatenated_bias)

        # DELETE the original added projection layers
        del self.add_q_proj
        del self.add_k_proj
        del self.add_v_proj

        if "add_q_proj" in self._modules:
            del self._modules["add_q_proj"]
        if "add_k_proj" in self._modules:
            del self._modules["add_k_proj"]
        if "add_v_proj" in self._modules:
            del self._modules["add_v_proj"]

    self.fused_projections = True
    logger.debug(
        f"Fused projections for {self.__class__.__name__} and deleted original layers"
    )


def patch_attention_fusion():
    """Apply the fusion patch that deletes original layers"""
    # Store original if needed
    Attention._original_fuse_projections = Attention.fuse_projections

    # Replace with our version that deletes layers
    Attention.fuse_projections = fuse_projections_with_deletion

    # Make unfuse a no-op since we can't unfuse after deletion
    Attention.unfuse_projections = lambda self: logger.warning(
        "Cannot unfuse projections - original layers have been deleted!"
    )

    logger.info("Patched Attention with aggressive fusion that deletes original layers")


# Alternative: Create properties that raise clear errors
def patch_attention_with_error_properties():
    """After fusion, make accessing deleted layers raise clear errors"""

    original_fuse = Attention.fuse_projections

    def fuse_with_error_properties(self, fuse=True):
        # Call original fusion
        original_fuse(self, fuse)

        if self.fused_projections:
            # Delete original layers
            for attr in [
                "to_q",
                "to_k",
                "to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
            ]:
                if hasattr(self, attr):
                    delattr(self, attr)

            # Add properties that raise helpful errors
            def make_error_property(name):
                def getter(self):
                    raise AttributeError(
                        f"'{name}' no longer exists - QKV projections have been fused. "
                        f"Use 'to_qkv' or 'to_kv' instead."
                    )

                return property(getter)

            # This won't work on already instantiated objects, but shows the idea
            self.__class__.to_q = make_error_property("to_q")
            self.__class__.to_k = make_error_property("to_k")
            self.__class__.to_v = make_error_property("to_v")

    Attention.fuse_projections = fuse_with_error_properties


# Usage:
# Call this once at the start of your training script
patch_attention_fusion()
