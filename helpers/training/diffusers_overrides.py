import torch
import torch.nn as nn
from diffusers.models.attention import Attention
import logging

logger = logging.getLogger(__name__)

# Configuration flag - set this based on your needs
PERMANENT_FUSION = True  # Set to False if you need unfuse capability

@torch.no_grad()
def fuse_projections_smart(self, fuse=True, permanent=None):
    """
    Fuse QKV projections with option for permanent (delete originals) or reversible fusion.
    
    Args:
        fuse: Whether to fuse (always True for compatibility)
        permanent: Override for PERMANENT_FUSION setting. If None, uses global setting.
    """
    if self.fused_projections:
        return  # Already fused
    
    # Determine if this should be permanent
    is_permanent = PERMANENT_FUSION if permanent is None else permanent
    
    device = self.to_q.weight.data.device
    dtype = self.to_q.weight.data.dtype

    if not self.is_cross_attention:
        # Fuse Q, K, V for self-attention
        concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]

        # Create fused layer
        self.to_qkv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
        self.to_qkv.weight.copy_(concatenated_weights)
        
        if self.use_bias:
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            self.to_qkv.bias.copy_(concatenated_bias)

        if is_permanent:
            # DELETE the original layers
            del self.to_q
            del self.to_k
            del self.to_v
            
            # Remove from _modules to ensure they're not accessible
            if 'to_q' in self._modules:
                del self._modules['to_q']
            if 'to_k' in self._modules:
                del self._modules['to_k']
            if 'to_v' in self._modules:
                del self._modules['to_v']

    else:
        # For cross-attention, keep to_q separate, only fuse k,v
        concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]

        self.to_kv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
        self.to_kv.weight.copy_(concatenated_weights)
        
        if self.use_bias:
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            self.to_kv.bias.copy_(concatenated_bias)

        if is_permanent:
            # DELETE the original k,v layers
            del self.to_k
            del self.to_v
            
            if 'to_k' in self._modules:
                del self._modules['to_k']
            if 'to_v' in self._modules:
                del self._modules['to_v']

    # Handle added projections for SD3 and others
    if (
        getattr(self, "add_q_proj", None) is not None
        and getattr(self, "add_k_proj", None) is not None
        and getattr(self, "add_v_proj", None) is not None
    ):
        concatenated_weights = torch.cat(
            [self.add_q_proj.weight.data, self.add_k_proj.weight.data, self.add_v_proj.weight.data]
        )
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]

        self.to_added_qkv = nn.Linear(
            in_features, out_features, bias=self.added_proj_bias, device=device, dtype=dtype
        )
        self.to_added_qkv.weight.copy_(concatenated_weights)
        
        if self.added_proj_bias:
            concatenated_bias = torch.cat(
                [self.add_q_proj.bias.data, self.add_k_proj.bias.data, self.add_v_proj.bias.data]
            )
            self.to_added_qkv.bias.copy_(concatenated_bias)

        if is_permanent:
            # DELETE the original added projection layers
            del self.add_q_proj
            del self.add_k_proj
            del self.add_v_proj
            
            if 'add_q_proj' in self._modules:
                del self._modules['add_q_proj']
            if 'add_k_proj' in self._modules:
                del self._modules['add_k_proj']
            if 'add_v_proj' in self._modules:
                del self._modules['add_v_proj']

    self.fused_projections = True
    fusion_type = "permanent" if is_permanent else "reversible"
    logger.debug(f"Fused projections for {self.__class__.__name__} ({fusion_type})")


@torch.no_grad()
def unfuse_projections_smart(self):
    """
    Unfuse the QKV projections back to their individual components.
    Will warn and return if fusion was permanent.
    """
    if not self.fused_projections:
        logger.debug("Projections are not fused, nothing to unfuse")
        return
    
    # Check if layers were deleted (permanent fusion)
    if not hasattr(self, 'to_q') and hasattr(self, 'to_qkv'):
        logger.warning(
            "Cannot unfuse projections - original layers were deleted during permanent fusion! "
            "Set PERMANENT_FUSION=False or use fuse_projections(permanent=False) for reversible fusion."
        )
        return
    
    logger.debug(f"Unfusing projections for {self.__class__.__name__}")
    
    # Handle self-attention unfusing
    if hasattr(self, 'to_qkv'):
        # Get device and dtype from fused layer
        device = self.to_qkv.weight.device
        dtype = self.to_qkv.weight.dtype
        
        # Get the concatenated weights and bias
        concatenated_weights = self.to_qkv.weight.data
        
        # Calculate dimensions
        total_dim = concatenated_weights.shape[0]
        q_dim = self.inner_dim
        k_dim = self.inner_kv_dim
        v_dim = self.inner_kv_dim
        
        # Verify dimensions
        assert total_dim == q_dim + k_dim + v_dim, \
            f"Dimension mismatch: {total_dim} != {q_dim} + {k_dim} + {v_dim}"
        
        # Split the weights
        q_weight = concatenated_weights[:q_dim]
        k_weight = concatenated_weights[q_dim:q_dim + k_dim]
        v_weight = concatenated_weights[q_dim + k_dim:]
        
        # Create individual linear layers
        self.to_q = nn.Linear(self.query_dim, q_dim, bias=self.use_bias, device=device, dtype=dtype)
        self.to_k = nn.Linear(self.cross_attention_dim, k_dim, bias=self.use_bias, device=device, dtype=dtype)
        self.to_v = nn.Linear(self.cross_attention_dim, v_dim, bias=self.use_bias, device=device, dtype=dtype)
        
        # Copy weights
        self.to_q.weight.data.copy_(q_weight)
        self.to_k.weight.data.copy_(k_weight)
        self.to_v.weight.data.copy_(v_weight)
        
        # Handle biases if they exist
        if self.use_bias and hasattr(self.to_qkv, 'bias') and self.to_qkv.bias is not None:
            concatenated_bias = self.to_qkv.bias.data
            q_bias = concatenated_bias[:q_dim]
            k_bias = concatenated_bias[q_dim:q_dim + k_dim]
            v_bias = concatenated_bias[q_dim + k_dim:]
            
            self.to_q.bias.data.copy_(q_bias)
            self.to_k.bias.data.copy_(k_bias)
            self.to_v.bias.data.copy_(v_bias)
        
        # Remove the fused layer
        del self.to_qkv
        if 'to_qkv' in self._modules:
            del self._modules['to_qkv']
            
        logger.debug("Unfused to_qkv -> to_q, to_k, to_v")
    
    # Handle cross-attention unfusing (fused K,V only)
    elif hasattr(self, 'to_kv'):
        # Get device and dtype
        device = self.to_kv.weight.device
        dtype = self.to_kv.weight.dtype
        
        # Get concatenated weights
        concatenated_weights = self.to_kv.weight.data
        
        # Calculate dimensions
        total_dim = concatenated_weights.shape[0]
        k_dim = self.inner_kv_dim
        v_dim = self.inner_kv_dim
        
        assert total_dim == k_dim + v_dim, \
            f"Dimension mismatch for KV: {total_dim} != {k_dim} + {v_dim}"
        
        # Split weights
        k_weight = concatenated_weights[:k_dim]
        v_weight = concatenated_weights[k_dim:]
        
        # Create individual layers
        self.to_k = nn.Linear(self.cross_attention_dim, k_dim, bias=self.use_bias, device=device, dtype=dtype)
        self.to_v = nn.Linear(self.cross_attention_dim, v_dim, bias=self.use_bias, device=device, dtype=dtype)
        
        # Copy weights
        self.to_k.weight.data.copy_(k_weight)
        self.to_v.weight.data.copy_(v_weight)
        
        # Handle biases
        if self.use_bias and hasattr(self.to_kv, 'bias') and self.to_kv.bias is not None:
            concatenated_bias = self.to_kv.bias.data
            k_bias = concatenated_bias[:k_dim]
            v_bias = concatenated_bias[k_dim:]
            
            self.to_k.bias.data.copy_(k_bias)
            self.to_v.bias.data.copy_(v_bias)
        
        # Remove fused layer
        del self.to_kv
        if 'to_kv' in self._modules:
            del self._modules['to_kv']
            
        logger.debug("Unfused to_kv -> to_k, to_v")
    
    # Handle added projections (SD3/Flux style)
    if hasattr(self, 'to_added_qkv'):
        # Get device and dtype
        device = self.to_added_qkv.weight.device
        dtype = self.to_added_qkv.weight.dtype
        
        # Get concatenated weights
        concatenated_weights = self.to_added_qkv.weight.data
        
        # Calculate dimensions
        total_dim = concatenated_weights.shape[0]
        q_dim = self.inner_dim
        k_dim = self.inner_kv_dim
        v_dim = self.inner_kv_dim
        
        assert total_dim == q_dim + k_dim + v_dim, \
            f"Dimension mismatch for added QKV: {total_dim} != {q_dim} + {k_dim} + {v_dim}"
        
        # Split weights
        add_q_weight = concatenated_weights[:q_dim]
        add_k_weight = concatenated_weights[q_dim:q_dim + k_dim]
        add_v_weight = concatenated_weights[q_dim + k_dim:]
        
        # Create individual layers
        self.add_q_proj = nn.Linear(self.added_kv_proj_dim, q_dim, bias=self.added_proj_bias, device=device, dtype=dtype)
        self.add_k_proj = nn.Linear(self.added_kv_proj_dim, k_dim, bias=self.added_proj_bias, device=device, dtype=dtype)
        self.add_v_proj = nn.Linear(self.added_kv_proj_dim, v_dim, bias=self.added_proj_bias, device=device, dtype=dtype)
        
        # Copy weights
        self.add_q_proj.weight.data.copy_(add_q_weight)
        self.add_k_proj.weight.data.copy_(add_k_weight)
        self.add_v_proj.weight.data.copy_(add_v_weight)
        
        # Handle biases
        if self.added_proj_bias and hasattr(self.to_added_qkv, 'bias') and self.to_added_qkv.bias is not None:
            concatenated_bias = self.to_added_qkv.bias.data
            add_q_bias = concatenated_bias[:q_dim]
            add_k_bias = concatenated_bias[q_dim:q_dim + k_dim]
            add_v_bias = concatenated_bias[q_dim + k_dim:]
            
            self.add_q_proj.bias.data.copy_(add_q_bias)
            self.add_k_proj.bias.data.copy_(add_k_bias)
            self.add_v_proj.bias.data.copy_(add_v_bias)
        
        # Remove fused layer
        del self.to_added_qkv
        if 'to_added_qkv' in self._modules:
            del self._modules['to_added_qkv']
            
        logger.debug("Unfused to_added_qkv -> add_q_proj, add_k_proj, add_v_proj")
    
    # Mark as unfused
    self.fused_projections = False
    logger.debug("Unfusing complete")


def patch_attention_flexible():
    """Apply flexible fusion/unfusion patches to Attention class"""
    # Store originals
    Attention._original_fuse_projections = Attention.fuse_projections
    Attention._original_unfuse_projections = getattr(Attention, 'unfuse_projections', None)
    
    # Apply our versions
    Attention.fuse_projections = fuse_projections_smart
    Attention.unfuse_projections = unfuse_projections_smart
    
    logger.info(f"Patched Attention with flexible fusion (permanent={PERMANENT_FUSION})")


# Convenience functions for different use cases
def enable_permanent_fusion():
    """Enable permanent fusion mode globally"""
    global PERMANENT_FUSION
    PERMANENT_FUSION = True
    logger.info("Enabled permanent QKV fusion mode")


def enable_reversible_fusion():
    """Enable reversible fusion mode globally"""
    global PERMANENT_FUSION
    PERMANENT_FUSION = False
    logger.info("Enabled reversible QKV fusion mode")

patch_attention_flexible()