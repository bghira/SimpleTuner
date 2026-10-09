"""Locally maintained CuTe attention for Ada, Hopper and RTX Blackwell."""

from .api import attention
from .api import automatic_kohaku_fa_scaled_dot_product_attention as automatic_scaled_dot_product_attention
from .api import kohaku_fa_scaled_dot_product_attention as scaled_dot_product_attention

__all__ = ["attention", "scaled_dot_product_attention", "automatic_scaled_dot_product_attention"]
