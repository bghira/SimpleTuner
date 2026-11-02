"""Preserve low components of key- and value-gradient operands."""

from .kernels import Attention


def schedule(program, dimension):
    """Apply last, preserving selector guards and shared object identities."""
    if dimension not in (16, 64, 128):
        raise ValueError("Compensated generic weights require D16, D64 or D128.")
    memo = {}

    def visit(owner):
        identity = id(owner)
        if identity in memo:
            return memo[identity]
        memo[identity] = owner
        # Custom subclasses keep their separately qualified ownership/math.
        if type(owner) is Attention and owner.mode == "kv":
            assert owner.centered
            narrow = dimension == 128 and owner.narrow_n
            columns = owner.columns if dimension == 16 and owner.logical_width else 32
            column_warps = 4 if narrow else 2 if dimension == 128 else 1
            replacement = Attention(
                "kv",
                owner.causal,
                owner.masked,
                columns=columns,
                width=max(32, dimension),
                warps=4,
                centered=True,
                vector=owner.vector,
                # Reconstructing explicitly replaces SM120 D64 TMA/double-
                # buffered staging with the qualified generic C32 owner.
                tma=False,
                pipelined=False,
                copy_cache_global=owner.copy_cache_global,
                reload_center=owner.reload_center,
                logical_width=owner.logical_width if dimension == 16 else 0,
                kv_column_warps=column_warps,
                kv_column_statistics=dimension == 128,
                narrow_n=narrow,
                full_visibility=owner.full_visibility,
                full_copies=owner.full_copies,
                packed_statistics=owner.packed_statistics,
            )
            replacement.baseline_precision = False
            replacement.compensated_keys = replacement.compensated_values = True
            replacement.scaled_key_residual = replacement.scaled_value_residual = True
            replacement.combined_key_products = replacement.combined_value_products = column_warps != 2
            memo[identity] = replacement
            return replacement
        for attribute in (
            "keyvalue",
            "fallback",
            "tuned",
            "short",
            "compact",
            "checked",
            "complete",
            "precise",
        ):
            child = getattr(owner, attribute, None)
            if child is not None:
                setattr(owner, attribute, visit(child))
        return owner

    return visit(program)
