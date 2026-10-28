"""Keep batch/head products above CUDA's grid-y limit on the grid-x axis."""

import cutlass
import cutlass.cute as cute


@cute.jit
def launch_grid(tiles, groups, widths=1):
    if cutlass.const_expr(groups > 65535):
        return (tiles * groups, 1, widths)
    return (tiles, groups, widths)


@cute.jit
def block_indices(tiles, groups):
    tile, group, width = cute.arch.block_idx()
    if cutlass.const_expr(groups > 65535):
        tile, group = tile % tiles, tile // tiles
    return tile, group, width
