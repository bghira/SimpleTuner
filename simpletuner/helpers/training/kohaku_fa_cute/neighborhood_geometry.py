"""Scalar spatial traversal shared by CPU qualification and CuTe JIT.

These functions contain only integer arithmetic and a static direction choice. Keep
them free of tensor operations: production allocates no spatial mask or index
table. Bounds may conservatively include unused tiles, but never omit an edge.
"""


def spatial_bounds(start, rows, height, width, window_h, window_w, inverse, minimum=min, maximum=max):
    last = minimum(start + rows - 1, height * width - 1)
    y0, y1 = start // width, last // width
    x0 = (start % width) * (y0 == y1)
    x1 = last % width + (width - 1 - last % width) * (y0 != y1)
    # inverse is a Python bool selected by the owner at compilation. Runtime
    # bounds use integer min/max and predicate arithmetic, so this same helper
    # also traces when applied through CuTe's programmatic JIT decorator.
    if inverse:
        # Invert start(q) <= key < start(q) + window. Boundary shifts
        # expand the inverse interval; a forward window about the key is wrong.
        y0 = maximum(y0 - (window_h - 1 - window_h // 2), 0) * (y0 >= window_h)
        y1 = minimum(height, y1 + window_h // 2 + 1 + window_h * (y1 >= height - window_h))
        x0 = maximum(x0 - (window_w - 1 - window_w // 2), 0) * (x0 >= window_w)
        x1 = minimum(width, x1 + window_w // 2 + 1 + window_w * (x1 >= width - window_w))
    else:
        y0 = minimum(maximum(y0 - window_h // 2, 0), height - window_h)
        y1 = minimum(maximum(y1 - window_h // 2, 0), height - window_h) + window_h
        x0 = minimum(maximum(x0 - window_w // 2, 0), width - window_w)
        x1 = minimum(maximum(x1 - window_w // 2, 0), width - window_w) + window_w
    return y0, y1, x0, x1


def spatial_tile(iteration, y0, x0, x1, width, columns, maximum=max):
    # Row alignment varies when width is not divisible by columns. Reserve
    # enough slots for any alignment and predicate unused slots before loading.
    slots = (x1 - x0 + 2 * columns - 2) // columns
    y = y0 + iteration // slots
    first = (y * width + x0) // columns
    previous_stop = ((y - 1) * width + x1 + columns - 1) // columns
    first = maximum(first, previous_stop * (y > y0))
    block = first + iteration % slots
    stop = (y * width + x1 + columns - 1) // columns
    return block, block < stop


def full_row_tile_bounds(y0, y1, width, columns):
    """Contiguous tile interval for a union of complete spatial rows."""
    return y0 * width // columns, (y1 * width + columns - 1) // columns


def normalize_exact_contiguous(selection):
    """Normalize a serialized experiment's forward/query/KV traversal flags."""
    if type(selection) is bool:
        return (selection,) * 3
    try:
        selected = tuple(selection)
    except TypeError as error:
        raise ValueError("Expected a bool or three bool traversal flags.") from error
    if len(selected) != 3 or any(type(value) is not bool for value in selected):
        raise ValueError("Expected a bool or three bool traversal flags.")
    return selected
