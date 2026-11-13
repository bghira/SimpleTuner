"""Spatial owners with normalized dQ and reconstructed dK corrections by default."""

from functools import lru_cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from .kernels import Attention, JointBackward, copy_vector, descriptor
from .neighborhood_geometry import full_row_tile_bounds, normalize_exact_contiguous, spatial_bounds, spatial_tile
from .neighborhood_schedule import DEFAULT_KEY_STATISTICS, DEFAULT_PRECISION, default_schedule

_bounds = cute.jit(spatial_bounds, preprocess=False)
_tile = cute.jit(spatial_tile, preprocess=False)
_full_rows = cute.jit(full_row_tile_bounds, preprocess=False)

# Explicit benchmark experiments only. A complete forward/backward call must
# run under one selection; changing this between forward and backward is not
# a supported public API. Benchmarks serialize these process-wide selections.
# None selects the bounded production scheduling policy. Arithmetic is identical
# on every device, including devices that retain the original C32 schedule.
_experiment_columns = None
_experiment_exact_contiguous = None
_experiment_precision = DEFAULT_PRECISION
_experiment_key_statistics = DEFAULT_KEY_STATISTICS


def set_experiment_key_statistics(mode):
    """Select an explicit KV statistics load schedule, returning its predecessor."""
    if mode not in ("direct", "column", "staged", "column_staged"):
        raise ValueError("Unknown neighborhood KV statistics schedule.")
    global _experiment_key_statistics
    previous, _experiment_key_statistics = _experiment_key_statistics, mode
    return previous


def set_experiment_precision(name):
    """Override the default method for a serialized benchmark control."""
    if name not in (
        "baseline",
        "fp32_p",
        "weighted_dq",
        "fp32_p_weighted_dq",
        "legacy_weighted",
        "reconstructed_dk",
        "weighted_dq_reconstructed_dk",
    ):
        raise ValueError("Unknown neighborhood precision experiment.")
    global _experiment_precision
    previous, _experiment_precision = _experiment_precision, name
    return previous


def requires_separate_delta():
    """Reconstructed dK owns a second FP32 row buffer, including by default."""
    return _experiment_precision in ("reconstructed_dk", "weighted_dq_reconstructed_dk")


def set_experiment_configuration(columns):
    """Select forward/query/KV tile widths for a serialized experiment.

    Return the previous immutable selection so callers can restore it. This
    chooses scheduling only: all arithmetic and precision helpers are unchanged.
    """
    selected = None if columns is None else tuple(columns)
    if selected is not None and (
        len(selected) != 3 or any(type(value) is not int or value not in (16, 32, 64, 128) for value in selected)
    ):
        raise ValueError("Expected three experimental column widths from 16, 32, 64, or 128.")
    global _experiment_columns
    previous, _experiment_columns = _experiment_columns, selected
    return previous


def set_experiment_exact_contiguous(enabled):
    """Select a traversal-only experiment, returning the previous selection."""
    selected = None if enabled is None else normalize_exact_contiguous(enabled)
    global _experiment_exact_contiguous
    previous, _experiment_exact_contiguous = _experiment_exact_contiguous, selected
    return previous


@lru_cache(maxsize=None)
def _device_schedule_metadata(device):
    properties = torch.cuda.get_device_properties(device)
    return properties.name, (properties.major, properties.minor), properties.multi_processor_count


def effective_schedule(tensors, spatial, window):
    """Resolve concrete cache-key schedules for the actual runtime operands."""
    if _experiment_columns is not None and _experiment_exact_contiguous is not None:
        return _experiment_columns, _experiment_exact_contiguous
    q, k, v = tensors[:3]
    name, capability, sm_count = _device_schedule_metadata(q.device.index)
    columns, traversal = default_schedule(
        device_name=name,
        capability=capability,
        sm_count=sm_count,
        shape=tuple(q.shape),
        spatial=spatial,
        window=window,
        dtype=str(q.dtype).split(".")[-1],
        contiguous=all(tensor.is_contiguous() for tensor in (q, k, v)),
    )
    return (
        columns if _experiment_columns is None else _experiment_columns,
        traversal if _experiment_exact_contiguous is None else _experiment_exact_contiguous,
    )


class NeighborhoodAttention(Attention):
    def __init__(self, mode, spatial, window, vector, columns=32, exact_contiguous=False, key_statistics="direct"):
        # Keep the qualified generic warp MMA and unchanged precision policy.
        # The selected columns, traversal and statistics loads change scheduling only.
        if key_statistics not in ("direct", "column", "staged", "column_staged"):
            raise ValueError("Unknown neighborhood KV statistics schedule.")
        if mode != "kv" and key_statistics != "direct":
            raise ValueError("Neighborhood statistics schedules require KV ownership.")
        super().__init__(
            mode,
            False,
            False,
            columns=columns,
            width=64,
            warps=4,
            centered=mode != "forward",
            vector=vector,
            kv_column_statistics=key_statistics in ("column", "column_staged"),
            stage_statistics=key_statistics in ("staged", "column_staged"),
        )
        self.height, self.image_width = spatial
        self.window_h, self.window_w = window
        # Every owner then covers complete image rows, including the tail. The
        # active tiles are one interval in exactly the original traversal order.
        self.exact_contiguous = exact_contiguous and self.rows % self.image_width == 0

    @cute.jit
    def visible(self, mask, batch, head, row, col, sq, sk, full):
        y = cutlass.min(cutlass.max(row // self.image_width - self.window_h // 2, 0), self.height - self.window_h)
        x = cutlass.min(cutlass.max(row % self.image_width - self.window_w // 2, 0), self.image_width - self.window_w)
        return (
            row < sq
            and col < sk
            and col // self.image_width >= y
            and col // self.image_width < y + self.window_h
            and col % self.image_width >= x
            and col % self.image_width < x + self.window_w
        )

    @cute.jit
    def block_visibility(self, blocks, batch, head, query, key, rows: cutlass.Constexpr, columns: cutlass.Constexpr):
        return True, False

    @cute.jit
    def traversal(self, start, length):
        bounds = _bounds(
            start,
            self.rows,
            self.height,
            self.image_width,
            self.window_h,
            self.window_w,
            self.mode == "kv",
            minimum=cutlass.min,
            maximum=cutlass.max,
        )
        y0, y1, x0, x1 = bounds
        if cutlass.const_expr(self.exact_contiguous):
            first, stop = _full_rows(y0, y1, self.image_width, self.columns)
            return first, stop, bounds
        slots = cute.ceil_div(x1 - x0 + self.columns - 1, self.columns)
        return 0, (y1 - y0) * slots, bounds

    @cute.jit
    def traversal_block(self, iteration, bounds):
        if cutlass.const_expr(self.exact_contiguous):
            return iteration, True
        y0, y1, x0, x1 = bounds
        return _tile(iteration, y0, x0, x1, self.image_width, self.columns, maximum=cutlass.max)


class NeighborhoodBackward(JointBackward):
    def __init__(
        self, spatial, window, vector, base_vector, columns=(32, 32), exact_contiguous=False, key_statistics="direct"
    ):
        query_columns, keyvalue_columns = columns
        query_exact, keyvalue_exact = (exact_contiguous,) * 2 if type(exact_contiguous) is bool else exact_contiguous
        query_configuration = (query_columns, 64, 4, False, True, False, False)
        keyvalue_configuration = (keyvalue_columns, 64, 4, False, True, False, False)
        super().__init__(False, False, query_configuration, keyvalue_configuration, vector, base_vector, True)
        self.query = NeighborhoodAttention("q", spatial, window, vector, query_columns, query_exact)
        self.keyvalue = NeighborhoodAttention(
            "kv", spatial, window, vector, keyvalue_columns, keyvalue_exact, key_statistics
        )


@lru_cache(maxsize=128)
def _compile(
    device,
    mode,
    spatial,
    window,
    descriptors,
    columns,
    exact_contiguous,
    precision=DEFAULT_PRECISION,
    key_statistics=DEFAULT_KEY_STATISTICS,
):
    dtypes = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.bool: cutlass.Boolean,
        torch.int32: cutlass.Int32,
    }
    arguments = [
        make_fake_tensor(
            dtypes[dtype], tuple(shape), stride=tuple(stride), memspace=cute.AddressSpace.gmem, assumed_align=alignment
        )
        for dtype, shape, stride, alignment in descriptors
    ]
    vector = copy_vector(tuple(descriptors[i] for i in (0, 1, 2, 4, 7, 9, 10, 11)))
    if mode == "forward":
        program = NeighborhoodAttention(mode, spatial, window, vector, columns[0], exact_contiguous[0])
    else:
        program = NeighborhoodBackward(
            spatial,
            window,
            vector,
            copy_vector((descriptors[4], descriptors[7])),
            columns[1:],
            exact_contiguous[1:],
            key_statistics,
        )
        reconstructed = precision in ("reconstructed_dk", "weighted_dq_reconstructed_dk")
        fp32_probability = precision in ("fp32_p", "fp32_p_weighted_dq", "legacy_weighted") or reconstructed
        weighted = precision in (
            "weighted_dq",
            "fp32_p_weighted_dq",
            "legacy_weighted",
            "weighted_dq_reconstructed_dk",
        )
        program.query.experiment_reconstruct_center = reconstructed
        program.keyvalue.experiment_reconstruct_center = reconstructed
        program.query.experiment_fp32_probability = fp32_probability
        program.keyvalue.experiment_fp32_probability = fp32_probability
        if weighted:
            # Reuse the qualified dual-product and row-sum ownership. The
            # actual-O center stays separate when reconstruction publishes t.
            program.query.baseline_precision = False
            program.query.experiment_weighted_dq = True
            program.query.experiment_normalized_mean = precision != "legacy_weighted"
    with torch.cuda.device(device):
        return cute.compile(
            program,
            *arguments,
            cutlass.Float32(1),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def launch(mode, tensors, spatial, window, scale):
    columns, traversal = effective_schedule(tensors, spatial, window)
    function = _compile(
        tensors[0].device.index,
        mode,
        spatial,
        window,
        tuple(descriptor(t) for t in tensors),
        columns,
        traversal,
        _experiment_precision,
        _experiment_key_statistics if mode == "backward" else "direct",
    )
    with torch.cuda.device(tensors[0].device):
        function(*tensors, scale)
