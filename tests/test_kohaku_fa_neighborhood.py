"""Qualification of vendored 2D neighborhood attention and its autograd contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from simpletuner.helpers.training.kohaku_fa_cute import api, neighborhood_attention
from simpletuner.helpers.training.kohaku_fa_cute.neighborhood_geometry import spatial_bounds, spatial_tile


def neighborhoods(spatial_shape, window_size):
    """Choose the legal window whose anchor is nearest the query on each axis.

    Enumerating every legal placement avoids reusing production's clamp or
    inverse formulas. The anchor is window//2, including even-size windows.
    """
    height, width = spatial_shape
    wh, ww = window_size
    if not 1 <= wh <= height or not 1 <= ww <= width:
        raise ValueError("Window must fit the spatial shape.")
    result = []
    for y in range(height):
        top = min(range(height - wh + 1), key=lambda candidate: abs(candidate + wh // 2 - y))
        for x in range(width):
            left = min(range(width - ww + 1), key=lambda candidate: abs(candidate + ww // 2 - x))
            result.append(tuple(row * width + col for row in range(top, top + wh) for col in range(left, left + ww)))
    return tuple(result)


class NeighborhoodGeometryTests(unittest.TestCase):
    def test_exhaustive_small_forward_and_inverse_tiles(self):
        for height in range(1, 6):
            for width in range(1, 6):
                for wh in range(1, height + 1):
                    for ww in range(1, width + 1):
                        edges = neighborhoods((height, width), (wh, ww))
                        inverse = tuple(tuple(q for q, keys in enumerate(edges) if key in keys) for key in range(len(edges)))
                        for rows in (1, 4, 64):
                            for columns in (8, 32):
                                for backward, adjacency in ((False, edges), (True, inverse)):
                                    for start in range(0, height * width, rows):
                                        bounds = spatial_bounds(start, rows, height, width, wh, ww, backward)
                                        y0, y1, x0, x1 = bounds
                                        slots = (x1 - x0 + 2 * columns - 2) // columns
                                        tiles = [
                                            block
                                            for i in range((y1 - y0) * slots)
                                            for block, selected in (spatial_tile(i, y0, x0, x1, width, columns),)
                                            if selected
                                        ]
                                        expected_tiles = {
                                            (y * width + x) // columns for y in range(y0, y1) for x in range(x0, x1)
                                        }
                                        self.assertEqual(set(tiles), expected_tiles)
                                        self.assertEqual(len(tiles), len(set(tiles)), "Repeated tiles double gradients")
                                        needed = {key for keys in adjacency[start : start + rows] for key in keys}
                                        self.assertTrue({key // columns for key in needed} <= set(tiles))
                                        if rows == 1:
                                            # For a single owner, inverse bounds must be exact.
                                            self.assertEqual(
                                                needed, {y * width + x for y in range(y0, y1) for x in range(x0, x1)}
                                            )

    def test_shifted_boundary_is_asymmetric(self):
        edges = neighborhoods((1, 5), (1, 3))
        self.assertEqual(edges, ((0, 1, 2), (0, 1, 2), (1, 2, 3), (2, 3, 4), (2, 3, 4)))
        self.assertIn(2, edges[0])
        self.assertNotIn(0, edges[2])
        self.assertEqual(spatial_bounds(2, 1, 1, 5, 1, 3, True), (0, 1, 0, 5))

    def test_large_width_skips_distant_spatial_tiles(self):
        for inverse in (False, True):
            bounds = spatial_bounds(128 * 1024 + 256, 64, 256, 1024, 7, 7, inverse)
            y0, y1, x0, x1 = bounds
            slots = (x1 - x0 + 62) // 32
            self.assertLess((y1 - y0) * slots, 40)
            self.assertGreater(256 * 1024 // 32, 8000)

    def test_unaligned_spatial_rows_and_partial_owner_tiles(self):
        for spatial, window in (((17, 131), (7, 5)), ((1, 65), (1, 3)), ((65, 1), (3, 1)), ((11, 33), (4, 6))):
            height, width = spatial
            edges = neighborhoods(spatial, window)
            inverse = [[] for _ in edges]
            for query, keys in enumerate(edges):
                for key in keys:
                    inverse[key].append(query)
            for backward, adjacency in ((False, edges), (True, inverse)):
                for start in range(0, height * width, 64):
                    y0, y1, x0, x1 = spatial_bounds(start, 64, height, width, *window, backward)
                    slots = (x1 - x0 + 62) // 32
                    tiles = [
                        block
                        for i in range((y1 - y0) * slots)
                        for block, selected in (spatial_tile(i, y0, x0, x1, width, 32),)
                        if selected
                    ]
                    expected = {(y * width + x) // 32 for y in range(y0, y1) for x in range(x0, x1)}
                    self.assertEqual(set(tiles), expected)
                    self.assertEqual(len(tiles), len(expected))
                    self.assertTrue({key // 32 for keys in adjacency[start : start + 64] for key in keys} <= expected)


class NeighborhoodAPITests(unittest.TestCase):
    def test_rejects_cpu_and_invalid_geometry(self):
        q = torch.zeros(1, 1, 15, 64, dtype=torch.bfloat16)
        for spatial, window in (((3, 5), 3), ((3, 5), 6), ((0, 5), 1), ((True, 5), 1), ((3, 5), (2, 0))):
            with self.subTest(spatial=spatial, window=window), self.assertRaises(ValueError):
                neighborhood_attention(q, q, q, spatial_shape=spatial, window_size=window)

    def test_explicit_dispatch_and_validation(self):
        from simpletuner.helpers.training.kohaku_fa_cute import neighborhood_ops

        q = SimpleNamespace(ndim=4, shape=(1, 2, 15, 64), device=torch.device("cuda", 0), dtype=torch.bfloat16)
        q.resolve_conj = q.resolve_neg = lambda: q
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
            patch.object(neighborhood_ops, "forward", return_value=(q, None, None)) as forward,
            patch.object(api, "extension", side_effect=AssertionError("Dense dispatch")),
        ):
            self.assertIs(neighborhood_attention(q, q, q, spatial_shape=(3, 5), window_size=3), q)
            forward.assert_called_once_with(q, q, q, 3, 5, 3, 3, 0.125)
            for scale in (float("nan"), float("inf"), 1e50):
                with self.assertRaises(ValueError):
                    neighborhood_attention(q, q, q, spatial_shape=(3, 5), window_size=3, scale=scale)
            for spatial, window in (((3, 6), 3), ((3, 5), True)):
                with self.assertRaises(ValueError):
                    neighborhood_attention(q, q, q, spatial_shape=spatial, window_size=window)

    def test_backward_saves_actual_output_and_geometry(self):
        from simpletuner.helpers.training.kohaku_fa_cute import neighborhood_ops

        inputs = [torch.randn(1, 2, 15, 64, dtype=torch.float16) for _ in range(3)]
        out = torch.zeros_like(inputs[0])
        stats = torch.empty(1, 2, 15)
        ctx = SimpleNamespace(mark_non_differentiable=lambda *args: None, set_materialize_grads=Mock())
        ctx.save_for_backward = lambda *args: setattr(ctx, "saved_tensors", args)
        neighborhood_ops.setup_context(ctx, (*inputs, 3, 5, 2, 3, -0.125), (out, stats, stats))
        ctx.set_materialize_grads.assert_called_once_with(False)
        with patch.object(neighborhood_ops, "backward", return_value=tuple(inputs)) as backward:
            self.assertEqual(len(neighborhood_ops.autograd_backward(ctx, out, None, None)), 8)
        self.assertIs(backward.call_args.args[3], out)
        self.assertEqual(backward.call_args.args[-5:], (3, 5, 2, 3, -0.125))
        with patch.object(neighborhood_ops, "backward", side_effect=AssertionError("Unused output backward")):
            self.assertEqual(neighborhood_ops.autograd_backward(ctx, None, None, None), (None,) * 8)

    def test_fake_and_compiled_fake_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        from simpletuner.helpers.training.kohaku_fa_cute import neighborhood_ops

        with FakeTensorMode():
            q = torch.empty(1, 2, 15, 64, dtype=torch.bfloat16, requires_grad=True)

            def call(q, k, v):
                return neighborhood_ops.forward(q, k, v, 3, 5, 2, 3, 0.125)[0]

            out = call(q, q, q)
            self.assertEqual(out.shape, q.shape)
            self.assertEqual(out.dtype, q.dtype)
            self.assertEqual(torch.autograd.grad(out.sum(), q)[0].shape, q.shape)
            compiled = torch.compile(call, backend="aot_eager", fullgraph=True)
            out = compiled(q, q, q)
            self.assertEqual(torch.autograd.grad(out.sum(), q)[0].shape, q.shape)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and the CuTe runtime")
class NeighborhoodGPUTests(unittest.TestCase):
    def test_outputs_and_gradients_match_enumerated_spatial_attention(self):
        from scripts.benchmark_attention_precision import fp64_reference

        torch.manual_seed(621)
        for dtype in (torch.float16, torch.bfloat16):
            for spatial, window in (((3, 5), (2, 3)), ((1, 5), (1, 3)), ((5, 1), (3, 1)), ((9, 17), (3, 3))):
                for scale in (0.0, -0.125, 0.125):
                    with self.subTest(dtype=dtype, spatial=spatial, window=window, scale=scale):
                        sequence = spatial[0] * spatial[1]
                        inputs = [
                            torch.randn(1, 2, sequence, 128, device="cuda", dtype=dtype)[..., ::2].requires_grad_()
                            for _ in range(3)
                        ]
                        dout = torch.randn_like(inputs[0])
                        out = neighborhood_attention(*inputs, spatial_shape=spatial, window_size=window, scale=scale)
                        actual = out, *torch.autograd.grad(out, inputs, dout)
                        mask = torch.zeros(sequence, sequence, dtype=torch.bool)
                        for query, keys in enumerate(neighborhoods(spatial, window)):
                            mask[query, list(keys)] = True
                        reference = fp64_reference([t.detach().cpu() for t in inputs], dout.cpu(), mask, scale=scale)
                        for got, expected in zip(actual, reference):
                            got = got.detach().cpu().double()
                            self.assertTrue(torch.isfinite(got).all())
                            if expected.norm() == 0:
                                self.assertEqual(got.count_nonzero(), 0)
                            else:
                                self.assertLess(((got - expected).norm() / expected.norm()).item(), 0.01)

    def test_compiled_geometry_changes_match_eager_gradients(self):
        torch.manual_seed(622)
        compiled = torch.compile(neighborhood_attention, fullgraph=True)
        inputs = [torch.randn(1, 2, 15, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        dout = torch.randn_like(inputs[0])
        for spatial, window in (((3, 5), (3, 3)), ((5, 3), (2, 3)), ((3, 5), (3, 3))):
            with self.subTest(spatial=spatial, window=window):
                out = compiled(*inputs, spatial_shape=spatial, window_size=window)
                actual = out, *torch.autograd.grad(out, inputs, dout)
                out = neighborhood_attention(*inputs, spatial_shape=spatial, window_size=window)
                expected = out, *torch.autograd.grad(out, inputs, dout)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, atol=0, rtol=0)
