import importlib.util
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


@unittest.skipUnless(importlib.util.find_spec("cutlass") is not None, "Requires the optional CuTe DSL runtime")
class CuTeConfigurationTests(unittest.TestCase):
    def _compiled_program(self, capability, mode, dimension, dtype, heads=8, kv_heads=None, *, causal=False):
        from simpletuner.helpers.training.kohaku_fa_cute import kernels

        q = torch.empty(1, heads, 512, dimension, dtype=dtype)
        k = torch.empty(1, heads if kv_heads is None else kv_heads, 512, dimension, dtype=dtype)
        placeholder = torch.empty(1, dtype=torch.int32)
        descriptors = tuple(kernels.descriptor(tensor) for tensor in (q, k, k, placeholder, q, q, q, q))
        kernels._compile_kernel.cache_clear()
        try:
            with (
                patch.object(torch.cuda, "get_device_capability", return_value=capability),
                patch.object(torch.cuda, "device", return_value=nullcontext()),
                patch.object(kernels.cute, "compile", side_effect=lambda program, *args, **kwargs: program),
            ):
                return kernels.compile_kernel(0, mode, causal, False, descriptors)
        finally:
            kernels._compile_kernel.cache_clear()

    def test_register_auxiliary_dispatch_requires_b300_bf16_d512_parallel_mha(self):
        from simpletuner.helpers.training.kohaku_fa_cute.blackwell_register_aux import AdaptiveBackward

        for capability, dimension, dtype, heads, kv_heads, expected in (
            ((10, 3), 512, torch.bfloat16, 8, 8, True),
            ((10, 0), 512, torch.bfloat16, 8, 8, False),
            ((10, 3), 512, torch.float16, 8, 8, False),
            ((10, 3), 256, torch.bfloat16, 8, 8, False),
            ((10, 3), 512, torch.bfloat16, 4, 4, False),
            ((10, 3), 512, torch.bfloat16, 8, 4, False),
        ):
            with self.subTest(capability=capability, dimension=dimension, dtype=dtype, heads=heads, kv_heads=kv_heads):
                program = self._compiled_program(capability, "backward", dimension, dtype, heads, kv_heads)
                self.assertEqual(isinstance(program, AdaptiveBackward), expected)

    def test_blackwell_forward_and_backward_receive_head_specific_transfer_widths(self):
        for capability in ((10, 0), (10, 3)):
            for dimension in (128, 256, 512):
                with self.subTest(capability=capability, dimension=dimension):
                    forward = self._compiled_program(capability, "forward", dimension, torch.float16)
                    backward = self._compiled_program(capability, "backward", dimension, torch.float16)
                    self.assertEqual(forward.dc, 128 if dimension == 512 else 64)
                    for worker in (backward.query, backward.keyvalue):
                        self.assertEqual(worker.dc, 128 if dimension == 128 else 64)

    def test_sm120_backward_shares_query_products_and_selects_kv_statistics(self):
        for dtype in (torch.float16, torch.bfloat16):
            for dimension in (64, 128, 256, 512):
                for causal in (False, True):
                    with self.subTest(dtype=dtype, dimension=dimension, causal=causal):
                        program = self._compiled_program((12, 0), "backward", dimension, dtype, causal=causal)
                        self.assertEqual(program.query.column_warps, {256: 4, 512: 2}.get(dimension, 1))
                        self.assertEqual(program.query.width, 256 if dimension in (256, 512) else dimension)
                        self.assertEqual(
                            program.keyvalue.stage_statistics,
                            dimension == 64 and causal or dimension == 256 and not causal,
                        )

    def test_sm120_query_selection_preserves_other_architectures_masks_and_unaligned_layouts(self):
        from simpletuner.helpers.training.kohaku_fa_cute import kernels

        for capability, masked, aligned in (
            ((8, 9), False, True),
            ((9, 0), False, True),
            ((10, 0), False, True),
            ((10, 3), False, True),
            ((12, 0), True, True),
            ((12, 0), False, False),
        ):
            with patch.object(torch.cuda, "get_device_capability", return_value=capability):
                for dimension in (256, 512):
                    with self.subTest(capability=capability, masked=masked, aligned=aligned, dimension=dimension):
                        descriptors = self._descriptors(dimension, 512, masked=masked, aligned=aligned)
                        expected = kernels.kernel_configuration(0, "q", descriptors)
                        self.assertEqual(kernels.backward_query_configuration(0, False, masked, descriptors), expected)

    def _descriptors(self, dimension, sequence, *, masked=False, aligned=True):
        from simpletuner.helpers.training.kohaku_fa_cute.kernels import descriptor

        width = dimension if aligned else dimension + 1
        tensors = [torch.empty(1, 2, sequence, width, dtype=torch.bfloat16)[..., :dimension] for _ in range(3)]
        mask = torch.ones(sequence, sequence, dtype=torch.bool) if masked else torch.empty(1, dtype=torch.int32)
        return tuple(descriptor(tensor) for tensor in (*tensors, mask, *([tensors[0]] * 4)))

    def test_blackwell_pipeline_thresholds_apply_to_forward_and_backward(self):
        from simpletuner.helpers.training.kohaku_fa_cute import kernels

        for capability in ((10, 0), (10, 3)):
            with patch.object(torch.cuda, "get_device_capability", return_value=capability):
                for dimension, threshold in ((64, 1024), (128, 512), (256, 256), (512, 256)):
                    for sequence in (threshold - 1, threshold, threshold + 1):
                        for mode in ("forward", "q", "kv"):
                            with self.subTest(capability=capability, dimension=dimension, sequence=sequence, mode=mode):
                                configuration = kernels.kernel_configuration(0, mode, self._descriptors(dimension, sequence))
                                self.assertEqual(configuration[2] == 12, sequence >= threshold)
                                if sequence >= threshold:
                                    self.assertEqual(configuration[0], 128)
                                self.assertEqual(configuration[4], mode != "forward")

    def test_masks_and_unaligned_layouts_do_not_select_tma_pipelines(self):
        from simpletuner.helpers.training.kohaku_fa_cute import kernels

        for capability in ((10, 0), (10, 3), (12, 0)):
            with patch.object(torch.cuda, "get_device_capability", return_value=capability):
                for masked, aligned in ((True, True), (False, False)):
                    for dimension in (64, 128, 256, 512):
                        for mode in ("forward", "q", "kv"):
                            with self.subTest(
                                capability=capability, masked=masked, aligned=aligned, dimension=dimension, mode=mode
                            ):
                                configuration = kernels.kernel_configuration(
                                    0, mode, self._descriptors(dimension, 1024, masked=masked, aligned=aligned)
                                )
                                self.assertNotEqual(configuration[2], 12)
                                self.assertFalse(configuration[5])
                                self.assertEqual(configuration[4], mode != "forward")

    def test_compiled_autograd_saves_output_for_centered_blackwell_backward(self):
        from simpletuner.helpers.training.kohaku_fa_cute import api

        q = torch.empty(1, 2, 17, 64, dtype=torch.bfloat16)
        output = torch.empty_like(q)
        stats = torch.empty(q.shape[:-1], dtype=torch.float32)
        blocks = torch.empty(1, dtype=torch.int32)
        for capability in ((8, 9), (9, 0), (10, 0), (10, 3), (12, 0)):
            with (
                self.subTest(capability=capability),
                patch.object(torch.cuda, "get_device_capability", return_value=capability),
            ):
                ctx = SimpleNamespace(save_for_backward=Mock(), mark_non_differentiable=Mock())
                api.setup_context(ctx, (q, q, q, None, 0.125, False), (output, stats, stats, blocks))
                self.assertTrue(ctx.centered)
                self.assertIs(ctx.save_for_backward.call_args.args[-1], output)
