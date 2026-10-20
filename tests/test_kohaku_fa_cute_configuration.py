import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


@unittest.skipUnless(importlib.util.find_spec("cutlass") is not None, "Requires the optional CuTe DSL runtime")
class CuTeConfigurationTests(unittest.TestCase):
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
