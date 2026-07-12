import os
import unittest
from types import SimpleNamespace

import torch


def _autograd_graph_contains(fn, needle: str) -> bool:
    seen = set()
    stack = [fn]
    while stack:
        current = stack.pop()
        if current is None or current in seen:
            continue
        seen.add(current)
        if needle in type(current).__name__:
            return True
        stack.extend(next_fn for next_fn, _ in getattr(current, "next_functions", ()))
    return False


class DynamoCudagraphWorkaroundTests(unittest.TestCase):
    def test_peft_lora_cudagraph_patch_clones_base_result(self):
        from peft import LoraConfig
        from peft.tuners.lora.layer import Linear

        from simpletuner.helpers.training import dynamo

        original_forward = getattr(Linear, "_simpletuner_original_forward", Linear.forward)
        original_patched = dynamo._PEFT_LORA_CUDAGRAPH_PATCHED
        had_original_attr = hasattr(Linear, "_simpletuner_original_forward")
        try:
            Linear.forward = original_forward
            if had_original_attr:
                delattr(Linear, "_simpletuner_original_forward")
            dynamo._PEFT_LORA_CUDAGRAPH_PATCHED = False

            patched = dynamo.patch_peft_lora_for_cudagraphs()

            config = LoraConfig(r=2, lora_alpha=1, target_modules=["linear"])
            layer = Linear(torch.nn.Linear(4, 4, bias=False), "default", config, r=2, lora_alpha=1)
            output = layer(torch.randn(2, 4, requires_grad=True))
        finally:
            Linear.forward = original_forward
            if had_original_attr:
                Linear._simpletuner_original_forward = original_forward
            elif hasattr(Linear, "_simpletuner_original_forward"):
                delattr(Linear, "_simpletuner_original_forward")
            dynamo._PEFT_LORA_CUDAGRAPH_PATCHED = original_patched

        self.assertTrue(patched)
        self.assertTrue(_autograd_graph_contains(output.grad_fn, "CloneBackward"))

    def test_mark_cudagraph_step_begin_patches_peft_when_inductor_cudagraphs_are_enabled(self):
        from peft.tuners.lora.layer import Linear

        from simpletuner.helpers.training import dynamo

        original_forward = getattr(Linear, "_simpletuner_original_forward", Linear.forward)
        original_patched = dynamo._PEFT_LORA_CUDAGRAPH_PATCHED
        had_original_attr = hasattr(Linear, "_simpletuner_original_forward")
        try:
            Linear.forward = original_forward
            if had_original_attr:
                delattr(Linear, "_simpletuner_original_forward")
            dynamo._PEFT_LORA_CUDAGRAPH_PATCHED = False

            with unittest.mock.patch.object(dynamo, "_inductor_cudagraphs_enabled", return_value=True):
                dynamo.mark_cudagraph_step_begin(SimpleNamespace(dynamo_backend=None))
            patched_attr_present = hasattr(Linear, "_simpletuner_original_forward")
        finally:
            Linear.forward = original_forward
            if had_original_attr:
                Linear._simpletuner_original_forward = original_forward
            elif hasattr(Linear, "_simpletuner_original_forward"):
                delattr(Linear, "_simpletuner_original_forward")
            dynamo._PEFT_LORA_CUDAGRAPH_PATCHED = original_patched

        self.assertTrue(patched_attr_present)

    def test_mark_cudagraph_step_begin_patches_peft_te_lora_when_available(self):
        from peft.tuners.lora.te import TeLinear

        from simpletuner.helpers.training import dynamo

        original_forward = getattr(TeLinear, "_simpletuner_original_forward", TeLinear.forward)
        original_patched = dynamo._PEFT_TE_LORA_CUDAGRAPH_PATCHED
        had_original_attr = hasattr(TeLinear, "_simpletuner_original_forward")
        try:
            TeLinear.forward = original_forward
            if had_original_attr:
                delattr(TeLinear, "_simpletuner_original_forward")
            dynamo._PEFT_TE_LORA_CUDAGRAPH_PATCHED = False

            with unittest.mock.patch.object(dynamo, "_inductor_cudagraphs_enabled", return_value=True):
                dynamo.mark_cudagraph_step_begin(SimpleNamespace(dynamo_backend=None))
            patched_attr_present = hasattr(TeLinear, "_simpletuner_original_forward")
        finally:
            TeLinear.forward = original_forward
            if had_original_attr:
                TeLinear._simpletuner_original_forward = original_forward
            elif hasattr(TeLinear, "_simpletuner_original_forward"):
                delattr(TeLinear, "_simpletuner_original_forward")
            dynamo._PEFT_TE_LORA_CUDAGRAPH_PATCHED = original_patched

        self.assertTrue(patched_attr_present)

    def test_inductor_cudagraph_tree_mode_enabled_from_accelerate_env(self):
        import torch._inductor.config as inductor_config

        from simpletuner.helpers.training import dynamo

        with (
            unittest.mock.patch.dict(
                os.environ,
                {
                    "ACCELERATE_DYNAMO_BACKEND": "INDUCTOR",
                    "ACCELERATE_DYNAMO_MODE": "reduce-overhead",
                },
            ),
            unittest.mock.patch.object(inductor_config.triton, "cudagraphs", False),
            unittest.mock.patch.object(inductor_config.triton, "cudagraph_trees", True),
        ):
            self.assertTrue(dynamo._inductor_cudagraphs_enabled(SimpleNamespace(dynamo_backend=None)))


if __name__ == "__main__":
    unittest.main()
