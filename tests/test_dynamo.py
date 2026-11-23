import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import patch

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
    def test_distillation_compile_cache_budget_is_scoped_and_preserves_larger_limits(self):
        from simpletuner.helpers.training.dynamo import dynamo_config_context

        for method in ("anyflow", "assistant_lora"):
            for initial_limit in (8, 64):
                with (
                    self.subTest(method=method, initial_limit=initial_limit),
                    torch._dynamo.config.patch(cache_size_limit=initial_limit),
                ):
                    with dynamo_config_context(SimpleNamespace(distillation_method=method)):
                        self.assertEqual(torch._dynamo.config.cache_size_limit, max(initial_limit, 32))
                    self.assertEqual(torch._dynamo.config.cache_size_limit, initial_limit)
                    with self.assertRaisesRegex(RuntimeError, "training failed"):
                        with dynamo_config_context(SimpleNamespace(distillation_method=method)):
                            self.assertEqual(torch._dynamo.config.cache_size_limit, max(initial_limit, 32))
                            raise RuntimeError("training failed")
                    self.assertEqual(torch._dynamo.config.cache_size_limit, initial_limit)
                    with dynamo_config_context(SimpleNamespace(distillation_method=None)):
                        self.assertEqual(torch._dynamo.config.cache_size_limit, initial_limit)

    def test_regional_checkpoint_replays_dynamo_config_on_a_different_thread(self):
        from simpletuner.helpers.training.dynamo import dynamo_config_context
        from simpletuner.helpers.training.gradient_checkpointing_interval import get_checkpoint_function

        config = SimpleNamespace(
            distillation_method="anyflow",
            dynamo_dynamic=True,
            dynamo_backend="inductor",
            dynamo_use_regional_compilation=True,
        )
        observed = []

        def settings():
            conf = torch._dynamo.config
            return conf.cache_size_limit, conf.capture_dynamic_output_shape_ops, conf.force_parameter_static_shapes

        def block(value):
            observed.append(settings())
            return value.sin().square()

        with ThreadPoolExecutor(max_workers=1) as worker:
            worker_defaults = worker.submit(settings).result()
            value = torch.tensor([0.5], requires_grad=True)
            with dynamo_config_context(config):
                expected = settings()
                loss = get_checkpoint_function()(block, value, use_reentrant=False).sum()
                worker.submit(loss.backward).result()
            self.assertEqual(observed, [expected, expected])
            self.assertEqual(worker.submit(settings).result(), worker_defaults)
        torch.testing.assert_close(value.grad, torch.sin(2 * value.detach()))

    def test_regional_checkpoint_replays_sdnq_allocator_on_a_different_thread(self):
        from torch.utils.checkpoint import checkpoint

        from simpletuner.helpers.training.dynamo import regional_checkpoint_function
        from simpletuner.helpers.training.sdnq_workarounds import _TRITON_ALLOCATOR_ATTR

        default = object()
        allocator = SimpleNamespace(**{_TRITON_ALLOCATOR_ATTR: True})
        allocator_state = ContextVar("sdnq_allocator", default=default)
        allocation = SimpleNamespace(_allocator=allocator_state)
        observed = []

        def block(value):
            observed.append(allocator_state.get())
            return value.sin().square()

        with (
            patch.dict(sys.modules, {"triton.runtime._allocation": allocation}),
            ThreadPoolExecutor(max_workers=1) as worker,
        ):
            self.assertIs(allocator_state.get(), default)
            self.assertIs(regional_checkpoint_function(checkpoint), checkpoint)
            token = allocator_state.set(allocator)
            try:
                value = torch.tensor([0.5], requires_grad=True)
                loss = regional_checkpoint_function(checkpoint)(block, value, use_reentrant=False).sum()
                worker.submit(loss.backward).result()
                self.assertEqual(observed, [allocator, allocator])
            finally:
                allocator_state.reset(token)
            self.assertIs(worker.submit(allocator_state.get).result(), default)
        torch.testing.assert_close(value.grad, torch.sin(2 * value.detach()))

    def test_compiled_checkpoint_boundary_supports_fullgraph(self):
        from simpletuner.helpers.training.dynamo import dynamo_config_context
        from simpletuner.helpers.training.gradient_checkpointing_interval import get_checkpoint_function

        config = SimpleNamespace(distillation_method="anyflow", dynamo_backend="eager", dynamo_use_regional_compilation=True)
        with dynamo_config_context(config):
            checkpoint = get_checkpoint_function()

            @torch.compile(backend="eager", fullgraph=True)
            def run(value):
                return checkpoint(torch.sin, value, use_reentrant=False)

            value = torch.tensor([0.5], requires_grad=True)
            run(value).sum().backward()
            torch.testing.assert_close(value.grad, value.detach().cos())

    def test_activation_checkpointing_downgrades_reduce_overhead_to_default(self):
        from simpletuner.helpers.training import dynamo

        config = SimpleNamespace(
            dynamo_backend="inductor",
            dynamo_mode="reduce-overhead",
            dynamo_use_regional_compilation=True,
            gradient_checkpointing=True,
        )
        with (
            unittest.mock.patch.object(dynamo.logger, "warning") as warning,
            unittest.mock.patch.dict(os.environ, {}, clear=True),
        ):
            self.assertTrue(dynamo.apply_checkpointing_cudagraph_compatibility(config))
            self.assertEqual(os.environ["TRAINING_DYNAMO_MODE"], "default")
            self.assertEqual(os.environ["ACCELERATE_DYNAMO_MODE"], "default")

        self.assertEqual(config.dynamo_mode, "default")
        self.assertTrue(config.dynamo_use_regional_compilation)
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[2], "https://github.com/pytorch/pytorch/issues/154306")

    def test_reduce_overhead_is_preserved_without_activation_checkpointing(self):
        from simpletuner.helpers.training import dynamo

        config = SimpleNamespace(
            dynamo_backend="inductor",
            dynamo_mode="reduce-overhead",
            gradient_checkpointing=False,
        )
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(dynamo.apply_checkpointing_cudagraph_compatibility(config))
            self.assertNotIn("TRAINING_DYNAMO_MODE", os.environ)

        self.assertEqual(config.dynamo_mode, "reduce-overhead")

    def test_inductor_wrapper_defaults_to_cpp(self):
        from simpletuner.helpers.training import dynamo

        inductor_config = SimpleNamespace(cpp_wrapper=False)
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch.object(dynamo.importlib, "import_module", return_value=inductor_config),
        ):
            wrapper = dynamo.configure_inductor_wrapper(SimpleNamespace(dynamo_backend="inductor"))

            self.assertEqual(wrapper, "cpp")
            self.assertEqual(os.environ["TORCHINDUCTOR_CPP_WRAPPER"], "1")
            self.assertTrue(inductor_config.cpp_wrapper)

    def test_inductor_wrapper_ignores_inactive_accelerate_backend(self):
        from simpletuner.helpers.training import dynamo

        inductor_config = SimpleNamespace(cpp_wrapper=False)
        with (
            unittest.mock.patch.dict(os.environ, {"ACCELERATE_DYNAMO_BACKEND": "NO"}, clear=True),
            unittest.mock.patch.object(dynamo.importlib, "import_module", return_value=inductor_config),
        ):
            wrapper = dynamo.configure_inductor_wrapper(SimpleNamespace(dynamo_backend="inductor"))

            self.assertEqual(wrapper, "cpp")
            self.assertEqual(os.environ["TORCHINDUCTOR_CPP_WRAPPER"], "1")
            self.assertTrue(inductor_config.cpp_wrapper)

    def test_inductor_wrapper_supports_legacy_python_mode(self):
        from simpletuner.helpers.training import dynamo

        inductor_config = SimpleNamespace(cpp_wrapper=True)
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch.object(dynamo.importlib, "import_module", return_value=inductor_config),
        ):
            wrapper = dynamo.configure_inductor_wrapper(SimpleNamespace(dynamo_backend="inductor", dynamo_wrapper="python"))

            self.assertEqual(wrapper, "python")
            self.assertEqual(os.environ["TORCHINDUCTOR_CPP_WRAPPER"], "0")
            self.assertFalse(inductor_config.cpp_wrapper)

    def test_inductor_wrapper_rejects_unknown_value(self):
        from simpletuner.helpers.training import dynamo

        with self.assertRaisesRegex(ValueError, "--dynamo_wrapper"):
            dynamo.configure_inductor_wrapper(SimpleNamespace(dynamo_wrapper="unknown"))

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

    def test_configured_inductor_overrides_outer_accelerate_no_backend(self):
        import torch._inductor.config as inductor_config

        from simpletuner.helpers.training import dynamo

        config = SimpleNamespace(dynamo_backend="inductor", dynamo_mode="reduce-overhead")
        with (
            unittest.mock.patch.dict(
                os.environ,
                {
                    "ACCELERATE_DYNAMO_BACKEND": "NO",
                    "ACCELERATE_DYNAMO_MODE": "default",
                },
            ),
            unittest.mock.patch.object(inductor_config.triton, "cudagraphs", False),
            unittest.mock.patch.object(inductor_config.triton, "cudagraph_trees", True),
        ):
            self.assertTrue(dynamo._inductor_cudagraphs_enabled(config))


if __name__ == "__main__":
    unittest.main()
