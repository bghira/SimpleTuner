import unittest
from types import MethodType

import torch
import torch._dynamo
from torch import nn

from accelerate.utils.operations import convert_outputs_to_fp32
from accelerate.utils.other import compile_regions

from simpletuner.helpers.training.wrappers import rebind_prepared_forward

OptimizedModule = torch._dynamo.eval_frame.OptimizedModule


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(4))

    def forward(self, x):
        return self.linear(x)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.gradient_checkpointing = False
        self.executed_by = []

    def forward(self, x):
        self.executed_by.append(self)
        if self.gradient_checkpointing:
            x = x * 2
        for block in self.blocks:
            x = block(x)
        return x


def _install_mixed_precision_forward(model):
    autocast_context = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    model._original_forward = model.forward
    model_forward_func = model.forward.__func__
    new_forward = autocast_context(model_forward_func)
    model.forward = MethodType(new_forward, model)
    model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
    return model


def _prepare_like_accelerate(backend="eager"):
    original = _Tiny()
    _install_mixed_precision_forward(original)
    twin = compile_regions(original, backend=backend)
    return original, twin


class RebindPreparedForwardTests(unittest.TestCase):
    def test_compile_regions_twin_inherits_forward_bound_to_the_original(self):
        original, twin = _prepare_like_accelerate()

        self.assertIsNot(twin, original)
        self.assertIsInstance(twin.blocks[0], OptimizedModule)
        self.assertNotIsInstance(original.blocks[0], OptimizedModule)
        self.assertIs(twin.__dict__["forward"].__self__, original)
        self.assertIs(twin.__dict__["_original_forward"].__self__, original)

        twin.gradient_checkpointing = True
        x = torch.ones(1, 4)
        out = twin(x)

        self.assertIs(twin.executed_by[-1], original)
        self.assertFalse(original.gradient_checkpointing)
        torch.testing.assert_close(out, x)

    def test_rebind_makes_the_twin_the_executing_module(self):
        original, twin = _prepare_like_accelerate()

        returned = rebind_prepared_forward(twin, original)

        self.assertIs(returned, twin)
        self.assertIs(twin.__dict__["forward"].__self__, twin)
        self.assertIs(twin.__dict__["_original_forward"].__self__, twin)
        self.assertIs(original.__dict__["forward"].__self__, original)

        twin.gradient_checkpointing = True
        x = torch.ones(1, 4)
        out = twin(x)

        self.assertIs(twin.executed_by[-1], twin)
        self.assertIsInstance(twin.executed_by[-1].blocks[0], OptimizedModule)
        torch.testing.assert_close(out, x * 2)

    def test_rebind_applies_with_the_inductor_backend(self):
        original, twin = _prepare_like_accelerate(backend="inductor")

        self.assertIs(twin.__dict__["forward"].__self__, original)
        rebind_prepared_forward(twin, original)
        self.assertIs(twin.__dict__["forward"].__self__, twin)

    def test_noop_when_prepared_is_original(self):
        original = _install_mixed_precision_forward(_Tiny())
        before = original.__dict__["forward"]

        self.assertIs(rebind_prepared_forward(original, original), original)
        self.assertIs(original.__dict__["forward"], before)
        self.assertIs(original.__dict__["forward"].__self__, original)

    def test_noop_when_original_is_none(self):
        original = _install_mixed_precision_forward(_Tiny())
        twin = compile_regions(original, backend="eager")
        before = twin.__dict__["forward"]

        self.assertIs(rebind_prepared_forward(twin, None), twin)
        self.assertIs(twin.__dict__["forward"], before)
        self.assertIs(twin.__dict__["forward"].__self__, original)

    def test_noop_when_accelerate_already_rebound_the_twin(self):
        original, twin = _prepare_like_accelerate()
        twin.forward = MethodType(twin.__dict__["forward"].__func__, twin)
        twin._original_forward = MethodType(twin.__dict__["_original_forward"].__func__, twin)
        before_forward = twin.__dict__["forward"]
        before_original_forward = twin.__dict__["_original_forward"]

        self.assertIs(rebind_prepared_forward(twin, original), twin)
        self.assertIs(twin.__dict__["forward"], before_forward)
        self.assertIs(twin.__dict__["_original_forward"], before_original_forward)
        self.assertIs(twin.__dict__["forward"].__self__, twin)
        self.assertIs(original.__dict__["forward"].__self__, original)

    def test_noop_without_an_instance_forward(self):
        original = _Tiny()
        twin = compile_regions(original, backend="eager")

        self.assertNotIn("forward", twin.__dict__)
        rebind_prepared_forward(twin, original)
        self.assertNotIn("forward", twin.__dict__)
        self.assertNotIn("_original_forward", twin.__dict__)

    def test_noop_for_plain_torch_compile_optimized_module(self):
        original = _install_mixed_precision_forward(_Tiny())
        compiled = torch.compile(original, backend="eager")
        before = compiled.__dict__.get("forward")

        self.assertIsInstance(compiled, OptimizedModule)
        self.assertIs(rebind_prepared_forward(compiled, original), compiled)
        self.assertIs(compiled.__dict__.get("forward"), before)
        self.assertIs(original.__dict__["forward"].__self__, original)

    def test_noop_for_a_module_wrapper_with_dot_module(self):
        original = _install_mixed_precision_forward(_Tiny())

        class _Wrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        wrapper = _Wrapper(original)

        self.assertNotIn("forward", wrapper.__dict__)
        self.assertIs(rebind_prepared_forward(wrapper, original), wrapper)
        self.assertNotIn("forward", wrapper.__dict__)
        self.assertIs(original.__dict__["forward"].__self__, original)


if __name__ == "__main__":
    unittest.main()
