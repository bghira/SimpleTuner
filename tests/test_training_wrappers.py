import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.training.wrappers import unwrap_model


class TrainingWrapperTests(unittest.TestCase):
    def test_unwrap_model_removes_ddp_and_compile_wrappers(self):
        leaf = SimpleNamespace()
        compiled = SimpleNamespace(_orig_mod=leaf)
        ddp = SimpleNamespace(module=compiled)

        with patch("simpletuner.helpers.training.wrappers.is_compiled_module", side_effect=lambda model: model is compiled):
            self.assertIs(unwrap_model(None, ddp), leaf)

    def test_unwrap_model_uses_accelerator_before_execution_wrappers(self):
        leaf = SimpleNamespace()
        ddp = SimpleNamespace(module=leaf)
        accelerator = SimpleNamespace(unwrap_model=lambda model, keep_fp32_wrapper=True: ddp)

        self.assertIs(unwrap_model(accelerator, SimpleNamespace()), leaf)

    def test_unwrap_model_does_not_unwrap_peft_model_internals(self):
        peft_model = SimpleNamespace(
            peft_config={"default": object()}, base_model=SimpleNamespace(), model=SimpleNamespace()
        )
        ddp = SimpleNamespace(module=peft_model)

        self.assertIs(unwrap_model(None, ddp), peft_model)

    def test_unwrap_model_keeps_fsdp_wrapper(self):
        fsdp = type("FullyShardedDataParallel", (), {"module": SimpleNamespace()})()

        self.assertIs(unwrap_model(None, fsdp), fsdp)


if __name__ == "__main__":
    unittest.main()
