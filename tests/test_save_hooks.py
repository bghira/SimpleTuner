import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from accelerate.utils import DistributedType

from simpletuner.helpers.training.save_hooks import SaveHookManager


class SaveHookManagerTests(unittest.TestCase):
    def test_ramtorch_prefetch_orders_are_saved_only_on_main_process(self):
        output_dir = "/tmp/checkpoint"

        with (
            patch("simpletuner.helpers.training.save_hooks.StateTracker.save_training_state"),
            patch("simpletuner.helpers.training.save_hooks.StateTracker.save_ramtorch_prefetch_orders") as save_orders,
        ):
            for is_main_process in (True, False):
                with self.subTest(is_main_process=is_main_process):
                    save_orders.reset_mock()
                    manager = object.__new__(SaveHookManager)
                    manager.accelerator = SimpleNamespace(
                        distributed_type=DistributedType.NO,
                        is_main_process=is_main_process,
                    )
                    manager.args = SimpleNamespace(model_type="full")
                    manager.training_state_path = "training_state.json"
                    manager._offload_models_during_save = Mock(return_value=nullcontext())
                    manager._save_ema_state = Mock()
                    manager._is_fsdp2 = Mock(return_value=False)
                    manager._save_full_model = Mock()

                    manager.save_model_hook([], [], output_dir)

                    self.assertEqual(save_orders.call_count, int(is_main_process))


if __name__ == "__main__":
    unittest.main()
