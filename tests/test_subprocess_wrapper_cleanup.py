import queue
import unittest
from unittest.mock import MagicMock, patch

try:
    from tests import test_setup
except ModuleNotFoundError:
    import test_setup  # noqa: F401

from simpletuner.simpletuner_sdk.subprocess_wrapper import SubprocessTrainerWrapper


class TestSubprocessWrapperCleanup(unittest.TestCase):
    def test_runs_the_supplied_trainer_class(self):
        instances = []

        class CustomTrainer:
            def __init__(self, config):
                self.config = config
                self.ran = False
                self.cleaned = False
                instances.append(self)

            def run(self):
                self.ran = True

            def cleanup(self):
                self.cleaned = True

        events = MagicMock()
        with patch("signal.signal"), patch("simpletuner.helpers.training.trainer.Trainer") as default_trainer:
            wrapper = SubprocessTrainerWrapper(command_queue=queue.Queue(), event_pipe=events)
            wrapper.run_trainer(CustomTrainer, {"model_family": "flux"})
        self.assertEqual(len(instances), 1)
        self.assertTrue(instances[0].ran)
        self.assertTrue(instances[0].cleaned)
        default_trainer.assert_not_called()
        self.assertEqual(instances[0].config, ["--model_family=flux"])
        self.assertEqual(
            [call.args[0]["data"]["status"] for call in events.send.call_args_list], ["starting", "running", "completed"]
        )
        wrapper.command_thread.join(timeout=2)
        self.assertFalse(wrapper.command_thread.is_alive())

    def test_cleanup_invokes_trainer_cleanup(self):
        """Wrapper cleanup should ask trainer to release GPU resources."""
        wrapper = SubprocessTrainerWrapper(command_queue=queue.Queue(), event_pipe=MagicMock())
        trainer = MagicMock()
        wrapper.trainer = trainer

        wrapper._cleanup()

        trainer.cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
