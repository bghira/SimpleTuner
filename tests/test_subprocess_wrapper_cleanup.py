import queue
import unittest
from unittest.mock import MagicMock

try:
    from tests import test_setup
except ModuleNotFoundError:
    import test_setup  # noqa: F401

from simpletuner.simpletuner_sdk.subprocess_wrapper import SubprocessTrainerWrapper


class TestSubprocessWrapperCleanup(unittest.TestCase):
    def test_cleanup_invokes_trainer_cleanup(self):
        """Wrapper cleanup should ask trainer to release GPU resources."""
        wrapper = SubprocessTrainerWrapper(command_queue=queue.Queue(), event_pipe=MagicMock())
        trainer = MagicMock()
        wrapper.trainer = trainer

        wrapper._cleanup()

        trainer.cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
