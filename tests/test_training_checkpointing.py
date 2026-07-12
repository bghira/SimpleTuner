import unittest
from unittest.mock import Mock, patch

from simpletuner.helpers.training import checkpointing


class TrainingCheckpointingTests(unittest.TestCase):
    def test_compiled_non_reentrant_checkpoint_disables_determinism_check(self):
        function = Mock()

        with (
            patch.dict("os.environ", {"TRAINING_DYNAMO_BACKEND": "inductor"}, clear=False),
            patch("simpletuner.helpers.training.checkpointing.torch.utils.checkpoint.checkpoint") as mock_checkpoint,
        ):
            checkpointing.checkpoint(function, "input", use_reentrant=False)

        mock_checkpoint.assert_called_once_with(function, "input", use_reentrant=False, determinism_check="none")

    def test_eager_non_reentrant_checkpoint_keeps_default_determinism_check(self):
        function = Mock()

        with (
            patch.dict("os.environ", {"TRAINING_DYNAMO_BACKEND": "no"}, clear=False),
            patch("simpletuner.helpers.training.checkpointing.torch.utils.checkpoint.checkpoint") as mock_checkpoint,
        ):
            checkpointing.checkpoint(function, "input", use_reentrant=False)

        mock_checkpoint.assert_called_once_with(function, "input", use_reentrant=False)

    def test_explicit_determinism_check_is_preserved(self):
        function = Mock()

        with (
            patch.dict("os.environ", {"TRAINING_DYNAMO_BACKEND": "inductor"}, clear=False),
            patch("simpletuner.helpers.training.checkpointing.torch.utils.checkpoint.checkpoint") as mock_checkpoint,
        ):
            checkpointing.checkpoint(function, "input", use_reentrant=False, determinism_check="default")

        mock_checkpoint.assert_called_once_with(function, "input", use_reentrant=False, determinism_check="default")


if __name__ == "__main__":
    unittest.main()
