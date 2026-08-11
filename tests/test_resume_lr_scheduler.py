import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simpletuner.helpers.training.trainer import Trainer
from simpletuner.helpers.utils.checkpoint_manager import CheckpointManager


class ResumeLRSchedulerTests(unittest.TestCase):
    @patch("simpletuner.helpers.training.trainer.AttentionBackendController.on_load_checkpoint")
    def test_constant_with_warmup_resume_restores_configured_lr(self, mock_attention_backend):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir, "checkpoint-100")
            checkpoint_dir.mkdir()
            trainer = object.__new__(Trainer)
            trainer.model = SimpleNamespace(reset_flow_custom_timestep_cursor=Mock())
            trainer.config = SimpleNamespace(
                output_dir=tmpdir,
                resume_from_checkpoint=str(checkpoint_dir),
                total_steps_remaining_at_start=100,
                global_resume_step=1,
                num_train_epochs=1,
                max_train_steps=100,
                musubi_blocks_to_swap=0,
                lr_scheduler="constant_with_warmup",
                learning_rate=0.001,
                is_schedulefree=False,
                overrode_max_train_steps=False,
                strict_epoch_limit=True,
                optimizer="adamw",
                delete_invalid_checkpoints=False,
            )
            trainer.accelerator = Mock(num_processes=1)
            trainer.accelerator.load_state = Mock()
            trainer.accelerator.wait_for_everyone = Mock()
            trainer.state = {"global_step": 0, "first_epoch": 1, "current_epoch": 1}
            trainer.optimizer = Mock(param_groups=[{"lr": 0.001}, {"lr": 0.0002}])
            trainer.distiller = None
            trainer.job_id = "test-job"
            trainer._emit_event = Mock()
            trainer.checkpoint_manager = CheckpointManager(tmpdir)
            scheduler_state = {"base_lrs": [0.25, 0.25], "_last_lr": [0.25, 0.25]}
            lr_scheduler = Mock()
            lr_scheduler.state_dict.return_value = scheduler_state
            trainer.accelerator.load_state.side_effect = lambda _checkpoint: trainer.optimizer.param_groups.__setitem__(
                slice(None),
                [{"lr": 0.25}, {"lr": 0.25}],
            )

            with (
                patch("simpletuner.helpers.training.state_tracker.StateTracker.get_data_backends", return_value={}),
                patch("simpletuner.helpers.training.state_tracker.StateTracker.get_global_step", return_value=100),
                patch("simpletuner.helpers.training.state_tracker.StateTracker.set_global_resume_step"),
                patch("simpletuner.helpers.training.state_tracker.StateTracker.get_training_state", return_value={}),
                patch("simpletuner.helpers.training.state_tracker.StateTracker.get_epoch", return_value=1),
                patch("simpletuner.helpers.training.state_tracker.StateTracker.set_epoch"),
            ):
                trainer.init_resume_checkpoint(lr_scheduler=lr_scheduler)

            self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.001)
            self.assertEqual(trainer.optimizer.param_groups[1]["lr"], 0.0002)
            self.assertEqual(scheduler_state["base_lrs"], [0.001, 0.0002])
            self.assertEqual(scheduler_state["_last_lr"], [0.001, 0.0002])
            trainer.accelerator.load_state.assert_called_once_with(str(checkpoint_dir))
            mock_attention_backend.assert_called_once_with(str(checkpoint_dir))

    @patch("simpletuner.helpers.training.trainer.AttentionBackendController.on_load_checkpoint")
    def test_constant_with_warmup_resume_rebases_progress_for_new_world_size(self, mock_attention_backend):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir, "checkpoint-200")
            checkpoint_dir.mkdir()
            trainer = object.__new__(Trainer)
            trainer.model = SimpleNamespace(reset_flow_custom_timestep_cursor=Mock())
            trainer.config = SimpleNamespace(
                output_dir=tmpdir,
                resume_from_checkpoint=str(checkpoint_dir),
                total_steps_remaining_at_start=25000,
                global_resume_step=1,
                num_train_epochs=1,
                max_train_steps=25000,
                musubi_blocks_to_swap=0,
                lr_scheduler="constant_with_warmup",
                lr_warmup_steps=1000,
                learning_rate=5e-5,
                is_schedulefree=False,
                overrode_max_train_steps=False,
                strict_epoch_limit=True,
                optimizer="adamw",
                delete_invalid_checkpoints=False,
            )
            trainer.accelerator = Mock(num_processes=8)
            trainer.accelerator.load_state = Mock()
            trainer.accelerator.wait_for_everyone = Mock()
            trainer.state = {"global_step": 0, "first_epoch": 1, "current_epoch": 1}
            trainer.optimizer = Mock(param_groups=[{"lr": 0.0, "initial_lr": 5e-5}])
            trainer.distiller = None
            trainer.job_id = "test-job"
            trainer._emit_event = Mock()
            trainer.checkpoint_manager = CheckpointManager(tmpdir)
            scheduler_state = {
                "base_lrs": [5e-5],
                "last_epoch": 800,
                "_step_count": 801,
                "_last_lr": [1e-5],
            }
            lr_scheduler = Mock(split_batches=False)
            lr_scheduler.state_dict.return_value = scheduler_state
            trainer.accelerator.load_state.side_effect = lambda _checkpoint: trainer.optimizer.param_groups.__setitem__(
                slice(None),
                [{"lr": 1e-5, "initial_lr": 5e-5}],
            )

            with (
                patch("simpletuner.helpers.training.state_tracker.StateTracker.get_data_backends", return_value={}),
                patch("simpletuner.helpers.training.state_tracker.StateTracker.get_global_step", return_value=200),
                patch("simpletuner.helpers.training.state_tracker.StateTracker.set_global_resume_step"),
                patch("simpletuner.helpers.training.state_tracker.StateTracker.get_training_state", return_value={}),
                patch("simpletuner.helpers.training.state_tracker.StateTracker.get_epoch", return_value=1),
                patch("simpletuner.helpers.training.state_tracker.StateTracker.set_epoch"),
            ):
                trainer.init_resume_checkpoint(lr_scheduler=lr_scheduler)

            self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 1e-5)
            self.assertEqual(trainer.optimizer.param_groups[0]["initial_lr"], 5e-5)
            self.assertEqual(scheduler_state["base_lrs"], [5e-5])
            self.assertEqual(scheduler_state["last_epoch"], 1600)
            self.assertEqual(scheduler_state["_step_count"], 1601)
            self.assertEqual(scheduler_state["_last_lr"], [1e-5])
            lr_scheduler.load_state_dict.assert_called_once_with(scheduler_state)
            trainer.accelerator.load_state.assert_called_once_with(str(checkpoint_dir))
            mock_attention_backend.assert_called_once_with(str(checkpoint_dir))


if __name__ == "__main__":
    unittest.main()
