import unittest
from types import SimpleNamespace

from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.validation import Evaluation


class EvaluationSchedulingTests(unittest.TestCase):
    def setUp(self):
        self._prev_args = StateTracker.get_args()

    def tearDown(self):
        StateTracker.set_args(self._prev_args)

    def _make_eval(self, **kwargs) -> Evaluation:
        config_defaults = {
            "eval_steps_interval": None,
            "eval_epoch_interval": None,
            "num_update_steps_per_epoch": 4,
            "weight_dtype": None,
        }
        config_defaults.update(kwargs)
        StateTracker.set_args(SimpleNamespace(**config_defaults))
        accelerator = SimpleNamespace(is_main_process=True)
        return Evaluation(accelerator=accelerator)

    def test_would_evaluate_step_interval_triggers_on_multiple(self):
        evaluator = self._make_eval(eval_steps_interval=2)
        training_state = {"global_step": 3, "global_resume_step": 0}
        self.assertFalse(evaluator.would_evaluate(training_state))

        training_state["global_step"] = 4
        self.assertTrue(evaluator.would_evaluate(training_state))

    def test_would_evaluate_epoch_fractional_interval(self):
        evaluator = self._make_eval(eval_epoch_interval=0.5, num_update_steps_per_epoch=4)
        training_state = {"global_step": 1, "global_resume_step": 0, "current_epoch": 1}
        self.assertFalse(evaluator.would_evaluate(training_state))

        training_state["global_step"] = 3
        self.assertTrue(evaluator.would_evaluate(training_state))

    def test_would_evaluate_warns_when_dual_schedule_configured(self):
        evaluator = self._make_eval(eval_steps_interval=2, eval_epoch_interval=1.0, num_update_steps_per_epoch=2)
        training_state = {"global_step": 1, "global_resume_step": 0, "current_epoch": 1}
        with self.assertLogs("Validation", level="WARNING") as captured_logs:
            evaluator.would_evaluate(training_state)
        self.assertTrue(
            any("eval_steps_interval and eval_epoch_interval" in message.lower() for message in captured_logs.output)
        )


if __name__ == "__main__":
    unittest.main()
