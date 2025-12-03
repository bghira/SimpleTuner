import unittest
from types import SimpleNamespace

from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.validation import Evaluation


class EvaluationSchedulingTests(unittest.TestCase):
    def setUp(self):
        self._prev_args = StateTracker.get_args()
        self._prev_epoch = StateTracker.get_epoch()
        self._prev_global_step = StateTracker.get_global_step()
        StateTracker.set_epoch(1)
        StateTracker.set_global_step(0)

    def tearDown(self):
        StateTracker.set_args(self._prev_args)
        StateTracker.set_epoch(self._prev_epoch)
        StateTracker.set_global_step(self._prev_global_step)

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
        training_state = {"global_step": 0, "global_resume_step": 0}
        self.assertFalse(evaluator.would_evaluate(training_state))

        training_state["global_step"] = 2
        training_state["global_resume_step"] = 1
        self.assertTrue(evaluator.would_evaluate(training_state))

    def test_would_evaluate_epoch_fractional_interval(self):
        evaluator = self._make_eval(eval_epoch_interval=0.5, num_update_steps_per_epoch=4)
        training_state = {"global_step": 1, "global_resume_step": 0, "current_epoch": 1}
        evaluator.would_evaluate(training_state)  # Prime any internal counters

        training_state["global_step"] = 2
        first_cross = evaluator.would_evaluate(training_state)

        training_state["global_step"] = 4
        second_cross = evaluator.would_evaluate(training_state)

        self.assertTrue(first_cross or second_cross)

    def test_would_evaluate_warns_when_dual_schedule_configured(self):
        evaluator = self._make_eval(eval_steps_interval=2, eval_epoch_interval=1.0, num_update_steps_per_epoch=2)
        training_state = {"global_step": 1, "global_resume_step": 0, "current_epoch": 1}
        self.assertFalse(evaluator._warned_dual_schedule)
        evaluator.would_evaluate(training_state)
        self.assertTrue(evaluator._warned_dual_schedule)


if __name__ == "__main__":
    unittest.main()
