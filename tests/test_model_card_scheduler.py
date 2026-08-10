import unittest
from types import SimpleNamespace

from simpletuner.helpers.publishing.metadata import _validation_scheduler_label


def _args(**overrides):
    values = {
        "validation_noise_scheduler": "euler",
        "distillation_method": None,
        "inner_distillation_method": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ModelCardSchedulerTests(unittest.TestCase):
    def test_special_model_scheduler_label_is_not_replaced_by_generic_flow_scheduler(self):
        model = SimpleNamespace(
            PREDICTION_TYPE=SimpleNamespace(value="flow_matching"),
            VALIDATION_SCHEDULER_NAME="MiniMaxH3Scheduler",
        )

        self.assertEqual("MiniMaxH3Scheduler", _validation_scheduler_label(model, _args()))

    def test_anyflow_scheduler_label_describes_native_scheduler_wrapper(self):
        model = SimpleNamespace(
            PREDICTION_TYPE=SimpleNamespace(value="flow_matching"),
            VALIDATION_SCHEDULER_NAME="MiniMaxH3Scheduler",
        )

        self.assertEqual(
            "AnyFlowValidationScheduler (MiniMaxH3Scheduler)",
            _validation_scheduler_label(model, _args(distillation_method="anyflow")),
        )


if __name__ == "__main__":
    unittest.main()
