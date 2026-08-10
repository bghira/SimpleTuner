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


def test_special_model_scheduler_label_is_not_replaced_by_generic_flow_scheduler():
    model = SimpleNamespace(
        PREDICTION_TYPE=SimpleNamespace(value="flow_matching"),
        VALIDATION_SCHEDULER_NAME="MiniMaxH3Scheduler",
    )

    assert _validation_scheduler_label(model, _args()) == "MiniMaxH3Scheduler"


def test_anyflow_scheduler_label_describes_native_scheduler_wrapper():
    model = SimpleNamespace(
        PREDICTION_TYPE=SimpleNamespace(value="flow_matching"),
        VALIDATION_SCHEDULER_NAME="MiniMaxH3Scheduler",
    )

    assert _validation_scheduler_label(model, _args(distillation_method="anyflow")) == (
        "AnyFlowValidationScheduler (MiniMaxH3Scheduler)"
    )
