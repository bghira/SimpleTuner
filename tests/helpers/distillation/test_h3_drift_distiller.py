import unittest
from types import SimpleNamespace

import torch

import tests.test_stubs  # noqa: F401
from simpletuner.helpers.distillation.factory import DistillerFactory
from simpletuner.helpers.distillation.h3_drift.distiller import H3DriftDistiller
from simpletuner.helpers.models.common import PredictionTypes


class _Adapter:
    def __init__(self):
        self.multiplier = 1.0
        self.calls = []

    def set_multiplier(self, value):
        self.multiplier = float(value)
        self.calls.append(self.multiplier)


class _H3Model:
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    NAME = "MiniMax H3"

    def __init__(self, adapter: _Adapter):
        self.adapter = adapter
        self.raise_when_disabled = False
        self.config = SimpleNamespace(lora_type="lycoris")
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            _lycoris_wrapped_network=adapter,
        )

    def model_predict(self, batch):
        if self.raise_when_disabled and self.adapter.multiplier == 0.0:
            raise RuntimeError("reference pass failed")
        offset = 3.0 if self.adapter.multiplier > 0.0 else 1.0
        video = torch.zeros(2, 1, 2, 2) + offset
        output = {
            "model_prediction": video,
            "hidden_states_buffer": {"layer": torch.ones(1)},
        }
        if batch.get("include_audio", True):
            output["audio_prediction"] = torch.zeros(2, 2, 3, 2) + offset
        else:
            output["audio_prediction"] = None
        return output


class H3DriftDistillerTests(unittest.TestCase):
    def test_computes_reference_prediction_loss_with_adapter_disabled(self):
        adapter = _Adapter()
        model = _H3Model(adapter)
        distiller = H3DriftDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "model_family": "minimaxh3",
                "loss_weight": 0.5,
                "sft_loss_weight": 0.25,
            },
        )
        batch = {"include_audio": True, "audio_latent_mask": torch.tensor([1.0, 0.0])}
        model_output = model.model_predict(batch)

        loss, logs = distiller.compute_distill_loss(batch, model_output, torch.tensor(4.0))

        self.assertAlmostEqual(float(loss), 3.0)
        self.assertAlmostEqual(logs["h3_drift_loss"], 4.0)
        self.assertAlmostEqual(logs["h3_drift_audio_loss"], 4.0)
        self.assertEqual(logs["h3_drift_audio_elements"], 12.0)
        self.assertEqual(adapter.calls, [0.0, 1.0])
        self.assertEqual(adapter.multiplier, 1.0)

    def test_video_only_batch_allows_missing_audio_prediction(self):
        adapter = _Adapter()
        model = _H3Model(adapter)
        distiller = H3DriftDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "model_family": "minimaxh3", "sft_loss_weight": 0.0},
        )
        batch = {"include_audio": False}
        model_output = model.model_predict(batch)

        loss, logs = distiller.compute_distill_loss(batch, model_output, torch.tensor(4.0))

        self.assertAlmostEqual(float(loss), 4.0)
        self.assertEqual(logs["h3_drift_audio_elements"], 0.0)

    def test_reference_pass_failure_reenables_adapter(self):
        adapter = _Adapter()
        model = _H3Model(adapter)
        model.raise_when_disabled = True
        distiller = H3DriftDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "model_family": "minimaxh3"},
        )
        batch = {"include_audio": True}
        model_output = model.model_predict(batch)

        with self.assertRaisesRegex(RuntimeError, "reference pass failed"):
            distiller.compute_distill_loss(batch, model_output, torch.tensor(0.0))

        self.assertEqual(adapter.calls, [0.0, 1.0])
        self.assertEqual(adapter.multiplier, 1.0)

    def test_factory_creates_h3_drift_distiller(self):
        adapter = _Adapter()
        model = _H3Model(adapter)

        distiller = DistillerFactory.create_distiller(
            "h3_drift",
            teacher_model=model,
            noise_scheduler=None,
            config={"distillation_config": {"h3_drift": {"balance": "modality"}}},
            model_type="lora",
            model_family="minimaxh3",
        )

        self.assertIsInstance(distiller, H3DriftDistiller)
        self.assertEqual(distiller.config["balance"], "modality")

    def test_rejects_non_h3_model_family(self):
        adapter = _Adapter()
        model = _H3Model(adapter)
        model.NAME = "Flux"

        with self.assertRaisesRegex(ValueError, "MiniMax-H3"):
            H3DriftDistiller(
                teacher_model=model,
                noise_scheduler=None,
                config={"model_type": "lora", "model_family": "flux"},
            )


if __name__ == "__main__":
    unittest.main()
