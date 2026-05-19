import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import tests.test_stubs  # noqa: F401
from simpletuner.helpers.distillation.factory import DistillerFactory
from simpletuner.helpers.distillation.flow_dpo.distiller import FlowDPODistiller
from simpletuner.helpers.models.common import PredictionTypes


class _Adapter:
    def __init__(self):
        self.multiplier = 1.0
        self.calls = []

    def set_multiplier(self, value):
        self.multiplier = float(value)
        self.calls.append(self.multiplier)


class _FlowModel:
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING

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
        target = batch["noise"] - batch["latents"]
        rejected = bool(torch.mean(batch["latents"]).item() > 0.5)
        if self.adapter.multiplier > 0.0:
            offset = 0.40 if rejected else 0.10
        else:
            offset = 0.10 if rejected else 0.30
        return {"model_prediction": target + offset * torch.ones_like(target)}


class _ConfigCaptureDistiller:
    last_config = None

    def __init__(self, teacher_model, student_model=None, *, noise_scheduler=None, config=None):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.noise_scheduler = noise_scheduler
        self.config = config or {}
        _ConfigCaptureDistiller.last_config = self.config


def _prepared_batch():
    latents = torch.zeros(2, 1, 2, 2)
    rejected_latents = torch.ones_like(latents)
    noise = torch.full_like(latents, 2.0)
    sigmas = torch.full((2, 1, 1, 1), 0.5)
    input_noise = noise.clone()
    return {
        "latents": latents,
        "conditioning_latents": rejected_latents,
        "conditioning_type": "reference_strict",
        "noise": noise,
        "input_noise": input_noise,
        "sigmas": sigmas,
        "timesteps": torch.full((2,), 500.0),
        "noisy_latents": (1 - sigmas) * latents + sigmas * input_noise,
    }


class FlowDPODistillerTests(unittest.TestCase):
    def test_computes_dpo_loss_with_adapter_reference_pass(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "auto_beta": False, "beta": 1.0},
        )
        batch = _prepared_batch()
        model_output = model.model_predict(batch)

        loss, logs = distiller.compute_distill_loss(batch, model_output, torch.tensor(0.0))

        self.assertGreater(float(loss), 0.0)
        self.assertGreater(logs["flow_dpo_margin"], 0.0)
        self.assertIn("flow_dpo_gradient_factor", logs)
        self.assertEqual(adapter.calls, [0.0, 1.0])
        self.assertEqual(adapter.multiplier, 1.0)

    def test_reference_pass_failure_reenables_adapter(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        model.raise_when_disabled = True
        distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "auto_beta": False},
        )
        batch = _prepared_batch()
        model_output = model.model_predict(batch)

        with self.assertRaisesRegex(RuntimeError, "reference pass failed"):
            distiller.compute_distill_loss(batch, model_output, torch.tensor(0.0))

        self.assertEqual(adapter.calls, [0.0, 1.0])
        self.assertEqual(adapter.multiplier, 1.0)

    def test_factory_creates_flow_dpo_distiller(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)

        distiller = DistillerFactory.create_distiller(
            "flow_dpo",
            teacher_model=model,
            noise_scheduler=None,
            config={"distillation_config": {"flow_dpo": {"auto_beta": False}}},
            model_type="lora",
        )

        self.assertIsInstance(distiller, FlowDPODistiller)
        self.assertFalse(distiller.config["auto_beta"])

    def test_registered_distiller_does_not_receive_runtime_defaults_unless_requested(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)

        with patch(
            "simpletuner.helpers.distillation.factory.DistillationRegistry.get", return_value=_ConfigCaptureDistiller
        ):
            DistillerFactory._create_registered_distiller(
                registry_key="perflow",
                teacher_model=model,
                noise_scheduler=None,
                distill_config={"loss_weight": 2.0},
            )

        self.assertEqual(_ConfigCaptureDistiller.last_config, {"loss_weight": 2.0})

    def test_registered_distiller_applies_explicit_runtime_defaults(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)

        with patch(
            "simpletuner.helpers.distillation.factory.DistillationRegistry.get", return_value=_ConfigCaptureDistiller
        ):
            DistillerFactory._create_registered_distiller(
                registry_key="flow_dpo",
                teacher_model=model,
                noise_scheduler=None,
                distill_config={"loss_weight": 2.0},
                runtime_config_defaults={"model_type": "lora"},
            )

        self.assertEqual(_ConfigCaptureDistiller.last_config, {"loss_weight": 2.0, "model_type": "lora"})

    def test_requires_low_rank_training(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)

        with self.assertRaisesRegex(ValueError, "only supports low-rank"):
            FlowDPODistiller(
                teacher_model=model,
                noise_scheduler=None,
                config={"model_type": "full"},
            )

    def test_requires_reference_strict_conditioning(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "auto_beta": False},
        )
        batch = _prepared_batch()
        batch["conditioning_type"] = "reference_loose"
        model_output = model.model_predict(batch)

        with self.assertRaisesRegex(ValueError, "conditioning_type=reference_strict"):
            distiller.compute_distill_loss(batch, model_output, torch.tensor(0.0))

    def test_empty_conditioning_latents_reports_missing_rejected_dataset(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "auto_beta": False},
        )
        batch = _prepared_batch()
        batch["conditioning_latents"] = []
        model_output = model.model_predict(batch)

        with self.assertRaisesRegex(ValueError, "conditioning_latents from the rejected-sample dataset"):
            distiller.compute_distill_loss(batch, model_output, torch.tensor(0.0))

    def test_rejected_latent_shape_must_match_preferred_latents(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "auto_beta": False},
        )
        batch = _prepared_batch()
        batch["conditioning_latents"] = torch.ones(2, 1, 3, 2)
        model_output = model.model_predict(batch)

        with self.assertRaisesRegex(ValueError, "must match preferred latents"):
            distiller.compute_distill_loss(batch, model_output, torch.tensor(0.0))

    def test_selects_reference_strict_latents_from_list(self):
        reference_latents = torch.ones(2, 1, 2, 2)
        mask_placeholder = torch.zeros_like(reference_latents)
        batch = _prepared_batch()
        batch["conditioning_latents"] = [mask_placeholder, reference_latents]
        batch["conditioning_latents_type"] = ["mask", "reference_strict"]

        selected = FlowDPODistiller._conditioning_latents(batch)

        self.assertTrue(torch.equal(selected, reference_latents))

    def test_rejected_batch_uses_input_noise_for_noisy_latents(self):
        batch = _prepared_batch()
        batch["noise"] = torch.full_like(batch["latents"], 2.0)
        batch["input_noise"] = torch.full_like(batch["latents"], 4.0)
        rejected_latents = torch.ones_like(batch["latents"])

        rejected_batch = FlowDPODistiller._build_rejected_batch(batch, rejected_latents)

        expected = (1 - batch["sigmas"]) * rejected_latents + batch["sigmas"] * batch["input_noise"]
        torch.testing.assert_close(rejected_batch["noisy_latents"], expected)

    def test_mask_for_loss_handles_video_mask_and_dilation(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "mask_dilate": 1},
        )
        prediction = torch.zeros(1, 1, 2, 4, 4)
        mask_image = torch.full((1, 1, 4, 4), -1.0)
        mask_image[:, :, 1, 1] = 1.0
        batch = {"loss_mask_type": "mask", "conditioning_pixel_values": mask_image}

        mask = distiller._mask_for_loss(batch, prediction)

        self.assertEqual(mask.shape, prediction.shape)
        self.assertTrue(torch.equal(mask[:, :, 0], mask[:, :, 1]))
        self.assertEqual(float(mask[0, 0, 0].sum()), 9.0)

    def test_mask_for_loss_reduces_segmentation_channels(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora"},
        )
        prediction = torch.zeros(1, 1, 2, 2)
        mask_image = torch.full((1, 2, 2, 2), -1.0)
        mask_image[:, 1, 0, 0] = 1.0
        batch = {"loss_mask_type": "segmentation", "conditioning_pixel_values": mask_image}

        mask = distiller._mask_for_loss(batch, prediction)

        self.assertEqual(mask.shape, prediction.shape)
        self.assertEqual(float(mask.sum()), 1.0)

    def test_masked_mean_counts_video_temporal_axis(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "norm_type": "masked_mean"},
        )
        prediction = torch.ones(1, 2, 3, 2, 2)
        target = torch.zeros_like(prediction)
        mask = torch.ones(1, 1, 1, 2, 2)

        sample_error = distiller._per_sample_error(prediction, target, mask)

        self.assertTrue(torch.equal(sample_error, torch.tensor([1.0])))

    def test_auto_beta_uses_positive_floor(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "auto_beta": True, "auto_beta_min": 1e-3},
        )

        beta = distiller._beta_for_margin(torch.tensor([1_000_000.0]))

        self.assertAlmostEqual(float(beta), 1e-3)

    def test_sft_loss_weight_mixes_original_loss(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        batch = _prepared_batch()
        model_output = model.model_predict(batch)
        base_distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "auto_beta": False},
        )
        mixed_distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "auto_beta": False, "sft_loss_weight": 0.5},
        )

        base_loss, _ = base_distiller.compute_distill_loss(batch, model_output, torch.tensor(2.0))
        mixed_loss, _ = mixed_distiller.compute_distill_loss(batch, model_output, torch.tensor(2.0))

        torch.testing.assert_close(mixed_loss, base_loss + torch.tensor(1.0))

    def test_anchor_regularizes_preferred_and_rejected_predictions(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        batch = _prepared_batch()
        model_output = model.model_predict(batch)
        base_distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "auto_beta": False},
        )
        anchored_distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "auto_beta": False, "anchor_alpha": 2.0},
        )

        base_loss, _ = base_distiller.compute_distill_loss(batch, model_output, torch.tensor(0.0))
        anchored_loss, logs = anchored_distiller.compute_distill_loss(batch, model_output, torch.tensor(0.0))

        self.assertIn("flow_dpo_anchor_loss", logs)
        torch.testing.assert_close(anchored_loss, base_loss + torch.tensor(0.13), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
