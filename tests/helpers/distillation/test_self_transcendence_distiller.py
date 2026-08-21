import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from simpletuner.helpers.distillation.factory import DistillationMethod, DistillerFactory
from simpletuner.helpers.distillation.self_transcendence.distiller import (
    SelfTranscendenceDistiller,
    SelfTranscendenceProjector,
)
from simpletuner.helpers.models.common import ModelTypes, PredictionTypes


class _Component(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.scale = nn.Parameter(torch.tensor(1.0))


class _Foundation:
    NAME = "test-transformer"
    MODEL_TYPE = ModelTypes.TRANSFORMER
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING

    def __init__(self):
        self.config = SimpleNamespace(model_type="lora", lora_type="standard")
        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.noise_schedule = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self.component = _Component()

    def get_trained_component(self, unwrap_model=False):
        return self.component

    def model_predict(self, batch):
        condition = batch["encoder_hidden_states"].float().flatten(1).mean(1).view(-1, 1, 1)
        hidden = condition.expand(-1, 4, 4) * self.component.scale
        return {"hidden_states_buffer": {"layer_3": hidden}}

    def get_prediction_target(self, batch):
        return batch["noise"] - batch["latents"]


def _config(stage="vae", **overrides):
    config = {
        "stage": stage,
        "student_block": 1,
        "teacher_block": 3 if stage == "self" else None,
        "weight": 0.5,
        "cfg_scale": 2.0,
        "timestep_min": 0.4,
        "timestep_max": 0.7,
        "projector_hidden_dim": 8,
    }
    config.update(overrides)
    return config


class SelfTranscendenceDistillerTests(unittest.TestCase):
    def test_factory_registers_method(self):
        self.assertEqual(DistillationMethod.from_string("self_transcendence"), DistillationMethod.SELF_TRANSCENDENCE)
        self.assertIsNotNone(
            DistillerFactory.training_batch_requirements("self_transcendence", {"distillation_config": _config()})
        )

    def test_self_stage_requests_unconditional_embeddings(self):
        requirements = SelfTranscendenceDistiller.training_batch_requirements(_config("self"))
        self.assertEqual(requirements, {"unconditional_text_embeddings"})

    def test_configuration_requires_stage_blocks_and_valid_range(self):
        with self.assertRaisesRegex(ValueError, "student_block"):
            SelfTranscendenceDistiller._normalized_config({"stage": "vae"})
        with self.assertRaisesRegex(ValueError, "teacher_block"):
            SelfTranscendenceDistiller._normalized_config({"stage": "self", "student_block": 1})
        with self.assertRaisesRegex(ValueError, "timestep range"):
            SelfTranscendenceDistiller._normalized_config(_config(timestep_min=0.8, timestep_max=0.2))

    def test_missing_teacher_adapter_fails_before_teacher_capture(self):
        foundation = _Foundation()
        settings = _config("self", teacher_adapter_path="missing.safetensors")
        SelfTranscendenceDistiller.prepare_model_for_adapter(foundation, settings)
        distiller = SelfTranscendenceDistiller(foundation, config=settings)
        with self.assertRaises(FileNotFoundError):
            distiller.pre_training_step(foundation, 0)

    def test_prepare_model_attaches_three_layer_projector(self):
        foundation = _Foundation()
        SelfTranscendenceDistiller.prepare_model_for_adapter(foundation, _config())
        projector = foundation.component.self_transcendence_projector
        self.assertIsInstance(projector, SelfTranscendenceProjector)
        self.assertEqual(sum(isinstance(module, nn.Linear) for module in projector.modules()), 3)

    def test_hidden_size_inference_accepts_common_transformer_config_names(self):
        component = SimpleNamespace(config=SimpleNamespace(d_model=6))
        self.assertEqual(SelfTranscendenceDistiller._infer_hidden_size(component), 6)

    def test_vae_stage_aligns_to_prediction_target_and_preserves_gradients(self):
        foundation = _Foundation()
        SelfTranscendenceDistiller.prepare_model_for_adapter(foundation, _config())
        distiller = SelfTranscendenceDistiller(foundation, config=_config())
        hidden = torch.randn(2, 4, 4, requires_grad=True)
        output = {"hidden_states_buffer": {"layer_1": hidden}}
        distiller.prepare_model_output(output)
        batch = {
            "latents": torch.randn(2, 2, 2, 2),
            "noise": torch.randn(2, 2, 2, 2),
            "sigmas": torch.tensor([0.5, 0.9]),
        }
        loss, logs = distiller.compute_distill_loss(batch, output, torch.tensor(1.0))
        loss.backward()
        self.assertGreater(loss.item(), 1.0)
        self.assertIsNotNone(hidden.grad)
        self.assertGreater(logs["self_transcendence/loss"], 0.0)
        self.assertEqual(output["hidden_states_buffer"], {})

    def test_vae_target_patchification_preserves_patch_values(self):
        latents = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        hidden = torch.zeros(1, 4, 4)
        target = SelfTranscendenceDistiller._vae_target_tokens(latents, hidden, token_count=4)
        expected = torch.tensor(
            [[[0.0, 1.0, 4.0, 5.0], [2.0, 3.0, 6.0, 7.0], [8.0, 9.0, 12.0, 13.0], [10.0, 11.0, 14.0, 15.0]]]
        )
        torch.testing.assert_close(target, expected)

    def test_vae_stage_uses_model_family_prediction_target(self):
        foundation = _Foundation()
        SelfTranscendenceDistiller.prepare_model_for_adapter(foundation, _config())
        distiller = SelfTranscendenceDistiller(foundation, config=_config())
        distiller.projector = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            distiller.projector.weight.copy_(torch.eye(4))
        output = {"hidden_states_buffer": {"layer_1": torch.zeros(1, 4, 4)}}
        distiller.prepare_model_output(output)
        _, logs = distiller.compute_distill_loss(
            {
                "latents": torch.zeros(1, 1, 2, 2),
                "noise": torch.full((1, 1, 2, 2), 2.0),
                "sigmas": torch.tensor([0.5]),
            },
            output,
            torch.tensor(1.0),
        )
        self.assertAlmostEqual(logs["self_transcendence/loss"], 4.0, places=5)

    def test_vae_target_uses_explicit_video_hidden_grid(self):
        latents = torch.arange(32, dtype=torch.float32).reshape(1, 1, 2, 4, 4)
        hidden = torch.zeros(1, 2, 4, 4)
        target = SelfTranscendenceDistiller._vae_target_tokens(latents, hidden, token_count=8)
        self.assertEqual(target.shape, (1, 8, 4))
        torch.testing.assert_close(target[0, 4], torch.tensor([16.0, 17.0, 20.0, 21.0]))

    def test_self_stage_uses_frozen_cfg_teacher_features(self):
        foundation = _Foundation()
        SelfTranscendenceDistiller.prepare_model_for_adapter(foundation, _config("self"))
        distiller = SelfTranscendenceDistiller(foundation, config=_config("self"))
        distiller.projector = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            distiller.projector.weight.copy_(torch.eye(4))
        distiller.pre_training_step(foundation, 0)

        student = torch.zeros(1, 4, 4, requires_grad=True)
        output = {"hidden_states_buffer": {"layer_1": student}}
        distiller.prepare_model_output(output)
        batch = {
            "latents": torch.randn(1, 2, 2, 2),
            "sigmas": torch.tensor([0.5]),
            "encoder_hidden_states": torch.ones(1, 2, 4),
            "prompt_embeds": torch.ones(1, 2, 4),
            "negative_encoder_hidden_states": torch.zeros(1, 2, 4),
            "negative_prompt_embeds": torch.zeros(1, 2, 4),
        }
        loss, logs = distiller.compute_distill_loss(batch, output, torch.tensor(1.0))
        self.assertAlmostEqual(logs["self_transcendence/loss"], 4.0, places=5)
        self.assertAlmostEqual(loss.item(), 3.0, places=5)

    def test_stop_step_keeps_zero_weight_projector_graph(self):
        foundation = _Foundation()
        settings = _config(stop_step=5)
        SelfTranscendenceDistiller.prepare_model_for_adapter(foundation, settings)
        distiller = SelfTranscendenceDistiller(foundation, config=settings)
        distiller.pre_training_step(foundation, 5)
        hidden = torch.randn(1, 4, 4, requires_grad=True)
        output = {"hidden_states_buffer": {"layer_1": hidden}}
        distiller.prepare_model_output(output)
        loss, logs = distiller.compute_distill_loss(
            {
                "latents": torch.randn(1, 2, 2, 2),
                "noise": torch.randn(1, 2, 2, 2),
                "sigmas": torch.tensor([0.5]),
            },
            output,
            torch.tensor(1.0),
        )
        loss.backward()
        self.assertEqual(logs["self_transcendence/weight"], 0.0)
        self.assertIsNotNone(hidden.grad)


if __name__ == "__main__":
    unittest.main()
