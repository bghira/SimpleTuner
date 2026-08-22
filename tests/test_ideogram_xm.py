import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from simpletuner.helpers.models.ideogram.model import Ideogram4
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class DummyAccelerator:
    device = torch.device("cpu")


def make_ideogram_shell(candidate_count: int = 2):
    model = Ideogram4.__new__(Ideogram4)
    model.config = SimpleNamespace(
        loss_type="l2",
        huber_schedule="constant",
        huber_c=0.1,
        weight_dtype=torch.float32,
    )
    model.accelerator = DummyAccelerator()
    model.xm_config = ExplorativeModelingConfig(
        enabled=True,
        candidate_count=candidate_count,
        training_target="noise",
        selection_scope="sample",
        block_size=0,
    )
    return model


class IdeogramXmTests(unittest.TestCase):
    def test_xm_validation_rejects_unsupported_training_target(self):
        model = make_ideogram_shell()
        model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target="route",
            selection_scope="sample",
            block_size=0,
        )

        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            model._validate_xm_support()

    def test_xm_validation_rejects_unsupported_selection_scope(self):
        model = make_ideogram_shell()
        model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target="noise",
            selection_scope="block",
            block_size=2,
        )

        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            model._validate_xm_support()

    def test_xm_noise_candidates_expand_conditioning_candidate_major(self):
        torch.manual_seed(11)
        model = make_ideogram_shell(candidate_count=3)
        latents = torch.arange(2 * 1 * 2 * 2, dtype=torch.float32).view(2, 1, 2, 2)
        batch = {
            "latents": latents.clone(),
            "sigmas": torch.tensor([0.25, 0.75]),
            "timesteps": torch.tensor([250.0, 750.0]),
            "prompt_embeds": torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3),
            "encoder_attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
        }

        model._prepare_xm_noise_candidates(batch)

        self.assertEqual(tuple(batch["latents"].shape), (6, 1, 2, 2))
        self.assertTrue(torch.equal(batch["latents"], latents.repeat(3, 1, 1, 1)))
        self.assertTrue(torch.equal(batch["timesteps"], torch.tensor([250.0, 750.0, 250.0, 750.0, 250.0, 750.0])))
        self.assertEqual(tuple(batch["prompt_embeds"].shape), (6, 4, 3))
        self.assertTrue(
            torch.equal(
                batch["encoder_attention_mask"],
                torch.tensor(
                    [
                        [1, 1, 1, 0],
                        [1, 1, 0, 0],
                        [1, 1, 1, 0],
                        [1, 1, 0, 0],
                        [1, 1, 1, 0],
                        [1, 1, 0, 0],
                    ],
                    dtype=torch.bool,
                ),
            )
        )
        sigma_grid = batch["sigmas"].view(6, 1, 1, 1)
        expected_noisy = (1.0 - sigma_grid) * batch["latents"] + sigma_grid * batch["noise"]
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_noisy))
        self.assertTrue(torch.allclose(batch["flow_target"], batch["noise"] - batch["latents"]))

    def test_model_predict_expands_xm_before_conditional_transformer_forward(self):
        model = make_ideogram_shell(candidate_count=2)
        model._new_hidden_state_buffer = mock.Mock(return_value={})
        captured = {}

        def forward(**kwargs):
            captured.update(kwargs)
            kwargs["hidden_states_buffer"]["layer_0"] = torch.arange(4 * 4 * 3, dtype=torch.float32).view(4, 4, 3)
            return torch.zeros_like(kwargs["x"])

        model.model = mock.Mock(side_effect=forward)
        model.unconditional_transformer = None
        batch = {
            "latents": torch.zeros(2, 128, 2, 2),
            "sigmas": torch.full((2,), 0.5),
            "timesteps": torch.tensor([500.0, 250.0]),
            "noisy_latents": torch.ones(2, 128, 2, 2),
            "encoder_hidden_states": torch.randn(2, 3, 5),
            "attention_mask": torch.ones(2, 3, dtype=torch.bool),
        }

        output = model.model_predict(batch)

        model.model.assert_called_once()
        self.assertEqual(tuple(batch["noisy_latents"].shape), (4, 128, 2, 2))
        self.assertEqual(tuple(captured["x"].shape), (4, 7, 128))
        self.assertEqual(tuple(captured["llm_features"].shape), (4, 7, 5))
        self.assertEqual(tuple(captured["position_ids"].shape), (4, 7, 3))
        self.assertEqual(tuple(output["model_prediction"].shape), (4, 128, 2, 2))
        self.assertEqual(output["xm_candidate_count"], 2)
        self.assertIn("layer_0", output["hidden_states_buffer"])

    def test_xm_loss_selects_winners_before_nextlat_auxiliary_loss(self):
        model = make_ideogram_shell(candidate_count=2)
        latents = torch.zeros(4, 1, 1, 1)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1, 1)
        target = noise - latents
        prediction = torch.tensor([5.0, 1.0, 2.0, -4.0]).view(4, 1, 1, 1)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "noisy_latents": noise,
            "sigmas": torch.ones(4, 1, 1, 1),
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["noise"], noise[[2, 1]]))
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)

        class DummyNextLat:
            enabled = True

            def __init__(self):
                self.seen_hidden = None

            def compute_loss(self, hidden_states_buffer, model_output):
                self.seen_hidden = hidden_states_buffer["layer_0"].clone()
                return model_output["model_prediction"].sum() * 0.0 + 0.25, {"nextlat_loss": 0.25}

        nextlat = DummyNextLat()
        model.nextlat_regularizer = nextlat
        model.internal_guidance_regularizer = None
        model._twinflow_active = lambda: False

        total_loss, aux_logs = model.auxiliary_loss(
            model_output=model_output,
            prepared_batch=prepared_batch,
            loss=loss,
            apply_layersync=False,
            clear_hidden_state_buffer=False,
        )

        self.assertAlmostEqual(total_loss.item(), 0.25)
        self.assertEqual(aux_logs["nextlat_loss"], 0.25)
        self.assertTrue(torch.equal(nextlat.seen_hidden, hidden[[2, 1]]))


if __name__ == "__main__":
    unittest.main()
