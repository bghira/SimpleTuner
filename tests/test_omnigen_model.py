import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.omnigen.model import OmniGen
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class OmniGenModelTests(unittest.TestCase):
    def setUp(self):
        self.model = OmniGen.__new__(OmniGen)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            base_weight_dtype=torch.float32,
            input_perturbation=0.0,
            scheduled_sampling_reflexflow=False,
            scheduled_sampling_max_step_offset=0,
            twinflow_enabled=False,
            crepa_self_flow=False,
            crepa_feature_source=None,
        )
        self.model._load_preprocessor = lambda: None
        self.model.processor = SimpleNamespace(
            process_multi_modal_prompt=lambda prompt, input_images=None: {"prompt": prompt},
            collator=lambda all_features: {
                "output_latents": torch.randn(1, 4, 2, 2),
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, 7, 7),
                "position_ids": torch.arange(7).view(1, 7),
                "input_img_latents": [],
                "input_image_sizes": {},
            },
        )
        self.model._new_hidden_state_buffer = MagicMock(return_value={"layer_7": torch.randn(1, 4, 8)})
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=3)
        self.model.model = MagicMock()
        self.model.model.config = SimpleNamespace(patch_size=1)
        self.model.model.return_value = (torch.randn(1, 4, 2, 2),)
        self.model.sample_flow_sigmas = OmniGen.sample_flow_sigmas.__get__(self.model, OmniGen)
        self.model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        self.model.diff2flow_bridge = None

    def _enable_xm(self, candidate_count: int = 2):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target="noise",
            selection_scope="sample",
            block_size=0,
        )

    def test_model_supports_crepa_self_flow(self):
        self.assertTrue(self.model.supports_crepa_self_flow())

    def test_model_does_not_use_text_embedding_cache(self):
        self.assertFalse(self.model.uses_text_embeddings_cache())

    def test_flow_matching_target_uses_omnigen_direction(self):
        latents = torch.tensor([1.0, 2.0])
        noise = torch.tensor([3.0, 5.0])

        target = self.model.get_flow_matching_target({"latents": latents, "noise": noise})

        self.assertTrue(torch.equal(target, latents - noise))

    def test_loss_uses_flow_matching_target_helper(self):
        latents = torch.tensor([1.0, 2.0])
        noise = torch.tensor([3.0, 5.0])
        prediction = latents - noise

        loss = self.model.loss({"latents": latents, "noise": noise}, {"model_prediction": prediction})

        self.assertEqual(loss.item(), 0.0)

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        self.model.config.crepa_self_flow_mask_ratio = 0.5
        batch = {
            "latents": torch.zeros(1, 4, 2, 2, dtype=torch.float32),
            "noise": torch.ones(1, 4, 2, 2, dtype=torch.float32),
            "timesteps": torch.tensor([0.2], dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
        }
        self.model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([0.8]), torch.tensor([0.8])))
        fake_mask_rand = torch.tensor([[[0.2, 0.7], [0.9, 0.1]]], dtype=torch.float32)

        with unittest.mock.patch("torch.rand", return_value=fake_mask_rand):
            result = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertEqual(sorted(round(float(x), 4) for x in result["timesteps"].view(-1)), [0.2, 0.2, 0.8, 0.8])
        self.assertEqual(result["sigmas"].shape, (1, 1, 2, 2))

    def test_model_predict_accepts_tokenwise_timesteps_and_capture_override(self):
        prepared_batch = {
            "noisy_latents": torch.randn(1, 4, 2, 2),
            "timesteps": torch.tensor([[200.0, 800.0, 200.0, 800.0]], dtype=torch.float32),
            "prompts": ["test"],
            "crepa_capture_block_index": 7,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertIsNotNone(result["crepa_hidden_states"])
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([[0.2, 0.8, 0.2, 0.8]])))

    def test_xm_noise_candidates_expand_with_omnigen_interpolation(self):
        self._enable_xm(candidate_count=3)
        batch = {
            "latents": torch.ones(2, 1, 2, 2),
            "noise": torch.zeros(2, 1, 2, 2),
            "input_noise": torch.zeros(2, 1, 2, 2),
            "noisy_latents": torch.zeros(2, 1, 2, 2),
            "timesteps": torch.tensor([0.25, 0.75]),
            "sigmas": torch.tensor([0.25, 0.75]),
            "prompts": ["a", "b"],
        }

        self.model._prepare_xm_noise_candidates(batch)

        self.assertEqual(tuple(batch["latents"].shape), (6, 1, 2, 2))
        self.assertEqual(batch["prompts"], ["a", "b", "a", "b", "a", "b"])
        self.assertTrue(torch.equal(batch["timesteps"], torch.tensor([0.25, 0.75, 0.25, 0.75, 0.25, 0.75])))
        t_view = batch["timesteps"].view(-1, 1, 1, 1)
        expected_noisy = t_view * batch["latents"] + (1.0 - t_view) * batch["noise"]
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_noisy))
        self.assertTrue(torch.equal(batch["flow_target"], batch["latents"] - batch["noise"]))

    def test_xm_loss_selects_winners_with_omnigen_direction(self):
        self._enable_xm(candidate_count=2)
        latents = torch.zeros(4, 1, 1, 1)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1, 1)
        target = latents - noise
        prediction = torch.tensor([5.0, -1.0, -2.0, 4.0]).view(4, 1, 1, 1)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "noisy_latents": noise,
            "sigmas": torch.ones(4),
            "timesteps": torch.tensor([0.1, 0.2, 0.1, 0.2]),
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = self.model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)

    def test_xm_rejects_crepa_self_flow(self):
        self._enable_xm(candidate_count=2)
        self.model.config.crepa_self_flow = True

        with self.assertRaisesRegex(ValueError, "CREPA self-flow"):
            self.model._validate_xm_support()


if __name__ == "__main__":
    unittest.main()
