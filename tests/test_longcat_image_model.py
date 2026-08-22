import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.longcat_image.model import LongCatImage
from simpletuner.helpers.models.longcat_image.pipeline import LongCatImagePipeline
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class LongCatImageModelTests(unittest.TestCase):
    def setUp(self):
        self.model = LongCatImage.__new__(LongCatImage)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            base_weight_dtype=torch.float32,
            model_flavour=None,
            twinflow_enabled=False,
            input_perturbation=0.0,
            scheduled_sampling_reflexflow=False,
            scheduled_sampling_max_step_offset=0,
            crepa_self_flow=False,
            crepa_feature_source=None,
            loss_type="l2",
            huber_schedule="constant",
            huber_c=0.1,
        )
        self.model.noise_schedule = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self.model._is_edit_flavour = lambda: False
        self.model.diff2flow_bridge = None
        self.model.nextlat_regularizer = None
        self.model.layersync_regularizer = None

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

    def test_pipeline_allows_missing_image_encoder(self):
        pipeline = LongCatImagePipeline(
            scheduler=MagicMock(),
            vae=MagicMock(),
            text_encoder=MagicMock(),
            tokenizer=MagicMock(),
            text_processor=MagicMock(),
            transformer=MagicMock(),
        )

        self.assertIsNone(pipeline.image_encoder)

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            base_weight_dtype=torch.float32,
            model_flavour=None,
            crepa_self_flow_mask_ratio=0.5,
        )
        self.model.sample_flow_sigmas = MagicMock(
            return_value=(torch.tensor([0.8], dtype=torch.float32), torch.tensor([800.0], dtype=torch.float32))
        )

        batch = {
            "latents": torch.zeros(1, 16, 4, 4, dtype=torch.float32),
            "input_noise": torch.ones(1, 16, 4, 4, dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
            "timesteps": torch.tensor([200.0], dtype=torch.float32),
        }
        fake_mask_rand = torch.tensor([[[0.2, 0.7], [0.9, 0.1]]], dtype=torch.float32)

        with patch("torch.rand", return_value=fake_mask_rand):
            result = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertEqual(result["sigmas"].shape, (1, 1, 4, 4))
        self.assertEqual(set(result["timesteps"].view(-1).tolist()), {200.0, 800.0})
        self.assertTrue(torch.equal(result["crepa_self_flow_mask"], fake_mask_rand < 0.5))

    def test_model_predict_uses_crepa_capture_block_override(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=3)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_3"] = torch.full((1, 4, 8), 3.0)
            kwargs["hidden_states_buffer"]["layer_7"] = torch.full((1, 4, 8), 7.0)
            return (torch.randn(1, 4, 64),)

        self.model.model = MagicMock(side_effect=_forward)

        prepared_batch = {
            "prompt_embeds": torch.randn(1, 2, 16),
            "timesteps": torch.tensor([500.0]),
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "latents": torch.randn(1, 16, 4, 4),
            "crepa_capture_block_index": 7,
        }

        with patch("simpletuner.helpers.models.longcat_image.model.pack_latents", return_value=torch.randn(1, 4, 16)):
            with patch(
                "simpletuner.helpers.models.longcat_image.model.unpack_latents",
                return_value=torch.randn(1, 16, 4, 4),
            ):
                result = self.model.model_predict(prepared_batch)

        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 7.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([0.5], dtype=torch.float32)))

    def test_model_predict_accepts_tokenwise_timesteps(self):
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.model = MagicMock(return_value=(torch.randn(1, 4, 64),))

        prepared_batch = {
            "prompt_embeds": torch.randn(1, 2, 16),
            "timesteps": torch.tensor([[100.0, 900.0]], dtype=torch.float32),
            "noisy_latents": torch.randn(1, 16, 2, 2),
            "latents": torch.randn(1, 16, 2, 2),
        }

        with patch("simpletuner.helpers.models.longcat_image.model.pack_latents", return_value=torch.randn(1, 2, 16)):
            with patch(
                "simpletuner.helpers.models.longcat_image.model.unpack_latents",
                return_value=torch.randn(1, 16, 2, 2),
            ):
                self.model.model_predict(prepared_batch)

        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([[0.1, 0.9]], dtype=torch.float32)))

    def test_model_predict_edit_mode_appends_clean_conditioning_timesteps(self):
        self.model.config = SimpleNamespace(
            weight_dtype=torch.float32, base_weight_dtype=torch.float32, model_flavour="edit"
        )
        self.model._is_edit_flavour = lambda: True
        hidden_states_buffer = {"layer_7": torch.arange(4 * 8, dtype=torch.float32).view(1, 4, 8)}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=3)
        self.model.model = MagicMock(return_value=(torch.randn(1, 4, 64),))

        prepared_batch = {
            "prompt_embeds": torch.randn(1, 2, 16),
            "timesteps": torch.tensor([[100.0, 900.0]], dtype=torch.float32),
            "noisy_latents": torch.randn(1, 16, 2, 2),
            "latents": torch.randn(1, 16, 2, 2),
            "conditioning_latents": torch.randn(1, 16, 2, 2),
            "crepa_capture_block_index": 7,
        }

        with patch(
            "simpletuner.helpers.models.longcat_image.model.pack_latents",
            side_effect=[torch.randn(1, 2, 16), torch.randn(1, 2, 16)],
        ):
            with patch(
                "simpletuner.helpers.models.longcat_image.model.unpack_latents",
                return_value=torch.randn(1, 16, 2, 2),
            ):
                result = self.model.model_predict(prepared_batch)

        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(
            torch.equal(
                transformer_kwargs["timestep"],
                torch.tensor([[0.1, 0.9, 0.0, 0.0]], dtype=torch.float32),
            )
        )
        self.assertEqual(result["crepa_hidden_states"].shape[1], 2)

    def test_validate_xm_support_rejects_unsupported_modes(self):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target="route",
            selection_scope="sample",
            block_size=0,
        )

        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            self.model._validate_xm_support()

        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target="noise",
            selection_scope="block",
            block_size=2,
        )

        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            self.model._validate_xm_support()

        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target="noise",
            selection_scope="sample",
            block_size=4,
        )

        with self.assertRaisesRegex(ValueError, "xm_block_size"):
            self.model._validate_xm_support()

        self._enable_xm(candidate_count=2)
        self.model.config.crepa_self_flow = True

        with self.assertRaisesRegex(ValueError, "CREPA self-flow"):
            self.model._validate_xm_support()

    def test_prepare_xm_noise_candidates_expands_longcat_conditioning_candidate_major(self):
        self._enable_xm(candidate_count=3)
        latents = torch.arange(8, dtype=torch.float32).view(2, 1, 2, 2)
        candidate_noise = torch.arange(24, dtype=torch.float32).view(6, 1, 2, 2)
        batch = {
            "latents": latents.clone(),
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": torch.zeros_like(latents),
            "sigmas": torch.full((2, 1, 1, 1), 0.25),
            "timesteps": torch.tensor([250.0, 750.0]),
            "prompt_embeds": torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4),
            "encoder_attention_mask": torch.ones(2, 3),
            "conditioning_latents": torch.arange(8, 16, dtype=torch.float32).view(2, 1, 2, 2),
            "conditioning_image_embeds": {
                "pixel_values": torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).view(2, 3, 2, 2),
            },
            self.model.FLOWMAP_R_TIMESTEP_BATCH_KEY: torch.tensor([100.0, 700.0]),
            "metadata": [{"id": 0}, {"id": 1}],
        }

        with patch("torch.randn_like", return_value=candidate_noise):
            self.model._prepare_xm_noise_candidates(batch)

        self.assertEqual(tuple(batch["latents"].shape), (6, 1, 2, 2))
        self.assertTrue(torch.equal(batch["latents"], latents.repeat(3, 1, 1, 1)))
        self.assertTrue(torch.equal(batch["noise"], candidate_noise))
        expected_noisy = 0.75 * batch["latents"] + 0.25 * candidate_noise
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_noisy))
        self.assertTrue(torch.equal(batch["flow_target"], candidate_noise - batch["latents"]))
        self.assertTrue(
            torch.equal(
                batch[self.model.FLOWMAP_R_TIMESTEP_BATCH_KEY],
                torch.tensor([100.0, 700.0, 100.0, 700.0, 100.0, 700.0]),
            )
        )
        self.assertEqual(batch["metadata"], [{"id": 0}, {"id": 1}, {"id": 0}, {"id": 1}, {"id": 0}, {"id": 1}])
        self.assertEqual(batch["xm_candidate_count"], 3)

    def test_model_predict_returns_xm_candidate_count(self):
        self._enable_xm(candidate_count=2)
        self.model._new_hidden_state_buffer = MagicMock(return_value={})

        class FakeTransformer:
            def __init__(self):
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return (torch.zeros_like(kwargs["hidden_states"]),)

        fake_transformer = FakeTransformer()
        self.model.model = fake_transformer
        prepared_batch = {
            "prompt_embeds": torch.randn(2, 3, 8),
            "timesteps": torch.tensor([250.0, 750.0]),
            "noisy_latents": torch.zeros(2, 16, 4, 4),
            "latents": torch.ones(2, 16, 4, 4),
            "noise": torch.zeros(2, 16, 4, 4),
            "input_noise": torch.zeros(2, 16, 4, 4),
            "sigmas": torch.full((2, 1, 1, 1), 0.5),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["xm_candidate_count"], 2)
        self.assertEqual(tuple(prepared_batch["latents"].shape), (4, 16, 4, 4))
        self.assertEqual(tuple(fake_transformer.kwargs["hidden_states"].shape), (4, 4, 64))
        self.assertEqual(tuple(result["model_prediction"].shape), (4, 16, 4, 4))

    def test_xm_loss_selects_winners_and_trims_metadata_before_nextlat(self):
        self._enable_xm(candidate_count=2)
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
            "metadata": [{"id": "c0s0"}, {"id": "c0s1"}, {"id": "c1s0"}, {"id": "c1s1"}],
            "xm_candidate_count": 2,
            "xm_original_batch_size": 2,
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "crepa_hidden_states": hidden.clone(),
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "metadata_out": ["c0s0", "c0s1", "c1s0", "c1s1"],
            "xm_candidate_count": 2,
        }

        loss, logs = self.model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertEqual(prepared_batch["metadata"], [{"id": "c1s0"}, {"id": "c0s1"}])
        self.assertEqual(model_output["metadata_out"], ["c1s0", "c0s1"])
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertNotIn("xm_candidate_count", model_output)
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)

        class FakeNextLat:
            enabled = True

            def __init__(self):
                self.hidden_shape = None
                self.prediction_shape = None

            def compute_loss(self, hidden_states_buffer, output):
                self.hidden_shape = tuple(hidden_states_buffer["layer_0"].shape)
                self.prediction_shape = tuple(output["model_prediction"].shape)
                return torch.tensor(0.5), {"nextlat_loss": 0.5}

        nextlat = FakeNextLat()
        self.model.nextlat_regularizer = nextlat
        aux_loss, aux_logs = self.model.auxiliary_loss(model_output, prepared_batch, loss)

        self.assertEqual(nextlat.hidden_shape, (2, 3, 2))
        self.assertEqual(nextlat.prediction_shape, (2, 1, 1, 1))
        self.assertAlmostEqual(aux_loss.item(), 0.5)
        self.assertEqual(aux_logs["nextlat_loss"], 0.5)


if __name__ == "__main__":
    unittest.main()
