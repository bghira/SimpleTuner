import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from accelerate import init_empty_weights

from simpletuner.helpers.models.hunyuanvideo.model import HunyuanVideo
from simpletuner.helpers.models.hunyuanvideo.transformer import HunyuanVideo15TimeEmbedding
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class HunyuanVideoModelTests(unittest.TestCase):
    def _xm_model(
        self,
        *,
        training_target: str = "noise",
        selection_scope: str = "sample",
        block_size: int = 0,
        candidate_count: int = 2,
    ) -> HunyuanVideo:
        model = HunyuanVideo.__new__(HunyuanVideo)
        model.config = SimpleNamespace(
            weight_dtype=torch.float32,
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
        model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target=training_target,
            selection_scope=selection_scope,
            block_size=block_size,
        )
        model.diff2flow_bridge = None
        return model

    def test_flowmap_gate_is_materialized_when_constructed_with_meta_buffers(self):
        with init_empty_weights(include_buffers=True):
            embedding = HunyuanVideo15TimeEmbedding(embedding_dim=8)

        self.assertEqual(embedding.flowmap_delta_emb_gate.device.type, "cpu")
        self.assertTrue(torch.equal(embedding.flowmap_delta_emb_gate, torch.tensor([0.25])))

    def test_set_flowmap_gate_materializes_meta_gate(self):
        with init_empty_weights(include_buffers=True):
            embedding = HunyuanVideo15TimeEmbedding(embedding_dim=8)
            embedding.flowmap_delta_emb_gate = torch.empty(1, device="meta")

        embedding.enable_flowmap_time_conditioning(gate_value=0.5, deltatime_type="r")

        self.assertEqual(embedding.flowmap_delta_emb_gate.device.type, "cpu")
        self.assertTrue(torch.equal(embedding.flowmap_delta_emb_gate, torch.tensor([0.5])))

    def test_load_text_encoder_registers_both_hunyuan_encoders_for_device_management(self):
        model = HunyuanVideo.__new__(HunyuanVideo)
        model.accelerator = SimpleNamespace(device=torch.device("cuda:0"))
        model.config = SimpleNamespace(
            hunyuan_text_encoder_path=None,
            glyph_byt5_repo="glyph/repo",
            glyph_byt5_fallback_repo="glyph/fallback",
        )
        model._ramtorch_text_encoders_requested = MagicMock(return_value=False)
        model._ramtorch_text_encoder_percent = MagicMock(return_value=1.0)
        model._apply_ramtorch_layers = MagicMock()

        qwen_tokenizer = MagicMock()
        byt5_tokenizer = MagicMock()
        text_encoder = MagicMock()
        text_encoder.to.return_value = text_encoder
        byt5_model = MagicMock()
        byt5_model.to.return_value = byt5_model

        with (
            patch(
                "simpletuner.helpers.models.hunyuanvideo.model.Qwen2Tokenizer.from_pretrained",
                return_value=qwen_tokenizer,
            ),
            patch(
                "simpletuner.helpers.models.hunyuanvideo.model.Qwen2_5_VLTextModel.from_pretrained",
                return_value=text_encoder,
            ),
            patch(
                "simpletuner.helpers.models.hunyuanvideo.model.ByT5Tokenizer.from_pretrained",
                return_value=byt5_tokenizer,
            ),
            patch(
                "simpletuner.helpers.models.hunyuanvideo.model.T5EncoderModel.from_pretrained",
                return_value=byt5_model,
            ),
            patch(
                "simpletuner.helpers.models.hunyuanvideo.model.hf_hub_download",
                side_effect=RuntimeError("no glyph checkpoint"),
            ),
        ):
            model.load_text_encoder(move_to_device=True)

        self.assertEqual(model.text_encoders, [text_encoder, byt5_model])
        self.assertEqual(model.tokenizers, [qwen_tokenizer, byt5_tokenizer])
        self.assertIs(model.text_encoder_1, text_encoder)
        self.assertIs(model.text_encoder_2, byt5_model)
        self.assertIs(model.get_text_encoder(1), byt5_model)

    def test_load_text_encoder_prefers_qwen_text_encoder_override(self):
        model = HunyuanVideo.__new__(HunyuanVideo)
        model.accelerator = SimpleNamespace(device=torch.device("cuda:0"))
        model.config = SimpleNamespace(
            qwen_text_encoder_model_name_or_path="custom/qwen",
            hunyuan_text_encoder_path="legacy/qwen",
            glyph_byt5_repo="glyph/repo",
            glyph_byt5_fallback_repo="glyph/fallback",
        )
        model._ramtorch_text_encoders_requested = MagicMock(return_value=False)
        model._ramtorch_text_encoder_percent = MagicMock(return_value=1.0)
        model._apply_ramtorch_layers = MagicMock()

        qwen_tokenizer = MagicMock()
        byt5_tokenizer = MagicMock()
        text_encoder = MagicMock()
        text_encoder.to.return_value = text_encoder
        byt5_model = MagicMock()
        byt5_model.to.return_value = byt5_model

        with (
            patch(
                "simpletuner.helpers.models.hunyuanvideo.model.Qwen2Tokenizer.from_pretrained",
                return_value=qwen_tokenizer,
            ) as mock_qwen_tokenizer,
            patch(
                "simpletuner.helpers.models.hunyuanvideo.model.Qwen2_5_VLTextModel.from_pretrained",
                return_value=text_encoder,
            ) as mock_qwen_text_encoder,
            patch(
                "simpletuner.helpers.models.hunyuanvideo.model.ByT5Tokenizer.from_pretrained",
                return_value=byt5_tokenizer,
            ),
            patch(
                "simpletuner.helpers.models.hunyuanvideo.model.T5EncoderModel.from_pretrained",
                return_value=byt5_model,
            ),
            patch(
                "simpletuner.helpers.models.hunyuanvideo.model.hf_hub_download",
                side_effect=RuntimeError("no glyph checkpoint"),
            ),
        ):
            model.load_text_encoder(move_to_device=True)

        mock_qwen_tokenizer.assert_called_once_with("custom/qwen")
        mock_qwen_text_encoder.assert_called_once_with("custom/qwen", torch_dtype=torch.bfloat16)

    def test_model_supports_crepa_self_flow(self):
        model = HunyuanVideo.__new__(HunyuanVideo)
        self.assertTrue(model.supports_crepa_self_flow())

    def test_check_user_config_maps_upstream_repo_to_diffusers_flavour_repo(self):
        model = HunyuanVideo.__new__(HunyuanVideo)
        model.config = SimpleNamespace(
            model_flavour="t2v-480p",
            pretrained_model_name_or_path=HunyuanVideo.UPSTREAM_MODEL_REPO,
        )

        model.check_user_config()

        self.assertEqual(
            model.config.pretrained_model_name_or_path,
            HunyuanVideo.HUGGINGFACE_PATHS["t2v-480p"],
        )

    def test_check_user_config_preserves_explicit_custom_model_path(self):
        model = HunyuanVideo.__new__(HunyuanVideo)
        model.config = SimpleNamespace(
            model_flavour="t2v-480p",
            pretrained_model_name_or_path="local-or-hub/custom-hunyuan-diffusers",
        )

        model.check_user_config()

        self.assertEqual(model.config.pretrained_model_name_or_path, "local-or-hub/custom-hunyuan-diffusers")

    def test_prepare_crepa_self_flow_batch_builds_tokenwise_student_and_teacher_views(self):
        model = HunyuanVideo.__new__(HunyuanVideo)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(weight_dtype=torch.float32, crepa_self_flow_mask_ratio=0.5)
        model.model = MagicMock(config=SimpleNamespace(patch_size=1, patch_size_t=1))
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        model.sample_flow_sigmas = MagicMock(
            return_value=(torch.tensor([0.8], dtype=torch.float32), torch.tensor([800.0], dtype=torch.float32))
        )

        batch = {
            "latents": torch.zeros(1, 2, 2, 2, 2, dtype=torch.float32),
            "input_noise": torch.ones(1, 2, 2, 2, 2, dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
            "timesteps": torch.tensor([200.0], dtype=torch.float32),
        }
        fake_mask_rand = torch.tensor(
            [[[[0.2, 0.7], [0.9, 0.1]], [[0.4, 0.6], [0.8, 0.3]]]],
            dtype=torch.float32,
        )

        with patch("torch.rand", return_value=fake_mask_rand):
            result = model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 8))
        self.assertEqual(result["sigmas"].shape, (1, 1, 2, 2, 2))
        self.assertEqual(result["crepa_teacher_timesteps"].shape, (1,))
        self.assertEqual(set(result["timesteps"].view(-1).tolist()), {200.0, 800.0})
        self.assertEqual(result["crepa_teacher_timesteps"].item(), 200.0)
        self.assertTrue(torch.equal(result["crepa_self_flow_mask"], fake_mask_rand < 0.5))

    def test_model_predict_preserves_tokenwise_timesteps_for_self_flow_capture(self):
        model = HunyuanVideo.__new__(HunyuanVideo)
        model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            twinflow_enabled=False,
            vision_num_semantic_tokens=4,
            vision_states_dim=6,
            text_embed_2_dim=4,
        )
        model.crepa_regularizer = MagicMock(block_index=3, use_backbone_features=False)
        model._new_hidden_state_buffer = MagicMock(return_value={})
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        model._is_i2v_like_flavour = lambda: False
        model._prepare_cond_latents = lambda conditioning_latents, latents, task_type: (
            torch.zeros_like(latents),
            torch.zeros(
                latents.shape[0],
                1,
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
                device=latents.device,
                dtype=latents.dtype,
            ),
        )

        captured = torch.randn(1, 2, 4, 8)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_7"] = captured
            return (torch.randn(1, 2, 2, 2, 2),)

        model.model = MagicMock(side_effect=_forward, config=SimpleNamespace(image_embed_dim=6, text_embed_2_dim=4))

        tokenwise_timesteps = torch.tensor([[100.0, 900.0, 100.0, 900.0, 100.0, 900.0, 100.0, 900.0]])
        prepared_batch = {
            "noisy_latents": torch.randn(1, 2, 2, 2, 2),
            "encoder_hidden_states": torch.randn(1, 3, 8),
            "encoder_attention_mask": torch.ones(1, 3),
            "timesteps": tokenwise_timesteps,
            "crepa_capture_block_index": 7,
        }

        result = model.model_predict(prepared_batch)

        self.assertIs(result["crepa_hidden_states"], captured)
        transformer_kwargs = model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], tokenwise_timesteps))
        self.assertEqual(transformer_kwargs["hidden_states"].shape, (1, 5, 2, 2, 2))

    def test_model_predict_moves_text_embeddings_to_latent_device(self):
        model = HunyuanVideo.__new__(HunyuanVideo)
        model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            twinflow_enabled=False,
            vision_num_semantic_tokens=4,
            vision_states_dim=6,
            text_embed_2_dim=4,
        )
        model.crepa_regularizer = None
        model._new_hidden_state_buffer = MagicMock(return_value={})
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        model._is_i2v_like_flavour = lambda: False
        model._prepare_cond_latents = lambda conditioning_latents, latents, task_type: (
            torch.zeros_like(latents),
            torch.zeros(
                latents.shape[0],
                1,
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
                device=latents.device,
                dtype=latents.dtype,
            ),
        )
        model._get_flowmap_r_timestep_forward_kwargs = MagicMock(return_value={})
        model._select_crepa_hidden_states = MagicMock(return_value=None)
        model.model = MagicMock(
            return_value=(torch.empty(1, 2, 2, 2, 2, device="meta"),),
            config=SimpleNamespace(image_embed_dim=6, text_embed_2_dim=4),
        )

        prepared_batch = {
            "noisy_latents": torch.empty(1, 2, 2, 2, 2, device="meta"),
            "encoder_hidden_states": torch.randn(1, 3, 8),
            "encoder_attention_mask": torch.ones(1, 3),
            "encoder_hidden_states_2": torch.randn(1, 2, 4),
            "encoder_attention_mask_2": torch.ones(1, 2),
            "timesteps": torch.tensor([100.0]),
        }

        model.model_predict(prepared_batch)

        transformer_kwargs = model.model.call_args.kwargs
        self.assertEqual(transformer_kwargs["encoder_hidden_states"].device.type, "meta")
        self.assertEqual(transformer_kwargs["encoder_attention_mask"].device.type, "meta")
        self.assertEqual(transformer_kwargs["encoder_hidden_states_2"].device.type, "meta")
        self.assertEqual(transformer_kwargs["encoder_attention_mask_2"].device.type, "meta")

    def test_xm_validation_rejects_unsupported_route_block_and_block_size(self):
        model = self._xm_model(training_target="route")
        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            model._validate_xm_support()

        model = self._xm_model(selection_scope="block")
        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            model._validate_xm_support()

        model = self._xm_model(block_size=2)
        with self.assertRaisesRegex(ValueError, "xm_block_size"):
            model._validate_xm_support()

    def test_xm_validation_rejects_crepa_self_flow(self):
        model = self._xm_model()
        model.config.crepa_self_flow = True

        with self.assertRaisesRegex(ValueError, "CREPA self-flow"):
            model._xm_noise_candidates_enabled()

    def test_xm_noise_candidates_expand_hunyuan_batch_candidate_major(self):
        model = self._xm_model(candidate_count=3)
        latents = torch.arange(2 * 1 * 2 * 2 * 2, dtype=torch.float32).view(2, 1, 2, 2, 2)
        candidate_noise = torch.arange(6 * 1 * 2 * 2 * 2, dtype=torch.float32).view(6, 1, 2, 2, 2)
        prompt_embeds = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
        prompt_embeds_2 = torch.arange(2 * 2 * 5, dtype=torch.float32).view(2, 2, 5)
        vision_states = torch.arange(2 * 4 * 6, dtype=torch.float32).view(2, 4, 6)
        conditioning_latents = torch.full((2, 1, 1, 2, 2), 7.0)
        conditioning_pixels = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
        batch = {
            "latents": latents.clone(),
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": torch.zeros_like(latents),
            "sigmas": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "timesteps": torch.tensor([250.0, 750.0], dtype=torch.float32),
            "encoder_hidden_states": prompt_embeds.clone(),
            "encoder_attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.int64),
            "encoder_hidden_states_2": prompt_embeds_2.clone(),
            "encoder_attention_mask_2": torch.tensor([[1, 0], [1, 1]], dtype=torch.int64),
            "vision_states": vision_states.clone(),
            "conditioning_latents": conditioning_latents.clone(),
            "conditioning_pixel_values": conditioning_pixels.clone(),
            "flowmap_r_timesteps": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "metadata": [{"id": "a"}, {"id": "b"}],
        }

        with patch("torch.randn_like", return_value=candidate_noise):
            model._prepare_xm_noise_candidates(batch, family_name=HunyuanVideo.NAME)

        self.assertEqual(tuple(batch["latents"].shape), (6, 1, 2, 2, 2))
        self.assertTrue(torch.equal(batch["latents"], latents.repeat(3, 1, 1, 1, 1)))
        self.assertTrue(torch.equal(batch["timesteps"], torch.tensor([250.0, 750.0] * 3)))
        self.assertTrue(torch.equal(batch["flowmap_r_timesteps"], torch.tensor([0.1, 0.2] * 3)))
        self.assertTrue(torch.equal(batch["encoder_hidden_states"], prompt_embeds.repeat(3, 1, 1)))
        self.assertTrue(torch.equal(batch["encoder_hidden_states_2"], prompt_embeds_2.repeat(3, 1, 1)))
        self.assertTrue(torch.equal(batch["vision_states"], vision_states.repeat(3, 1, 1)))
        self.assertTrue(torch.equal(batch["conditioning_latents"], conditioning_latents.repeat(3, 1, 1, 1, 1)))
        self.assertTrue(torch.equal(batch["conditioning_pixel_values"], conditioning_pixels.repeat(3, 1, 1, 1)))
        self.assertEqual(batch["metadata"], [{"id": "a"}, {"id": "b"}, {"id": "a"}, {"id": "b"}, {"id": "a"}, {"id": "b"}])

        sigma_grid = batch["sigmas"].view(6, 1, 1, 1, 1)
        expected_noisy = (1.0 - sigma_grid) * batch["latents"] + sigma_grid * candidate_noise
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_noisy))
        self.assertTrue(torch.equal(batch["noise"], candidate_noise))
        self.assertTrue(torch.allclose(batch["flow_target"], candidate_noise - batch["latents"]))
        self.assertEqual(batch["xm_candidate_count"], 3)
        self.assertEqual(batch["xm_original_batch_size"], 2)

    def test_xm_model_predict_returns_candidate_count(self):
        model = self._xm_model(candidate_count=2)
        model._prepare_xm_noise_candidates = MagicMock()
        model._model_predict_single = MagicMock(return_value={"model_prediction": torch.zeros(4, 1, 1, 1, 1)})
        batch = {"latents": torch.zeros(2, 1, 1, 1, 1)}

        result = model.model_predict(batch)

        model._prepare_xm_noise_candidates.assert_called_once()
        self.assertIs(model._prepare_xm_noise_candidates.call_args.args[0], batch)
        self.assertEqual(model._prepare_xm_noise_candidates.call_args.kwargs["family_name"], HunyuanVideo.NAME)
        self.assertEqual(result["xm_candidate_count"], 2)

    def test_xm_loss_selects_winners_and_trims_sample_aligned_values(self):
        model = self._xm_model(candidate_count=2)
        latents = torch.zeros(4, 1, 1, 1, 1)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1, 1, 1)
        target = noise - latents
        prediction = torch.tensor([5.0, 1.0, 2.0, -4.0]).view(4, 1, 1, 1, 1)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "input_noise": noise,
            "noisy_latents": noise,
            "sigmas": torch.ones(4),
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
            "metadata": ["a0", "b0", "a1", "b1"],
            "nested": {"labels": ("a0", "b0", "a1", "b1")},
            "xm_candidate_count": 2,
            "xm_original_batch_size": 2,
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "crepa_hidden_states": hidden.clone(),
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(model_output["crepa_hidden_states"], hidden[[2, 1]]))
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["noise"], noise[[2, 1]]))
        self.assertEqual(prepared_batch["metadata"], ["a1", "b0"])
        self.assertEqual(prepared_batch["nested"]["labels"], ("a1", "b0"))
        self.assertNotIn("xm_candidate_count", prepared_batch)
        self.assertNotIn("xm_candidate_count", model_output)
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)


if __name__ == "__main__":
    unittest.main()
