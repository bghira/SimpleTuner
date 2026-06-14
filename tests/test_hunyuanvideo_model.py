import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from accelerate import init_empty_weights

from simpletuner.helpers.models.hunyuanvideo.model import HunyuanVideo
from simpletuner.helpers.models.hunyuanvideo.transformer import HunyuanVideo15TimeEmbedding


class HunyuanVideoModelTests(unittest.TestCase):
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

    def test_model_supports_crepa_self_flow(self):
        model = HunyuanVideo.__new__(HunyuanVideo)
        self.assertTrue(model.supports_crepa_self_flow())

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


if __name__ == "__main__":
    unittest.main()
