import types
import unittest
from unittest.mock import MagicMock, patch

import torch
from transformers import Mistral3Config

from simpletuner.helpers.models.ernie.model import Ernie
from simpletuner.helpers.models.ernie.transformer import ErnieImageTransformer2DModel
from simpletuner.helpers.models.tae.types import Flux2TAESpec
from simpletuner.helpers.training.tread import TREADRouter


class DummyAccelerator:
    device = torch.device("cpu")


class DummyTransformer:
    def __init__(self):
        self.received = None

    def __call__(self, hidden_states, timestep, text_bth, text_lens, return_dict=False, **kwargs):
        self.received = (hidden_states, timestep, text_bth, text_lens, kwargs)
        return (torch.zeros_like(hidden_states),)


class ErnieModelTests(unittest.TestCase):
    def _build_model(self):
        model = Ernie.__new__(Ernie)
        model.accelerator = DummyAccelerator()
        model.config = types.SimpleNamespace(
            weight_dtype=torch.float32,
            model_family="ernie",
            pretrained_model_name_or_path=None,
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
            flow_schedule_shift=1.0,
            revision=None,
            variant=None,
        )
        model.model = DummyTransformer()
        return model

    def test_patch_text_encoder_config_repairs_ministral3_typo(self):
        config_dict = {
            "model_type": "mistral3",
            "text_config": {
                "model_type": "ministral3",
                "hidden_size": 3072,
                "intermediate_size": 9216,
                "num_hidden_layers": 2,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 96,
                "vocab_size": 32000,
            },
            "vision_config": {
                "model_type": "pixtral",
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_hidden_layers": 2,
                "num_attention_heads": 16,
                "head_dim": 64,
                "image_size": 1540,
                "patch_size": 14,
                "num_channels": 3,
            },
        }

        patched = Ernie._patch_text_encoder_config_dict(config_dict)

        self.assertEqual(config_dict["text_config"]["model_type"], "ministral3")
        self.assertEqual(patched["text_config"]["model_type"], "ministral")
        self.assertEqual(patched["text_config"]["sliding_window"], 4096)
        mistral3_config = Mistral3Config.from_dict(patched)
        self.assertEqual(mistral3_config.text_config.model_type, "ministral")
        self.assertEqual(mistral3_config.text_config.sliding_window, 4096)

    @patch("simpletuner.helpers.models.ernie.model.PreTrainedTokenizerFast")
    def test_build_ernie_tokenizer_uses_tokenizer_json(self, mock_tokenizer_cls):
        tokenizer = MagicMock()
        mock_tokenizer_cls.return_value = tokenizer

        built = Ernie._build_ernie_tokenizer(
            "/tmp/tokenizer.json",
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "clean_up_tokenization_spaces": False,
                "model_max_length": 2048,
            },
        )

        mock_tokenizer_cls.assert_called_once_with(
            tokenizer_file="/tmp/tokenizer.json",
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            clean_up_tokenization_spaces=False,
        )
        self.assertIs(built, tokenizer)
        self.assertEqual(tokenizer.model_max_length, 2048)

    @patch("simpletuner.helpers.models.ernie.model.AutoModel.from_pretrained")
    def test_load_text_encoder_uses_patched_config(self, mock_from_pretrained):
        model = self._build_model()
        text_encoder = MagicMock()
        mock_from_pretrained.return_value = text_encoder
        sentinel_config = MagicMock()
        model._ramtorch_text_encoders_requested = lambda: False
        model.load_text_tokenizer = MagicMock()
        model._load_ernie_text_encoder_config = MagicMock(return_value=sentinel_config)
        model._resolve_text_encoder_path = MagicMock(return_value="baidu/ERNIE-Image")

        model.load_text_encoder()

        model.load_text_tokenizer.assert_called_once_with()
        model._load_ernie_text_encoder_config.assert_called_once_with()
        mock_from_pretrained.assert_called_once_with(
            pretrained_model_name_or_path="baidu/ERNIE-Image",
            config=sentinel_config,
            variant=None,
            revision=None,
            subfolder="text_encoder",
            torch_dtype=torch.float32,
        )
        text_encoder.to.assert_called_once_with(torch.device("cpu"), dtype=torch.float32)
        text_encoder.eval.assert_called_once_with()
        text_encoder.requires_grad_.assert_called_once_with(False)
        self.assertEqual(model.text_encoders, [text_encoder])
        self.assertIs(model.text_encoder_1, text_encoder)

    @patch("simpletuner.helpers.models.ernie.model.AutoModel.from_pretrained")
    def test_load_text_encoder_applies_ramtorch_when_requested(self, mock_from_pretrained):
        model = self._build_model()
        text_encoder = MagicMock()
        mock_from_pretrained.return_value = text_encoder
        model.load_text_tokenizer = MagicMock()
        model._load_ernie_text_encoder_config = MagicMock(return_value=MagicMock())
        model._resolve_text_encoder_path = MagicMock(return_value="baidu/ERNIE-Image")
        model._ramtorch_text_encoders_requested = lambda: True
        model._ramtorch_text_encoder_percent = lambda: None
        model._apply_ramtorch_layers = MagicMock()

        model.load_text_encoder()

        model._apply_ramtorch_layers.assert_called_once_with(
            text_encoder,
            "text_encoder_1",
            full_ramtorch=True,
            percent=None,
        )
        text_encoder.to.assert_not_called()

    def test_encode_prompts_uses_language_model_hidden_states(self):
        model = self._build_model()

        class _Tokenizer:
            def __init__(self):
                self.called = None

            def __call__(self, prompts, **kwargs):
                del prompts
                self.called = kwargs
                return types.SimpleNamespace(
                    input_ids=torch.tensor([[1, 2, 0, 0]]),
                    attention_mask=torch.tensor([[1, 1, 0, 0]]),
                )

        class _LanguageModel:
            def __init__(self):
                self.called = None

            def __call__(self, **kwargs):
                self.called = kwargs
                hs = torch.randn(1, 4, Ernie.TEXT_EMBED_DIM)
                return types.SimpleNamespace(hidden_states=[hs, hs + 1, hs + 2])

        class _TextEncoder:
            def __init__(self):
                self.language_model = _LanguageModel()

            def get_input_embeddings(self):
                def _embed(input_ids):
                    return torch.randn(input_ids.shape[0], input_ids.shape[1], Ernie.TEXT_EMBED_DIM)

                return _embed

        model.tokenizers = [_Tokenizer()]
        model.text_encoders = [_TextEncoder()]

        encoded = model._encode_prompts(["hello"])

        self.assertEqual(tuple(encoded["prompt_embeds"].shape), (1, 4, Ernie.TEXT_EMBED_DIM))
        self.assertTrue(torch.equal(encoded["attention_mask"], torch.tensor([[True, True, False, False]])))
        self.assertIn("inputs_embeds", model.text_encoders[0].language_model.called)
        self.assertNotIn("input_ids", model.text_encoders[0].language_model.called)
        self.assertFalse(model.tokenizers[0].called["padding"])
        self.assertTrue(model.tokenizers[0].called["truncation"])
        self.assertNotIn("max_length", model.tokenizers[0].called)

    def test_encode_prompts_falls_back_to_last_hidden_state(self):
        model = self._build_model()

        class _Tokenizer:
            def __call__(self, prompts, **kwargs):
                del prompts, kwargs
                return types.SimpleNamespace(
                    input_ids=torch.tensor([[1, 2, 0]]),
                    attention_mask=torch.tensor([[1, 1, 0]]),
                )

        class _LanguageModel:
            def __call__(self, **kwargs):
                del kwargs
                return types.SimpleNamespace(last_hidden_state=torch.randn(1, 3, Ernie.TEXT_EMBED_DIM))

        class _TextEncoder:
            def __init__(self):
                self.language_model = _LanguageModel()

            def get_input_embeddings(self):
                def _embed(input_ids):
                    return torch.randn(input_ids.shape[0], input_ids.shape[1], Ernie.TEXT_EMBED_DIM)

                return _embed

        model.tokenizers = [_Tokenizer()]
        model.text_encoders = [_TextEncoder()]

        encoded = model._encode_prompts(["hello"])

        self.assertEqual(tuple(encoded["prompt_embeds"].shape), (1, 3, Ernie.TEXT_EMBED_DIM))

    def test_encode_prompts_honors_explicit_tokenizer_max_length(self):
        model = self._build_model()
        model.config.tokenizer_max_length = 256

        class _Tokenizer:
            def __init__(self):
                self.called = None

            def __call__(self, prompts, **kwargs):
                del prompts
                self.called = kwargs
                return types.SimpleNamespace(
                    input_ids=torch.tensor([[1, 2, 3]]),
                    attention_mask=torch.tensor([[1, 1, 1]]),
                )

        class _LanguageModel:
            def __call__(self, **kwargs):
                del kwargs
                hs = torch.randn(1, 3, Ernie.TEXT_EMBED_DIM)
                return types.SimpleNamespace(hidden_states=[hs, hs + 1, hs + 2])

        class _TextEncoder:
            def __init__(self):
                self.language_model = _LanguageModel()

            def get_input_embeddings(self):
                def _embed(input_ids):
                    return torch.randn(input_ids.shape[0], input_ids.shape[1], Ernie.TEXT_EMBED_DIM)

                return _embed

        model.tokenizers = [_Tokenizer()]
        model.text_encoders = [_TextEncoder()]

        model._encode_prompts(["hello"])

        self.assertEqual(model.tokenizers[0].called["max_length"], 256)

    def test_convert_text_embed_for_pipeline_masks_tokens(self):
        model = self._build_model()
        prompt_embeds = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 1]], dtype=torch.bool)

        converted = model.convert_text_embed_for_pipeline(
            {"prompt_embeds": prompt_embeds, "attention_mask": attention_mask}
        )["prompt_embeds"]

        self.assertEqual(len(converted), 2)
        self.assertTrue(torch.equal(converted[0], prompt_embeds[0][:2]))
        self.assertTrue(torch.equal(converted[1], prompt_embeds[1][[0, 2, 3]]))

    def test_collate_prompt_embeds_pads_variable_length_sequences(self):
        model = self._build_model()
        output = model.collate_prompt_embeds(
            [
                {
                    "prompt_embeds": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
                    "attention_mask": torch.tensor([[1, 1]], dtype=torch.bool),
                },
                {
                    "prompt_embeds": torch.tensor([[[5.0, 6.0]]]),
                    "attention_mask": torch.tensor([[1]], dtype=torch.bool),
                },
            ]
        )

        self.assertEqual(tuple(output["prompt_embeds"].shape), (2, 2, 2))
        self.assertEqual(tuple(output["attention_masks"].shape), (2, 2))
        self.assertTrue(torch.equal(output["attention_masks"], torch.tensor([[True, True], [True, False]])))
        self.assertTrue(torch.equal(output["prompt_embeds"][1, 1], torch.zeros(2)))

    def test_transformer_device_tracks_x_embedder_for_offload(self):
        transformer = ErnieImageTransformer2DModel(
            hidden_size=12,
            num_attention_heads=2,
            num_layers=1,
            ffn_hidden_size=24,
            in_channels=4,
            out_channels=4,
            text_in_dim=12,
            rope_axes_dim=(2, 2, 2),
        )

        self.assertIsNone(transformer.onload_device)
        self.assertEqual(transformer.device, next(transformer.x_embedder.parameters()).device)

    def test_validation_preview_uses_flux2_tae_spec(self):
        self.assertIsInstance(Ernie.VALIDATION_PREVIEW_SPEC, Flux2TAESpec)

    def test_transformer_accepts_timestep_sign_and_skip_layers(self):
        transformer = ErnieImageTransformer2DModel(
            hidden_size=12,
            num_attention_heads=2,
            num_layers=2,
            ffn_hidden_size=24,
            in_channels=4,
            out_channels=4,
            text_in_dim=12,
            rope_axes_dim=(2, 2, 2),
            enable_time_sign_embed=True,
        )

        output = transformer(
            hidden_states=torch.randn(1, 4, 2, 2),
            timestep=torch.tensor([0.5]),
            timestep_sign=torch.tensor([-1.0]),
            text_bth=torch.randn(1, 1, 12),
            text_lens=torch.tensor([1]),
            skip_layers=[1],
            return_dict=False,
        )[0]

        self.assertEqual(tuple(output.shape), (1, 4, 2, 2))

    def test_transformer_tread_routes_4d_rotary_embeddings(self):
        transformer = ErnieImageTransformer2DModel(
            hidden_size=12,
            num_attention_heads=2,
            num_layers=1,
            ffn_hidden_size=24,
            in_channels=4,
            out_channels=4,
            text_in_dim=12,
            rope_axes_dim=(2, 2, 2),
        )
        transformer.set_router(
            TREADRouter(seed=1, device=torch.device("cpu")),
            [{"start_layer_idx": 0, "end_layer_idx": 0, "selection_ratio": 0.5}],
        )

        with torch.enable_grad():
            output = transformer(
                hidden_states=torch.randn(1, 4, 2, 2, requires_grad=True),
                timestep=torch.tensor([0.5]),
                text_bth=torch.randn(1, 1, 12),
                text_lens=torch.tensor([1]),
                return_dict=False,
            )[0]

        self.assertEqual(tuple(output.shape), (1, 4, 2, 2))

    def test_transformer_requires_time_sign_embedding_when_timestep_sign_used(self):
        transformer = ErnieImageTransformer2DModel(
            hidden_size=12,
            num_attention_heads=2,
            num_layers=1,
            ffn_hidden_size=24,
            in_channels=4,
            out_channels=4,
            text_in_dim=12,
            rope_axes_dim=(2, 2, 2),
        )

        with self.assertRaisesRegex(ValueError, "enable_time_sign_embed"):
            transformer(
                hidden_states=torch.randn(1, 4, 2, 2),
                timestep=torch.tensor([0.5]),
                timestep_sign=torch.tensor([-1.0]),
                text_bth=torch.randn(1, 1, 12),
                text_lens=torch.tensor([1]),
                return_dict=False,
            )

    def test_pretrained_load_args_enable_twinflow_and_musubi(self):
        model = self._build_model()
        model.config.twinflow_enabled = True
        model.config.musubi_blocks_to_swap = 4
        model.config.musubi_block_swap_device = "cpu"

        args = model.pretrained_load_args({})

        self.assertTrue(args["enable_time_sign_embed"])
        self.assertEqual(args["musubi_blocks_to_swap"], 4)
        self.assertEqual(args["musubi_block_swap_device"], "cpu")

    def test_model_predict_shapes_and_timesteps(self):
        model = self._build_model()
        latents = torch.randn(2, 128, 8, 8)
        timesteps = torch.tensor([100.0, 500.0])
        prompt_embeds = torch.randn(2, 5, 6)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)

        output = model.model_predict(
            {
                "noisy_latents": latents,
                "timesteps": timesteps,
                "encoder_hidden_states": prompt_embeds,
                "encoder_attention_mask": attention_mask,
            }
        )

        noise_pred = output["model_prediction"]
        self.assertEqual(noise_pred.shape, latents.shape)

        received_latents, received_t, text_bth, text_lens, kwargs = model.model.received
        self.assertEqual(received_latents.shape, latents.shape)
        self.assertTrue(torch.allclose(received_t, torch.tensor([100.0, 500.0])))
        self.assertTrue(torch.equal(text_lens, torch.tensor([3, 2])))
        self.assertEqual(text_bth.shape[:2], (2, 3))
        self.assertNotIn("hidden_states_buffer", kwargs)
        self.assertIsNone(output["hidden_states_buffer"])

    def test_model_predict_passes_twinflow_time_sign(self):
        model = self._build_model()
        model.config.twinflow_enabled = True
        time_sign = torch.tensor([-1.0, 1.0])

        model.model_predict(
            {
                "noisy_latents": torch.randn(2, 128, 8, 8),
                "timesteps": torch.tensor([100.0, 500.0]),
                "encoder_hidden_states": torch.randn(2, 5, 6),
                "encoder_attention_mask": torch.ones(2, 5, dtype=torch.bool),
                "twinflow_time_sign": time_sign,
            }
        )

        kwargs = model.model.received[-1]
        self.assertIs(kwargs["timestep_sign"], time_sign)


if __name__ == "__main__":
    unittest.main()
