import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from simpletuner.helpers.models.common import PipelineTypes, TextEmbedCacheKey
from simpletuner.helpers.models.minimaxmusic.condition_encoder import MiniMaxMusic3ConditionEncoder
from simpletuner.helpers.models.minimaxmusic.model import MiniMaxMusic
from simpletuner.helpers.models.minimaxmusic.modular_pipeline import (
    MINIMAX_MUSIC_SWIGLU_GATE_FIRST_METADATA_KEY,
    MiniMaxMusic3ModularPipeline,
    _convert_minimax_music_comfy_lora_to_diffusers,
    _convert_minimax_music_diffusers_lora_to_comfyui,
)
from simpletuner.helpers.models.minimaxmusic.transformer import MiniMaxMusic3AttnProcessor, MiniMaxMusic3Transformer1DModel
from simpletuner.helpers.models.minimaxmusic.vocoder import MiniMaxMusic3DAV
from simpletuner.helpers.models.registry import ModelRegistry


def _tiny_transformer(enable_time_sign_embed: bool = False, **kwargs) -> MiniMaxMusic3Transformer1DModel:
    return MiniMaxMusic3Transformer1DModel(
        in_channels=4,
        condition_dim=8,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=6,
        ff_inner_dim=16,
        rotary_dim=4,
        fourier_embedding_dim=8,
        enable_time_sign_embed=enable_time_sign_embed,
        **kwargs,
    )


class FakeLoraTarget:
    def __init__(self, *, swiglu_gate_first: bool = False):
        self.config = SimpleNamespace(swiglu_gate_first=swiglu_gate_first)
        self.calls = []
        self.flowmap_deltatime_type = None
        self.flowmap_gate_value = None

    def enable_flowmap_time_conditioning(self, gate_value: float = 0.25, deltatime_type: str = "r"):
        self.flowmap_gate_value = gate_value
        self.flowmap_deltatime_type = deltatime_type

    def load_lora_adapter(self, state_dict, **kwargs):
        self.calls.append((state_dict, kwargs))


class MiniMaxMusicModelTests(unittest.TestCase):
    def _build_model(self, transformer=None):
        model = MiniMaxMusic.__new__(MiniMaxMusic)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"), is_local_main_process=True)
        model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            base_weight_dtype=torch.float32,
            flow_schedule_shift=1.0,
            input_perturbation=0.0,
            input_perturbation_steps=None,
            logit_mean=0.0,
            logit_std=1.0,
            twinflow_enabled=False,
            distillation_method=None,
            distillation_config={},
            lora_type="standard",
            controlnet=False,
            model_family="minimaxmusic",
            model_flavour="music3",
            pretrained_model_name_or_path=None,
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
        )
        model.model = transformer or _tiny_transformer(enable_time_sign_embed=True)
        model.condition_encoder = None
        model.crepa_regularizer = None
        model.layersync_regularizer = None
        model.vae = None
        model._new_hidden_state_buffer = MagicMock(return_value={})
        model.get_trained_component = lambda: model.model
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        model._twinflow_active = lambda: False
        return model

    def test_registry_exposes_minimaxmusic_family(self):
        registered = ModelRegistry.get("minimaxmusic")

        self.assertIsNotNone(registered)
        self.assertEqual(registered.NAME, "MiniMax Music 3")
        self.assertIn("music3", registered.get_flavour_choices())

    @patch("simpletuner.helpers.training.offloaded_gradient_checkpointer.offloaded_checkpoint")
    def test_unsloth_checkpointing_backend_uses_offloaded_checkpoint(self, offloaded_checkpoint):
        transformer = _tiny_transformer().train()
        transformer.gradient_checkpointing = True
        transformer.set_gradient_checkpointing_backend("unsloth")
        transformer.set_gradient_checkpointing_interval(2)
        offloaded_checkpoint.side_effect = lambda function, *args, **kwargs: function(*args)

        transformer(
            hidden_states=torch.randn(1, 4, 6, requires_grad=True),
            timestep=torch.ones(1),
            encoder_hidden_states=torch.randn(1, 6, 8),
        )

        offloaded_checkpoint.assert_called_once()
        self.assertFalse(offloaded_checkpoint.call_args.kwargs["use_reentrant"])

    def test_minimaxmusic_supports_audio_only_training(self):
        self.assertTrue(MiniMaxMusic.supports_audio_only_training())

    def test_minimaxmusic_inherits_model_card_schedule_hook(self):
        model = self._build_model()

        self.assertEqual(model.custom_model_card_schedule_info(), "")

    def test_minimaxmusic_validation_audio_sample_rate(self):
        model = self._build_model()

        self.assertEqual(model.validation_audio_sample_rate(), 44100)

    @patch("simpletuner.helpers.models.minimaxmusic.transformer.dispatch_attention_fn")
    def test_attention_processor_casts_sdnq_fp32_outputs_to_autocast_dtype(self, dispatch_attention_fn):
        dispatch_attention_fn.side_effect = lambda query, key, value, **kwargs: query

        class FP32Projection(torch.nn.Module):
            def forward(self, hidden_states):
                return hidden_states.float()

        attention = SimpleNamespace(
            to_q=FP32Projection(),
            to_k=FP32Projection(),
            to_v=FP32Projection(),
            to_out=torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()]),
            heads=2,
            head_dim=4,
        )
        hidden_states = torch.randn(1, 3, 8)
        rotary_emb = (torch.ones(3, 4), torch.zeros(3, 4))

        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = MiniMaxMusic3AttnProcessor()(attention, hidden_states, rotary_emb)

        query, key, value = dispatch_attention_fn.call_args.args[:3]
        self.assertEqual(query.dtype, torch.bfloat16)
        self.assertEqual(key.dtype, torch.bfloat16)
        self.assertEqual(value.dtype, torch.bfloat16)
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_validation_kwargs_restore_raw_prompt_for_modular_pipeline(self):
        model = self._build_model()
        model.config.validation_lyrics = "[verse]\nhello"
        prompt_embeds = torch.randn(1, 2, 3)
        pipeline_kwargs = {
            "prompt": None,
            "_validation_prompt_text": "bright synth pop",
            "frame_hiddens": prompt_embeds,
            "prompt_embeds": prompt_embeds,
            "attention_masks": torch.tensor([2]),
        }

        updated = model.update_pipeline_call_kwargs(pipeline_kwargs)

        self.assertEqual(updated["prompt"], "bright synth pop")
        self.assertEqual(updated["lyrics"], "[verse]\nhello")
        self.assertNotIn("frame_hiddens", updated)
        self.assertNotIn("prompt_embeds", updated)
        self.assertNotIn("attention_masks", updated)

    def test_validation_kwargs_replace_blank_prompt_library_lyrics(self):
        model = self._build_model()
        model.config.validation_lyrics = "[verse]\nconfigured"
        pipeline_kwargs = {
            "prompt": "instrumental jazz",
            "_validation_prompt_text": "instrumental jazz",
            "lyrics": "",
        }

        updated = model.update_pipeline_call_kwargs(pipeline_kwargs)

        self.assertEqual(updated["lyrics"], "[verse]\nconfigured")

    def test_validation_kwargs_fall_back_to_prompt_without_configured_lyrics(self):
        model = self._build_model()
        model.config.validation_lyrics = ""
        pipeline_kwargs = {
            "prompt": "cinematic ambient instrumental",
            "_validation_prompt_text": "cinematic ambient instrumental",
            "lyrics": "",
        }

        updated = model.update_pipeline_call_kwargs(pipeline_kwargs)

        self.assertEqual(updated["lyrics"], "cinematic ambient instrumental")

    def test_load_text_encoder_restores_tokenizer_after_validation_clear(self):
        model = self._build_model()
        model.language_model = object()
        model.rvq_depth_decoder = object()
        model.condition_encoder = object()
        model.tokenizers = None

        def restore_tokenizer():
            model.tokenizers = ["tokenizer"]
            model.tokenizer_1 = "tokenizer"

        model.load_text_tokenizer = MagicMock(side_effect=restore_tokenizer)

        model.load_text_encoder(move_to_device=False)

        model.load_text_tokenizer.assert_called_once_with()
        self.assertEqual(model.tokenizers, ["tokenizer"])

    def test_unload_text_encoder_clears_minimax_conditioning_aliases(self):
        model = self._build_model()
        model.language_model = object()
        model.rvq_depth_decoder = object()
        model.condition_encoder = object()
        model.text_encoder_1 = model.language_model
        model.text_encoders = [model.language_model]
        model.tokenizers = ["tokenizer"]

        model.unload_text_encoder()

        self.assertIsNone(model.language_model)
        self.assertIsNone(model.rvq_depth_decoder)
        self.assertIsNone(model.condition_encoder)
        self.assertIsNone(model.text_encoder_1)
        self.assertIsNone(model.text_encoders)
        self.assertIsNone(model.tokenizers)

    def test_cached_frame_hiddens_only_reload_condition_encoder(self):
        model = self._build_model()
        condition_encoder = torch.nn.Identity()

        def restore_condition_encoder(move_to_device=True):
            del move_to_device
            model.condition_encoder = condition_encoder
            return condition_encoder

        model.load_condition_encoder = MagicMock(side_effect=restore_condition_encoder)
        model.load_text_encoder = MagicMock()
        frame_hiddens = torch.randn(1, 5, 8)

        condition = model._condition_from_frame_hiddens(frame_hiddens, latent_length=5)

        model.load_condition_encoder.assert_called_once_with(move_to_device=True)
        model.load_text_encoder.assert_not_called()
        torch.testing.assert_close(condition, frame_hiddens)

    @patch("simpletuner.helpers.models.minimaxmusic.model.MiniMaxMusic3ModularPipeline")
    def test_get_pipeline_restores_tokenizer_after_validation_clear(self, mock_pipeline_cls):
        pipeline = MagicMock()
        mock_pipeline_cls.return_value = pipeline
        model = self._build_model()
        model.pipelines = {}
        model.language_model = object()
        model.rvq_depth_decoder = object()
        model.condition_encoder = object()
        model.tokenizers = None
        model.vae = object()
        model.guider = object()

        def restore_text_stack(move_to_device=True):
            del move_to_device
            model.tokenizers = ["tokenizer"]
            model.tokenizer_1 = "tokenizer"

        model.load_text_encoder = MagicMock(side_effect=restore_text_stack)

        result = model.get_pipeline()

        self.assertIs(result, pipeline)
        model.load_text_encoder.assert_called_once_with(move_to_device=True)
        self.assertEqual(pipeline.update_components.call_args.kwargs["tokenizer"], "tokenizer")

    def test_text_embed_cache_uses_audio_sample_context_keys(self):
        model = self._build_model()

        self.assertEqual(model.text_embed_cache_key(), TextEmbedCacheKey.DATASET_AND_FILENAME)
        self.assertTrue(model.requires_text_embed_image_context())
        self.assertFalse(model.should_precompute_dropout_caption())
        self.assertFalse(model.use_text_cache_dropout_sentinel())
        self.assertTrue(model.uses_image_context_dropout_caption_cache())
        self.assertEqual(
            model.text_embed_cache_key_value(prompt="", default_key="music:track.wav", metadata={}),
            "music:track.wav:__caption_dropout__",
        )
        self.assertEqual(
            model.text_embed_cache_key_value(prompt="bright synth pop", default_key="music:track.wav", metadata={}),
            "music:track.wav",
        )

    def test_constructed_model_initializes_vae_slot(self):
        config = SimpleNamespace(
            model_family="minimaxmusic",
            model_flavour="music3",
            pretrained_model_name_or_path="MiniMaxAI/MiniMax-Music3",
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
        )
        model = MiniMaxMusic(config=config, accelerator=SimpleNamespace())

        self.assertTrue(hasattr(model, "vae"))
        self.assertTrue(hasattr(model, "controlnet"))
        self.assertIsNone(model.controlnet)
        self.assertIsNone(model.vae)

    def test_text_embed_cache_metadata_includes_lyrics_and_audio_duration(self):
        model = self._build_model()

        metadata = model.text_embed_cache_metadata_for_sample(
            example={
                "lyrics": "[verse]\nhello world",
                "audio_duration": 12.5,
                "image_metadata": {"duration_seconds": 10.0},
            },
            latent=torch.zeros(1),
            prompt="bright synth pop",
            data_backend_id="music",
            dataset_relative_path="track_01.wav",
        )

        self.assertEqual(metadata["prompt"], "bright synth pop")
        self.assertEqual(metadata["lyrics"], "[verse]\nhello world")
        self.assertEqual(metadata["audio_duration"], 12.5)
        self.assertEqual(metadata["duration_seconds"], 10.0)

    def test_text_embed_cache_metadata_accepts_training_sample_metadata(self):
        model = self._build_model()
        sample = SimpleNamespace(image_metadata={"lyrics": "la la", "duration": 8.0})

        metadata = model.text_embed_cache_metadata_for_sample(
            example=sample,
            latent=None,
            prompt="warm piano ballad",
            data_backend_id=None,
            dataset_relative_path=None,
        )

        self.assertEqual(metadata["prompt"], "warm piano ballad")
        self.assertEqual(metadata["lyrics"], "la la")
        self.assertEqual(metadata["duration"], 8.0)

    def test_text_embed_cache_metadata_for_filepath_uses_audio_metadata_backend(self):
        model = self._build_model()
        metadata_backend = MagicMock()
        metadata_backend.get_metadata_by_filepath.return_value = {
            "lyrics": "[chorus]\nshine",
            "duration_seconds": 1.0,
            "bucket_duration_seconds": 1.0,
        }

        metadata = model.text_embed_cache_metadata_for_filepath(
            init_backend={"metadata_backend": metadata_backend},
            image_path="track.wav",
            prompt="bright synth pop",
            data_backend_id="music",
            dataset_relative_path="track.wav",
        )

        metadata_backend.get_metadata_by_filepath.assert_called_once_with("track.wav")
        self.assertEqual(metadata["prompt"], "bright synth pop")
        self.assertEqual(metadata["lyrics"], "[chorus]\nshine")
        self.assertEqual(metadata["duration_seconds"], 1.0)
        self.assertEqual(metadata["bucket_duration_seconds"], 1.0)

    def test_audio_duration_for_context_accepts_audio_bucket_duration(self):
        model = self._build_model()
        model.config.validation_audio_duration = 30.0

        self.assertEqual(model._audio_duration_for_context({"bucket_duration_seconds": 1.0}), 1.0)

    def test_dav_encodes_raw_audio_to_stereo_latents(self):
        dav = MiniMaxMusic3DAV(
            latent_channels=4,
            channel_latent_channels=2,
            encoder_dim=2,
            encoder_rates=(2,),
            encoder_latent_dim=4,
            decoder_input_dim=4,
            decoder_hidden_dim=4,
            upsampling_ratios=(2,),
        )

        latents = dav.encode(torch.randn(1, 1, 16))
        waveform = dav.decode(latents)

        self.assertEqual(latents.shape, (1, 4, 8))
        self.assertEqual(waveform.shape, (1, 2, 16))

    def test_encode_cache_batch_uses_dav_encoder(self):
        model = self._build_model()
        dav = MiniMaxMusic3DAV(
            latent_channels=4,
            channel_latent_channels=2,
            encoder_dim=2,
            encoder_rates=(2,),
            encoder_latent_dim=4,
            decoder_input_dim=4,
            decoder_hidden_dim=4,
            upsampling_ratios=(2,),
        )

        latents = model.encode_cache_batch(dav, torch.randn(1, 1, 16))

        self.assertEqual(latents.shape, (1, 4, 8))

    def test_load_vae_accepts_diffusers_audio_vae_subfolder(self):
        model = self._build_model()
        dav = MiniMaxMusic3DAV(
            latent_channels=4,
            channel_latent_channels=2,
            encoder_dim=2,
            encoder_rates=(2,),
            encoder_latent_dim=4,
            decoder_input_dim=4,
            decoder_hidden_dim=4,
            upsampling_ratios=(2,),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_vae_dir = f"{tmpdir}/audio_vae"
            dav.save_pretrained(audio_vae_dir)
            model.config.pretrained_vae_model_name_or_path = tmpdir

            loaded = model.load_vae(move_to_device=False)

        self.assertIsInstance(loaded, MiniMaxMusic3DAV)
        self.assertEqual(loaded.config.latent_channels, 4)

    def test_condition_encoder_resamples_frame_hidden_states(self):
        encoder = MiniMaxMusic3ConditionEncoder(
            condition_hidden_dim=8,
            num_condition_layers=2,
            out_dim=8,
            input_sampling_rate=24000,
            input_hop_length=960,
            output_sampling_rate=44100,
            output_hop_length=512,
        )

        output = encoder(torch.randn(1, 5, 16))

        self.assertEqual(output.shape, (1, 17, 8))

    def test_transformer_supports_tiny_forward_and_hidden_capture(self):
        transformer = _tiny_transformer(enable_time_sign_embed=True)
        transformer.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
        hidden_states_buffer = {}

        output = transformer(
            hidden_states=torch.randn(1, 4, 6),
            timestep=torch.tensor([0.5]),
            encoder_hidden_states=torch.randn(1, 6, 8),
            timestep_sign=torch.tensor([-1.0]),
            r_timestep=torch.tensor([0.8]),
            skip_layers=[99],
            hidden_states_buffer=hidden_states_buffer,
            output_hidden_states=True,
            hidden_state_layer=1,
            return_dict=True,
        )

        self.assertEqual(output.sample.shape, (1, 4, 6))
        self.assertEqual(output.hidden_states.shape, (1, 6, 12))
        self.assertEqual(sorted(hidden_states_buffer), ["layer_0", "layer_1"])

    def test_transformer_activates_configured_musubi_block_swap(self):
        transformer = _tiny_transformer(enable_time_sign_embed=True)
        manager = MagicMock()
        manager.activate.return_value = True
        manager.is_managed_block.side_effect = lambda index: index == 1
        transformer._musubi_block_swap = manager

        transformer(
            hidden_states=torch.randn(1, 4, 6),
            timestep=torch.tensor([0.5]),
            encoder_hidden_states=torch.randn(1, 6, 8),
            return_dict=True,
        )

        manager.activate.assert_called_once()
        manager.stream_in.assert_called_once_with(transformer.transformer_blocks[1], torch.device("cpu"))
        manager.stream_out.assert_called_once_with(transformer.transformer_blocks[1])

    def test_transformer_accepts_tokenwise_timesteps_for_self_flow(self):
        transformer = _tiny_transformer(enable_time_sign_embed=True)
        transformer.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="t-r")

        output = transformer(
            hidden_states=torch.randn(1, 4, 5),
            timestep=torch.tensor([[0.2, 0.8, 0.2, 0.8, 0.2]]),
            encoder_hidden_states=torch.randn(1, 5, 8),
            timestep_sign=torch.tensor([1.0]),
            r_timestep=torch.tensor([[0.1, 0.7, 0.1, 0.7, 0.1]]),
            return_dict=False,
        )

        self.assertEqual(output[0].shape, (1, 4, 5))

    def test_transformer_propagates_swiglu_gate_first_config(self):
        transformer = _tiny_transformer(swiglu_gate_first=True)

        self.assertTrue(transformer.config.swiglu_gate_first)
        self.assertTrue(transformer.transformer_blocks[0].swiglu_gate_first)

    def test_transformer_supports_peft_adapter_injection(self):
        from peft import LoraConfig

        transformer = _tiny_transformer()

        transformer.add_adapter(
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff_in", "ff_out", "proj_in", "proj_out"],
            )
        )

        trainable_lora_parameters = [
            name for name, parameter in transformer.named_parameters() if "lora_" in name and parameter.requires_grad
        ]
        self.assertTrue(trainable_lora_parameters)
        self.assertIn("transformer_blocks.0.attn.to_q.lora_A.default.weight", trainable_lora_parameters)

    def test_transformer_requires_initialized_timestep_conditioning(self):
        transformer = _tiny_transformer()
        kwargs = {
            "hidden_states": torch.randn(1, 4, 4),
            "timestep": torch.tensor([0.5]),
            "encoder_hidden_states": torch.randn(1, 4, 8),
        }

        with self.assertRaisesRegex(ValueError, "enable_time_sign_embed"):
            transformer(**kwargs, timestep_sign=torch.tensor([1.0]))
        with self.assertRaisesRegex(ValueError, "FlowMap"):
            transformer(**kwargs, r_timestep=torch.tensor([0.25]))

    def test_prepare_batch_uses_precomputed_latents_and_data_ward_flow_target(self):
        model = self._build_model()
        model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([0.25, 0.75]), torch.tensor([0.75, 0.25])))
        batch = {
            "latent_batch": torch.randn(2, 6, 4),
            "encoder_hidden_states": torch.randn(2, 6, 8),
        }

        prepared = model.prepare_batch(batch, state={"global_step": 0})

        self.assertEqual(prepared["latents"].shape, (2, 4, 6))
        self.assertEqual(prepared["encoder_hidden_states"].shape, (2, 6, 8))
        self.assertEqual(prepared["sigmas"].shape, (2, 1, 1))
        torch.testing.assert_close(prepared["timesteps"], torch.tensor([0.75, 0.25]))
        torch.testing.assert_close(model.get_prediction_target(prepared), prepared["latents"] - prepared["noise"])

    def test_prepare_batch_accepts_audio_only_collate_latents(self):
        model = self._build_model()
        model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([0.4]), torch.tensor([0.6])))
        audio_latents = torch.randn(1, 4, 6)
        batch = {
            "latent_batch": None,
            "audio_latent_batch": audio_latents,
            "prompt_embeds": torch.randn(1, 6, 8),
        }

        prepared = model.prepare_batch(batch, state={"global_step": 0})

        torch.testing.assert_close(prepared["latents"], audio_latents)
        self.assertEqual(prepared["encoder_hidden_states"].shape, (1, 6, 8))
        torch.testing.assert_close(prepared["timesteps"], torch.tensor([0.6]))

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        model = self._build_model()
        model.config.crepa_self_flow_mask_ratio = 0.5
        model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([0.7]), torch.tensor([0.3])))
        batch = {
            "latents": torch.zeros(1, 4, 5, dtype=torch.float32),
            "input_noise": torch.ones(1, 4, 5, dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
            "timesteps": torch.tensor([0.8], dtype=torch.float32),
        }
        fake_mask_rand = torch.tensor([[0.2, 0.7, 0.1, 0.9, 0.4]], dtype=torch.float32)

        with patch("torch.rand", return_value=fake_mask_rand):
            result = model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["sigmas"].shape, (1, 1, 5))
        self.assertEqual(result["timesteps"].shape, (1, 5))
        torch.testing.assert_close(result["timesteps"], torch.tensor([[0.3, 0.8, 0.3, 0.8, 0.3]]))
        self.assertTrue(torch.equal(result["crepa_self_flow_mask"], fake_mask_rand < 0.5))
        torch.testing.assert_close(result["crepa_teacher_timesteps"], torch.tensor([0.8]))

    def test_model_predict_forwards_feature_kwargs_and_captures_hidden_states(self):
        transformer = _tiny_transformer(enable_time_sign_embed=True)
        transformer.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
        model = self._build_model(transformer)
        model.config.twinflow_enabled = True
        model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=1, wants_hidden_states=lambda: True)
        hidden_states_buffer = {}
        model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        result = model.model_predict(
            {
                "noisy_latents": torch.randn(2, 4, 6),
                "encoder_hidden_states": torch.randn(2, 6, 8),
                "timesteps": torch.tensor([250.0, 750.0]),
                "sigmas": torch.tensor([0.25, 0.75]),
                "twinflow_time_sign": torch.tensor([-1.0, 1.0]),
                MiniMaxMusic.FLOWMAP_R_TIMESTEP_BATCH_KEY: torch.tensor([0.8, 0.6]),
            }
        )

        self.assertEqual(result["model_prediction"].shape, (2, 4, 6))
        self.assertEqual(result["crepa_hidden_states"].shape, (2, 6, 12))
        self.assertEqual(sorted(hidden_states_buffer), ["layer_0", "layer_1"])

    def test_timestep_helpers_convert_noise_sigmas_to_minimax_data_time(self):
        model = self._build_model()

        converted = model.flow_matching_timesteps_from_sigmas(torch.tensor([0.25, 0.75]))
        torch.testing.assert_close(converted, torch.tensor([0.75, 0.25]))

        scalar_timesteps = model._timesteps_for_transformer(
            {"timesteps": torch.tensor([250.0]), "sigmas": torch.tensor([0.25])},
            torch.zeros(1, 4, 6),
        )
        torch.testing.assert_close(scalar_timesteps, torch.tensor([0.75]))

        tokenwise_timesteps = model._timesteps_for_transformer(
            {
                "timesteps": torch.tensor([[250.0, 750.0, 250.0]], dtype=torch.float32),
                "sigmas": torch.tensor([[[0.25, 0.75, 0.25]]], dtype=torch.float32),
            },
            torch.zeros(1, 4, 3),
        )
        torch.testing.assert_close(tokenwise_timesteps, torch.tensor([[0.75, 0.25, 0.75]]))

    def test_comfy_lora_conversion_maps_music_names_and_splits_qkv(self):
        qkv_down = torch.randn(4, 12)
        qkv_up = torch.randn(36, 4)
        ff_down = torch.randn(2, 12)
        value_rows = torch.full((3, 2), 2.0)
        gate_rows = torch.full((3, 2), 1.0)
        state_dict = {
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_A.weight": qkv_down,
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_B.weight": qkv_up,
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.alpha": torch.tensor(4.0),
            "diffusion_model.diffusion_transformer.transformer.layers.0.ff.ff.0.proj.lora_A.weight": ff_down,
            "diffusion_model.diffusion_transformer.transformer.layers.0.ff.ff.0.proj.lora_B.weight": torch.cat(
                (value_rows, gate_rows), dim=0
            ),
            "diffusion_model.diffusion_transformer.to_timestep_embed.0.lora_A.weight": torch.randn(2, 8),
            "diffusion_model.diffusion_transformer.to_timestep_embed.0.lora_B.weight": torch.randn(12, 2),
        }

        converted, network_alphas = _convert_minimax_music_comfy_lora_to_diffusers(
            state_dict,
            target_prefix="transformer",
        )

        q_key = "transformer.transformer_blocks.0.attn.to_q.lora.down.weight"
        k_key = "transformer.transformer_blocks.0.attn.to_k.lora.down.weight"
        v_key = "transformer.transformer_blocks.0.attn.to_v.lora.down.weight"
        self.assertTrue(torch.equal(converted[q_key], qkv_down))
        self.assertTrue(torch.equal(converted[k_key], qkv_down))
        self.assertTrue(torch.equal(converted[v_key], qkv_down))
        self.assertTrue(torch.equal(converted["transformer.transformer_blocks.0.attn.to_q.lora.up.weight"], qkv_up[:12]))
        self.assertTrue(torch.equal(converted["transformer.transformer_blocks.0.attn.to_k.lora.up.weight"], qkv_up[12:24]))
        self.assertTrue(torch.equal(converted["transformer.transformer_blocks.0.attn.to_v.lora.up.weight"], qkv_up[24:]))
        self.assertTrue(
            torch.equal(
                converted["transformer.transformer_blocks.0.ff_in.lora.up.weight"],
                torch.cat((value_rows, gate_rows), dim=0),
            )
        )
        self.assertIn("transformer.time_embed.linear_1.lora.down.weight", converted)
        self.assertEqual(network_alphas["transformer.transformer_blocks.0.attn.to_q.alpha"], 4.0)
        self.assertEqual(network_alphas["transformer.transformer_blocks.0.attn.to_k.alpha"], 4.0)
        self.assertEqual(network_alphas["transformer.transformer_blocks.0.attn.to_v.alpha"], 4.0)

    def test_comfy_lora_conversion_swaps_swiglu_rows_for_gate_first_target(self):
        value_rows = torch.full((3, 2), 2.0)
        gate_rows = torch.full((3, 2), 1.0)

        converted, _network_alphas = _convert_minimax_music_comfy_lora_to_diffusers(
            {
                "diffusion_transformer.transformer.layers.0.ff.ff.0.proj.lora_A.weight": torch.randn(2, 12),
                "diffusion_transformer.transformer.layers.0.ff.ff.0.proj.lora_B.weight": torch.cat(
                    (value_rows, gate_rows), dim=0
                ),
            },
            target_prefix="transformer",
            target_swiglu_gate_first=True,
        )

        self.assertTrue(
            torch.equal(
                converted["transformer.transformer_blocks.0.ff_in.lora.up.weight"],
                torch.cat((gate_rows, value_rows), dim=0),
            )
        )

    def test_comfy_lora_export_fuses_qkv_and_maps_swiglu(self):
        state_dict = {}
        expected_deltas = []
        for index, projection in enumerate(("to_q", "to_k", "to_v"), start=1):
            down = torch.full((2, 3), float(index))
            up = torch.full((4, 2), float(index + 3))
            prefix = f"transformer.transformer_blocks.0.attn.{projection}"
            state_dict[f"{prefix}.lora_A.weight"] = down
            state_dict[f"{prefix}.lora_B.weight"] = up
            expected_deltas.append(up @ down)
        gate_rows = torch.full((2, 2), 1.0)
        value_rows = torch.full((2, 2), 2.0)
        state_dict["transformer.transformer_blocks.0.ff_in.lora_A.weight"] = torch.randn(2, 3)
        state_dict["transformer.transformer_blocks.0.ff_in.lora_B.weight"] = torch.cat((gate_rows, value_rows), dim=0)

        converted = _convert_minimax_music_diffusers_lora_to_comfyui(
            state_dict,
            source_swiglu_gate_first=True,
        )

        prefix = "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv"
        fused_down = converted[f"{prefix}.lora_A.weight"]
        fused_up = converted[f"{prefix}.lora_B.weight"]
        self.assertEqual(tuple(fused_down.shape), (6, 3))
        self.assertEqual(tuple(fused_up.shape), (12, 6))
        self.assertEqual(converted[f"{prefix}.alpha"].item(), 6.0)
        self.assertTrue(torch.equal(fused_up @ fused_down, torch.cat(expected_deltas, dim=0)))
        self.assertTrue(
            torch.equal(
                converted["diffusion_model.diffusion_transformer.transformer.layers.0.ff.ff.0.proj.lora_B.weight"],
                torch.cat((value_rows, gate_rows), dim=0),
            )
        )

    def test_lora_loader_detects_native_comfy_music_layout(self):
        pipe = MiniMaxMusic3ModularPipeline.__new__(MiniMaxMusic3ModularPipeline)
        pipe.transformer = FakeLoraTarget()
        pipe.lora_state_dict = lambda _path, **_kwargs: {
            "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_A.weight": torch.randn(2, 12),
            "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_B.weight": torch.randn(36, 2),
        }

        pipe.load_lora_weights("unused", adapter_name="music", lora_format="comfyui")

        loaded_state_dict, kwargs = pipe.transformer.calls[0]
        self.assertIn("transformer.transformer_blocks.0.attn.to_q.lora.down.weight", loaded_state_dict)
        self.assertIn("transformer.transformer_blocks.0.attn.to_k.lora.down.weight", loaded_state_dict)
        self.assertIn("transformer.transformer_blocks.0.attn.to_v.lora.down.weight", loaded_state_dict)
        self.assertEqual(kwargs["adapter_name"], "music")

    def test_lora_loader_uses_metadata_for_ambiguous_diffusers_swiglu_layout(self):
        pipe = MiniMaxMusic3ModularPipeline.__new__(MiniMaxMusic3ModularPipeline)
        pipe.transformer = FakeLoraTarget(swiglu_gate_first=False)
        gate_rows = torch.full((2, 4), 1.0)
        value_rows = torch.full((2, 4), 2.0)
        state_dict = {
            "transformer.transformer_blocks.0.ff_in.lora_A.weight": torch.randn(4, 12),
            "transformer.transformer_blocks.0.ff_in.lora_B.weight": torch.cat((gate_rows, value_rows), dim=0),
        }
        pipe.lora_state_dict = lambda _path, **_kwargs: (
            state_dict,
            {"transformer.swiglu_gate_first": True},
        )

        pipe.load_lora_weights("unused", adapter_name="music")

        loaded_state_dict, _kwargs = pipe.transformer.calls[0]
        self.assertTrue(
            torch.equal(
                loaded_state_dict["transformer.transformer_blocks.0.ff_in.lora_B.weight"],
                torch.cat((value_rows, gate_rows), dim=0),
            )
        )

    def test_model_comfy_lora_save_uses_native_music_exporter(self):
        model = object.__new__(MiniMaxMusic)
        model.config = SimpleNamespace(
            controlnet=False,
            lora_format="comfyui",
            model_family="minimaxmusic",
            model_flavour="music3",
        )
        model.model = _tiny_transformer()
        model.accelerator = SimpleNamespace(unwrap_model=lambda model, keep_fp32_wrapper=True: model)
        pipeline_class = model.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(pipeline_class, "save_lora_weights") as save_lora_weights:
            model.save_lora_weights(
                tmpdir,
                transformer_lora_layers={"transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 3)},
            )
            save_function = save_lora_weights.call_args.kwargs["save_function"]
            output_path = f"{tmpdir}/music-comfy.safetensors"
            shared_down = torch.randn(2, 3)
            weights = {}
            for projection in ("to_q", "to_k", "to_v"):
                prefix = f"transformer.transformer_blocks.0.attn.{projection}"
                weights[f"{prefix}.lora_A.weight"] = shared_down.clone()
                weights[f"{prefix}.lora_B.weight"] = torch.randn(4, 2)
            save_function(weights, output_path)
            saved = load_file(output_path)
            with safe_open(output_path, framework="pt", device="cpu") as handle:
                adapter_metadata = json.loads(handle.metadata()["lora_adapter_metadata"])

        self.assertIn("diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_A.weight", saved)
        self.assertNotIn("diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight", saved)
        self.assertFalse(adapter_metadata[MINIMAX_MUSIC_SWIGLU_GATE_FIRST_METADATA_KEY])

    def test_training_resume_imports_native_comfy_lora_keys(self):
        wrapper = object.__new__(MiniMaxMusic)
        wrapper.config = SimpleNamespace(
            controlnet=False,
            lora_format=None,
            model_flavour="music3",
        )
        wrapper.model = _tiny_transformer(swiglu_gate_first=False)
        wrapper.controlnet = None
        wrapper.text_encoders = []
        wrapper.accelerator = SimpleNamespace(
            unwrap_model=lambda model, keep_fp32_wrapper=True: model,
        )
        native_state_dict = {
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_A.weight": torch.randn(2, 12),
            "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_B.weight": torch.randn(36, 2),
            "diffusion_model.diffusion_transformer.transformer.layers.0.ff.ff.0.proj.lora_A.weight": torch.randn(2, 12),
            "diffusion_model.diffusion_transformer.transformer.layers.0.ff.ff.0.proj.lora_B.weight": torch.randn(32, 2),
        }
        pipeline_class = wrapper.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]

        with (
            patch.object(pipeline_class, "lora_state_dict", return_value=native_state_dict),
            patch("peft.utils.set_peft_model_state_dict") as set_peft_state,
        ):
            set_peft_state.return_value = SimpleNamespace(unexpected_keys=[])
            wrapper.load_lora_weights([wrapper.model], "unused")

        loaded = set_peft_state.call_args.args[1]
        self.assertIn("transformer_blocks.0.attn.to_q.lora_A.weight", loaded)
        self.assertIn("transformer_blocks.0.attn.to_k.lora_A.weight", loaded)
        self.assertIn("transformer_blocks.0.attn.to_v.lora_A.weight", loaded)
        self.assertIn("transformer_blocks.0.ff_in.lora_A.weight", loaded)
        self.assertNotIn("diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_A.weight", loaded)


class MiniMaxMusicAnyFlowValidationWrapperTests(unittest.TestCase):
    def test_install_pipeline_hooks_wraps_transformer_and_injects_r(self):
        import numpy as np
        from diffusers import FlowMatchEulerDiscreteScheduler

        from simpletuner.helpers.distillation.anyflow.scheduler import AnyFlowValidationScheduler
        from simpletuner.helpers.models.minimaxmusic.modular_blocks import MiniMaxMusic3Blocks

        pipeline = MiniMaxMusic3ModularPipeline(MiniMaxMusic3Blocks())
        transformer = _tiny_transformer()
        transformer.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, invert_sigmas=True)
        pipeline.update_components(transformer=transformer, scheduler=scheduler)

        validation_scheduler = AnyFlowValidationScheduler(scheduler, num_train_timesteps=1)
        validation_scheduler.install_pipeline_hooks(pipeline, component_names=("transformer",))

        # The denoise blocks resolve the component exactly like this attribute access,
        # so a registry-backed pipeline must expose the wrapper here, not the raw module.
        wrapped = pipeline.transformer
        self.assertTrue(getattr(wrapped, "_anyflow_validation_wrapper", False))

        received = {}

        def capture_forward(*args, **kwargs):
            received.update(kwargs)
            return (torch.zeros(1, 4, 6),)

        transformer.forward = capture_forward

        sigmas = np.linspace(1.0, 0.25, 4)
        scheduler.set_timesteps(sigmas=sigmas, device="cpu")
        wrapped(
            hidden_states=torch.zeros(1, 4, 6),
            timestep=scheduler.timesteps[0].expand(1),
            encoder_hidden_states=torch.zeros(1, 6, 8),
            return_dict=False,
        )

        r_timestep = received.get("r_timestep")
        self.assertIsNotNone(r_timestep)
        self.assertTrue(torch.all(r_timestep >= received["timestep"]))
        self.assertTrue(torch.all(r_timestep <= 1.0))


class MiniMaxMusicExampleConfigTests(unittest.TestCase):
    def test_memory_tier_examples_use_tested_acceleration_settings(self):
        examples_root = Path(__file__).resolve().parents[1] / "simpletuner" / "examples"
        expected = {
            "24g": {"duration": 30, "quantize_via": "cpu", "blocks_to_swap": 35, "stride": 6},
            "32g": {"duration": 40, "quantize_via": "cpu", "blocks_to_swap": 18, "stride": 6},
            "48g": {"duration": 60, "quantize_via": "accelerator", "blocks_to_swap": 0, "stride": 12},
        }

        for tier, settings in expected.items():
            with self.subTest(tier=tier):
                config_path = examples_root / f"minimaxmusic-music3-{tier}.peft-lora" / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                data_path = examples_root / f"minimaxmusic-audio-{tier}.json"
                data_config = json.loads(data_path.read_text(encoding="utf-8"))

                self.assertEqual(config["attention_mechanism"], "flash-attn-varlen-hub")
                self.assertTrue(config["trust_remote_code"])
                self.assertEqual(config["base_model_precision"], "int8-sdnq")
                self.assertEqual(config["text_encoder_1_precision"], "int8-sdnq")
                self.assertTrue(config["sdnq_use_hadamard"])
                self.assertEqual(config["sdnq_hadamard_group_size"], 256)
                self.assertEqual(config["quantize_via"], settings["quantize_via"])
                self.assertTrue(config["dynamo_use_regional_compilation"])
                self.assertEqual(config["gradient_checkpointing_interval"], 2)
                self.assertEqual(config["gradient_checkpointing_segment_stride"], settings["stride"])
                self.assertEqual(config.get("musubi_blocks_to_swap", 0), settings["blocks_to_swap"])
                self.assertEqual(config["validation_audio_duration"], settings["duration"])
                self.assertEqual(data_config[0]["split"], "test")
                self.assertEqual(data_config[0]["audio"]["max_duration_seconds"], settings["duration"])


if __name__ == "__main__":
    unittest.main()
