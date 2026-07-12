import inspect
import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.models.common import TextEmbedCacheKey
from simpletuner.helpers.models.krea2 import Krea2, Krea2LoraLoaderMixin, Krea2Pipeline, Krea2Transformer2DModel
from simpletuner.helpers.models.krea2.transformer import Krea2Attention, _krea2_rope_freqs_dtype
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.tread import TREADRouter


class FakeKrea2Transformer:
    def __init__(self):
        self.config = SimpleNamespace(patch_size=2)
        self.last_call = None

    def __call__(self, **kwargs):
        self.last_call = kwargs
        return (torch.zeros_like(kwargs["hidden_states"]),)


class FakeLatentDistribution:
    def __init__(self, latents):
        self.latents = latents

    def sample(self):
        return self.latents


class Krea2VendoredModelTests(unittest.TestCase):
    def test_model_registry_resolves_krea2(self):
        registry_entry = ModelRegistry.get("krea2")
        model_class = registry_entry.get_real_class() if hasattr(registry_entry, "get_real_class") else registry_entry
        self.assertIs(model_class, Krea2)

    def test_krea2_reference_latents_config_field_is_parseable(self):
        from simpletuner.helpers.configuration.cmd_args import get_argument_parser

        parser = get_argument_parser()
        args = parser.parse_args(
            [
                "--model_family",
                "krea2",
                "--output_dir",
                "/tmp/simpletuner-test",
                "--model_type",
                "lora",
                "--optimizer",
                "adamw_bf16",
                "--data_backend_config",
                "/tmp/backend.json",
                "--krea2_reference_latents",
                "true",
            ]
        )

        self.assertTrue(args.krea2_reference_latents)

    def test_model_components_are_local_simpletuner_classes(self):
        self.assertEqual(Krea2Pipeline.__module__, "simpletuner.helpers.models.krea2.pipeline")
        self.assertEqual(Krea2Transformer2DModel.__module__, "simpletuner.helpers.models.krea2.transformer")
        self.assertEqual(Krea2LoraLoaderMixin.__module__, "simpletuner.helpers.models.krea2.lora_pipeline")
        self.assertEqual(Krea2.PROCESSOR_PATH, "Qwen/Qwen3-VL-4B-Instruct")
        self.assertIsNone(Krea2.PROCESSOR_SUBFOLDER)

    def test_rope_frequency_dtype_matches_pypi_diffusers_device_support(self):
        self.assertEqual(_krea2_rope_freqs_dtype(torch.device("cpu")), torch.float64)
        self.assertEqual(_krea2_rope_freqs_dtype(torch.device("cuda")), torch.float64)
        self.assertEqual(_krea2_rope_freqs_dtype(torch.device("mps")), torch.float32)

    def test_lora_loader_targets_transformer(self):
        self.assertEqual(Krea2LoraLoaderMixin._lora_loadable_modules, ["transformer"])
        self.assertEqual(Krea2LoraLoaderMixin.transformer_name, "transformer")

    def test_lora_save_accepts_transformer_metadata(self):
        parameters = inspect.signature(Krea2LoraLoaderMixin.save_lora_weights).parameters
        self.assertIn("transformer_lora_adapter_metadata", parameters)

    def test_pipeline_accepts_reference_image_for_validation(self):
        parameters = inspect.signature(Krea2Pipeline.__call__).parameters
        self.assertIn("reference_image", parameters)

    def test_pipeline_accepts_cfg_zero_star_and_skip_layer_guidance(self):
        parameters = inspect.signature(Krea2Pipeline.__call__).parameters
        self.assertIn("use_cfg_zero_star", parameters)
        self.assertIn("skip_guidance_layers", parameters)
        self.assertIn("skip_layer_guidance_scale", parameters)

    def test_reference_latents_enable_reference_dataset_hooks(self):
        model = Krea2.__new__(Krea2)
        model.config = SimpleNamespace(krea2_reference_latents=True)

        self.assertTrue(model.supports_conditioning_dataset())
        self.assertTrue(model.requires_conditioning_dataset())
        self.assertTrue(model.requires_conditioning_validation_inputs())
        self.assertTrue(model.requires_validation_edit_captions())
        self.assertTrue(model.requires_text_embed_image_context())
        self.assertTrue(model.requires_conditioning_latents())
        self.assertFalse(model.should_precompute_validation_negative_prompt())
        self.assertEqual(model.text_embed_cache_key(), TextEmbedCacheKey.DATASET_AND_FILENAME)

    def test_reference_latents_maps_validation_image_to_reference_image(self):
        model = Krea2.__new__(Krea2)
        model.config = SimpleNamespace(krea2_reference_latents=True)

        kwargs = model.update_pipeline_call_kwargs({"image": "reference"})

        self.assertEqual(kwargs, {"reference_image": "reference"})

    def test_reference_latents_disabled_keeps_text_to_image_hooks(self):
        model = Krea2.__new__(Krea2)
        model.config = SimpleNamespace(krea2_reference_latents=False, control=False, controlnet=False)

        self.assertTrue(model.supports_conditioning_dataset())
        self.assertFalse(model.requires_conditioning_dataset())
        self.assertFalse(model.requires_conditioning_validation_inputs())
        self.assertFalse(model.requires_validation_edit_captions())
        self.assertFalse(model.requires_text_embed_image_context())
        self.assertFalse(model.requires_conditioning_latents())
        self.assertTrue(model.should_precompute_validation_negative_prompt())
        self.assertEqual(model.text_embed_cache_key(), TextEmbedCacheKey.CAPTION)

    def test_fused_qkv_lora_targets_use_fused_projection(self):
        model = Krea2.__new__(Krea2)
        model.config = SimpleNamespace(fuse_qkv_projections=True)

        self.assertEqual(model.get_lora_target_layers(), ["to_qkv", "to_out.0"])

    def test_krea2_attention_fused_projection_matches_unfused_path(self):
        attention = Krea2Attention(hidden_size=8, num_heads=2, num_kv_heads=1)
        hidden_states = torch.randn(2, 5, 8)

        unfused = attention(hidden_states)
        attention.fuse_projections()
        fused = attention(hidden_states)

        self.assertTrue(torch.allclose(fused, unfused, atol=1e-6, rtol=1e-6))
        self.assertTrue(hasattr(attention, "to_qkv"))

        attention.unfuse_projections()
        self.assertFalse(hasattr(attention, "to_qkv"))
        self.assertFalse(attention.fused_projections)

    def test_model_supports_crepa_self_flow(self):
        model = Krea2.__new__(Krea2)

        self.assertTrue(model.supports_crepa_self_flow())

    def test_pretrained_load_args_enable_twinflow_and_musubi(self):
        model = Krea2.__new__(Krea2)
        model.config = SimpleNamespace(
            twinflow_enabled=True,
            musubi_blocks_to_swap=3,
            musubi_block_swap_device="cpu",
        )

        args = model.pretrained_load_args({})

        self.assertTrue(args["enable_time_sign_embed"])
        self.assertEqual(args["musubi_blocks_to_swap"], 3)
        self.assertEqual(args["musubi_block_swap_device"], "cpu")

    def test_transformer_captures_hidden_states_and_accepts_skip_layers(self):
        transformer = Krea2Transformer2DModel(
            in_channels=4,
            num_layers=2,
            attention_head_dim=6,
            num_attention_heads=1,
            num_key_value_heads=1,
            intermediate_size=8,
            timestep_embed_dim=8,
            text_hidden_dim=6,
            num_text_layers=2,
            text_num_attention_heads=1,
            text_num_key_value_heads=1,
            text_intermediate_size=8,
            num_layerwise_text_blocks=1,
            num_refiner_text_blocks=1,
            axes_dims_rope=(2, 2, 2),
        )
        hidden_states_buffer = {}

        output = transformer(
            hidden_states=torch.randn(1, 4, 4),
            encoder_hidden_states=torch.randn(1, 3, 2, 6),
            timestep=torch.tensor([0.5]),
            position_ids=torch.zeros(7, 3, dtype=torch.long),
            encoder_attention_mask=torch.ones(1, 3, dtype=torch.int64),
            skip_layers=[1],
            hidden_states_buffer=hidden_states_buffer,
            return_dict=False,
        )

        self.assertEqual(tuple(output[0].shape), (1, 4, 4))
        self.assertIn("layer_0", hidden_states_buffer)
        self.assertIn("layer_1", hidden_states_buffer)
        self.assertEqual(tuple(hidden_states_buffer["layer_0"].shape[:2]), (1, 4))

    def test_transformer_accepts_tread_routing(self):
        transformer = Krea2Transformer2DModel(
            in_channels=4,
            num_layers=2,
            attention_head_dim=6,
            num_attention_heads=1,
            num_key_value_heads=1,
            intermediate_size=8,
            timestep_embed_dim=8,
            text_hidden_dim=6,
            num_text_layers=2,
            text_num_attention_heads=1,
            text_num_key_value_heads=1,
            text_intermediate_size=8,
            num_layerwise_text_blocks=1,
            num_refiner_text_blocks=1,
            axes_dims_rope=(2, 2, 2),
        )
        transformer.train()
        transformer.set_router(
            TREADRouter(seed=1, device=torch.device("cpu")),
            [{"start_layer_idx": 0, "end_layer_idx": 0, "selection_ratio": 0.5}],
        )

        with torch.enable_grad():
            output = transformer(
                hidden_states=torch.randn(1, 4, 4),
                encoder_hidden_states=torch.randn(1, 3, 2, 6),
                timestep=torch.tensor([0.5]),
                position_ids=torch.zeros(7, 3, dtype=torch.long),
                encoder_attention_mask=torch.ones(1, 3, dtype=torch.int64),
                return_dict=False,
            )

        self.assertEqual(tuple(output[0].shape), (1, 4, 4))

    def test_transformer_supports_twinflow_and_anyflow_timestep_conditioning(self):
        transformer = Krea2Transformer2DModel(
            in_channels=4,
            num_layers=1,
            attention_head_dim=6,
            num_attention_heads=1,
            num_key_value_heads=1,
            intermediate_size=8,
            timestep_embed_dim=8,
            text_hidden_dim=6,
            num_text_layers=2,
            text_num_attention_heads=1,
            text_num_key_value_heads=1,
            text_intermediate_size=8,
            num_layerwise_text_blocks=1,
            num_refiner_text_blocks=1,
            axes_dims_rope=(2, 2, 2),
            enable_time_sign_embed=True,
        )
        transformer.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
        kwargs = {
            "hidden_states": torch.randn(1, 4, 4),
            "encoder_hidden_states": torch.randn(1, 3, 2, 6),
            "timestep": torch.tensor([0.5]),
            "position_ids": torch.zeros(7, 3, dtype=torch.long),
            "encoder_attention_mask": torch.ones(1, 3, dtype=torch.int64),
            "return_dict": False,
        }

        output = transformer(**kwargs, timestep_sign=torch.ones(1), r_timestep=torch.zeros(1))

        self.assertEqual(tuple(output[0].shape), (1, 4, 4))

    def test_transformer_requires_initialization_for_timestep_conditioning(self):
        transformer = Krea2Transformer2DModel(
            in_channels=4,
            num_layers=1,
            attention_head_dim=6,
            num_attention_heads=1,
            num_key_value_heads=1,
            intermediate_size=8,
            timestep_embed_dim=8,
            text_hidden_dim=6,
            num_text_layers=2,
            text_num_attention_heads=1,
            text_num_key_value_heads=1,
            text_intermediate_size=8,
            num_layerwise_text_blocks=1,
            num_refiner_text_blocks=1,
            axes_dims_rope=(2, 2, 2),
        )
        kwargs = {
            "hidden_states": torch.randn(1, 4, 4),
            "encoder_hidden_states": torch.randn(1, 3, 2, 6),
            "timestep": torch.tensor([0.5]),
            "position_ids": torch.zeros(7, 3, dtype=torch.long),
            "encoder_attention_mask": torch.ones(1, 3, dtype=torch.int64),
            "return_dict": False,
        }

        with self.assertRaisesRegex(ValueError, "enable_time_sign_embed"):
            transformer(**kwargs, timestep_sign=torch.ones(1))
        with self.assertRaisesRegex(ValueError, "FlowMap"):
            transformer(**kwargs, r_timestep=torch.zeros(1))

    def test_vae_encode_hooks_use_qwen_image_vae_rank_and_normalization(self):
        model = Krea2.__new__(Krea2)
        vae = SimpleNamespace(config=SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0], z_dim=2))
        model.get_vae = lambda: vae

        image_batch = torch.zeros(1, 3, 64, 64)
        preprocessed = model.pre_vae_encode_transform_sample(image_batch)
        self.assertEqual(tuple(preprocessed.shape), (1, 3, 1, 64, 64))

        vae_output = SimpleNamespace(latent_dist=FakeLatentDistribution(torch.ones(1, 2, 1, 8, 8) * 5.0))
        latents = model.post_vae_encode_transform_sample(vae_output)

        self.assertEqual(tuple(latents.shape), (1, 2, 8, 8))
        self.assertTrue(torch.allclose(latents[:, 0], torch.full((1, 8, 8), 2.0)))
        self.assertTrue(torch.allclose(latents[:, 1], torch.full((1, 8, 8), 0.75)))

    def test_transformer_accepts_cached_int64_encoder_attention_mask(self):
        transformer = Krea2Transformer2DModel(
            in_channels=4,
            num_layers=1,
            attention_head_dim=6,
            num_attention_heads=1,
            num_key_value_heads=1,
            intermediate_size=8,
            timestep_embed_dim=8,
            text_hidden_dim=6,
            num_text_layers=2,
            text_num_attention_heads=1,
            text_num_key_value_heads=1,
            text_intermediate_size=8,
            num_layerwise_text_blocks=1,
            num_refiner_text_blocks=1,
            axes_dims_rope=(2, 2, 2),
        )

        output = transformer(
            hidden_states=torch.randn(1, 4, 4),
            encoder_hidden_states=torch.randn(1, 3, 2, 6),
            timestep=torch.tensor([0.5]),
            position_ids=torch.zeros(7, 3, dtype=torch.long),
            encoder_attention_mask=torch.ones(1, 3, dtype=torch.int64),
            return_dict=False,
        )

        self.assertEqual(tuple(output[0].shape), (1, 4, 4))

    def test_model_predict_appends_reference_latent_tokens_and_crops_output(self):
        model = Krea2.__new__(Krea2)
        model.config = SimpleNamespace(krea2_reference_latents=True, weight_dtype=torch.float32)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.model = FakeKrea2Transformer()
        model.unwrap_model = lambda model: model

        batch = {
            "noisy_latents": torch.ones(1, 16, 4, 4),
            "latents": torch.zeros(1, 16, 4, 4),
            "conditioning_latents": torch.full((1, 16, 4, 4), 2.0),
            "timesteps": torch.tensor([500.0]),
            "prompt_embeds": torch.zeros(1, 3, 2, 4),
            "encoder_attention_mask": torch.ones(1, 3, dtype=torch.int64),
        }

        result = model.model_predict(batch)

        self.assertEqual(tuple(result["model_prediction"].shape), (1, 16, 4, 4))
        self.assertEqual(tuple(model.model.last_call["hidden_states"].shape), (1, 8, 64))
        self.assertEqual(tuple(model.model.last_call["position_ids"].shape), (11, 3))
        self.assertTrue(torch.equal(model.model.last_call["timestep"], torch.tensor([0.5])))


if __name__ == "__main__":
    unittest.main()
