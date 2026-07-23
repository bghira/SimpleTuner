import inspect
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from simpletuner.helpers.acceleration import AccelerationBackend
from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.mageflow.pipeline import MageFlowPipeline, _mageflow_velocity
from simpletuner.helpers.models.mageflow.pipeline_edit import MageFlowEditPipeline
from simpletuner.helpers.models.mageflow.transformer import MageFlowTransformer2DModel
from simpletuner.helpers.models.mageflow.vendor.models.modules import _attn_backend as mageflow_attn_backend
from simpletuner.helpers.models.mageflow.vendor.models.modules.text_encoder import _resolve_hf_attn_impl
from simpletuner.helpers.models.mageflow.vendor.pipeline import _build_pack_ctx, _lens_to_cu
from simpletuner.helpers.models.registry import ModelRegistry


def _tiny_transformer(depth=1, enable_time_sign_embed=False):
    return MageFlowTransformer2DModel(
        in_channels=4,
        out_channels=4,
        hidden_size=24,
        num_heads=2,
        axes_dim=[4, 4, 4],
        depth=depth,
        context_in_dim=8,
        attn_type="sdpa",
        enable_time_sign_embed=enable_time_sign_embed,
    )


def _mageflow_class():
    registry_entry = ModelRegistry.get("mageflow")
    return registry_entry.get_real_class() if hasattr(registry_entry, "get_real_class") else registry_entry


class MageFlowModelTests(unittest.TestCase):
    def test_model_metadata_contains_mageflow(self):
        metadata_path = Path(__file__).parent.parent / "simpletuner/helpers/models/model_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        self.assertIn("mageflow", metadata)
        self.assertEqual(metadata["mageflow"]["class_name"], "MageFlow")
        self.assertEqual(
            metadata["mageflow"]["flavour_choices"],
            ["base", "default", "turbo", "edit-base", "edit", "edit-turbo"],
        )

    def test_registry_resolves_mageflow(self):
        model_cls = _mageflow_class()
        self.assertEqual(model_cls.NAME, "Mage-Flow")
        self.assertEqual(model_cls.DEFAULT_MODEL_FLAVOUR, "base")
        self.assertIn("edit-turbo", model_cls.get_flavour_choices())
        self.assertFalse(model_cls.DDP_FIND_UNUSED_PARAMETERS)
        self.assertIn("MageFlowTransformerBlock", model_cls.MODEL_CLASS._no_split_modules)

    def test_context_parallel_is_rejected(self):
        model_cls = _mageflow_class()
        model = object.__new__(model_cls)
        model.config = SimpleNamespace(
            model_flavour="base",
            aspect_bucket_alignment=16,
            tokenizer_max_length=None,
            validation_num_inference_steps=4,
            context_parallel_size=2,
        )

        with self.assertRaisesRegex(ValueError, "context_parallel_size"):
            model.check_user_config()

    def test_acceleration_presets_include_ramtorch_and_musubi_levels(self):
        model_cls = _mageflow_class()
        presets = model_cls.get_acceleration_presets()
        ramtorch_levels = {preset.level for preset in presets if preset.backend is AccelerationBackend.RAMTORCH}
        musubi_levels = {preset.level for preset in presets if preset.backend is AccelerationBackend.MUSUBI_BLOCK_SWAP}

        self.assertEqual(ramtorch_levels, {"light", "balanced", "aggressive"})
        self.assertEqual(musubi_levels, {"light", "balanced", "aggressive"})

    def test_pretrained_load_args_include_musubi_settings(self):
        model_cls = _mageflow_class()
        model = object.__new__(model_cls)
        model.config = SimpleNamespace(
            attention_mechanism="flash-attn-varlen-hub",
            musubi_blocks_to_swap=6,
            musubi_block_swap_device="cpu",
            twinflow_enabled=False,
        )

        args = model.pretrained_load_args({})

        self.assertEqual(args["attn_type"], "flash-attn-varlen-hub")
        self.assertEqual(args["musubi_blocks_to_swap"], 6)
        self.assertEqual(args["musubi_block_swap_device"], "cpu")

    def test_mageflow_attention_backend_uses_packed_dispatch_for_hub_flash(self):
        calls = {}

        class DummyPackedBackend:
            def varlen_unpacked(
                self,
                query_unpad,
                key_unpad,
                value_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                *,
                causal=False,
                dropout_p=0.0,
                softmax_scale=None,
            ):
                calls.update(
                    {
                        "query": query_unpad,
                        "key": key_unpad,
                        "value": value_unpad,
                        "cu_seqlens_q": cu_seqlens_q,
                        "cu_seqlens_k": cu_seqlens_k,
                        "max_seqlen_q": max_seqlen_q,
                        "max_seqlen_k": max_seqlen_k,
                        "causal": causal,
                        "dropout_p": dropout_p,
                        "softmax_scale": softmax_scale,
                    }
                )
                return query_unpad + value_unpad

        q = torch.randn(3, 2, 4)
        k = torch.randn(3, 2, 4)
        v = torch.randn(3, 2, 4)
        cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)

        try:
            with patch.object(
                mageflow_attn_backend.simpletuner_attention_backend,
                "get_packed_attention_backend",
                return_value=DummyPackedBackend(),
            ) as get_backend:
                mageflow_attn_backend.set_attn_backend("flash-attn-varlen-hub")
                output = mageflow_attn_backend.flash_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=3,
                    max_seqlen_k=3,
                    causal=True,
                    softmax_scale=0.5,
                    dropout_p=0.0,
                    window_size=(-1, -1),
                )
        finally:
            mageflow_attn_backend.set_attn_backend("sdpa")

        get_backend.assert_called_once_with("flash-attn-varlen-hub")
        self.assertTrue(torch.equal(output, q + v))
        self.assertIs(calls["query"], q)
        self.assertIs(calls["key"], k)
        self.assertIs(calls["value"], v)
        self.assertIs(calls["cu_seqlens_q"], cu_seqlens)
        self.assertIs(calls["cu_seqlens_k"], cu_seqlens)
        self.assertEqual(calls["max_seqlen_q"], 3)
        self.assertEqual(calls["max_seqlen_k"], 3)
        self.assertTrue(calls["causal"])
        self.assertEqual(calls["softmax_scale"], 0.5)

    def test_mageflow_text_encoder_hub_flash_loads_hf_module_with_sdpa(self):
        self.assertEqual(_resolve_hf_attn_impl("diffusers"), "sdpa")
        self.assertEqual(_resolve_hf_attn_impl("flash-attn-varlen-hub"), "sdpa")

    def test_mageflow_pack_metadata_dtypes_do_not_follow_bf16_default_dtype(self):
        previous_dtype = torch.get_default_dtype()
        device = torch.device("cpu")
        try:
            torch.set_default_dtype(torch.bfloat16)
            txt = torch.randn(1, 5, 8)
            neg_txt = torch.randn(1, 5, 8)
            txt_cu = _lens_to_cu([5], device)
            neg_cu = _lens_to_cu([5], device)
            ctx = _build_pack_ctx(
                torch.zeros(1, 4, 3, dtype=torch.float32),
                _lens_to_cu([4], device),
                [[(1, 2, 2)]],
                [4],
                txt,
                txt_cu,
                torch.ones(1, 5, dtype=torch.bool),
                torch.randn(1, 8),
                neg_txt,
                neg_cu,
                torch.ones(1, 5, dtype=torch.bool),
                torch.randn(1, 8),
                5.0,
                False,
                True,
                device,
            )
        finally:
            torch.set_default_dtype(previous_dtype)

        self.assertEqual(ctx["txt_ids"].dtype, torch.float32)
        self.assertEqual(ctx["neg_ids"].dtype, torch.float32)
        self.assertEqual(ctx["d_txt_ids"].dtype, torch.float32)
        self.assertEqual(ctx["d_txt_mask"].dtype, torch.bool)
        self.assertEqual(ctx["img_cu"].dtype, torch.int32)
        self.assertEqual(ctx["d_img_cu"].dtype, torch.int32)

    def test_mageflow_restores_qwen_text_rotary_inv_freq_to_fp32(self):
        model_cls = _mageflow_class()

        class RotaryEmbedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace()
                self.attention_scaling = 0.25
                rounded = torch.tensor([1.0, 0.1001], dtype=torch.bfloat16)
                self.register_buffer("inv_freq", rounded, persistent=False)
                self.register_buffer("original_inv_freq", rounded.clone(), persistent=False)

            @staticmethod
            def compute_default_rope_parameters(config, device):
                del config
                return torch.tensor([1.0, 0.100123], device=device, dtype=torch.float32), 1.0

        text_encoder = SimpleNamespace(language_model=SimpleNamespace(rotary_emb=RotaryEmbedding()))

        model_cls._ensure_qwen3vl_text_rotary_precision(text_encoder, torch.device("cpu"))

        rotary_emb = text_encoder.language_model.rotary_emb
        self.assertEqual(rotary_emb.inv_freq.dtype, torch.float32)
        self.assertEqual(rotary_emb.original_inv_freq.dtype, torch.float32)
        self.assertTrue(torch.equal(rotary_emb.inv_freq, torch.tensor([1.0, 0.100123], dtype=torch.float32)))
        self.assertTrue(torch.equal(rotary_emb.original_inv_freq, rotary_emb.inv_freq))
        self.assertEqual(rotary_emb.attention_scaling, 1.0)

    def test_pretrained_load_args_enable_twinflow_time_sign_embedding(self):
        model_cls = _mageflow_class()
        model = object.__new__(model_cls)
        model.config = SimpleNamespace(
            attention_mechanism="sdpa",
            musubi_blocks_to_swap=0,
            musubi_block_swap_device="cpu",
            twinflow_enabled=True,
        )

        args = model.pretrained_load_args({})

        self.assertTrue(args["enable_time_sign_embed"])

    def test_edit_flavour_uses_edit_pipeline(self):
        model_cls = _mageflow_class()
        config = SimpleNamespace(
            model_flavour="edit-turbo",
            aspect_bucket_alignment=16,
            tokenizer_max_length=None,
            validation_using_datasets=False,
            validation_num_inference_steps=4,
        )
        model = object.__new__(model_cls)
        model.config = config
        model.PIPELINE_CLASSES = dict(model_cls.PIPELINE_CLASSES)
        model.check_user_config()
        self.assertIs(model.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG], MageFlowEditPipeline)
        self.assertFalse(model.requires_conditioning_dataset())
        self.assertTrue(model.supports_conditioning_dataset())
        self.assertFalse(model.requires_conditioning_validation_inputs())
        self.assertFalse(model.requires_validation_edit_captions())
        self.assertFalse(model.requires_text_embed_image_context())

    def test_small_transformer_instantiates(self):
        model = _tiny_transformer()
        self.assertEqual(model.config.in_channels, 4)
        self.assertEqual(len(model.transformer_blocks), 1)

    def test_gradient_checkpointing_uses_supplied_function(self):
        checkpoint_calls = []

        def checkpoint_func(function, *args, **kwargs):
            checkpoint_calls.append(dict(kwargs))
            kwargs.pop("use_reentrant", None)
            return function(*args)

        model = _tiny_transformer()
        model.train()
        model.enable_gradient_checkpointing(checkpoint_func)

        output = model(
            img=torch.randn(1, 4, 4),
            txt=torch.randn(1, 3, 8),
            timesteps=torch.tensor([0.5]),
            img_shapes=[[(1, 2, 2)]],
            img_cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            txt_cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
        )

        self.assertEqual(output.shape, (1, 4, 4))
        self.assertEqual(len(checkpoint_calls), 1)
        self.assertEqual(checkpoint_calls[0], {"use_reentrant": False})

    def test_transformer_supports_twinflow_time_sign(self):
        model = _tiny_transformer(enable_time_sign_embed=True)

        output = model(
            img=torch.randn(1, 4, 4),
            txt=torch.randn(1, 3, 8),
            timesteps=torch.tensor([0.5]),
            timestep_sign=torch.tensor([-1.0]),
            img_shapes=[[(1, 2, 2)]],
            img_cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            txt_cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
        )

        self.assertEqual(output.shape, (1, 4, 4))

    def test_transformer_rejects_time_sign_without_twinflow_embedding(self):
        model = _tiny_transformer()

        with self.assertRaisesRegex(ValueError, "enable_time_sign_embed"):
            model(
                img=torch.randn(1, 4, 4),
                txt=torch.randn(1, 3, 8),
                timesteps=torch.tensor([0.5]),
                timestep_sign=torch.tensor([-1.0]),
                img_shapes=[[(1, 2, 2)]],
                img_cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
                txt_cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
            )

    def test_transformer_captures_hidden_states_and_skips_layers(self):
        model = _tiny_transformer(depth=2)
        hidden_states_buffer = {}

        output = model(
            img=torch.randn(1, 4, 4),
            txt=torch.randn(1, 3, 8),
            timesteps=torch.tensor([0.5]),
            img_shapes=[[(1, 2, 2)]],
            img_cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            txt_cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
            hidden_states_buffer=hidden_states_buffer,
            skip_layers=[1],
            return_dict=False,
        )

        self.assertEqual(output[0].shape, (1, 4, 4))
        self.assertEqual(set(hidden_states_buffer), {"layer_0", "layer_1"})
        self.assertEqual(hidden_states_buffer["layer_1"].shape, (1, 4, 24))

    def test_transformer_has_musubi_constructor_args_and_cpu_forward(self):
        model = MageFlowTransformer2DModel(
            in_channels=4,
            out_channels=4,
            hidden_size=24,
            num_heads=2,
            axes_dim=[4, 4, 4],
            depth=2,
            context_in_dim=8,
            attn_type="sdpa",
            musubi_blocks_to_swap=1,
            musubi_block_swap_device="cpu",
        )

        self.assertIsNotNone(model._musubi_block_swap)
        self.assertTrue(model._musubi_block_swap.is_managed_block(1))
        output = model(
            img=torch.randn(1, 4, 4),
            txt=torch.randn(1, 3, 8),
            timesteps=torch.tensor([0.5]),
            img_shapes=[[(1, 2, 2)]],
            img_cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            txt_cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
        )

        self.assertEqual(output.shape, (1, 4, 4))

    def test_model_predict_uses_wrapped_model_for_distributed_training(self):
        model_cls = _mageflow_class()

        class WrappedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.called = False

            def forward(self, img, **kwargs):
                del kwargs
                self.called = True
                return torch.zeros_like(img)

        wrapper = WrappedModel()
        model = object.__new__(model_cls)
        model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            twinflow_enabled=False,
            model_flavour="base",
            controlnet=False,
        )
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.model = wrapper
        model.crepa_regularizer = None
        model.layersync_regularizer = None

        result = model.model_predict(
            {
                "noisy_latents": torch.randn(1, 4, 2, 2),
                "prompt_embeds": torch.randn(1, 3, 8),
                "attention_masks": torch.ones(1, 3, dtype=torch.bool),
                "timesteps": torch.tensor([500.0]),
            }
        )

        self.assertTrue(wrapper.called)
        self.assertEqual(result["model_prediction"].shape, (1, 4, 2, 2))
        self.assertIsNone(result["hidden_states_buffer"])

    def test_model_predict_passes_layersync_crepa_hidden_state_buffer(self):
        model_cls = _mageflow_class()

        class WrappedModel(torch.nn.Module):
            def forward(self, img, hidden_states_buffer=None, **kwargs):
                del kwargs
                if hidden_states_buffer is not None:
                    hidden_states_buffer["layer_0"] = torch.full((1, 4, 8), 3.0)
                return torch.zeros_like(img)

        model = object.__new__(model_cls)
        model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            twinflow_enabled=False,
            model_flavour="base",
            controlnet=False,
        )
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.model = WrappedModel()
        model.layersync_regularizer = SimpleNamespace(wants_hidden_states=lambda: True)
        model.crepa_regularizer = SimpleNamespace(wants_hidden_states=lambda: True, block_index=0)

        result = model.model_predict(
            {
                "noisy_latents": torch.randn(1, 4, 2, 2),
                "prompt_embeds": torch.randn(1, 3, 8),
                "attention_masks": torch.ones(1, 3, dtype=torch.bool),
                "timesteps": torch.tensor([500.0]),
            }
        )

        self.assertIsNotNone(result["hidden_states_buffer"])
        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 3.0)))

    def test_crepa_self_flow_remains_disabled_for_tokenwise_timesteps(self):
        model_cls = _mageflow_class()
        model = object.__new__(model_cls)
        self.assertFalse(model.supports_crepa_self_flow())

    def test_pipeline_signatures_include_guidance_features(self):
        signature = inspect.signature(MageFlowPipeline.__call__)
        self.assertIn("use_cfg_zero_star", signature.parameters)
        self.assertIn("skip_guidance_layers", signature.parameters)
        edit_signature = inspect.signature(MageFlowEditPipeline.__call__)
        self.assertIn("use_cfg_zero_star", edit_signature.parameters)
        self.assertIn("skip_guidance_layers", edit_signature.parameters)

    def test_edit_pipeline_without_image_uses_text_to_image_path(self):
        pipe = object.__new__(MageFlowEditPipeline)

        with patch.object(MageFlowPipeline, "__call__", return_value="text-only") as text_to_image:
            result = MageFlowEditPipeline.__call__(
                pipe,
                prompt="a subject",
                image=None,
                height=768,
                width=512,
                num_inference_steps=4,
                return_dict=False,
            )

        self.assertEqual(result, "text-only")
        self.assertEqual(text_to_image.call_args.kwargs["height"], 768)
        self.assertEqual(text_to_image.call_args.kwargs["width"], 512)

    def test_mageflow_velocity_applies_skip_layer_guidance(self):
        class FakeTransformer:
            def __init__(self):
                self.skip_calls = []

            def __call__(self, img, txt, timesteps, skip_layers=None, **kwargs):
                del txt, timesteps, kwargs
                self.skip_calls.append(skip_layers)
                if skip_layers:
                    return torch.full_like(img, 0.25)
                if img.shape[1] == 8:
                    return torch.cat([torch.full_like(img[:, :4], 2.0), torch.full_like(img[:, :4], 1.0)], dim=1)
                return torch.full_like(img, 2.0)

        ctx = {
            "na": 1,
            "has_neg": True,
            "batch_cfg": True,
            "cfg": 3.0,
            "renorm": False,
            "txt": torch.randn(1, 3, 8),
            "txt_cu": torch.tensor([0, 3], dtype=torch.int32),
            "img_cu": torch.tensor([0, 4], dtype=torch.int32),
            "img_shapes": [[(1, 2, 2)]],
            "d_txt": torch.randn(1, 6, 8),
            "d_txt_cu": torch.tensor([0, 3, 6], dtype=torch.int32),
            "d_img_cu": torch.tensor([0, 4, 8], dtype=torch.int32),
            "d_img_shapes": [[(1, 2, 2), (1, 2, 2)]],
        }

        velocity = _mageflow_velocity(
            FakeTransformer(),
            torch.zeros(1, 4, 4),
            ctx,
            0.5,
            step_index=1,
            num_inference_steps=10,
            skip_guidance_layers=[1],
            skip_layer_guidance_start=0.0,
            skip_layer_guidance_stop=1.0,
            skip_layer_guidance_scale=2.0,
            use_cfg_zero_star=False,
        )

        self.assertTrue(torch.allclose(velocity, torch.full_like(velocity, 7.5)))


if __name__ == "__main__":
    unittest.main()
