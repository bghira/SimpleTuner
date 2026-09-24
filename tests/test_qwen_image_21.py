import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

from simpletuner.helpers.models.common import ImageModelFoundation, PipelineTypes
from simpletuner.helpers.models.qwen_image.autoencoder_21 import AutoencoderKLQwenImage21
from simpletuner.helpers.models.qwen_image.model import QwenImage
from simpletuner.helpers.models.qwen_image.pipeline_21 import QwenImage21Pipeline
from simpletuner.helpers.models.qwen_image.transformer import QwenImageTransformer2DModel
from simpletuner.helpers.models.qwen_image.transformer_21 import (
    QwenImage21AttnProcessor,
    QwenImage21FlexAttnProcessor,
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
    _qwenimage21_prepare_qkv,
    apply_rotary_emb_qwen,
)
from simpletuner.helpers.training.adapter import load_lora_weights
from simpletuner.helpers.training.save_hooks import _collect_anyflow_sidecar_state, _materialize_state_dict_for_save
from tests.test_qwen_prompt_encoding import PromptEncodingQwen


class QwenImage21Tests(unittest.TestCase):
    def tearDown(self):
        torch._dynamo.reset()

    def make_transformer(self, num_layers=2):
        torch.manual_seed(42)
        return QwenImage21Transformer2DModel(
            in_channels=4,
            out_channels=4,
            num_layers=num_layers,
            attention_head_dim=16,
            num_attention_heads=2,
            context_in_dim=8,
            mlp_ratio=2,
            axes_dims_rope=(4, 6, 6),
        )

    def inputs(self):
        return dict(
            hidden_states=torch.randn(2, 8, 4),
            encoder_hidden_states=torch.randn(2, 5, 8),
            encoder_hidden_states_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
            timestep=torch.tensor([0.3, 0.7]),
            img_shapes=[[(1, 2, 4)]] * 2,
            img_mask=torch.tensor([[0, 0, 0, 0, 0, 1, 1]] * 2, dtype=torch.bool),
            return_dict=False,
        )

    def test_flavour_components_are_isolated(self):
        def init(model, config, accelerator):
            model.config = config
            model.accelerator = accelerator

        with patch.object(ImageModelFoundation, "__init__", init), patch.object(QwenImage, "_validate_xm_support"):
            new = QwenImage(SimpleNamespace(model_flavour="v2.1"), None)
            old = QwenImage(SimpleNamespace(model_flavour="v2.0"), None)
            edit = QwenImage(SimpleNamespace(model_flavour="edit-v3"), None)
        self.assertEqual(QwenImage.DEFAULT_MODEL_FLAVOUR, "v2.1")
        self.assertIs(new.MODEL_CLASS, QwenImage21Transformer2DModel)
        self.assertIs(new.AUTOENCODER_CLASS, AutoencoderKLQwenImage21)
        self.assertIs(new.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG], QwenImage21Pipeline)
        self.assertEqual((new.LATENT_CHANNEL_COUNT, new.vae_scale_factor), (64, 16))
        self.assertIs(old.MODEL_CLASS, QwenImageTransformer2DModel)
        self.assertEqual((old.LATENT_CHANNEL_COUNT, old.vae_scale_factor), (16, 8))
        self.assertIs(edit.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG], QwenImage.EDIT_PLUS_PIPELINE_CLASS)
        self.assertEqual(QwenImage.max_swappable_blocks(new.config), 31)
        self.assertEqual(QwenImage.max_swappable_blocks(old.config), 59)

    def test_vae_input_adds_opaque_alpha_only_for_21(self):
        model = QwenImage.__new__(QwenImage)
        model.config = SimpleNamespace(model_flavour="v2.1")
        rgb = torch.randn(2, 3, 32, 64)
        rgba = model.pre_vae_encode_transform_sample(rgb)
        torch.testing.assert_close(rgba[:, :3, 0], rgb)
        torch.testing.assert_close(rgba[:, 3], torch.ones(2, 1, 32, 64))
        torch.testing.assert_close(model.pre_vae_encode_transform_sample(rgba), rgba)
        model.config.model_flavour = "v2.0"
        self.assertEqual(model.pre_vae_encode_transform_sample(rgb).shape, (2, 3, 1, 32, 64))

    def test_prompt_encoding_restores_cache_mask_when_upstream_omits_it(self):
        embeddings = torch.randn(2, 5, 8)
        pipeline = SimpleNamespace(encode_prompt=Mock(return_value=(embeddings, None, torch.zeros(2, 5))))
        model = PromptEncodingQwen(pipeline, flavour="v2.1")
        encoded, mask = model._encode_prompts(["one", "two"])
        torch.testing.assert_close(encoded, embeddings)
        torch.testing.assert_close(mask, torch.ones(2, 5, dtype=torch.long))

    def test_collation_elides_only_redundant_21_padding_masks(self):
        model = QwenImage.__new__(QwenImage)
        for flavour in ("v2.1", "v2.0"):
            model.config = SimpleNamespace(model_flavour=flavour)
            samples = [{"prompt_embeds": torch.randn(5, 8), "attention_masks": torch.ones(5)}] * 2
            collated = model.collate_prompt_embeds(samples)
            if flavour == "v2.1":
                self.assertIsNone(collated["attention_masks"])
            else:
                torch.testing.assert_close(collated["attention_masks"], torch.ones(2, 5))
            samples[1] = {"prompt_embeds": torch.randn(3, 8), "attention_masks": torch.ones(3)}
            collated = model.collate_prompt_embeds(samples)
            torch.testing.assert_close(
                collated["attention_masks"], torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0]])
            )

    def test_vae_round_trip_uses_16x_compression_and_rgba(self):
        vae = AutoencoderKLQwenImage21(
            base_dim=4,
            decoder_base_dim=4,
            z_dim=4,
            dim_mult=[1, 1, 1, 1, 1],
            num_res_blocks=1,
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        ).eval()
        image = torch.randn(1, 4, 1, 32, 64)
        with torch.no_grad():
            latent = vae.encode(image).latent_dist.mode()
            decoded = vae.decode(latent).sample
        self.assertEqual(latent.shape, (1, 4, 1, 2, 4))
        self.assertEqual(decoded.shape, image.shape)
        self.assertTrue(torch.isfinite(decoded).all())

    def test_training_packs_unpatched_latents_and_selects_target_tokens(self):
        model = QwenImage.__new__(QwenImage)
        model.config = SimpleNamespace(model_flavour="v2.1", weight_dtype=torch.float32)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        latents = torch.randn(2, 4, 2, 4)
        prompt = torch.randn(2, 5, 8)
        model.model = Mock(
            return_value=(torch.cat([torch.full((2, 5, 4), float("nan")), latents.flatten(2).transpose(1, 2)], 1),)
        )
        prediction = model._model_predict_standard(
            dict(noisy_latents=latents, latents=latents, prompt_embeds=prompt, timesteps=torch.tensor([300, 700]))
        )["model_prediction"]
        torch.testing.assert_close(prediction, latents)
        kwargs = model.model.call_args.kwargs
        self.assertEqual(kwargs["img_shapes"], [[(1, 2, 4)]] * 2)
        torch.testing.assert_close(kwargs["timestep"], torch.tensor([0.3, 0.7]))

    def test_real_rope_matches_complex_forward_and_backward(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            shape = (2, 1088, 32, 128) if device == "cuda" else (2, 13, 2, 16)
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(device=device, dtype=dtype):
                    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                    reference_x = x.detach().clone().requires_grad_()
                    frequency_shape = (shape[1], shape[-1] // 2)
                    frequencies = torch.polar(
                        torch.ones(frequency_shape, device=device), torch.randn(frequency_shape, device=device)
                    )
                    actual = apply_rotary_emb_qwen(x, frequencies)
                    complex_x = torch.view_as_complex(reference_x.float().unflatten(-1, (-1, 2)))
                    expected = torch.view_as_real(complex_x * frequencies.unsqueeze(1)).flatten(3).to(dtype)
                    if device == "cuda":
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(actual, expected)
                    gradient = torch.randn_like(actual)
                    actual.backward(gradient)
                    expected.backward(gradient)
                    torch.testing.assert_close(x.grad, reference_x.grad)

    def test_static_rope_matches_image_layout(self):
        model = self.make_transformer()
        mask = torch.tensor([False] * 5 + [True] * 8)
        reference = model.pos_embed([(1, 2, 4)], mask, torch.device("cpu"))
        actual = model.pos_embed.text_to_image(5, 2, 4, torch.device("cpu"))
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_block_attention_matches_dense_mask_and_gradients(self):
        class DenseProcessor:
            def __call__(self, attn, hidden_states, rotary_emb, key_valid, **kwargs):
                q, k, v, length = _qwenimage21_prepare_qkv(attn, hidden_states, rotary_emb, None, None, None)
                positions = torch.arange(length)
                allowed = (positions[:, None] >= positions[None, :]) | (
                    (positions[:, None] >= 5) & (positions[None, :] >= 5)
                )
                allowed = allowed[None, None]
                if key_valid is not None:
                    allowed = allowed & key_valid[:, None, None, :]
                out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), allowed)
                return attn.to_out[1](attn.to_out[0](out.transpose(1, 2).flatten(2)))

        for padded in (False, True):
            with self.subTest(padded=padded):
                model = self.make_transformer()
                reference = copy.deepcopy(model)
                reference.set_attn_processor(DenseProcessor())
                inputs = self.inputs()
                if not padded:
                    inputs["encoder_hidden_states_mask"] = None
                actual = model(**inputs)[0]
                expected = reference(**inputs)[0]
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                actual.square().mean().backward()
                expected.square().mean().backward()
                for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual_param.grad, expected_param.grad, atol=2e-6, rtol=2e-5)

    def test_varlen_does_not_silently_discard_causal_padding_mask(self):
        model = self.make_transformer()
        processor = QwenImage21AttnProcessor()
        processor._attention_backend = "_flash_3_varlen_hub"
        model.set_attn_processor(processor)
        with self.assertRaisesRegex(ValueError, "varlen padding mask cannot represent"):
            model(**self.inputs())
        processor._attention_backend = None
        with patch.object(_AttentionBackendRegistry, "_active_backend", AttentionBackendName._FLASH_3_VARLEN_HUB):
            with self.assertRaisesRegex(ValueError, "varlen padding mask cannot represent"):
                model(**self.inputs())

    def test_flex_configuration_installs_block_causal_processor(self):
        model = QwenImage.__new__(QwenImage)
        model.config = SimpleNamespace(model_flavour="v2.1", attention_mechanism="flex")
        model.model = self.make_transformer()
        model.unwrap_model = Mock(return_value=model.model)
        model._maybe_load_assistant_lora = Mock()
        with patch.object(ImageModelFoundation, "post_model_load_setup"):
            model.post_model_load_setup()
        for block in model.model.transformer_blocks:
            self.assertIsInstance(block.attn.processor, QwenImage21FlexAttnProcessor)

    def test_flex_preserves_padded_block_causal_attention(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = self.make_transformer().to(device)
                reference = copy.deepcopy(model)
                model.set_attn_processor(QwenImage21FlexAttnProcessor())
                inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in self.inputs().items()}
                if device == "cuda":
                    model.compile_repeated_blocks(fullgraph=True)
                # PyTorch FlexAttention supports CPU forward, but requires CUDA for backward.
                with torch.set_grad_enabled(device == "cuda"):
                    actual = model(**inputs)[0]
                    expected = reference(**inputs)[0]
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                if device == "cuda":
                    actual.square().mean().backward()
                    expected.square().mean().backward()
                    for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual_param.grad, expected_param.grad, atol=2e-6, rtol=2e-5)

    def test_interval_checkpointing_preserves_gradients(self):
        for stride in (None, 4):
            with self.subTest(stride=stride):
                model = self.make_transformer(num_layers=6)
                reference = copy.deepcopy(model)
                model.enable_gradient_checkpointing()
                model.set_gradient_checkpointing_interval(2)
                model.set_gradient_checkpointing_segment_stride(stride)
                groups = []
                active_group = None

                def record_block(index):
                    def hook(*_args):
                        if active_group is not None:
                            active_group.append(index)

                    return hook

                for index, block in enumerate(model.transformer_blocks):
                    block.register_forward_pre_hook(record_block(index))
                checkpoint_fn = model._gradient_checkpointing_func

                def checkpoint(function, *args):
                    nonlocal active_group
                    active_group = []
                    result = checkpoint_fn(function, *args)
                    groups.append(active_group)
                    active_group = None
                    return result

                model._gradient_checkpointing_func = checkpoint
                inputs = self.inputs()
                model(**inputs)[0].square().mean().backward()
                reference(**inputs)[0].square().mean().backward()
                self.assertEqual(groups, [[0, 1], [2, 3], [4, 5]] if stride is None else [[0, 1], [4, 5]])
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad)

    def test_text_to_image_compiles_without_graph_breaks(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = self.make_transformer().to(device)
                inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in self.inputs().items()}
                compiled = torch.compile(model, backend="eager", fullgraph=True)
                for _ in range(2):
                    actual = compiled(**inputs)[0]
                    torch.testing.assert_close(actual, model(**inputs)[0])
                actual.square().mean().backward()
                self.assertIsNotNone(model.img_in.weight.grad)

    def test_segmented_checkpointing_compiles_without_graph_breaks(self):
        for stride in (None, 4):
            with self.subTest(stride=stride):
                model = self.make_transformer(num_layers=6)
                model.enable_gradient_checkpointing()
                model.set_gradient_checkpointing_interval(2)
                model.set_gradient_checkpointing_segment_stride(stride)
                reference = copy.deepcopy(model)
                reference.disable_gradient_checkpointing()
                inputs = self.inputs()
                compiled = torch.compile(model, backend="eager", fullgraph=True)
                actual = compiled(**inputs)[0]
                expected = reference(**inputs)[0]
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward()
                expected.square().mean().backward()
                for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual_param.grad, expected_param.grad)

    def test_lora_training_compiles_without_graph_breaks(self):
        model = self.make_transformer()
        model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        compiled(**self.inputs())[0].square().mean().backward()
        gradients = [param.grad for name, param in model.named_parameters() if "lora_B" in name]
        self.assertTrue(gradients)
        self.assertTrue(all(grad is not None and torch.isfinite(grad).all() for grad in gradients))
        self.assertTrue(any(grad.abs().sum() > 0 for grad in gradients))

    @unittest.skipUnless(torch.cuda.is_available(), "Inductor CUDA kernel verification requires CUDA")
    def test_inductor_lora_forward_and_gradients(self):
        model = self.make_transformer().cuda()
        model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        reference = copy.deepcopy(model)
        inputs = {key: value.cuda() if torch.is_tensor(value) else value for key, value in self.inputs().items()}
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        actual = compiled(**inputs)[0]
        expected = reference(**inputs)[0]
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=2e-4, atol=2e-5)

    def test_cached_validation_matches_full_forward(self):
        model = self.make_transformer().eval()
        inputs = self.inputs()
        cache = QwenImage21KVCache(2)
        with torch.no_grad():
            model(**inputs, kv_cache=cache, kv_cache_mode="extract")
            inputs["timestep"] = torch.tensor([0.2, 0.4])
            cached = model(**inputs, kv_cache=cache, kv_cache_mode="cached")[0]
            expected = model(**inputs)[0][:, -8:]
        torch.testing.assert_close(cached, expected, rtol=2e-5, atol=2e-5)

    def test_flowmap_endpoints_preserve_causal_prefix(self):
        model = self.make_transformer()
        inputs = self.inputs()
        baseline = model(**inputs)[0]
        with self.assertRaisesRegex(ValueError, "enable_flowmap_time_conditioning"):
            model(**inputs, r_timestep=inputs["timestep"])
        model.enable_flowmap_time_conditioning()
        matched = model(**inputs, r_timestep=inputs["timestep"])[0]
        torch.testing.assert_close(matched, baseline)
        changed = model(**inputs, r_timestep=torch.zeros(2))[0]
        torch.testing.assert_close(changed[:, :5], matched[:, :5])
        self.assertFalse(torch.allclose(changed[:, -8:], matched[:, -8:]))
        changed[:, -8:].square().mean().backward()
        gradients = [p.grad for p in model.time_text_embed.delta_timestep_embedder.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_flowmap_configuration_round_trip_and_capture(self):
        model = self.make_transformer()
        model.enable_flowmap_time_conditioning(gate_value=0.4, deltatime_type="t-r")
        inputs = dict(self.inputs(), r_timestep=torch.tensor([0.1, 0.2]))
        expected = model(**inputs)[0]
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = QwenImage21Transformer2DModel.from_pretrained(directory)
        self.assertEqual(restored.config.deltatime_type, "t-r")
        self.assertEqual(restored.config.gate_value, 0.4)
        compiled = torch.compile(restored, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(**inputs)[0], expected)

    def test_model_predict_normalizes_both_flowmap_endpoints(self):
        model = QwenImage.__new__(QwenImage)
        model.config = SimpleNamespace(model_flavour="v2.1", weight_dtype=torch.float32, crepa_enabled=False)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.model = Mock(return_value=(torch.zeros(2, 11, 64),))
        model._get_flowmap_r_timestep_forward_kwargs = lambda batch: {"r_timestep": batch["flowmap_r_timesteps"]}
        latents = torch.zeros(2, 64, 2, 4)
        model._model_predict_21(
            dict(
                noisy_latents=latents,
                prompt_embeds=torch.zeros(2, 3, 8),
                timesteps=torch.tensor([300, 700]),
                flowmap_r_timesteps=torch.tensor([100, 200]),
            )
        )
        kwargs = model.model.call_args.kwargs
        torch.testing.assert_close(kwargs["timestep"], torch.tensor([0.3, 0.7]))
        torch.testing.assert_close(kwargs["r_timestep"], torch.tensor([0.1, 0.2]))

    def test_flowmap_compiles_with_varying_padded_caption_lengths(self):
        model = self.make_transformer()
        model.enable_flowmap_time_conditioning()
        model.enable_gradient_checkpointing()
        model.set_gradient_checkpointing_interval(2)
        model.add_adapter(
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                modules_to_save=["time_text_embed.delta_timestep_embedder"],
            )
        )
        reference = copy.deepcopy(model)
        compiled = torch.compile(model, backend="eager", fullgraph=True, dynamic=True)
        for length in (5, 9, 7):
            with self.subTest(caption_length=length):
                inputs = self.inputs()
                inputs["encoder_hidden_states"] = torch.randn(2, length, 8)
                inputs["encoder_hidden_states_mask"] = torch.ones(2, length, dtype=torch.long)
                inputs["encoder_hidden_states_mask"][1, -2:] = 0
                inputs["img_mask"] = torch.tensor([[False] * length + [True, True]] * 2)
                inputs["r_timestep"] = torch.tensor([0.1, 0.2])
                model.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                actual = compiled(**inputs)[0]
                expected = reference(**inputs)[0]
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward()
                expected.square().mean().backward()
                for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual_param.grad, expected_param.grad)

    def test_regionally_compiled_adapter_roundtrips_to_eager_and_compiled_models(self):
        model = self.make_transformer()
        model.enable_flowmap_time_conditioning()
        adapter_config = LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["to_q"],
            modules_to_save=["time_text_embed.delta_timestep_embedder"],
        )
        model.add_adapter(adapter_config)
        with torch.no_grad():
            model.time_text_embed.delta_timestep_embedder.modules_to_save["default"].linear_2.weight.add_(0.2)
            model.transformer_blocks[0].attn.to_q.lora_B["default"].weight.fill_(0.1)
        inputs = dict(self.inputs(), r_timestep=torch.tensor([0.1, 0.2]))
        expected = model(**inputs)[0]
        model.time_text_embed = torch.compile(model.time_text_embed, backend="eager")
        for index, block in enumerate(model.transformer_blocks):
            model.transformer_blocks[index] = torch.compile(block, backend="eager")

        legacy_state = get_peft_model_state_dict(model)
        self.assertTrue(any("._orig_mod." in key for key in legacy_state))
        sidecar = _collect_anyflow_sidecar_state(model)
        self.assertTrue(sidecar)
        self.assertTrue(all("_orig_mod" not in key for key in sidecar))
        saved = _materialize_state_dict_for_save({**legacy_state, **sidecar})
        self.assertTrue(all("_orig_mod" not in key for key in saved))
        for state in (legacy_state, saved):
            for compiled in (False, True):
                with self.subTest(legacy=state is legacy_state, compiled=compiled):
                    restored = self.make_transformer()
                    restored.enable_flowmap_time_conditioning()
                    restored.add_adapter(adapter_config)
                    if compiled:
                        restored.time_text_embed = torch.compile(restored.time_text_embed, backend="eager")
                        for index, block in enumerate(restored.transformer_blocks):
                            restored.transformer_blocks[index] = torch.compile(block, backend="eager")
                    additional, missing = load_lora_weights(
                        {"transformer": restored},
                        "unused",
                        state_dict={f"transformer.{name}": value for name, value in state.items()},
                    )
                    self.assertFalse(additional)
                    self.assertFalse(missing)
                    torch.testing.assert_close(restored(**inputs)[0], expected)

    def test_flowmap_adapter_preserves_frozen_reference_and_sidecar(self):
        model = self.make_transformer()
        model.enable_flowmap_time_conditioning()
        foundation = QwenImage.__new__(QwenImage)
        foundation.config = SimpleNamespace(model_flavour="v2.1")
        foundation.get_trained_component = lambda **kwargs: model
        saved_modules = foundation.get_lora_save_layers()
        self.assertIn("time_text_embed.delta_timestep_embedder", saved_modules)
        model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"], modules_to_save=saved_modules))
        inputs = dict(self.inputs(), r_timestep=torch.tensor([0.1, 0.2]))
        delta = model.time_text_embed.delta_timestep_embedder
        self.assertTrue(all(p.requires_grad for p in delta.modules_to_save["default"].parameters()))
        model.disable_lora()
        reference = model(**inputs)[0]
        model.enable_lora()
        with torch.no_grad():
            delta.modules_to_save["default"].linear_2.weight.add_(0.2)
        trained = model(**inputs)[0]
        self.assertFalse(torch.allclose(trained, reference))
        model.disable_lora()
        torch.testing.assert_close(model(**inputs)[0], reference)
        model.enable_lora()
        sidecar = _collect_anyflow_sidecar_state(model)
        self.assertTrue(sidecar)
        restored = self.make_transformer()
        restored.enable_flowmap_time_conditioning()
        restored.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"], modules_to_save=saved_modules))
        with tempfile.TemporaryDirectory() as directory:
            filename = f"{directory}/adapter.safetensors"
            save_file({f"transformer.{name}": value for name, value in sidecar.items()}, filename)
            load_lora_weights({"transformer": restored}, filename)
        torch.testing.assert_close(restored(**inputs)[0], trained)
        restored.add_adapter(
            LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"], modules_to_save=saved_modules),
            adapter_name="discriminator",
        )
        restored.set_adapter("discriminator")
        torch.testing.assert_close(restored(**inputs)[0], reference)
        restored.set_adapter("default")
        torch.testing.assert_close(restored(**inputs)[0], trained)
        restored.disable_lora()
        torch.testing.assert_close(restored(**inputs)[0], reference)
        restored.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"]), adapter_name="plain")
        restored.enable_lora()
        restored.set_adapter("plain")
        torch.testing.assert_close(restored(**inputs)[0], reference)
        restored.set_adapter("default")
        torch.testing.assert_close(restored(**inputs)[0], trained)


if __name__ == "__main__":
    unittest.main()
