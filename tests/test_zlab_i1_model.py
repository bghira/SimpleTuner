import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin

from simpletuner.helpers.models.zlab_i1.model import ZLabI1
from simpletuner.helpers.models.zlab_i1.pipeline import ZlabI1Pipeline
from simpletuner.helpers.models.zlab_i1.transformer import ZlabI1Transformer2DModel
from simpletuner.helpers.utils import ramtorch as ramtorch_utils
from simpletuner.helpers.training.layersync import LayerSyncRegularizer
from simpletuner.helpers.training.tread import TREADRouter


class DummyVAE(torch.nn.Module):
    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return torch.device("cpu")

    def decode(self, latents):
        return SimpleNamespace(sample=latents[:, :3])


class DummyLoadVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0)
        self.tiling_enabled = False

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls()

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return torch.device("cpu")

    def enable_tiling(self):
        self.tiling_enabled = True

    def to(self, *args, **kwargs):
        return self


class StubRamtorchLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, skip_init=False):
        super().__init__(in_features, out_features, bias=bias, device="cpu", dtype=dtype)


def tiny_transformer():
    return ZlabI1Transformer2DModel(
        input_size=4,
        image_resolution=32,
        patch_size=2,
        in_channels=3,
        hidden_size=24,
        depth=3,
        num_heads=2,
        mlp_ratio=1.0,
        text_embed_dim=6,
        text_num_tokens=5,
    )


class ZlabI1FeatureTests(unittest.TestCase):
    def setUp(self):
        self.transformer = tiny_transformer()
        self.latents = torch.randn(1, 3, 4, 4)
        self.timesteps = torch.zeros(1)
        self.prompt_embeds = torch.randn(1, 5, 6)
        self.attention_mask = torch.ones(1, 5, dtype=torch.bool)

    def test_transformer_exposes_hidden_states_for_auxiliary_losses(self):
        hidden_states_buffer = {}

        output = self.transformer(
            self.latents,
            self.timesteps,
            self.prompt_embeds,
            self.attention_mask,
            hidden_states_buffer=hidden_states_buffer,
        )

        self.assertEqual(output.shape, self.latents.shape)
        self.assertEqual(sorted(hidden_states_buffer), ["layer_0", "layer_1", "layer_2"])
        for hidden in hidden_states_buffer.values():
            self.assertEqual(hidden.shape, torch.Size([1, 4, 24]))

    def test_scratch_init_has_deterministic_null_caption_and_no_trainable_timestep_embedder(self):
        self.assertTrue(torch.equal(self.transformer.text_encoder_adapter.learnable_null_caption, torch.zeros_like(self.transformer.text_encoder_adapter.learnable_null_caption)))
        self.assertFalse(any(param.requires_grad for param in self.transformer.t_embedder.parameters()))

    def test_prediction_target_matches_upstream_rectified_flow_velocity(self):
        model = ZLabI1.__new__(ZLabI1)
        latents = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])
        noise = torch.tensor([[[[-0.5, 1.0], [0.25, -1.5]]]])

        target = model.get_prediction_target({"latents": latents, "noise": noise})

        self.assertTrue(torch.equal(target, latents - noise))

    def test_prediction_target_prefers_explicit_parent_student_target(self):
        model = ZLabI1.__new__(ZLabI1)
        explicit_target = torch.randn(1, 3, 4, 4)

        target = model.get_prediction_target(
            {
                "latents": torch.randn(1, 3, 4, 4),
                "noise": torch.randn(1, 3, 4, 4),
                "target": explicit_target,
            }
        )

        self.assertIs(target, explicit_target)

    def test_transformer_skip_layers_and_tread_routes_keep_output_shape(self):
        skipped = self.transformer(
            self.latents,
            self.timesteps,
            self.prompt_embeds,
            self.attention_mask,
            skip_layers=[1],
        )
        self.assertEqual(skipped.shape, self.latents.shape)
        self.assertTrue(torch.isfinite(skipped).all())

        for route in (
            {"start_layer_idx": 0, "end_layer_idx": 1, "selection_ratio": 0.5},
            {"start_layer_idx": 2, "end_layer_idx": 2, "selection_ratio": 0.5},
        ):
            self.transformer.train()
            self.transformer.set_router(TREADRouter(seed=123, device="cpu"), [route])
            latents = self.latents.detach().clone().requires_grad_()
            routed = self.transformer(latents, self.timesteps, self.prompt_embeds, self.attention_mask)
            self.assertEqual(routed.shape, self.latents.shape)
            self.assertTrue(torch.isfinite(routed).all())
            routed.mean().backward()

    def test_transformer_has_musubi_constructor_args_and_cpu_forward(self):
        transformer = ZlabI1Transformer2DModel(
            input_size=4,
            image_resolution=32,
            patch_size=2,
            in_channels=3,
            hidden_size=24,
            depth=3,
            num_heads=2,
            mlp_ratio=1.0,
            text_embed_dim=6,
            text_num_tokens=5,
            musubi_blocks_to_swap=1,
            musubi_block_swap_device="cpu",
        )

        self.assertIsNotNone(transformer._musubi_block_swap)
        self.assertTrue(transformer._musubi_block_swap.is_managed_block(2))
        output = transformer(self.latents, self.timesteps, self.prompt_embeds, self.attention_mask)
        self.assertEqual(output.shape, self.latents.shape)

    def test_transformer_supports_torch_compile_batched_forward(self):
        transformer = tiny_transformer().eval()
        compiled = torch.compile(transformer, backend="aot_eager")
        latents = torch.randn(2, 3, 4, 4)
        timesteps = torch.zeros(2)
        prompt_embeds = torch.randn(2, 5, 6)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)

        with torch.no_grad():
            output = compiled(latents, timesteps, prompt_embeds, attention_mask)

        self.assertEqual(output.shape, latents.shape)
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_musubi_streams_i1_blocks_on_cuda_forward(self):
        transformer = ZlabI1Transformer2DModel(
            input_size=4,
            image_resolution=32,
            patch_size=2,
            in_channels=3,
            hidden_size=24,
            depth=3,
            num_heads=2,
            mlp_ratio=1.0,
            text_embed_dim=6,
            text_num_tokens=5,
            musubi_blocks_to_swap=1,
            musubi_block_swap_device="cpu",
        ).to("cuda")

        output = transformer(
            self.latents.to("cuda"),
            self.timesteps.to("cuda"),
            self.prompt_embeds.to("cuda"),
            self.attention_mask.to("cuda"),
        )

        self.assertEqual(output.shape, self.latents.shape)
        managed_block = list(transformer.out_blocks)[-1]
        self.assertEqual(next(managed_block.parameters()).device.type, "cpu")

    def test_pipeline_supports_cfg_zero_star_step_skip_and_skip_layer_guidance(self):
        self.transformer.eval()
        self.transformer.set_router(TREADRouter(seed=123, device="cpu"), [])
        pipeline = ZlabI1Pipeline(
            transformer=self.transformer,
            vae=DummyVAE(),
            text_encoder=None,
            tokenizer=None,
        )

        calls = [
            {"guidance_scale": 0.0},
            {"guidance_scale": 1.0},
            {"guidance_scale": 4.0, "use_cfg_zero_star": True, "zero_steps": 0},
            {"guidance_scale": 4.0, "no_cfg_until_timestep": 1},
            {
                "guidance_scale": 4.0,
                "skip_guidance_layers": [1],
                "skip_layer_guidance_start": 0.0,
                "skip_layer_guidance_stop": 1.0,
            },
        ]
        for kwargs in calls:
            with self.subTest(kwargs=kwargs):
                output = pipeline(
                    prompt_embeds=self.prompt_embeds,
                    attention_mask=self.attention_mask,
                    height=32,
                    width=32,
                    num_inference_steps=2,
                    output_type="latent",
                    **kwargs,
                ).images
                self.assertEqual(output.shape, self.latents.shape)
                self.assertTrue(torch.isfinite(output).all())

    def test_pipeline_does_not_force_transformer_to_device_when_musubi_is_active(self):
        transformer = ZlabI1Transformer2DModel(
            input_size=4,
            image_resolution=32,
            patch_size=2,
            in_channels=3,
            hidden_size=24,
            depth=3,
            num_heads=2,
            mlp_ratio=1.0,
            text_embed_dim=6,
            text_num_tokens=5,
            musubi_blocks_to_swap=1,
            musubi_block_swap_device="cpu",
        ).eval()
        pipeline = ZlabI1Pipeline(
            transformer=transformer,
            vae=DummyVAE(),
            text_encoder=None,
            tokenizer=None,
        )

        with patch.object(transformer, "to", wraps=transformer.to) as to_spy:
            output = pipeline(
                prompt_embeds=self.prompt_embeds,
                attention_mask=self.attention_mask,
                height=32,
                width=32,
                num_inference_steps=1,
                output_type="latent",
                guidance_scale=1.0,
            ).images

        self.assertEqual(output.shape, self.latents.shape)
        to_spy.assert_not_called()

    def test_model_predict_returns_hidden_state_buffer_for_layersync_and_crepa(self):
        model = ZLabI1.__new__(ZLabI1)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(base_weight_dtype=torch.float32)
        model.model = tiny_transformer()
        model.layersync_regularizer = SimpleNamespace(wants_hidden_states=lambda: True)
        model.crepa_regularizer = SimpleNamespace(wants_hidden_states=lambda: True, block_index=1)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 32, 4, 4),
            "timesteps": torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
            "prompt_embeds": torch.randn(1, 5, 6),
            "attention_mask": torch.ones(1, 5, dtype=torch.bool),
            "crepa_capture_block_index": 2,
        }
        model.model = ZlabI1Transformer2DModel(
            input_size=4,
            image_resolution=32,
            patch_size=2,
            in_channels=32,
            hidden_size=24,
            depth=3,
            num_heads=2,
            mlp_ratio=1.0,
            text_embed_dim=6,
            text_num_tokens=5,
        )

        result = model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, prepared_batch["noisy_latents"].shape)
        self.assertIn("layer_2", result["hidden_states_buffer"])
        self.assertTrue(torch.equal(result["crepa_hidden_states"], result["hidden_states_buffer"]["layer_2"]))

        regularizer = LayerSyncRegularizer(
            SimpleNamespace(layersync_enabled=True, layersync_student_block=1, layersync_teacher_block=2, layersync_lambda=0.1)
        )
        loss, logs = regularizer.compute_loss(result["hidden_states_buffer"])
        self.assertIsNotNone(loss)
        self.assertIn("layersync_loss", logs)

    def test_model_predict_supports_batched_prompt_masks_and_conditioning_force_keep(self):
        model = ZLabI1.__new__(ZLabI1)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(base_weight_dtype=torch.float32, tread_config={"routes": []})
        model.model = ZlabI1Transformer2DModel(
            input_size=4,
            image_resolution=32,
            patch_size=2,
            in_channels=32,
            hidden_size=24,
            depth=3,
            num_heads=2,
            mlp_ratio=1.0,
            text_embed_dim=6,
            text_num_tokens=5,
        )
        model.layersync_regularizer = None
        model.crepa_regularizer = None

        conditioning_mask = torch.full((2, 1, 4, 4), -1.0)
        conditioning_mask[0] = 1.0
        conditioning_mask[1, :, :2, :2] = 1.0
        prepared_batch = {
            "noisy_latents": torch.randn(2, 32, 4, 4),
            "timesteps": torch.tensor([[100.0, 200.0, 300.0, 400.0], [400.0, 300.0, 200.0, 100.0]]),
            "prompt_embeds": torch.randn(2, 5, 6),
            "attention_mask": torch.tensor([[[1, 1, 1, 0, 0]], [[1, 1, 0, 0, 0]]], dtype=torch.bool),
            "conditioning_pixel_values": conditioning_mask,
            "loss_mask_type": "mask",
        }

        with patch.object(model.model, "forward", wraps=model.model.forward) as forward_spy:
            result = model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, prepared_batch["noisy_latents"].shape)
        force_keep_mask = forward_spy.call_args.kwargs["force_keep_mask"]
        self.assertEqual(force_keep_mask.shape, torch.Size([2, 4]))
        self.assertTrue(torch.equal(force_keep_mask[0], torch.tensor([True, True, True, True])))
        self.assertTrue(torch.equal(force_keep_mask[1], torch.tensor([True, False, False, False])))

    def test_model_predict_rejects_mismatched_batched_inputs(self):
        model = ZLabI1.__new__(ZLabI1)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(base_weight_dtype=torch.float32)
        model.model = ZlabI1Transformer2DModel(
            input_size=4,
            image_resolution=32,
            patch_size=2,
            in_channels=32,
            hidden_size=24,
            depth=3,
            num_heads=2,
            mlp_ratio=1.0,
            text_embed_dim=6,
            text_num_tokens=5,
        )

        prepared_batch = {
            "noisy_latents": torch.randn(2, 32, 4, 4),
            "timesteps": torch.tensor([100.0, 200.0]),
            "prompt_embeds": torch.randn(1, 5, 6),
            "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        }

        with self.assertRaisesRegex(ValueError, "prompt_embeds batch size"):
            model.model_predict(prepared_batch)

    def test_shared_masked_loss_handles_batched_i1_latents(self):
        model = ZLabI1.__new__(ZLabI1)
        model.config = SimpleNamespace(
            loss_type="l2",
            scheduled_sampling_reflexflow=False,
            masked_loss_probability=1.0,
        )
        model.diff2flow_bridge = None

        conditioning_mask = torch.full((2, 1, 4, 4), -1.0)
        conditioning_mask[0] = 1.0
        conditioning_mask[1, :, :, :2] = 1.0
        prepared_batch = {
            "latents": torch.zeros(2, 32, 4, 4),
            "noise": torch.zeros(2, 32, 4, 4),
            "timesteps": torch.tensor([100.0, 200.0]),
            "conditioning_pixel_values": conditioning_mask,
            "loss_mask_type": "mask",
        }
        model_output = {"model_prediction": torch.ones(2, 32, 4, 4)}

        loss = model.loss(prepared_batch, model_output)

        self.assertTrue(torch.isclose(loss, torch.tensor(0.75)))

    def test_pretrained_load_args_pass_musubi_config(self):
        model = ZLabI1.__new__(ZLabI1)
        model.config = SimpleNamespace(twinflow_enabled=False, musubi_blocks_to_swap=4, musubi_block_swap_device="cpu")

        args = model.pretrained_load_args({})

        self.assertEqual(args["musubi_blocks_to_swap"], 4)
        self.assertEqual(args["musubi_block_swap_device"], "cpu")

    def test_diffusers_safetensors_layout_loads_from_transformer_subfolder(self):
        source = tiny_transformer()
        with TemporaryDirectory() as tmpdir:
            transformer_dir = Path(tmpdir) / "transformer"
            source.save_pretrained(transformer_dir, safe_serialization=True)

            loaded = ZlabI1Transformer2DModel.from_pretrained(
                tmpdir,
                subfolder="transformer",
                use_safetensors=True,
            )

        self.assertIsInstance(loaded, ZlabI1Transformer2DModel)
        self.assertEqual(loaded.config.hidden_size, source.config.hidden_size)
        self.assertEqual(loaded.config.depth, source.config.depth)

    def test_diffusers_loader_preserves_safetensors_and_variant_kwargs(self):
        sentinel = object()
        with patch.object(ModelMixin, "from_pretrained", return_value=sentinel) as from_pretrained:
            loaded = ZlabI1Transformer2DModel.from_pretrained(
                "not-zlab-upstream",
                use_safetensors=False,
                variant="fp16",
            )

        self.assertIs(loaded, sentinel)
        kwargs = from_pretrained.call_args.kwargs
        self.assertFalse(kwargs["use_safetensors"])
        self.assertEqual(kwargs["variant"], "fp16")

    def test_acceleration_presets_include_ramtorch_and_musubi(self):
        presets = ZLabI1.get_acceleration_presets()
        preset_backends = {preset.backend.name for preset in presets}

        self.assertIn("RAMTORCH", preset_backends)
        self.assertIn("MUSUBI_BLOCK_SWAP", preset_backends)

    def test_ramtorch_conversion_applies_to_i1_transformer(self):
        model = ZLabI1.__new__(ZLabI1)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(
            ramtorch=True,
            ramtorch_target_modules=["in_blocks.0.attn.qkv_image"],
            ramtorch_transformer_percent=None,
            ramtorch_disable_sync_hooks=True,
            ramtorch_disable_extensions=True,
        )
        transformer = tiny_transformer()
        stub_imports = {
            "Linear": StubRamtorchLinear,
            "replace_all": lambda module, device=None: None,
            "broadcast_zero_params": None,
            "create_zero_param_groups": None,
            "setup_grad_sharding_hooks": None,
        }

        with patch.object(ramtorch_utils, "ensure_available", return_value=stub_imports):
            replaced = model._apply_ramtorch_layers(transformer, "transformer")

        self.assertEqual(replaced, 1)
        self.assertIsInstance(transformer.in_blocks[0].attn.qkv_image, StubRamtorchLinear)
        self.assertTrue(getattr(transformer.in_blocks[0].attn.qkv_image.weight, "is_ramtorch", False))

    def test_load_vae_enables_tiling_for_i1(self):
        model = ZLabI1.__new__(ZLabI1)
        model.AUTOENCODER_CLASS = DummyLoadVAE
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(
            vae_path="stub-vae",
            model_family="zlab_i1",
            revision=None,
            variant=None,
            delete_model_after_load=False,
            vae_enable_tiling=True,
            vae_enable_slicing=False,
            crepa_drop_vae_encoder=False,
            ramtorch=False,
        )
        model._single_file_checkpoint_path = lambda: None
        model.post_vae_load_setup = lambda: None

        model.load_vae(move_to_device=False)

        self.assertTrue(model.vae.tiling_enabled)

    def test_twinflow_forward_uses_i1_model_predict_contract(self):
        model = ZLabI1.__new__(ZLabI1)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(base_weight_dtype=torch.float32, weight_dtype=torch.float32)
        model.model = ZlabI1Transformer2DModel(
            input_size=4,
            image_resolution=32,
            patch_size=2,
            in_channels=32,
            hidden_size=24,
            depth=3,
            num_heads=2,
            mlp_ratio=1.0,
            text_embed_dim=6,
            text_num_tokens=5,
        )
        model.layersync_regularizer = None
        model.crepa_regularizer = None
        model.diff2flow_bridge = None

        prepared_batch = {
            "noisy_latents": torch.randn(1, 32, 4, 4),
            "timesteps": torch.tensor([250.0]),
            "sigmas": torch.tensor([0.25]),
            "prompt_embeds": torch.randn(1, 5, 6),
            "attention_mask": torch.ones(1, 5, dtype=torch.bool),
        }

        prediction = model._twinflow_forward(
            prepared_batch=prepared_batch,
            noisy_latents=prepared_batch["noisy_latents"],
            sigmas=prepared_batch["sigmas"],
            use_grad=False,
        )

        self.assertEqual(prediction.shape, prepared_batch["noisy_latents"].shape)
        self.assertTrue(torch.isfinite(prediction).all())

    def test_crepa_self_flow_batch_uses_i1_patch_tokens(self):
        model = ZLabI1.__new__(ZLabI1)
        model.config = SimpleNamespace(crepa_self_flow_mask_ratio=0.5)
        model.sample_flow_sigmas = lambda batch, state: (
            torch.tensor([0.75], dtype=torch.float32),
            torch.tensor([750.0], dtype=torch.float32),
        )
        batch = {
            "latents": torch.zeros(1, 32, 4, 4),
            "input_noise": torch.ones(1, 32, 4, 4),
            "sigmas": torch.tensor([0.25], dtype=torch.float32),
            "timesteps": torch.tensor([250.0], dtype=torch.float32),
        }

        updated = model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(updated["timesteps"].shape, torch.Size([1, 4]))
        self.assertEqual(updated["crepa_teacher_timesteps"].shape, torch.Size([1]))
        self.assertEqual(updated["crepa_teacher_noisy_latents"].shape, torch.Size([1, 32, 4, 4]))


if __name__ == "__main__":
    unittest.main()
