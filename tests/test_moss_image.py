import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args
from simpletuner.helpers.models.common import PredictionTypes
from simpletuner.helpers.models.flux2.latents import denormalize_latents, normalize_latents, patchify, unpatchify
from simpletuner.helpers.models.moss_image.model import MOSSImage
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline, MOSSImageScheduler
from simpletuner.helpers.models.moss_image.text import encode_prompts
from simpletuner.helpers.models.moss_image.transformer import MOSSImageTransformer
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.layersync import LayerSyncRegularizer
from simpletuner.helpers.utils.hidden_state_buffer import HiddenStateBuffer

FIXTURE = json.loads(Path(__file__).with_name("fixtures").joinpath("moss_image_transformer.json").read_text())


def tiny_transformer(**kwargs):
    config = dict(FIXTURE["config"], **kwargs)
    torch.manual_seed(FIXTURE["seed"])
    return MOSSImageTransformer(**config)


class MOSSImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.model = tiny_transformer()
        self.image = torch.linspace(-1, 1, 96).reshape(2, 8, 2, 3)
        self.text = torch.linspace(-2, 2, 160).reshape(2, 5, 16)
        self.mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
        self.time = torch.tensor([700.0, 233.0])

    def predict(self, model=None, **kwargs):
        return (model or self.model)(self.image, self.time, self.text, self.mask, **kwargs).sample

    def pipeline(self, model=None):
        pipeline = MOSSImagePipeline(model or self.model)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline

    def sample(self, pipeline=None, **kwargs):
        inputs = dict(
            prompt_embeds=self.text,
            prompt_attention_mask=self.mask,
            negative_prompt_embeds=-self.text,
            negative_prompt_attention_mask=self.mask,
            height=32,
            width=48,
            num_inference_steps=3,
            guidance_scale=3.0,
            uncond_skip_layers=[1],
            latents=self.image,
            output_type="latent",
        )
        inputs.update(kwargs)
        return (pipeline or self.pipeline())(**inputs).images

    def test_forward_matches_the_pinned_upstream(self):
        with torch.no_grad():
            torch.testing.assert_close(self.predict(), torch.tensor(FIXTURE["expected"]), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(
                self.predict(skip_layers=[1]), torch.tensor(FIXTURE["skip_layer_1"]), atol=1e-6, rtol=1e-5
            )

    def test_sampling_matches_independent_euler_integration(self):
        expected = self.image.clone()
        sigmas = [1.0, 10 / 11, 5 / 7, 0.0]
        with torch.no_grad():
            for start, end in zip(sigmas[:-1], sigmas[1:]):
                time = torch.full((2,), start * 1000)
                positive = self.model(expected, time, self.text, self.mask).sample
                negative = self.model(expected, time, -self.text, self.mask, skip_layers=[1]).sample
                expected = expected + (end - start) * (negative + 3 * (positive - negative))
        torch.testing.assert_close(self.sample(), expected, atol=1e-5, rtol=1e-4)

    def test_padding_cannot_influence_text_refinement_or_joint_attention(self):
        expected = self.predict()
        changed = self.text.clone()
        changed[1, 2:] = 1e6
        actual = self.model(self.image, self.time, changed, self.mask).sample
        torch.testing.assert_close(actual, expected)
        single = self.model(self.image[1:], self.time[1:], changed[1:, :2]).sample
        torch.testing.assert_close(single, expected[1:])

    def test_attention_processors_use_the_shared_dispatcher(self):
        from simpletuner.helpers.models.moss_image import transformer

        with patch.object(transformer, "dispatch_attention_fn", wraps=transformer.dispatch_attention_fn) as dispatch:
            self.predict()
        self.assertEqual(dispatch.call_count, 5)
        self.assertTrue(all(call.args[0].shape[0] == 2 for call in dispatch.call_args_list))
        self.assertEqual(len(self.model.attn_processors), 5)

    def test_batched_outputs_and_gradients_match_unpadded_individual_samples(self):
        for tokenwise in (False, True):
            with self.subTest(tokenwise=tokenwise):
                batched = tiny_transformer(enable_time_sign_embed=True)
                batched.enable_flowmap_time_conditioning(0.25, "t-r")
                with torch.no_grad():
                    batched.time_sign_embed.weight[1].fill_(0.2)
                individual = copy.deepcopy(batched)
                image, text = self.image.clone().requires_grad_(), self.text.clone().requires_grad_()
                mask = torch.tensor([[0, 1, 1, 0, 1], [1, 0, 1, 0, 0]], dtype=torch.bool)
                times = self.time[:, None].expand(-1, 6) if tokenwise else self.time
                ends = times / 2
                signs = torch.tensor([-1, 1])
                output = batched(image, times, text, mask, r_timestep=ends, timestep_sign=signs).sample
                output.square().mean().backward()
                image_grad, text_grad = image.grad.clone(), text.grad.clone()
                image.grad, text.grad = None, None
                rows = []
                for index in range(2):
                    row = individual(
                        image[index : index + 1],
                        times[index : index + 1],
                        text[index : index + 1, mask[index]],
                        r_timestep=ends[index : index + 1],
                        timestep_sign=signs[index : index + 1],
                    ).sample
                    (row.square().sum() / output.numel()).backward()
                    rows.append(row)
                torch.testing.assert_close(output, torch.cat(rows), atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(image_grad, image.grad, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(text_grad, text.grad, atol=1e-6, rtol=1e-5)
                for (name, parameter), (_, expected) in zip(batched.named_parameters(), individual.named_parameters()):
                    with self.subTest(parameter=name):
                        torch.testing.assert_close(parameter.grad, expected.grad, atol=1e-6, rtol=1e-5)

    def test_masked_nan_embeddings_and_empty_captions(self):
        expected = self.predict()
        text = self.text.clone().requires_grad_()
        changed = text.masked_fill(~self.mask[:, :, None], float("nan"))
        output = self.model(self.image, self.time, changed, self.mask).sample
        torch.testing.assert_close(output, expected)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(text.grad).all())
        self.assertEqual(text.grad[~self.mask].abs().sum().item(), 0)
        for mask in (torch.zeros_like(self.mask), self.mask.new_zeros(2, 0)):
            with self.subTest(shape=mask.shape), self.assertRaisesRegex(ValueError, "at least one valid text token"):
                self.model(self.image, self.time, self.text[:, : mask.shape[1]], mask)

    def test_attention_activation_offload_preserves_gradients(self):
        eager = self.predict()
        eager.square().mean().backward()
        expected = {name: parameter.grad.clone() for name, parameter in self.model.named_parameters()}
        model = tiny_transformer()
        model.set_gradient_checkpointing_offload_attention(True)
        prediction = self.predict(model)
        torch.testing.assert_close(prediction, eager)
        prediction.square().mean().backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, expected[name], atol=1e-6, rtol=1e-5)

    def test_validation_embedding_keyword_interface(self):
        foundation = object.__new__(MOSSImage)
        entry = {"prompt_embeds": self.text[0], "attention_masks": self.mask[0]}
        positive = foundation.convert_text_embed_for_pipeline(text_embedding=entry)
        negative = foundation.convert_negative_text_embed_for_pipeline(text_embedding=entry)
        self.assertEqual(positive["prompt_embeds"].shape, (1, 5, 16))
        self.assertEqual(negative["negative_prompt_attention_mask"].shape, (1, 5))
        torch.testing.assert_close(positive["prompt_embeds"], negative["negative_prompt_embeds"])

    def test_full_and_ffn_checkpoint_gradients_match_eager(self):
        eager = self.predict()
        eager.square().mean().backward()
        expected = {name: parameter.grad.clone() for name, parameter in self.model.named_parameters()}
        for backend in ("torch", "torch-ffn"):
            for interval, stride in ((1, None), (2, None), (2, 3)):
                with self.subTest(backend=backend, interval=interval, stride=stride):
                    model = tiny_transformer()
                    model.enable_gradient_checkpointing()
                    model.set_gradient_checkpointing_backend(backend)
                    model.set_gradient_checkpointing_interval(interval)
                    model.set_gradient_checkpointing_segment_stride(stride)
                    prediction = self.predict(model)
                    torch.testing.assert_close(prediction, eager)
                    prediction.square().mean().backward()
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(parameter.grad, expected[name], atol=1e-6, rtol=1e-5)

    def test_tokenwise_times_and_hidden_state_capture_keep_spatial_tokens(self):
        scalar = self.predict()
        states = HiddenStateBuffer(capture_layers={0, 2})
        prediction = self.model(
            self.image, self.time[:, None].expand(-1, 6), self.text, self.mask, hidden_states_buffer=states
        ).sample
        torch.testing.assert_close(prediction, scalar, atol=1e-6, rtol=1e-5)
        self.assertEqual(set(states), {"layer_0", "layer_2"})
        self.assertEqual(states["layer_0"].shape, (2, 6, 32))
        regularizer = LayerSyncRegularizer(
            SimpleNamespace(
                layersync_enabled=True,
                layersync_student_block=0,
                layersync_teacher_block=2,
                layersync_lambda=0.2,
            )
        )
        loss, logs = regularizer.compute_loss(states)
        self.assertTrue(torch.isfinite(loss))
        (prediction.square().mean() + loss).backward()
        self.assertIsNotNone(self.model.layers[0].proj["q"].weight.grad)

    def test_anyflow_interval_conditioning_is_trainable_and_roundtrips(self):
        self.model.enable_flowmap_time_conditioning(gate_value=0.0, deltatime_type="t-r")
        baseline = self.predict()
        conditioned = self.predict(r_timestep=torch.tensor([400.0, 100.0]))
        torch.testing.assert_close(conditioned, baseline)
        self.model.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="t-r")
        self.predict(r_timestep=torch.tensor([400.0, 100.0])).square().mean().backward()
        self.assertGreater(self.model.delta_timestep_embedder[0].weight.grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            restored = MOSSImageTransformer.from_pretrained(directory)
            torch.testing.assert_close(
                self.predict(restored, r_timestep=torch.tensor([400.0, 100.0])),
                self.predict(r_timestep=torch.tensor([400.0, 100.0])),
            )

    def test_twinflow_sign_is_neutral_on_load_and_saved_with_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            model = MOSSImageTransformer.from_pretrained(directory, enable_time_sign_embed=True)
        torch.testing.assert_close(model.time_sign_embed.weight, torch.zeros_like(model.time_sign_embed.weight))
        model.add_adapter(
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=MOSSImage.DEFAULT_LORA_TARGET,
                modules_to_save=["time_sign_embed"],
            )
        )
        self.predict(model, timestep_sign=torch.tensor([-1, 1])).square().mean().backward()
        state = get_peft_model_state_dict(model)
        self.assertIn("time_sign_embed.weight", state)
        self.assertTrue(any(parameter.grad is not None for parameter in model.time_sign_embed.parameters()))

    def test_twinflow_adapter_reload_restores_signed_conditioning(self):
        signed = tiny_transformer(enable_time_sign_embed=True)
        signed.add_adapter(
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=MOSSImage.DEFAULT_LORA_TARGET,
                modules_to_save=["time_sign_embed"],
            )
        )
        with torch.no_grad():
            signed.time_sign_embed.modules_to_save["default"].weight.fill_(0.4)
        trained_pipeline = self.pipeline(signed)
        restored_pipeline = self.pipeline(tiny_transformer())
        with tempfile.TemporaryDirectory() as directory:
            MOSSImagePipeline.save_lora_weights(
                directory,
                transformer_lora_layers=get_peft_model_state_dict(signed),
                transformer_lora_adapter_metadata=signed.peft_config["default"].to_dict(),
            )
            restored_pipeline.load_lora_weights(directory)
        torch.testing.assert_close(self.sample(restored_pipeline), self.sample(trained_pipeline))

    def test_anyflow_adapter_reload_restores_interval_conditioning(self):
        self.model.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="t-r")
        self.model.crepa_projector = torch.nn.Linear(32, 4)
        self.model.add_adapter(
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=MOSSImage.DEFAULT_LORA_TARGET,
                modules_to_save=["delta_timestep_embedder", "crepa_projector"],
            )
        )
        prediction = self.predict(r_timestep=torch.tensor([400.0, 100.0]))
        prediction.square().mean().backward()
        trainable = self.model.delta_timestep_embedder.modules_to_save["default"]
        self.assertGreater(trainable[0].weight.grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            MOSSImagePipeline.save_lora_weights(
                directory,
                transformer_lora_layers=get_peft_model_state_dict(self.model),
                transformer_lora_adapter_metadata=self.model.peft_config["default"].to_dict(),
            )
            with self.assertRaisesRegex(ValueError, "flowmap_deltatime_type"):
                self.pipeline(tiny_transformer()).load_lora_weights(directory)
            restored = self.pipeline(tiny_transformer())
            restored.load_lora_weights(directory, flowmap_deltatime_type="t-r", flowmap_gate_value=0.25)
        torch.testing.assert_close(self.sample(restored), self.sample())

    def test_anyflow_metadata_reload_preserves_accelerator_placement(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            self.skipTest("No accelerator available for adapter placement regression")
        self.model.to(device)
        self.model.enable_flowmap_time_conditioning(0.25, "r")
        self.model.add_adapter(
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=MOSSImage.DEFAULT_LORA_TARGET,
                modules_to_save=["delta_timestep_embedder"],
            )
        )
        restored = self.pipeline(tiny_transformer().to(device))
        with tempfile.TemporaryDirectory() as directory:
            MOSSImagePipeline.save_lora_weights(
                directory,
                transformer_lora_layers=get_peft_model_state_dict(self.model),
                transformer_lora_adapter_metadata=self.model.peft_config["default"].to_dict(),
            )
            restored.load_lora_weights(directory, flowmap_deltatime_type="r")
        delta = restored.transformer.delta_timestep_embedder
        self.assertTrue(all(parameter.device.type == device.type for parameter in delta.parameters()))
        torch.testing.assert_close(self.sample(restored), self.sample(), atol=1e-5, rtol=1e-5)

    def test_foundation_defaults_keep_training_and_inference_schedules_separate(self):
        from simpletuner.helpers.configuration.cmd_args import get_default_config

        config = SimpleNamespace(
            **dict(
                get_default_config(),
                model_family="moss_image",
                model_flavour=None,
                pretrained_model_name_or_path=None,
                model_type="lora",
                vae_path=None,
            )
        )
        foundation = MOSSImage(config, SimpleNamespace(device=torch.device("cpu")))
        foundation.check_user_config()
        self.assertEqual(foundation.config.model_flavour, "sft")
        self.assertEqual(foundation.config.pretrained_model_name_or_path, "OpenMOSS-Team/MOSS-Image-SFT")
        self.assertEqual(foundation.noise_schedule.config.shift, 1.0)
        self.assertEqual(foundation.config.flow_schedule_shift, 1.0)
        self.assertEqual(foundation.config.moss_image_validation_shift, 5.0)
        self.assertEqual(foundation.config.moss_image_uncond_skip_layers, [22])
        self.assertEqual(foundation.config.vae_dtype, "fp32")
        self.assertEqual(foundation.config.tokenizer_max_length, 4096)
        self.assertTrue(foundation.VALIDATION_REQUIRES_EMPTY_NEGATIVE_PROMPT)

    def test_model_card_uses_native_pipeline_and_preserves_anyflow_settings(self):
        from simpletuner.helpers.configuration.cmd_args import get_default_config
        from simpletuner.helpers.publishing.metadata import _model_card_family_tag, _model_imports

        config = SimpleNamespace(
            **dict(
                get_default_config(),
                model_family="moss_image",
                model_flavour="sft",
                model_type="lora",
                pretrained_model_name_or_path=None,
                vae_path=None,
            )
        )
        foundation = MOSSImage(config, SimpleNamespace(device=torch.device("cpu")))
        foundation.model = self.model
        foundation.unwrap_model = lambda model=None: self.model if model is None else model
        self.model.enable_flowmap_time_conditioning(0.25, "t-r")
        example = foundation.custom_model_card_code_example("example/moss-lora")
        self.assertIn("MOSSImagePipeline", example)
        self.assertIn("OpenMOSS-Team/MOSS-Image-SFT", example)
        self.assertIn('flowmap_deltatime_type="t-r"', example)
        self.assertIn("flowmap_gate_value=0.25", example)
        self.assertNotIn(str(Path.home()), example)
        self.assertEqual(_model_card_family_tag("moss_image"), "moss-image")
        config.lora_type = "lycoris"
        self.assertEqual(foundation.custom_model_card_code_example(), "")
        with patch("simpletuner.helpers.publishing.metadata.StateTracker.get_model", return_value=None):
            self.assertIn("moss_image.pipeline import MOSSImagePipeline", _model_imports(config))

    def test_native_anyflow_distiller_builds_a_finite_trainable_image_target(self):
        from simpletuner.helpers.configuration.cmd_args import get_default_config
        from simpletuner.helpers.distillation.anyflow.distiller import AnyFlowDistiller

        config = SimpleNamespace(
            **dict(
                get_default_config(),
                model_family="moss_image",
                model_flavour="base",
                pretrained_model_name_or_path=None,
                model_type="lora",
                vae_path=None,
                weight_dtype=torch.float32,
                lora_dropout=0.0,
            )
        )
        foundation = MOSSImage(
            config,
            SimpleNamespace(
                device=torch.device("cpu"),
                num_processes=1,
                process_index=0,
                is_main_process=True,
            ),
        )
        foundation.model = self.model
        foundation.unwrap_model = lambda model=None: self.model if model is None else model
        foundation._new_hidden_state_buffer = lambda: None
        distill_config = dict(model_type="lora", fuse_guidance_scale=1.0, seed=42)
        AnyFlowDistiller.prepare_model_for_adapter(foundation, distill_config)
        self.model.add_adapter(
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=MOSSImage.DEFAULT_LORA_TARGET,
                modules_to_save=foundation.get_lora_save_layers(),
            )
        )
        distiller = AnyFlowDistiller(
            teacher_model=foundation,
            student_model=foundation,
            noise_scheduler=foundation.noise_schedule,
            config=distill_config,
        )
        noise = torch.randn_like(self.image)
        sigmas = (self.time / 1000).reshape(2, 1, 1, 1)
        batch = dict(
            latents=self.image,
            noise=noise,
            input_noise=noise,
            sigmas=sigmas,
            timesteps=self.time,
            noisy_latents=(1 - sigmas) * self.image + sigmas * noise,
            prompt_embeds=self.text,
            encoder_attention_mask=self.mask,
        )
        prepared = distiller.prepare_batch(batch, model=foundation, state={})
        output = foundation.model_predict(prepared_batch=prepared)
        loss, logs = distiller.compute_distill_loss(prepared, output, torch.tensor(0.0))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        delta = self.model.delta_timestep_embedder.modules_to_save["default"]
        self.assertIsNotNone(delta[0].weight.grad)

    def test_lora_optimization_and_pipeline_adapter_roundtrip(self):
        base = copy.deepcopy(self.model)
        self.model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=MOSSImage.DEFAULT_LORA_TARGET))
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=0.01)
        before = self.predict().detach()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = (self.predict() - 0.5).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        after = self.predict().detach()
        self.assertGreater((after - before).abs().max().item(), 1e-5)
        with tempfile.TemporaryDirectory() as directory:
            MOSSImagePipeline.save_lora_weights(directory, transformer_lora_layers=get_peft_model_state_dict(self.model))
            self.pipeline(base).load_lora_weights(directory)
        torch.testing.assert_close(self.predict(base), after)

    def test_pipeline_roundtrip_preserves_optional_components_and_schedule(self):
        pipeline = self.pipeline()
        with tempfile.TemporaryDirectory() as directory:
            pipeline.save_pretrained(directory)
            restored = MOSSImagePipeline.from_pretrained(directory, torch_dtype=torch.float32)
        restored.set_progress_bar_config(disable=True)
        self.assertIsNone(restored.text_encoder)
        self.assertIsNone(restored.tokenizer)
        self.assertIsNone(restored.vae)
        self.assertEqual(restored.scheduler.config.shift, 5.0)
        torch.testing.assert_close(self.sample(restored), self.sample(pipeline))

    def test_preview_callbacks_replace_latents_and_interrupt_sampling(self):
        called = []

        def callback(pipeline, index, timestep, tensors):
            called.append(index)
            self.assertEqual(set(tensors), {"latents"})
            pipeline._interrupt = True
            return {"latents": torch.zeros_like(tensors["latents"])}

        result = self.sample(callback_on_step_end=callback)
        self.assertEqual(called, [0])
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_zero_star_rescale_and_skip_guidance_are_finite(self):
        baseline = self.sample()
        result = self.sample(use_cfg_zero_star=True, guidance_rescale=0.5, slg_layers=[0], slg_scale=2.0)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater((result - baseline).abs().max().item(), 1e-5)
        for invalid in ([True], [-1], [3]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.sample(uncond_skip_layers=invalid)

    def test_spatial_vae_packing_and_normalization_are_invertible(self):
        latent = torch.randn(2, 32, 4, 6)
        vae = SimpleNamespace(bn=torch.nn.BatchNorm2d(128, affine=False), config=SimpleNamespace(batch_norm_eps=1e-4))
        vae.bn.running_mean.copy_(torch.randn(128))
        vae.bn.running_var.copy_(torch.rand(128) + 0.1)
        packed = normalize_latents(patchify(latent), vae)
        torch.testing.assert_close(unpatchify(denormalize_latents(packed, vae)), latent)
        model = MOSSImage.__new__(MOSSImage)
        model.vae = vae
        torch.testing.assert_close(model.pre_latent_decode(packed), unpatchify(packed))
        torch.testing.assert_close(model.pre_latent_decode(model.denormalize_latents_for_preview(packed)), latent)
        self.assertTrue(model.supports_validation_preview())

    def test_vae_matches_the_released_small_decoder(self):
        config = json.loads(Path(__file__).with_name("fixtures").joinpath("moss_image_vae.json").read_text())
        with torch.device("meta"):
            vae = MOSSImage.AUTOENCODER_CLASS.from_config(config)
        self.assertEqual(vae.decoder.conv_in.out_channels, 384)
        self.assertEqual(vae.encoder.conv_in.out_channels, 128)
        self.assertEqual(vae.bn.num_features, 128)

    def test_native_flavours_parse_and_resolve_without_remote_code(self):
        self.assertEqual(ModelRegistry.get("moss_image").NAME, "MOSS-Image")
        for flavour in ("base", "sft", "base-edit", "sft-edit"):
            args = parse_cmdline_args(
                input_args=[
                    "--model_family=moss_image",
                    f"--model_flavour={flavour}",
                    "--model_type=lora",
                    "--optimizer=adamw_bf16",
                    "--output_dir=output",
                    "--data_backend_config=config/data.json",
                    "--moss_image_uncond_skip_layers=[]",
                    "--validation_guidance_skip_layers=[24]",
                ],
                exit_on_error=True,
            )
            self.assertEqual(args.model_flavour, flavour)
            self.assertEqual(args.moss_image_uncond_skip_layers, "[]")
            self.assertEqual(args.validation_guidance_skip_layers, [24])

    def flow_foundation(self, **overrides):
        from simpletuner.helpers.configuration.cmd_args import get_default_config

        config = SimpleNamespace(
            **dict(
                get_default_config(),
                model_family="moss_image",
                model_flavour="sft",
                pretrained_model_name_or_path=None,
                model_type="lora",
                vae_path=None,
                **overrides,
            )
        )
        foundation = MOSSImage(config, SimpleNamespace(device=torch.device("cpu")))
        foundation._get_dataset_timestep_sampling_offset = lambda batch: 0.0
        return foundation

    def test_native_noise_distribution_matches_report_equation_four(self):
        foundation = self.flow_foundation()
        batch = {"latents": torch.empty(200000, 1), "noise": torch.empty(200000, 1)}
        with torch.random.fork_rng():
            torch.manual_seed(51)
            sigmas, timesteps = foundation.sample_flow_sigmas(batch, {})
        self.assertAlmostEqual(sigmas.mean().item(), 0.7441694326, delta=0.003)
        self.assertAlmostEqual((sigmas >= 0.7).float().mean().item(), 0.6541584951, delta=0.004)
        torch.testing.assert_close(timesteps, sigmas * 1000)
        torch.testing.assert_close(timesteps, timesteps.round(), atol=0.0001, rtol=0)
        self.assertTrue(torch.all((sigmas >= 0.001) & (sigmas <= 1)))
        self.assertTrue(torch.all(foundation._noise_sampling_probabilities >= 0.00001))

    def test_native_noise_scale_zero_and_dataset_offset(self):
        batch = {"latents": torch.empty(10000, 1), "noise": torch.empty(10000, 1)}
        foundation = self.flow_foundation(flow_sigmoid_scale=0.0)
        sigmas, _ = foundation.sample_flow_sigmas(batch, {})
        self.assertGreater((sigmas == 0.788).float().mean().item(), 0.97)
        foundation.config.flow_sigmoid_scale = 1.0
        foundation.sample_flow_sigmas(batch, {})
        original = foundation._noise_sampling_probabilities.clone()
        foundation._get_dataset_timestep_sampling_offset = lambda batch: 0.5
        foundation.sample_flow_sigmas(batch, {})
        grid = torch.arange(1, 1001) / 1000
        self.assertGreater((grid * foundation._noise_sampling_probabilities).sum(), (grid * original).sum())

    def test_native_sampling_remains_finite_for_large_dataset_offsets(self):
        foundation = self.flow_foundation()
        batch = {"latents": torch.empty(128, 1), "noise": torch.empty(128, 1)}
        for offset in [-10.0, 10.0]:
            with self.subTest(offset=offset):
                foundation._get_dataset_timestep_sampling_offset = lambda batch: offset
                sigmas, _ = foundation.sample_flow_sigmas(batch, {})
                self.assertTrue(torch.isfinite(sigmas).all())
                self.assertAlmostEqual(foundation._noise_sampling_probabilities.sum().item(), 1.0, places=6)

    def test_explicit_schedules_bypass_native_sampling(self):
        batch = {"latents": torch.empty(8, 1), "noise": torch.empty(8, 1)}
        for options in [
            {"flow_use_uniform_schedule": True},
            {"flow_use_beta_schedule": True},
            {"flow_custom_timesteps": [250, 750]},
        ]:
            with self.subTest(options=options):
                foundation = self.flow_foundation(**options)
                with patch.object(
                    foundation, "sample_default_flow_sigmas", side_effect=AssertionError("native sampler called")
                ):
                    sigmas, timesteps = foundation.sample_flow_sigmas(batch, {})
                self.assertEqual(sigmas.shape, (8,))
                self.assertTrue(torch.all((sigmas >= 0) & (sigmas <= 1)))

    def test_scheduler_shifts_uniform_rollout_grid_once(self):
        expected = {1: [1, 0.75, 0.5, 0.25, 0], 3: [1, 0.9, 0.75, 0.5, 0], 5: [1, 0.9375, 5 / 6, 0.625, 0]}
        from simpletuner.helpers.distillation.anyflow.distiller import AnyFlowDistiller

        for shift, values in expected.items():
            with self.subTest(shift=shift):
                scheduler = MOSSImageScheduler(shift=shift)
                scheduler.set_timesteps(4)
                torch.testing.assert_close(scheduler.sigmas, torch.tensor(values))
                distiller = object.__new__(AnyFlowDistiller)
                distiller.config = {"schedule_shift": shift}
                distiller.noise_scheduler = scheduler
                rollout = distiller._apply_scheduler_shift(torch.linspace(1, 0, 5))
                torch.testing.assert_close(scheduler.sigmas, rollout)
        scheduler.set_timesteps(1)
        torch.testing.assert_close(scheduler.sigmas, torch.tensor([1.0, 0.0]))
        scheduler.set_timesteps(50)
        self.assertEqual(len(scheduler.timesteps), 50)
        self.assertAlmostEqual(scheduler.sigmas[-2].item(), 5 * 0.02 / (1 + 4 * 0.02), places=6)

    def test_scheduler_matches_training_grid_and_preserves_float32_updates(self):
        scheduler = MOSSImageScheduler()
        scheduler.set_timesteps(50)
        self.assertEqual(scheduler.timesteps[0].item(), 1000)
        torch.testing.assert_close(scheduler.sigmas[:-1] * 1000, scheduler.timesteps)
        self.assertTrue(torch.all(scheduler.sigmas[:-1] >= scheduler.sigmas[1:]))
        output = self.sample()
        self.assertEqual(output.dtype, torch.float32)

    def test_mps_forward_and_backward(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        model = self.model.to("mps")
        result = model(self.image.to("mps"), self.time.to("mps"), self.text.to("mps"), self.mask.to("mps")).sample
        self.assertTrue(torch.isfinite(result).all())
        result.square().mean().backward()
        self.assertIsNotNone(model.layers[0].proj["q"].weight.grad)


if __name__ == "__main__":
    unittest.main()
