import unittest
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import torch

from simpletuner.helpers.models.boogu_image.lora_pipeline import BooguImageLoraLoaderMixin
from simpletuner.helpers.models.boogu_image.embeddings import apply_rotary_emb
from simpletuner.helpers.models.boogu_image.model import BooguImage
from simpletuner.helpers.models.boogu_image.pipeline import BooguImagePipeline, retrieve_timesteps
from simpletuner.helpers.models.boogu_image.rope import BooguImageDoubleStreamRotaryPosEmbed
from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.flux.model import Flux
from simpletuner.helpers.training.attention_backend import _DIFFUSERS_BACKEND_ALIASES
from simpletuner.helpers.training.quantisation import _torchao_filter_fn


class BooguImageModelTests(unittest.TestCase):
    def _config(self, flavour):
        return SimpleNamespace(
            model_flavour=flavour,
            pretrained_model_name_or_path=None,
            pretrained_transformer_model_name_or_path=None,
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
            revision=None,
            variant=None,
        )

    def test_flavour_paths_include_requested_variants(self):
        self.assertEqual(BooguImage.HUGGINGFACE_PATHS["v0.1-base"], "SimpleTuner/Boogu-Image-0.1-Base")
        self.assertEqual(BooguImage.HUGGINGFACE_PATHS["v0.1-base-fp8"], "SimpleTuner/Boogu-Image-0.1-Base")
        self.assertEqual(BooguImage.HUGGINGFACE_PATHS["v0.1-turbo"], "SimpleTuner/Boogu-Image-0.1-Turbo")
        self.assertEqual(BooguImage.HUGGINGFACE_PATHS["v0.1-turbo-fp8"], "SimpleTuner/Boogu-Image-0.1-Turbo")
        self.assertEqual(BooguImage.HUGGINGFACE_PATHS["v0.1-edit"], "SimpleTuner/Boogu-Image-0.1-Edit")
        self.assertEqual(BooguImage.HUGGINGFACE_PATHS["v0.1-edit-fp8"], "SimpleTuner/Boogu-Image-0.1-Edit")

    def test_model_components_are_local_simpletuner_classes(self):
        self.assertEqual(BooguImage.MODEL_CLASS.__module__, "simpletuner.helpers.models.boogu_image.transformer")
        self.assertEqual(
            BooguImage.PIPELINE_CLASSES[next(iter(BooguImage.PIPELINE_CLASSES))].__module__,
            "simpletuner.helpers.models.boogu_image.pipeline",
        )
        self.assertIn("BooguImageSingleStreamTransformerBlock", BooguImage.MODEL_CLASS._repeated_blocks)

    def test_assistant_lora_enabled_for_turbo_only_with_placeholder_path(self):
        self.assertEqual(BooguImage.ASSISTANT_LORA_FLAVOURS, ["v0.1-turbo", "v0.1-turbo-fp8"])
        self.assertIsNone(BooguImage.ASSISTANT_LORA_PATH)

    def test_validation_preview_uses_flux_tae_spec(self):
        self.assertIs(BooguImage.VALIDATION_PREVIEW_SPEC, Flux.VALIDATION_PREVIEW_SPEC)
        self.assertEqual(BooguImage.VALIDATION_PREVIEW_SPEC.repo_id, "madebyollin/taef1")

    def test_validation_uses_boogu_inference_scheduler(self):
        model = object.__new__(BooguImage)

        self.assertTrue(model.requires_special_scheduler_setup())

    def test_special_scheduler_models_load_pipeline_scheduler_without_training_scheduler(self):
        class DummyPipeline:
            last_kwargs = None

            @classmethod
            def from_pretrained(cls, **kwargs):
                cls.last_kwargs = kwargs
                return SimpleNamespace(**kwargs)

        class DummyScheduler:
            def __init__(self, name):
                self.name = name
                self.config = {"name": name}

            @classmethod
            def from_config(cls, config):
                return cls(config["name"])

        inference_scheduler = DummyScheduler("inference")
        model = object.__new__(BooguImage)
        model.model = object()
        model.vae = object()
        model.text_encoders = None
        model.tokenizers = None
        model.pipelines = {}
        model.noise_schedule = DummyScheduler("training")
        model.PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: DummyPipeline}
        model.config = SimpleNamespace(controlnet=False, revision=None, local_files_only=False)
        model._model_config_path = lambda: "repo/model"
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped

        with patch(
            "simpletuner.helpers.models.boogu_image.model.FlowMatchEulerDiscreteScheduler.from_pretrained",
            return_value=inference_scheduler,
        ):
            model._load_pipeline(PipelineTypes.TEXT2IMG)

        self.assertIs(DummyPipeline.last_kwargs["scheduler"], inference_scheduler)
        self.assertIsNot(DummyPipeline.last_kwargs["scheduler"], model.noise_schedule)

    def test_hub_kernel_attention_aliases_are_available(self):
        self.assertEqual(_DIFFUSERS_BACKEND_ALIASES["flash-attn-hub"].value, "flash_hub")
        self.assertEqual(_DIFFUSERS_BACKEND_ALIASES["flash-attn-3-hub"].value, "_flash_3_hub")
        self.assertEqual(_DIFFUSERS_BACKEND_ALIASES["flash-attn-4-hub"].value, "flash_4_hub")

    def test_lora_save_accepts_simpletuner_adapter_metadata(self):
        parameters = inspect.signature(BooguImageLoraLoaderMixin.save_lora_weights).parameters
        self.assertIn("metadata_kwargs", parameters)

    def test_fp8_pretrained_load_args_preserve_safetensors(self):
        model = object.__new__(BooguImage)
        model.config = self._config("v0.1-turbo-fp8")
        args = BooguImage.pretrained_load_args(model, {"use_safetensors": True})
        self.assertTrue(args["use_safetensors"])

    def test_non_fp8_pretrained_load_args_preserve_safetensors(self):
        model = object.__new__(BooguImage)
        model.config = self._config("v0.1-turbo")
        args = BooguImage.pretrained_load_args(model, {"use_safetensors": True})
        self.assertTrue(args["use_safetensors"])

    def test_edit_flavours_require_conditioning(self):
        model = object.__new__(BooguImage)
        model.config = self._config("v0.1-edit-fp8")
        self.assertTrue(model.requires_conditioning_dataset())
        self.assertTrue(model.requires_conditioning_latents())
        self.assertTrue(model.requires_text_embed_image_context())

    def test_model_predict_uses_trained_component_dispatch(self):
        model = object.__new__(BooguImage)
        model.config = SimpleNamespace(weight_dtype=torch.float32)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model._freqs_cis = lambda: "freqs"
        dispatch_calls = []
        forward_calls = []

        class FakeTransformer:
            def __call__(self, *args, **kwargs):
                forward_calls.append((args, kwargs))
                return SimpleNamespace(sample="prediction")

        def get_trained_component(*, base_model=False):
            dispatch_calls.append(base_model)
            return FakeTransformer()

        model.get_trained_component = get_trained_component
        prepared_batch = {
            "noisy_latents": torch.zeros(1, 16, 8, 8),
            "timesteps": torch.tensor(0.5),
            "encoder_hidden_states": torch.zeros(1, 4, 8),
            "encoder_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        }

        result = model.model_predict(prepared_batch)

        self.assertEqual(result, {"model_prediction": "prediction"})
        self.assertEqual(dispatch_calls, [True])
        self.assertEqual(len(forward_calls), 1)
        self.assertEqual(forward_calls[0][0][3], "freqs")

    def test_boogu_flow_target_points_from_noise_to_latents(self):
        model = object.__new__(BooguImage)
        latents = torch.tensor([1.0, 3.0])
        noise = torch.tensor([4.0, -1.0])

        target = model.get_prediction_target({"latents": latents, "noise": noise})

        self.assertTrue(torch.equal(target, latents - noise))

    def test_boogu_flow_timesteps_are_normalized_clean_progress(self):
        model = object.__new__(BooguImage)

        def sample_flow_sigmas(self, batch, state):
            return torch.tensor([0.25, 0.75]), torch.tensor([250.0, 750.0])

        parent = BooguImage.__mro__[1]
        original = parent.sample_flow_sigmas
        try:
            parent.sample_flow_sigmas = sample_flow_sigmas
            noise_sigmas, timesteps = model.sample_flow_sigmas(batch={}, state={})
        finally:
            parent.sample_flow_sigmas = original

        self.assertTrue(torch.equal(noise_sigmas, torch.tensor([0.25, 0.75])))
        self.assertTrue(torch.equal(timesteps, torch.tensor([0.75, 0.25])))

    def test_torchao_filter_skips_boogu_reference_image_modules(self):
        self.assertFalse(_torchao_filter_fn(torch.nn.Linear(16, 16), "ref_image_refiner.0.attn.to_q"))
        self.assertTrue(_torchao_filter_fn(torch.nn.Linear(16, 16), "context_refiner.0.attn.to_q"))

    def test_boogu_rotary_complex_inputs_use_real_valued_math(self):
        x = torch.randn(2, 5, 3, 8, dtype=torch.bfloat16)
        angles = torch.randn(2, 5, 4)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        expected = torch.view_as_real(
            torch.view_as_complex(x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2))
            * freqs_cis.unsqueeze(2)
        ).flatten(-2).to(x.dtype)

        with patch("torch.view_as_complex", side_effect=AssertionError("complex path used")):
            actual = apply_rotary_emb(x, freqs_cis, use_real=False)

        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2))

    def test_boogu_double_stream_rope_returns_real_rotary_tuples(self):
        embedder = BooguImageDoubleStreamRotaryPosEmbed(
            theta=10000,
            axes_dim=(4, 4, 4),
            axes_lens=(8, 8, 8),
            patch_size=2,
        )
        freqs_cis = embedder.get_freqs_cis(embedder.axes_dim, embedder.axes_lens, embedder.theta)

        output = embedder(
            freqs_cis=freqs_cis,
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            l_effective_ref_img_len=[[4]],
            l_effective_img_len=[4],
            ref_img_sizes=[[(4, 4)]],
            img_sizes=[(4, 4)],
            device=torch.device("cpu"),
        )

        for rotary in (output[0], output[1], output[2], output[3], output[6]):
            self.assertIsInstance(rotary, tuple)
            self.assertEqual(len(rotary), 2)
            self.assertFalse(torch.is_complex(rotary[0]))
            self.assertFalse(torch.is_complex(rotary[1]))

    def test_validation_kwargs_are_mapped_to_boogu_pipeline_names(self):
        model = object.__new__(BooguImage)
        pipeline_kwargs = model.update_pipeline_call_kwargs(
            {
                "prompt": "a ceramic mug",
                "negative_prompt": "blurry",
                "num_images_per_prompt": 2,
                "guidance_scale": 4.0,
                "guidance_scale_real": 1.0,
            }
        )

        self.assertNotIn("prompt", pipeline_kwargs)
        self.assertNotIn("negative_prompt", pipeline_kwargs)
        self.assertNotIn("num_images_per_prompt", pipeline_kwargs)
        self.assertNotIn("guidance_scale", pipeline_kwargs)
        self.assertNotIn("guidance_scale_real", pipeline_kwargs)
        self.assertEqual(pipeline_kwargs["instruction"], "a ceramic mug")
        self.assertEqual(pipeline_kwargs["negative_instruction"], "blurry")
        self.assertEqual(pipeline_kwargs["num_images_per_instruction"], 2)
        self.assertEqual(pipeline_kwargs["text_guidance_scale"], 4.0)

    def test_validation_kwargs_keep_precomputed_instruction_embeds(self):
        model = object.__new__(BooguImage)
        instruction_embeds = torch.zeros(1, 4, 8)
        pipeline_kwargs = model.update_pipeline_call_kwargs(
            {
                "prompt": None,
                "negative_prompt": None,
                "instruction_embeds": instruction_embeds,
                "num_images_per_prompt": 1,
                "guidance_scale": 4.0,
            }
        )

        self.assertIs(pipeline_kwargs["instruction_embeds"], instruction_embeds)
        self.assertNotIn("instruction", pipeline_kwargs)
        self.assertNotIn("negative_instruction", pipeline_kwargs)
        self.assertEqual(pipeline_kwargs["num_images_per_instruction"], 1)
        self.assertEqual(pipeline_kwargs["text_guidance_scale"], 4.0)

    def test_pipeline_accepts_precomputed_instruction_embeds_without_instruction(self):
        pipeline = object.__new__(BooguImagePipeline)
        pipeline.enable_inner_devices_manager = False
        instruction_embeds = torch.zeros(1, 4, 8)
        instruction_attention_mask = torch.ones(1, 4, dtype=torch.bool)

        result = pipeline.encode_instruction(
            instruction=None,
            do_classifier_free_guidance=False,
            device=torch.device("cpu"),
            instruction_embeds=instruction_embeds,
            instruction_attention_mask=instruction_attention_mask,
        )

        self.assertTrue(torch.equal(result[0], instruction_embeds))
        self.assertTrue(torch.equal(result[1], instruction_attention_mask))
        self.assertIsNone(result[2])
        self.assertIsNone(result[3])

    def test_retrieve_timesteps_filters_unsupported_scheduler_kwargs(self):
        class SchedulerWithoutNumTokens:
            order = 1

            def set_timesteps(self, num_inference_steps, device=None):
                self.timesteps = torch.arange(num_inference_steps, device=device)

        scheduler = SchedulerWithoutNumTokens()
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps=3,
            device=torch.device("cpu"),
            num_tokens=4096,
        )

        self.assertEqual(num_inference_steps, 3)
        self.assertTrue(torch.equal(timesteps, torch.arange(3)))


if __name__ == "__main__":
    unittest.main()
