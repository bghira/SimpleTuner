from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry
from simpletuner.helpers.models.generation import base_model_generation_context
from simpletuner.helpers.training.state_tracker import StateTracker


class AssistantLoRADistiller(DistillationBase):
    """Train a positive assistant adapter on fresh samples from the frozen base model."""

    def __init__(self, teacher_model, student_model=None, *, noise_scheduler=None, config=None):
        options = {"num_inference_steps": 40, "resolutions": [[1024, 1024]], "seed": 42}
        options.update(config or {})
        super().__init__(teacher_model, student_model, options)
        if not self.low_rank_distillation or options.get("model_type") != "lora":
            raise ValueError("Assistant LoRA requires adapter training on a shared base model.")
        if getattr(teacher_model, "assistant_lora_loaded", False):
            raise ValueError("Disable the existing assistant adapter when training a new assistant LoRA.")
        if not isinstance(options["num_inference_steps"], int) or options["num_inference_steps"] < 1:
            raise ValueError("Assistant LoRA num_inference_steps must be a positive integer.")
        resolutions = options["resolutions"]
        if not isinstance(resolutions, list) or not resolutions:
            raise ValueError("Assistant LoRA resolutions must be a nonempty list of [width, height] pairs.")
        for resolution in resolutions:
            if (
                not isinstance(resolution, (list, tuple))
                or len(resolution) != 2
                or any(not isinstance(value, int) or value <= 0 for value in resolution)
            ):
                raise ValueError("Assistant LoRA resolutions must contain positive integer [width, height] pairs.")
        self.pipeline = teacher_model.get_latent_generation_pipeline()
        self.pipeline.set_progress_bar_config(disable=True)
        multiple = self.pipeline.vae_scale_factor * 2
        if any(value % multiple for resolution in resolutions for value in resolution):
            raise ValueError(f"Assistant LoRA resolution dimensions must be divisible by {multiple}.")
        self._generation_index = 0
        self._generated_samples = 0
        self._seed = 42 if options["seed"] is None else int(options["seed"])

    def consumes_caption_batches(self):
        return True

    def prepare_caption_batch(self, caption_batch, model, state):
        captions = caption_batch.get("captions")
        if not captions:
            raise ValueError("Assistant LoRA requires a nonempty caption batch.")
        backend_id = caption_batch["data_backend_id"]
        cache = StateTracker.get_data_backend(backend_id)["text_embed_cache"]
        text_output = cache.compute_embeddings_for_prompts(captions, return_concat=True, split_between_processes=False)
        text_output = model.collate_prompt_embeds([text_output])
        text_output = {
            key: (
                value.to(
                    device=model.accelerator.device,
                    dtype=model.config.weight_dtype if value.is_floating_point() else value.dtype,
                )
                if torch.is_tensor(value)
                else value
            )
            for key, value in text_output.items()
        }
        width, height = self.config["resolutions"][self._generation_index % len(self.config["resolutions"])]
        rank = model.accelerator.process_index
        world_size = model.accelerator.num_processes
        seeds = [self._seed + (self._generated_samples + index) * world_size + rank for index in range(len(captions))]
        generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]
        kwargs = model.convert_text_embed_for_pipeline(text_output)
        kwargs.update(
            width=width,
            height=height,
            num_inference_steps=self.config["num_inference_steps"],
            generator=generators,
            output_type="latent",
            guidance_scale_real=1.0,
        )
        kwargs = model.update_pipeline_call_kwargs(kwargs)
        self.pipeline.transformer = model.get_trained_component()
        original_scheduler = self.pipeline.scheduler
        try:
            self.pipeline.scheduler = copy.deepcopy(original_scheduler)
            with base_model_generation_context(model):
                generated = self.pipeline(**kwargs).images
                latents = model.unpack_generated_latents(generated, height=height, width=width).detach()
        finally:
            self.pipeline.scheduler = original_scheduler
        self._generation_index += 1
        self._generated_samples += len(captions)
        synthetic_batch = {
            "latent_batch": latents,
            "prompts": list(captions),
            "prompt_embeds": text_output["prompt_embeds"],
            "add_text_embeds": text_output.get("pooled_prompt_embeds"),
            "batch_time_ids": text_output.get("batch_time_ids"),
            "encoder_attention_mask": text_output.get("attention_masks"),
            "conditioning_pixel_values": None,
            "conditioning_latents": None,
            "conditioning_image_embeds": None,
            "is_regularisation_data": False,
            "is_i2v_data": False,
            "data_backend_id": backend_id,
            "captions": list(captions),
            "records": caption_batch.get("records", []),
            "assistant_generation_seeds": seeds,
            "assistant_generation_resolution": [width, height],
        }
        return model.prepare_batch(synthetic_batch, state)

    def _generation_config(self):
        return {
            "seed": self._seed,
            "resolutions": [list(resolution) for resolution in self.config["resolutions"]],
            "num_inference_steps": self.config["num_inference_steps"],
        }

    def on_save_checkpoint(self, step, ckpt_dir):
        Path(ckpt_dir, "assistant_lora_state.json").write_text(
            json.dumps(
                {
                    "generation_index": self._generation_index,
                    "generated_samples": self._generated_samples,
                    "generation_config": self._generation_config(),
                }
            )
        )

    def on_load_checkpoint(self, ckpt_dir):
        saved = json.loads(Path(ckpt_dir, "assistant_lora_state.json").read_text())
        if saved["generation_config"] != self._generation_config():
            raise ValueError("Assistant LoRA resume requires unchanged seed, resolutions and num_inference_steps.")
        self._generation_index = int(saved["generation_index"])
        self._generated_samples = int(saved["generated_samples"])


DistillationRegistry.register(
    "assistant_lora",
    AssistantLoRADistiller,
    data_requirements=[DatasetType.CAPTION],
    is_data_generator=True,
    requires_distillation_cache=False,
    requirement_notes="Generates fresh base-model latents from cached caption embeddings; no terminal latent cache.",
)
