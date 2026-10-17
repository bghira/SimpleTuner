"""MOSS-Image text-to-image training and validation integration."""

import json

import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2Tokenizer, Qwen3_5Model

from simpletuner.helpers.configuration.registry import ConfigRegistry, make_default_rule, make_override_rule
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.flux2.latents import denormalize_latents, normalize_latents, patchify, unpatchify
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline, MOSSImageScheduler
from simpletuner.helpers.models.moss_image.prompting import MossImageJSONSchemaConverter
from simpletuner.helpers.models.moss_image.text import MAX_TEXT_LENGTH, encode_prompts
from simpletuner.helpers.models.moss_image.transformer import MOSSImageTransformer, _validate_skip_layers
from simpletuner.helpers.models.prompt_upsampling import TransformersJSONPromptUpsampler
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.models.tae.types import Flux2TAESpec
from simpletuner.helpers.models.text import collate_embeddings
from simpletuner.helpers.musubi_block_swap import apply_musubi_pretrained_defaults


class MOSSImage(ImageModelFoundation):
    NAME = "MOSS-Image"
    PROMPT_UPSAMPLING_CONFIG_KEY = "moss_image_prompt_upsample"
    PROMPT_UPSAMPLING_MODEL_ID = "Qwen/Qwen3.5-27B"
    MODEL_DESCRIPTION = "MOSS-Image text-to-image generation with native text and layout control"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    MODEL_CLASS = MOSSImageTransformer
    MODEL_SUBFOLDER = "transformer"
    DEFAULT_MODEL_FLAVOUR = "sft"
    HUGGINGFACE_PATHS = {"base": "OpenMOSS-Team/MOSS-Image-Base", "sft": "OpenMOSS-Team/MOSS-Image-SFT"}
    MODEL_LICENSE = "apache-2.0"
    AUTOENCODER_CLASS = AutoencoderKLFlux2
    AUTOENCODER_SCALING_FACTOR = 1.0
    LATENT_CHANNEL_COUNT = 128
    VAE_SCALE_FACTOR = 16
    USES_DYNAMIC_SHIFT = False
    VALIDATION_USES_NEGATIVE_PROMPT = True
    VALIDATION_REQUIRES_EMPTY_NEGATIVE_PROMPT = True
    VALIDATION_PREVIEW_SPEC = Flux2TAESpec()
    DEFAULT_LORA_TARGET = [
        "proj.q",
        "proj.k",
        "proj.v",
        "proj.o",
        "qkv",
        "text_embedder.blocks.0.proj",
        "text_embedder.blocks.1.proj",
    ]
    DEFAULT_LYCORIS_TARGET = ["MOSSImageTransformerBlock", "MOSSImageTextBlock"]
    PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: MOSSImagePipeline}
    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen3.5-9B",
            "tokenizer": Qwen2Tokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": Qwen3_5Model,
            "subfolder": "text_encoder",
        }
    }

    def __init__(self, config, accelerator):
        config.aspect_bucket_alignment = 16
        config.vae_dtype = "fp32"
        config.flow_schedule_auto_shift = False
        config.flow_schedule_shift = 1.0
        if config.tokenizer_max_length is None:
            config.tokenizer_max_length = MAX_TEXT_LENGTH
        super().__init__(config, accelerator)
        self.vae_scale_factor = self.VAE_SCALE_FACTOR

    @classmethod
    def register_config_requirements(cls):
        ConfigRegistry.register_rules(
            "moss_image",
            [
                make_override_rule("aspect_bucket_alignment", 16, "MOSS-Image dimensions must be divisible by 16."),
                make_override_rule("vae_dtype", "fp32", "MOSS-Image uses its VAE in float32."),
                make_override_rule("flow_schedule_auto_shift", False, "MOSS-Image uses a static flow schedule."),
                make_override_rule("flow_schedule_shift", 1.0, "MOSS-Image uses its unshifted pretrained training grid."),
                make_default_rule("tokenizer_max_length", MAX_TEXT_LENGTH, "MOSS-Image supports up to 4096 text tokens."),
            ],
        )

    def check_user_config(self):
        super().check_user_config()
        if self.config.train_text_encoder:
            raise ValueError("MOSS-Image training keeps the pretrained Qwen3.5 text encoder frozen.")
        if not 1 <= self.config.tokenizer_max_length <= MAX_TEXT_LENGTH:
            raise ValueError("MOSS-Image tokenizer_max_length must be between 1 and 4096.")
        if self.config.fuse_qkv_projections:
            raise ValueError("MOSS-Image does not support fuse_qkv_projections.")
        if self.config.moss_image_validation_shift <= 0:
            raise ValueError("MOSS-Image validation shift must be positive.")
        layers = self.config.moss_image_uncond_skip_layers
        if isinstance(layers, str):
            layers = json.loads(layers)
        self.config.moss_image_uncond_skip_layers = list(_validate_skip_layers(layers, 42, "moss_image_uncond_skip_layers"))
        if self.config.validation_guidance_skip_layers is not None:
            _validate_skip_layers(self.config.validation_guidance_skip_layers, 42, "validation_guidance_skip_layers")

    def pretrained_load_args(self, args):
        return apply_musubi_pretrained_defaults(self.config, super().pretrained_load_args(args))

    def setup_training_noise_schedule(self):
        self.noise_schedule = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        return self.config, self.noise_schedule

    def get_json_schema_converter(self):
        if getattr(self.config, "moss_image_auto_json", False) or self.prompt_upsampling_enabled():
            return MossImageJSONSchemaConverter()
        return None

    def create_prompt_upsampler(self):
        return TransformersJSONPromptUpsampler(
            model_id=self.get_prompt_upsampler_model_id(),
            converter=self.get_json_schema_converter(),
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
            text_encoders=self.text_encoders,
        )

    def get_prompt_upsampler_model_id(self):
        return self.config.moss_image_prompt_upsampler_model_id

    def text_embed_cache_metadata_for_filepath(self, *, init_backend, image_path, **kwargs):
        metadata = super().text_embed_cache_metadata_for_filepath(init_backend=init_backend, image_path=image_path, **kwargs)
        if self.prompt_upsampling_enabled():
            metadata.update(image_paths=[str(image_path)], data_backend_ids=[init_backend["id"]])
        return metadata

    def text_embed_cache_metadata_for_sample(self, *, example, **kwargs):
        metadata = super().text_embed_cache_metadata_for_sample(example=example, **kwargs)
        if self.prompt_upsampling_enabled():
            metadata.update(image_paths=[str(example["image_path"])], data_backend_ids=[example["data_backend_id"]])
        return metadata

    def _encode_prompts(self, prompts, is_negative_prompt=False):
        if not self.text_encoders:
            self.load_text_encoder()
        encoder = self.text_encoders[0]
        encoder.to(self.accelerator.device)
        return encode_prompts(
            self.tokenizers[0], encoder, prompts, self.accelerator.device, self.config.tokenizer_max_length
        )

    def _format_text_embedding(self, embedding):
        embeddings, masks = embedding
        return {"prompt_embeds": embeddings, "attention_masks": masks}

    def collate_prompt_embeds(self, entries):
        return collate_embeddings(entries) if entries else {}

    def convert_text_embed_for_pipeline(self, text_embedding):
        embeds, mask = text_embedding["prompt_embeds"], text_embedding["attention_masks"]
        if embeds.ndim == 2:
            embeds, mask = embeds.unsqueeze(0), mask.unsqueeze(0)
        return {"prompt_embeds": embeds, "prompt_attention_mask": mask}

    def convert_negative_text_embed_for_pipeline(self, text_embedding):
        return {"negative_" + key: value for key, value in self.convert_text_embed_for_pipeline(text_embedding).items()}

    def load_vae(self, move_to_device=True):
        super().load_vae(move_to_device=move_to_device)
        self.vae.to(dtype=torch.float32)
        return self.vae

    def post_vae_encode_transform_sample(self, sample):
        return normalize_latents(patchify(sample.latent_dist.mode()), self.vae)

    def pre_latent_decode(self, latents):
        return unpatchify(latents)

    def denormalize_latents_for_preview(self, latents):
        return denormalize_latents(latents, self.vae)

    def _load_pipeline(self, pipeline_type=PipelineTypes.TEXT2IMG, load_base_model=True):
        if pipeline_type not in self.PIPELINE_CLASSES:
            raise ValueError("MOSS-Image supports text-to-image generation with text or JSON layout prompts.")
        if pipeline_type not in self.pipelines:
            self.pipelines[pipeline_type] = MOSSImagePipeline(
                transformer=self.unwrap_model(model=self.model) if self.model is not None else None,
                scheduler=MOSSImageScheduler(shift=self.config.moss_image_validation_shift),
                vae=self.vae,
                text_encoder=self.text_encoders[0] if self.text_encoders else None,
                tokenizer=self.tokenizers[0] if self.tokenizers else None,
            )
        return self.pipelines[pipeline_type]

    def get_latent_generation_pipeline(self):
        return MOSSImagePipeline(
            transformer=self.get_trained_component(unwrap_model=True),
            scheduler=MOSSImageScheduler(shift=self.config.moss_image_validation_shift),
        )

    def unpack_generated_latents(self, latents, *, height, width):
        if tuple(latents.shape[1:]) != (self.LATENT_CHANNEL_COUNT, height // 16, width // 16):
            raise ValueError("MOSS-Image generated latents must be normalized 128-channel spatial grids.")
        return latents

    def update_pipeline_call_kwargs(self, kwargs):
        kwargs = dict(super().update_pipeline_call_kwargs(kwargs))
        kwargs["uncond_skip_layers"] = self.config.moss_image_uncond_skip_layers
        if "guidance_scale_real" in kwargs:
            kwargs["guidance_scale"] = kwargs.pop("guidance_scale_real")
        for source, target in {
            "skip_guidance_layers": "slg_layers",
            "skip_layer_guidance_scale": "slg_scale",
            "skip_layer_guidance_start": "slg_start",
            "skip_layer_guidance_stop": "slg_stop",
        }.items():
            if source in kwargs:
                kwargs[target] = kwargs.pop(source)
        return kwargs

    def custom_model_card_code_example(self, repo_id=None):
        if self.config.lora_type == "lycoris":
            return ""
        from simpletuner.helpers.publishing.metadata import _model_card_base_model

        destination = repo_id or "your-user/your-moss-lora"
        base = _model_card_base_model(self.config)
        model_id = base if "lora" in self.config.model_type else destination
        adapter_options = ""
        component = self.get_trained_component(unwrap_model=True)
        if component is not None and component.flowmap_deltatime_type is not None:
            adapter_options = (
                f", flowmap_deltatime_type={json.dumps(component.flowmap_deltatime_type)}"
                f", flowmap_gate_value={float(component.flowmap_delta_emb_gate.detach().cpu().item())}"
            )
        load_adapter = (
            f"\npipe.load_lora_weights({json.dumps(destination)}{adapter_options})"
            if "lora" in self.config.model_type
            else ""
        )
        prompt = json.dumps(self.config.validation_prompt or "A detailed studio photograph of a blue ceramic teapot.")
        return f"""```python
import torch
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline

pipe = MOSSImagePipeline.from_pretrained({json.dumps(model_id)}, torch_dtype=torch.bfloat16)
{load_adapter}
pipe.to("cuda")
image = pipe({prompt}, height=1024, width=1024, num_inference_steps=50, guidance_scale=3.0).images[0]
image.save("output.png")
```"""

    def get_lora_save_layers(self):
        modules = list(super().get_lora_save_layers() or [])
        if getattr(self.config, "twinflow_enabled", False):
            modules.append("time_sign_embed")
        component = self.get_trained_component(unwrap_model=True)
        if component is not None and component.delta_timestep_embedder is not None:
            modules.append("delta_timestep_embedder")
        return modules or None

    def supports_crepa_self_flow(self):
        return True

    def _prepare_crepa_self_flow_batch(self, batch, state):
        return self._prepare_image_crepa_self_flow_batch(batch, state, patch_size=1)

    def model_predict(self, prepared_batch):
        states = self._new_hidden_state_buffer()
        prediction = self.model(
            hidden_states=prepared_batch["noisy_latents"].to(self.accelerator.device, self.config.weight_dtype),
            timestep=prepared_batch["timesteps"].to(self.accelerator.device),
            encoder_hidden_states=prepared_batch["prompt_embeds"].to(self.accelerator.device, self.config.weight_dtype),
            encoder_attention_mask=prepared_batch.get("encoder_attention_mask"),
            timestep_sign=prepared_batch.get("twinflow_time_sign"),
            hidden_states_buffer=states,
            return_dict=False,
            **self._get_flowmap_r_timestep_forward_kwargs(prepared_batch),
        )[0]
        regularizer = getattr(self, "crepa_regularizer", None)
        capture = prepared_batch.get("crepa_capture_block_index", getattr(regularizer, "block_index", None))
        return {
            "model_prediction": prediction,
            "hidden_states_buffer": states,
            "crepa_hidden_states": (
                states.get(f"layer_{int(capture)}") if states is not None and capture is not None else None
            ),
        }


ModelRegistry.register("moss_image", MOSSImage)
MOSSImage.register_config_requirements()
