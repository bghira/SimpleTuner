"""FLUX.3 Image Klein training, cached conditioning, and validation integration."""

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, Qwen3VLModel

from simpletuner.helpers.assistant_lora import load_assistant_adapter
from simpletuner.helpers.configuration.registry import ConfigRegistry, make_default_rule, make_override_rule
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.flux2.autoencoder import AutoencoderKLFlux2
from simpletuner.helpers.models.flux3_image import denormalize_latents, image_ids, normalize_latents, patchify, unpatchify
from simpletuner.helpers.models.flux3_image.pipeline import FLOW_SHIFT, Flux3ImagePipeline
from simpletuner.helpers.models.flux3_image.prompt_upsampler import PROMPT_UPSAMPLER_MODEL_ID, Flux3ImagePromptUpsampler
from simpletuner.helpers.models.flux3_image.text import MAX_TEXT_LENGTH, collate_embeddings, encode_prompts
from simpletuner.helpers.models.flux3_image.transformer import Flux3ImageTransformer2DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.state_tracker import StateTracker


class Flux3Image(ImageModelFoundation):
    NAME = "FLUX.3 Image Klein"
    MODEL_DESCRIPTION = "FLUX.3 image generation and reference-image editing"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    DDP_FIND_UNUSED_PARAMETERS = True
    USES_DYNAMIC_SHIFT = False
    AUTOENCODER_CLASS = AutoencoderKLFlux2
    AUTOENCODER_SCALING_FACTOR = 1.0
    LATENT_CHANNEL_COUNT = 128
    VAE_SCALE_FACTOR = 16
    MODEL_CLASS = Flux3ImageTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    VALIDATION_USES_NEGATIVE_PROMPT = True
    VALIDATION_REQUIRES_EMPTY_NEGATIVE_PROMPT = True
    PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: Flux3ImagePipeline, PipelineTypes.IMG2IMG: Flux3ImagePipeline}
    DEFAULT_MODEL_FLAVOUR = "klein-base"
    HUGGINGFACE_PATHS = {
        "klein-base": "black-forest-labs/flux-3-image-klein-base",
        "klein": "black-forest-labs/flux-3-image-klein",
    }
    MODEL_LICENSE = "other"
    PROMPT_UPSAMPLING_CONFIG_KEY = "flux3_prompt_upsample"
    PROMPT_UPSAMPLING_MODEL_ID = PROMPT_UPSAMPLER_MODEL_ID
    ASSISTANT_LORA_FLAVOURS = ("klein",)
    DEFAULT_LORA_TARGET = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]
    DEFAULT_LYCORIS_TARGET = ["Flux3ImageTransformerBlock"]
    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen3-VL-4B",
            "tokenizer": AutoTokenizer,
            "tokenizer_subfolder": "text_encoder",
            "model": Qwen3VLModel,
            "subfolder": "text_encoder",
        }
    }

    def __init__(self, config, accelerator):
        config.aspect_bucket_alignment = 16
        config.vae_dtype = "fp32"
        config.flow_schedule_auto_shift = False
        config.flow_schedule_shift = FLOW_SHIFT
        if config.tokenizer_max_length is None:
            config.tokenizer_max_length = MAX_TEXT_LENGTH
        self.PREDICTION_TYPE = PredictionTypes.FLOW_MAP if config.model_flavour == "klein" else PredictionTypes.FLOW_MATCHING
        super().__init__(config, accelerator)
        self.vae_scale_factor = self.VAE_SCALE_FACTOR

    @classmethod
    def register_config_requirements(cls):
        ConfigRegistry.register_rules(
            "flux3_image",
            [
                make_override_rule("aspect_bucket_alignment", 16, "FLUX.3 images require dimensions divisible by 16."),
                make_override_rule("vae_dtype", "fp32", "FLUX.3 uses a float32 image VAE."),
                make_override_rule("flow_schedule_auto_shift", False, "FLUX.3 uses static flow shifting."),
                make_override_rule(
                    "flow_schedule_shift", FLOW_SHIFT, "FLUX.3 uses its pretrained static flow shift of 6.93."
                ),
                make_default_rule("tokenizer_max_length", MAX_TEXT_LENGTH, "FLUX.3 supports up to 8192 text tokens."),
            ],
        )

    def setup_training_noise_schedule(self):
        self.noise_schedule = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=self.config.flow_schedule_shift, use_dynamic_shifting=False
        )
        return self.config, self.noise_schedule

    def check_user_config(self):
        super().check_user_config()
        if not 1 <= self.config.tokenizer_max_length <= MAX_TEXT_LENGTH:
            raise ValueError("FLUX.3 tokenizer_max_length must be between 1 and 8192.")
        if self.config.fuse_qkv_projections:
            raise ValueError("FLUX.3 image training does not support fuse_qkv_projections.")
        if self.config.model_flavour == "klein":
            self.config.validation_guidance = 1.0
            self.config.validation_guidance_real = 1.0

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if not self.text_encoders:
            self.load_text_encoder()
        encoder = self.text_encoders[0]
        encoder.to(self.accelerator.device)
        return encode_prompts(
            self.tokenizers[0], encoder, prompts, self.accelerator.device, self.config.tokenizer_max_length
        )

    def create_prompt_upsampler(self):
        return Flux3ImagePromptUpsampler(self.accelerator.device, self.config.weight_dtype, self.text_encoders)

    def _format_text_embedding(self, text_embedding):
        embeddings, masks = text_embedding
        return {"prompt_embeds": embeddings, "attention_masks": masks}

    def collate_prompt_embeds(self, text_encoder_output):
        return collate_embeddings(text_encoder_output) if text_encoder_output else {}

    def convert_text_embed_for_pipeline(self, text_embedding):
        embeddings = text_embedding["prompt_embeds"]
        mask = text_embedding["attention_masks"]
        if embeddings.ndim == 2:
            embeddings, mask = embeddings.unsqueeze(0), mask.unsqueeze(0)
        return {"prompt_embeds": embeddings, "prompt_attention_mask": mask}

    def convert_negative_text_embed_for_pipeline(self, text_embedding):
        return {"negative_" + key: value for key, value in self.convert_text_embed_for_pipeline(text_embedding).items()}

    def load_vae(self, move_to_device=True):
        super().load_vae(move_to_device=move_to_device)
        self.vae.to(dtype=torch.float32)
        return self.vae

    def post_vae_encode_transform_sample(self, sample):
        return normalize_latents(patchify(sample.latent_dist.mode()), self.vae)

    def pre_latent_decode(self, latents):
        return unpatchify(denormalize_latents(latents, self.vae))

    def _load_pipeline(self, pipeline_type=PipelineTypes.TEXT2IMG, load_base_model=True):
        if pipeline_type not in self.PIPELINE_CLASSES:
            raise ValueError("FLUX.3 Image Klein supports text-to-image and reference-image editing pipelines.")
        if pipeline_type not in self.pipelines:
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000, shift=self.config.flow_schedule_shift, use_dynamic_shifting=False
            )
            self.pipelines[pipeline_type] = Flux3ImagePipeline(
                transformer=self.unwrap_model(model=self.model) if self.model is not None else None,
                vae=self.vae,
                text_encoder=self.text_encoders[0] if self.text_encoders else None,
                tokenizer=self.tokenizers[0] if self.tokenizers else None,
                scheduler=scheduler,
            )
        return self.pipelines[pipeline_type]

    def get_latent_generation_pipeline(self):
        return Flux3ImagePipeline(
            transformer=self.get_trained_component(unwrap_model=True),
            scheduler=FlowMatchEulerDiscreteScheduler(shift=self.config.flow_schedule_shift),
        )

    def unpack_generated_latents(self, latents, *, height, width):
        expected = (self.LATENT_CHANNEL_COUNT, height // 16, width // 16)
        if tuple(latents.shape[1:]) != expected:
            raise ValueError(f"Expected generated FLUX.3 image latents with trailing shape {expected}.")
        return latents

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        pipeline_kwargs = dict(pipeline_kwargs)
        if "guidance_scale_real" in pipeline_kwargs:
            pipeline_kwargs["guidance_scale"] = pipeline_kwargs.pop("guidance_scale_real")
        return pipeline_kwargs

    def post_model_load_setup(self):
        super().post_model_load_setup()
        path = getattr(self.config, "assistant_lora_path", None)
        if path and self.supports_assistant_lora(self.config) and self.config.model_type == "lora":
            self.assistant_lora_loaded = load_assistant_adapter(
                transformer=self.unwrap_model(model=self.model),
                pipeline_cls=Flux3ImagePipeline,
                lora_path=path,
                adapter_name=self.assistant_adapter_name,
                weight_name=getattr(self.config, "assistant_lora_weight_name", None),
                low_cpu_mem_usage=getattr(self.config, "low_cpu_mem_usage", False),
            )

    def get_lora_save_layers(self):
        layers = list(super().get_lora_save_layers() or [])
        if getattr(self.config, "twinflow_enabled", False):
            layers.append("time_sign_embed")
        return layers or None

    def supports_crepa_self_flow(self):
        return self.config.model_flavour != "klein"

    def _prepare_crepa_self_flow_batch(self, batch, state):
        return self._prepare_image_crepa_self_flow_batch(batch, state, patch_size=1)

    def requires_conditioning_latents(self):
        return True

    def supports_conditioning_dataset(self):
        return True

    def requires_validation_edit_captions(self):
        if not getattr(self.config, "validation_using_datasets", False):
            return False
        return any(backend.get("conditioning_data") for backend in StateTracker.get_data_backends(_type="image").values())

    def prepare_batch_conditions(self, batch, state):
        references = batch.get("conditioning_latents")
        if references is not None:
            if torch.is_tensor(references):
                references = list(references.unbind(1)) if references.ndim == 5 else [references]
            if len(references) > 10:
                raise ValueError("FLUX.3 Image Klein supports at most ten reference images.")
            batch_size = batch["latents"].shape[0]
            tokens, positions = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
            for index, reference in enumerate(references):
                if len(reference) != batch_size:
                    raise ValueError("Reference image batches must match the training batch size.")
                for sample_index, latent in enumerate(reference):
                    if latent.ndim != 3 or latent.shape[0] != self.LATENT_CHANNEL_COUNT:
                        raise ValueError("FLUX.3 reference caches must contain normalized 128-channel packed latents.")
                    latent = latent.to(self.accelerator.device, self.config.weight_dtype)
                    tokens[sample_index].append(latent.flatten(1).transpose(0, 1))
                    positions[sample_index].append(image_ids(*latent.shape[-2:], latent.device, reference_index=index))
            if references:
                tokens = [torch.cat(sample, dim=0) for sample in tokens]
                positions = [torch.cat(sample, dim=0) for sample in positions]
                batch["conditioning_packed_latents"] = pad_sequence(tokens, batch_first=True)
                batch["conditioning_ids"] = pad_sequence(positions, batch_first=True)
                batch["conditioning_attention_mask"] = pad_sequence(
                    [torch.ones(value.shape[0], device=value.device, dtype=torch.bool) for value in tokens], batch_first=True
                )
        return super().prepare_batch_conditions(batch, state)

    def model_predict(self, prepared_batch):
        latent = prepared_batch["noisy_latents"]
        batch_size, channels, height, width = latent.shape
        dtype = self.config.weight_dtype
        device = self.accelerator.device
        packed = latent.flatten(2).transpose(1, 2).to(device=device, dtype=dtype)
        timestep = prepared_batch["timesteps"].to(device=device, dtype=dtype) / 1000
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        timestep = timestep.reshape(batch_size, -1)
        target_time = None
        if self.unwrap_model(model=self.model).config.flow_map:
            if timestep.shape[1] != 1:
                raise ValueError("Tokenwise self-flow is not supported for the native distilled FLUX.3 checkpoint.")
            target_time = timestep
        states = self._new_hidden_state_buffer()
        prediction = self.model(
            hidden_states=packed,
            encoder_hidden_states=prepared_batch["prompt_embeds"].to(device=device, dtype=dtype),
            encoder_attention_mask=prepared_batch.get("encoder_attention_mask"),
            timestep=timestep,
            target_timestep=target_time,
            timestep_sign=prepared_batch.get("twinflow_time_sign"),
            img_ids=image_ids(height, width, device),
            conditioning_hidden_states=prepared_batch.get("conditioning_packed_latents"),
            conditioning_ids=prepared_batch.get("conditioning_ids"),
            conditioning_attention_mask=prepared_batch.get("conditioning_attention_mask"),
            hidden_states_buffer=states,
            return_dict=False,
        )[0]
        crepa = getattr(self, "crepa_regularizer", None)
        capture = prepared_batch.get("crepa_capture_block_index", getattr(crepa, "block_index", None))
        return {
            "model_prediction": prediction.transpose(1, 2).reshape(batch_size, channels, height, width),
            "hidden_states_buffer": states,
            "crepa_hidden_states": (
                states.get(f"layer_{int(capture)}") if states is not None and capture is not None else None
            ),
        }


ModelRegistry.register("flux3_image", Flux3Image)
Flux3Image.register_config_requirements()
