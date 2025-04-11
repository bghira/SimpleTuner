import torch, os, logging, einops
from helpers.training.wrappers import gather_dict_of_tensors_shapes, move_dict_of_tensors_to_device
from helpers.models.common import (
    ImageModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from transformers import (
    PreTrainedTokenizer,
    T5EncoderModel,
    AutoTokenizer,
    LlamaForCausalLM,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)

HiDreamImageTransformer2DModel = None
HiDreamImagePipeline = None
try:
    from helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
    from helpers.models.hidream.pipeline import HiDreamImagePipeline
except Exception as e:
    print(f"HiDream not available: {e}")
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)
is_primary_process = True
if os.environ.get("RANK") is not None:
    if int(os.environ.get("RANK")) != 0:
        is_primary_process = False
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if is_primary_process else "ERROR"
)


class HiDream(ImageModelFoundation):
    NAME = "HiDream"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default seems to help more with HiDream.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = HiDreamImageTransformer2DModel
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: HiDreamImagePipeline,
        # PipelineTypes.IMG2IMG: None,
        # PipelineTypes.CONTROLNET: None,
    }
    MODEL_SUBFOLDER = "transformer"
    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "dev"
    HUGGINGFACE_PATHS = {
        "dev": "HiDream-ai/HiDream-I1-Dev",
        "full": "HiDream-ai/HiDream-I1-Full",
        "fast": "HiDream-ai/HiDream-I1-Fast",
    }
    MODEL_LICENSE = "other"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "CLIP-L/14",
            "tokenizer": CLIPTokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": CLIPTextModelWithProjection,
        },
        "text_encoder_2": {
            "name": "CLIP-G/14",
            "tokenizer": CLIPTokenizer,
            "subfolder": "text_encoder_2",
            "tokenizer_subfolder": "tokenizer_2",
            "model": CLIPTextModelWithProjection,
        },
        "text_encoder_3": {
            "name": "T5 XXL v1.1",
            "tokenizer": AutoTokenizer,
            "subfolder": "text_encoder_3",
            "tokenizer_subfolder": "tokenizer_3",
            "model": T5EncoderModel,
        },
        "text_encoder_4": {
            "name": "Llama",
            "tokenizer": AutoTokenizer,
            "subfolder": None,
            "tokenizer_subfolder": None,
            "model": LlamaForCausalLM,
            "path": "meta-llama/Llama-3.1-8B-Instruct",
            # "required_quantisation_level": "int4_weight_only",
        },
    }

    def post_vae_load_setup(self):
        # we have to differently scale VAE inputs due to the patches.
        self.AUTOENCODER_SCALING_FACTOR *= 2

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        self.config:
            text_embedding (torch.Tensor): The embed to adjust.

        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        t5_embeds, llama_embeds, pooled_prompt_embeds = text_embedding

        return {
            "t5_prompt_embeds": t5_embeds.squeeze(0),
            "llama_prompt_embeds": llama_embeds.squeeze(0),
            "pooled_prompt_embeds": pooled_prompt_embeds.squeeze(0),
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "prompt_embeds": [
                text_embedding["t5_embeds"].unsqueeze(0),
                text_embedding["llama_embeds"].unsqueeze(0),
            ],
            "pooled_prompt_embeds": text_embedding["pooled_prompt_embeds"].unsqueeze(0),
        }

    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "negative_prompt_embeds": [
                text_embedding["t5_embeds"].unsqueeze(0),
                text_embedding["llama_embeds"].unsqueeze(0),
            ],
            "negative_pooled_prompt_embeds": text_embedding[
                "pooled_prompt_embeds"
            ].unsqueeze(0),
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode prompts for an HiDream model into a single tensor.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """
        t5_embeds, llama_embeds, pooled_prompt_embeds = self.pipelines[
            PipelineTypes.TEXT2IMG
        ]._encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            prompt_3=prompts,
            prompt_4=prompts,
            device=self.accelerator.device,
            dtype=self.config.base_weight_dtype,
            num_images_per_prompt=1,
            max_sequence_length=self.config.tokenizer_max_length or 128,
        )
        # verbose declarations are simply for clarity.
        return t5_embeds, llama_embeds, pooled_prompt_embeds

    def collate_prompt_embeds(self, text_encoder_output: dict) -> dict:
        return move_dict_of_tensors_to_device({
            "t5_prompt_embeds": torch.stack(
                [e["t5_prompt_embeds"] for e in text_encoder_output], dim=0
            ),
            "llama_prompt_embeds": torch.stack(
                [e["llama_prompt_embeds"] for e in text_encoder_output], dim=0
            ),
            "pooled_prompt_embeds": torch.stack(
                [e["pooled_prompt_embeds"] for e in text_encoder_output], dim=0
            ),
        }, self.accelerator.device)

    def model_predict(self, prepared_batch):
        logger.debug(f"Prompt embeds: {prepared_batch['text_encoder_output']}")
        logger.debug(
            f"Input shapes:"
            f"\n{prepared_batch['noisy_latents'].shape}"
            f"\n{prepared_batch['timesteps'].shape}"
            f"\n{gather_dict_of_tensors_shapes(prepared_batch['text_encoder_output'])}"
            f"\nT5: {prepared_batch['text_encoder_output']['t5_prompt_embeds'].shape if hasattr(prepared_batch['text_encoder_output']['t5_prompt_embeds'], 'shape') else [x.shape for x in prepared_batch['text_encoder_output']['t5_prompt_embeds']]}"
            f"\nLlama: {prepared_batch['text_encoder_output']['llama_prompt_embeds'].shape if hasattr(prepared_batch['text_encoder_output']['llama_prompt_embeds'], 'shape') else [x.shape for x in prepared_batch['text_encoder_output']['llama_prompt_embeds']]}"
            f"\nCLIP L + G: {prepared_batch['text_encoder_output']['pooled_prompt_embeds'].shape}"
        )

        if (
            prepared_batch["noisy_latents"].shape[-2]
            != prepared_batch["noisy_latents"].shape[-1]
        ):
            B, C, H, W = prepared_batch["noisy_latents"].shape
            pH, pW = (
                H // self.model.config.patch_size,
                W // self.model.config.patch_size,
            )

            img_sizes = torch.tensor([pH, pW], dtype=torch.int64).reshape(-1)
            img_ids = torch.zeros(pH, pW, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW)[None, :]
            img_ids = img_ids.reshape(pH * pW, -1)
            img_ids_pad = torch.zeros(self.model.max_seq, 3)
            img_ids_pad[: pH * pW, :] = img_ids

            img_sizes = img_sizes.unsqueeze(0).to(
                prepared_batch["noisy_latents"].device
            )
            img_ids = img_ids_pad.unsqueeze(0).to(
                prepared_batch["noisy_latents"].device
            )
            if self.do_classifier_free_guidance:
                img_sizes = img_sizes.repeat(2 * B, 1)
                img_ids = img_ids.repeat(2 * B, 1, 1)
        else:
            img_sizes = img_ids = None

        latent_model_input = prepared_batch["noisy_latents"]
        if latent_model_input.shape[-2] != latent_model_input.shape[-1]:
            B, C, H, W = latent_model_input.shape
            patch_size = self.model.config.patch_size
            pH, pW = H // patch_size, W // patch_size
            out = torch.zeros(
                (B, C, self.model.max_seq, patch_size * patch_size),
                dtype=latent_model_input.dtype,
                device=latent_model_input.device,
            )
            latent_model_input = einops.rearrange(
                latent_model_input,
                "B C (H p1) (W p2) -> B C (H W) (p1 p2)",
                p1=patch_size,
                p2=patch_size,
            )
            out[:, :, 0 : pH * pW] = latent_model_input
            latent_model_input = out

        return {
            "model_prediction": self.model(
                hidden_states=latent_model_input.to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                timesteps=prepared_batch["timesteps"],
                t5_hidden_states=prepared_batch["text_encoder_output"][
                    "t5_prompt_embeds"
                ],
                llama_hidden_states=prepared_batch["text_encoder_output"][
                    "llama_prompt_embeds"
                ],
                pooled_embeds=prepared_batch["text_encoder_output"][
                    "pooled_prompt_embeds"
                ],
                img_sizes=img_sizes,
                img_ids=img_ids,
                return_dict=False,
            )[0]
            * -1  # the model is trained with inverted velocity :(
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        t5_max_length = 128
        if (
            self.config.tokenizer_max_length is None
            or int(self.config.tokenizer_max_length) > t5_max_length
        ):
            if not self.config.i_know_what_i_am_doing:
                logger.warning(
                    f"Updating T5 XXL tokeniser max length to {t5_max_length} for {self.NAME}."
                )
                self.config.tokenizer_max_length = t5_max_length
            else:
                logger.warning(
                    f"-!- {self.NAME} supports a max length of {t5_max_length} tokens, but you have supplied `--i_know_what_i_am_doing`, so this limit will not be enforced. -!-"
                )
                logger.warning(
                    f"The model will begin to collapse after a short period of time, if the model you are continuing from has not been tuned beyond {t5_max_length} tokens."
                )

        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                "MM-DiT requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64

        self.config.vae_enable_tiling = True
        self.config.vae_enable_slicing = True

    def custom_model_card_schedule_info(self):
        output_args = []
        if self.config.flow_schedule_auto_shift:
            output_args.append("flow_schedule_auto_shift")
        if self.config.flow_schedule_shift is not None:
            output_args.append(f"shift={self.config.flow_schedule_shift}")
        if self.config.flow_use_beta_schedule:
            output_args.append(
                f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}"
            )
            output_args.append(
                f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}"
            )
        if self.config.flow_use_uniform_schedule:
            output_args.append(f"flow_use_uniform_schedule")
        output_str = (
            f" (extra parameters={output_args})"
            if output_args
            else " (no special parameters set)"
        )

        return output_str
