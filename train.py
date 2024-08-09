#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 pseudoterminalX, CaptnSeraph and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import hashlib
import json
import copy
import random
import logging
import math
import os
import sys

# Quiet down, you.
os.environ["ACCELERATE_LOG_LEVEL"] = "WARNING"
from helpers import log_format  # noqa
from helpers.arguments import parse_args
from helpers.caching.memory import reclaim_memory
from helpers.training.validation import Validation, prepare_validation_prompt_list
from helpers.training.state_tracker import StateTracker
from helpers.data_backend.factory import BatchFetcher
from helpers.training.deepspeed import deepspeed_zero_init_disabled_context_manager
from helpers.training.wrappers import unwrap_model
from helpers.data_backend.factory import configure_multi_databackend
from helpers.data_backend.factory import random_dataloader_iterator
from helpers.training.custom_schedule import (
    generate_timestep_weights,
    segmented_timestep_selection,
)
from helpers.training.min_snr_gamma import compute_snr
from helpers.prompts import PromptHandler
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

filelock_logger = get_logger("filelock")
connection_logger = get_logger("urllib3.connectionpool")
training_logger = get_logger("training-loop")

# More important logs.
target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)
training_logger_level = os.environ.get("SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL", "INFO")
training_logger.setLevel(training_logger_level)

# Less important logs.
filelock_logger.setLevel("WARNING")
connection_logger.setLevel("WARNING")
import torch
import diffusers
import accelerate
import transformers
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from helpers.training.model_freeze import freeze_transformer_blocks
from helpers.training.custom_schedule import get_sd3_sigmas
from transformers import PretrainedConfig, CLIPTokenizer
from helpers.sdxl.pipeline import StableDiffusionXLPipeline
from diffusers import StableDiffusion3Pipeline

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
)

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from helpers.training.ema import EMAModel
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from transformers.utils import ContextManagers

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

SCHEDULER_NAME_MAP = {
    "euler": EulerDiscreteScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "unipc": UniPCMultistepScheduler,
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
}


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: str,
    args,
    subfolder: str = "text_encoder",
):
    if args.smoldit:
        from transformers import AutoModelForSeq2SeqLM

        return AutoModelForSeq2SeqLM
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    elif model_class == "UMT5EncoderModel":
        from transformers import UMT5EncoderModel

        return UMT5EncoderModel
    elif model_class == "ChatGLMModel":
        from diffusers.pipelines.kolors.text_encoder import ChatGLMModel

        return ChatGLMModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def get_tokenizers(args):
    tokenizer_1, tokenizer_2, tokenizer_3 = None, None, None
    try:
        if args.smoldit:
            from transformers import AutoTokenizer

            tokenizer_1 = AutoTokenizer.from_pretrained(
                "EleutherAI/pile-t5-base", pad_token="[PAD]"
            )
            return tokenizer_1, tokenizer_2, tokenizer_3

        tokenizer_kwargs = {
            "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
            "subfolder": "tokenizer",
            "revision": args.revision,
        }
        is_t5_model = False
        if args.pixart_sigma:
            from transformers import T5Tokenizer

            tokenizer_cls = T5Tokenizer
            is_t5_model = True
        elif args.kolors:
            from diffusers.pipelines.kolors.tokenizer import ChatGLMTokenizer

            tokenizer_cls = ChatGLMTokenizer
            tokenizer_1 = tokenizer_cls.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=args.revision,
                use_fast=False,
            )
        else:
            tokenizer_1 = CLIPTokenizer.from_pretrained(**tokenizer_kwargs)

        if is_t5_model:
            text_encoder_path = (
                args.pretrained_t5_model_name_or_path
                if args.pretrained_t5_model_name_or_path is not None
                else args.pretrained_model_name_or_path
            )
            logger.info(
                f"Tokenizer path: {text_encoder_path}, custom T5 model path: {args.pretrained_t5_model_name_or_path} revision: {args.revision}"
            )
            try:
                tokenizer_1 = tokenizer_cls.from_pretrained(
                    text_encoder_path,
                    subfolder="tokenizer",
                    revision=args.revision,
                    use_fast=False,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer 1: {e}, attempting no subfolder"
                )
                tokenizer_1 = T5Tokenizer.from_pretrained(
                    text_encoder_path,
                    subfolder=None,
                    revision=args.revision,
                    use_fast=False,
                )
    except Exception as e:
        import traceback

        logger.warning(
            "Primary tokenizer (CLIP-L/14) failed to load. Continuing to test whether we have just the secondary tokenizer.."
            f"\nError: -> {e}"
            f"\nTraceback: {traceback.format_exc()}"
        )
        if args.sd3:
            raise e
    from transformers import T5TokenizerFast

    if not any([args.pixart_sigma, args.kolors]):
        try:
            tokenizer_2_cls = CLIPTokenizer
            if args.flux:
                tokenizer_2_cls = T5TokenizerFast
            tokenizer_2 = tokenizer_2_cls.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer_2",
                revision=args.revision,
                use_fast=False,
            )
            if tokenizer_1 is None:
                logger.info("Seems that we are training an SDXL refiner model.")
                StateTracker.is_sdxl_refiner(True)
                if args.validation_using_datasets is None:
                    logger.warning(
                        "Since we are training the SDXL refiner and --validation_using_datasets was not specified, it is now being enabled."
                    )
                    args.validation_using_datasets = True
        except Exception as e:
            logger.warning(
                f"Could not load secondary tokenizer (OpenCLIP-G/14). Cannot continue: {e}"
            )
        if not tokenizer_1 and not tokenizer_2:
            raise Exception("Failed to load tokenizer")
    else:
        if not tokenizer_1:
            raise Exception("Failed to load tokenizer")

    if args.sd3:
        try:
            tokenizer_3 = T5TokenizerFast.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer_3",
                revision=args.revision,
                use_fast=True,
            )
        except:
            raise ValueError(
                "Could not load tertiary tokenizer (T5-XXL v1.1). Cannot continue."
            )
    return tokenizer_1, tokenizer_2, tokenizer_3


def main():
    StateTracker.set_model_type("sdxl")
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if args.sd3:
        StateTracker.set_model_type("sd3")
    if args.flux:
        StateTracker.set_model_type("flux")
        from helpers.models.flux import (
            prepare_latent_image_ids,
            pack_latents,
            unpack_latents,
            update_flux_schedule_to_fast,
        )
    if args.pixart_sigma:
        StateTracker.set_model_type("pixart_sigma")
    if args.legacy:
        StateTracker.set_model_type("legacy")
    if args.kolors:
        StateTracker.set_model_type("kolors")
    if args.smoldit:
        from diffusers.models.embeddings import get_2d_rotary_pos_embed
        from helpers.models.smoldit import get_resize_crop_region_for_grid

        StateTracker.set_model_type("smoldit")

    StateTracker.set_args(args)
    if not args.preserve_data_backend_cache:
        StateTracker.delete_cache_files(
            preserve_data_backend_cache=args.preserve_data_backend_cache
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    from accelerate import InitProcessGroupKwargs
    from datetime import timedelta

    # Create the custom configuration
    process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=5400)
    )  # 1.5 hours
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[process_group_kwargs],
    )
    StateTracker.set_accelerator(accelerator)
    webhook_handler = None
    if args.webhook_config is not None:
        from helpers.webhooks.handler import WebhookHandler

        webhook_handler = WebhookHandler(
            args.webhook_config,
            accelerator,
            f"{args.tracker_project_name} {args.tracker_run_name}",
        )
        StateTracker.set_webhook_handler(webhook_handler)
        webhook_handler.send(
            message="SimpleTuner has launched. Hold onto your butts!",
            store_response=True,
        )

    # Make one log on every process with the configuration for debugging.
    # logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    if (
        hasattr(accelerator.state, "deepspeed_plugin")
        and accelerator.state.deepspeed_plugin is not None
    ):
        if "lora" in args.model_type:
            logger.error(
                "LoRA can not be trained with DeepSpeed. Please disable DeepSpeed via 'accelerate config' before reattempting."
            )
            sys.exit(1)
        if (
            "gradient_accumulation_steps"
            in accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            args.gradient_accumulation_steps = (
                accelerator.state.deepspeed_plugin.deepspeed_config[
                    "gradient_accumulation_steps"
                ]
            )
            logger.info(
                f"Updated gradient_accumulation_steps to the value provided by DeepSpeed: {args.gradient_accumulation_steps}"
            )

    # If passed along, set the training seed now.
    if args.seed is not None and args.seed != 0:
        set_seed(args.seed, args.seed_for_each_device)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            from helpers.publishing.huggingface import HubManager

            hub_manager = HubManager(config=args)

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        logger.info(
            "Enabling tf32 precision boost for NVIDIA devices due to --allow_tf32."
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.cuda.is_available():
        logger.warning(
            "If using an Ada or Ampere NVIDIA device, --allow_tf32 could add a bit more performance."
        )

    if args.lr_scale:
        logger.info(f"Scaling learning rate ({args.learning_rate}), due to --lr_scale")
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    is_quantized = (
        False
        if (args.base_model_precision == "no_change" or "lora" not in args.model_type)
        else True
    )
    weight_dtype = (
        torch.bfloat16
        if (
            (args.mixed_precision == "bf16" or torch.backends.mps.is_available())
            or (args.base_model_default_dtype == "bf16" and is_quantized)
        )
        else torch.float32
    )
    StateTracker.set_weight_dtype(weight_dtype)

    # Load scheduler, tokenizer and models.
    logger.info("Load tokenizers")
    # SDXL style models use text encoder and tokenizer 1 and 2, while SD3 will use all three.
    # Pixart Sigma just uses one T5 XXL model.
    # If --disable_text_encoder is provided, we will skip loading entirely.
    tokenizer_1, tokenizer_2, tokenizer_3 = get_tokenizers(args)
    text_encoder_1, text_encoder_2, text_encoder_3 = None, None, None
    text_encoders = []
    tokenizers = []
    if args.kolors:
        logger.info("Loading Kolors ChatGLM language model..")
        text_encoder_path = args.pretrained_model_name_or_path
        text_encoder_subfolder = "text_encoder"
    elif args.smoldit:
        text_encoder_path = "EleutherAI/pile-t5-base"
        text_encoder_subfolder = None
    elif args.flux:
        text_encoder_path = args.pretrained_model_name_or_path
        text_encoder_subfolder = "text_encoder"
    elif args.pixart_sigma:
        text_encoder_path = (
            args.pretrained_t5_model_name_or_path
            if args.pretrained_t5_model_name_or_path is not None
            else args.pretrained_model_name_or_path
        )
        # Google's version of the T5 XXL model doesn't have a subfolder :()
        text_encoder_subfolder = "text_encoder"
    else:
        # sdxl and sd3 use the sd 1.5 clip-L/14 as number one.
        # sd2.x uses openclip vit-H/14
        logger.info("Load CLIP text encoder..")
        text_encoder_path = args.pretrained_model_name_or_path
        text_encoder_subfolder = "text_encoder"
    if tokenizer_1 is not None:
        text_encoder_cls_1 = import_model_class_from_model_name_or_path(
            text_encoder_path, args.revision, args, subfolder=text_encoder_subfolder
        )
    if tokenizer_2 is not None:
        text_encoder_cls_2 = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path,
            args.revision,
            args,
            subfolder="text_encoder_2",
        )
    if tokenizer_3 is not None and args.sd3:
        text_encoder_cls_3 = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path,
            args.revision,
            args,
            subfolder="text_encoder_3",
        )

    # Load scheduler and models
    flow_matching = False
    if (args.sd3 and args.flow_matching_loss != "diffusion") or args.flux:
        # Stable Diffusion 3 uses rectified flow.
        flow_matching = True
        from diffusers import FlowMatchEulerDiscreteScheduler

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        noise_scheduler_copy = copy.deepcopy(
            update_flux_schedule_to_fast(
                args=args, noise_scheduler_to_copy=noise_scheduler
            )
        )

    else:
        if args.legacy:
            args.rescale_betas_zero_snr = True
            args.training_scheduler_timestep_spacing = "trailing"

        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
            rescale_betas_zero_snr=args.rescale_betas_zero_snr,
            timestep_spacing=args.training_scheduler_timestep_spacing,
        )
        args.prediction_type = noise_scheduler.config.prediction_type
        if flow_matching and args.flow_matching_loss == "diffusion":
            logger.warning(
                "Since --flow_matching_loss=diffusion, we will be reparameterising the model to v-prediction diffusion objective. This will break things for a while. Perhaps forever.."
            )
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder_variant = args.variant
        if tokenizer_1 is not None and not args.smoldit:
            if args.pixart_sigma:
                logger.info(
                    f"Loading T5-XXL v1.1 text encoder from {text_encoder_path}/{text_encoder_subfolder}.."
                )
            elif args.flux:
                logger.info(
                    f"Loading OpenAI CLIP-L text encoder from {text_encoder_path}/{text_encoder_subfolder}.."
                )
            elif args.kolors:
                logger.info(
                    f"Loading ChatGLM language model from {text_encoder_path}/{text_encoder_subfolder}.."
                )
                text_encoder_variant = "fp16"
            else:
                logger.info(
                    f"Loading CLIP text encoder from {text_encoder_path}/{text_encoder_subfolder}.."
                )
            text_encoder_1 = text_encoder_cls_1.from_pretrained(
                text_encoder_path,
                subfolder=text_encoder_subfolder,
                revision=args.revision,
                variant=text_encoder_variant,
                torch_dtype=weight_dtype,
            )
        elif args.smoldit:
            text_encoder_1 = text_encoder_cls_1.from_pretrained(
                "EleutherAI/pile-t5-base",
                torch_dtype=weight_dtype,
            ).encoder
            text_encoder_1.eval()

        if tokenizer_2 is not None:
            if args.flux:
                logger.info(
                    f"Loading T5 XXL v1.1 text encoder from {args.pretrained_model_name_or_path}/text_encoder_2.."
                )
            else:
                logger.info("Loading LAION OpenCLIP-G/14 text encoder..")
            text_encoder_2 = text_encoder_cls_2.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="text_encoder_2",
                revision=args.revision,
                torch_dtype=weight_dtype,
                variant=args.variant,
            )
        if tokenizer_3 is not None and args.sd3:
            logger.info("Loading T5-XXL v1.1 text encoder..")
            text_encoder_3 = text_encoder_cls_3.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="text_encoder_3",
                torch_dtype=weight_dtype,
                revision=args.revision,
                variant=args.variant,
            )

        logger.info(f"Load VAE: {vae_path}")
        vae_kwargs = {
            "pretrained_model_name_or_path": vae_path,
            "subfolder": "vae",
            "revision": args.revision,
            "force_upcast": False,
            "variant": args.variant,
        }
        try:
            vae = AutoencoderKL.from_pretrained(**vae_kwargs)
        except:
            logger.warning(
                "Couldn't load VAE with default path. Trying without a subfolder.."
            )
            vae_kwargs["subfolder"] = None
            vae = AutoencoderKL.from_pretrained(**vae_kwargs)

    if tokenizer_1 is not None:
        logger.info("Moving text encoder to GPU.")
        text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    if tokenizer_2 is not None:
        logger.info("Moving text encoder 2 to GPU.")
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    if tokenizer_1 is not None or tokenizer_2 is not None:
        tokenizers = [tokenizer_1, tokenizer_2]
        text_encoders = [text_encoder_1, text_encoder_2]

    if tokenizer_3 is not None:
        logger.info("Moving text encoder 3 to GPU.")
        text_encoder_3.to(accelerator.device, dtype=weight_dtype)
        tokenizers = [tokenizer_1, tokenizer_2, tokenizer_3]
        text_encoders = [text_encoder_1, text_encoder_2, text_encoder_3]

    # Freeze vae and text_encoders
    if vae is not None:
        vae.requires_grad_(False)
    if text_encoder_1 is not None:
        text_encoder_1.requires_grad_(False)
    if text_encoder_2 is not None:
        text_encoder_2.requires_grad_(False)
    if text_encoder_3 is not None:
        text_encoder_3.requires_grad_(False)
    if webhook_handler is not None:
        webhook_handler.send(
            message=f"Loading model: `{args.pretrained_model_name_or_path}`..."
        )

    if vae is not None:
        # The VAE is in bfloat16 to avoid NaN losses.
        vae_dtype = torch.bfloat16
        if hasattr(args, "vae_dtype"):
            # Let's use a case-switch for convenience: bf16, fp16, fp32, none/default
            if args.vae_dtype == "bf16":
                vae_dtype = torch.bfloat16
            elif args.vae_dtype == "fp16":
                raise ValueError(
                    "fp16 is not supported for SDXL's VAE. Please use bf16 or fp32."
                )
            elif args.vae_dtype == "fp32":
                vae_dtype = torch.float32
            elif args.vae_dtype == "none" or args.vae_dtype == "default":
                vae_dtype = torch.bfloat16
        logger.info(
            f"Loading VAE onto accelerator, converting from {vae.dtype} to {vae_dtype}"
        )
        vae.to(accelerator.device, dtype=vae_dtype)
        StateTracker.set_vae_dtype(vae_dtype)
    StateTracker.set_vae(vae)

    # Create a DataBackend, so that we can access our dataset.
    prompt_handler = None
    if not args.disable_compel and not any([args.sd3, args.pixart_sigma, args.kolors]):
        # Only CLIP works with prompt weighting.
        prompt_handler = PromptHandler(
            args=args,
            text_encoders=[text_encoder_1, text_encoder_2],
            tokenizers=[tokenizer_1, tokenizer_2],
            accelerator=accelerator,
            model_type=StateTracker.get_model_type(),
        )

    try:
        if webhook_handler is not None:
            webhook_handler.send(
                message="Configuring data backends... (this may take a while!)"
            )
        configure_multi_databackend(
            args,
            accelerator=accelerator,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            prompt_handler=prompt_handler,
        )
    except Exception as e:
        import traceback

        logger.error(f"{e}, traceback: {traceback.format_exc()}")
        if webhook_handler is not None:
            webhook_handler.send(
                message=f"Failed to load data backends: {e}", message_level="critical"
            )

        sys.exit(0)

    if accelerator.is_main_process:
        if args.flux:
            (
                validation_prompts,
                validation_shortnames,
                validation_negative_prompt_embeds,
                validation_negative_pooled_embeds,
                validation_negative_time_ids,
            ) = prepare_validation_prompt_list(
                args=args, embed_cache=StateTracker.get_default_text_embed_cache()
            )
        else:
            (
                validation_prompts,
                validation_shortnames,
                validation_negative_prompt_embeds,
                validation_negative_pooled_embeds,
            ) = prepare_validation_prompt_list(
                args=args, embed_cache=StateTracker.get_default_text_embed_cache()
            )
    else:
        validation_prompts = None
        validation_shortnames = None
        validation_negative_prompt_embeds = None
        validation_negative_pooled_embeds = None
    accelerator.wait_for_everyone()

    if args.model_type == "full" or not args.train_text_encoder:
        # Grab GPU memory used:
        if torch.cuda.is_available():
            memory_before_unload = torch.cuda.memory_allocated() / 1024**3
        elif torch.backends.mps.is_available():
            memory_before_unload = torch.mps.current_allocated_memory() / 1024**3
        else:
            logger.warning(
                "CUDA, ROCm, or Apple MPS not detected here. We cannot report VRAM reductions."
            )
            memory_before_unload = 0
        if accelerator.is_main_process:
            logger.info("Unloading text encoders, as they are not being trained.")

        if text_encoder_1 is not None:
            text_encoder_1 = text_encoder_1.to("cpu")
        if text_encoder_2 is not None:
            text_encoder_2 = text_encoder_2.to("cpu")
        if text_encoder_3 is not None:
            text_encoder_3 = text_encoder_3.to("cpu")
        del text_encoder_1, text_encoder_2, text_encoder_3
        text_encoder_1 = None
        text_encoder_2 = None
        text_encoder_3 = None
        text_encoders = []
        if prompt_handler is not None:
            prompt_handler.text_encoders = []
        for backend_id, backend in StateTracker.get_data_backends().items():
            if "text_embed_cache" in backend:
                backend["text_embed_cache"].text_encoders = None
                backend["text_embed_cache"].pipeline = None
        reclaim_memory()
        if torch.cuda.is_available():
            memory_after_unload = torch.cuda.memory_allocated() / 1024**3
        elif torch.backends.mps.is_available():
            memory_after_unload = torch.mps.current_allocated_memory() / 1024**3
        else:
            memory_after_unload = 0
        memory_saved = memory_after_unload - memory_before_unload
        logger.info(
            f"After nuking text encoders from orbit, we freed {abs(round(memory_saved, 2))} GB of VRAM."
            " The real memories were the friends we trained a model on along the way."
        )

    pretrained_load_args = {
        "revision": args.revision,
        "variant": args.variant,
    }
    unet = None
    transformer = None
    if args.sd3:
        # Stable Diffusion 3 uses a Diffusion transformer.
        logger.info("Loading Stable Diffusion 3 diffusion transformer..")
        try:
            from diffusers import SD3Transformer2DModel
        except Exception as e:
            logger.error(
                f"Can not load SD3 model class. This release requires the latest version of Diffusers: {e}"
            )
        transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            **pretrained_load_args,
        )
    elif args.flux:
        from diffusers.models import FluxTransformer2DModel

        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
            **pretrained_load_args,
        )
    elif args.pixart_sigma:
        from diffusers.models import PixArtTransformer2DModel

        transformer = PixArtTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
            **pretrained_load_args,
        )
    elif args.smoldit:
        logger.info("Loading SmolDiT model..")
        if args.validation_noise_scheduler is None:
            args.validation_noise_scheduler = "ddpm"
        transformer_variant = None
        from helpers.models.smoldit import SmolDiT2DModel, SmolDiTConfigurations

        if args.smoldit_config not in SmolDiTConfigurations:
            raise ValueError(
                f"Invalid SmolDiT size configuration: {args.smoldit_config}"
            )

        transformer = SmolDiT2DModel(**SmolDiTConfigurations[args.smoldit_config])
        if "lora" in args.model_type:
            raise ValueError("SmolDiT does not yet support LoRA training.")
    else:
        logger.info("Loading U-net..")
        unet_variant = args.variant
        if (
            args.kolors
            and args.pretrained_model_name_or_path.lower()
            == "kwai-kolors/kolors-diffusers"
        ):
            unet_variant = "fp16"
        pretrained_load_args["variant"] = unet_variant
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", **pretrained_load_args
        )
    disable_accelerator = os.environ.get("SIMPLETUNER_DISABLE_ACCELERATOR", False)

    model_type_label = "SDXL"
    if StateTracker.is_sdxl_refiner():
        model_type_label = "SDXL Refiner"
    if args.sd3:
        model_type_label = "Stable Diffusion 3"
    if args.pixart_sigma:
        model_type_label = "PixArt Sigma"
    if args.legacy:
        model_type_label = "Stable Diffusion 1.x/2.x"
    if "deepfloyd" in args.model_type:
        model_type_label = "DeepFloyd-IF"
    if args.kolors:
        model_type_label = "Kwai Kolors"

    enable_adamw_bf16 = True if weight_dtype == torch.bfloat16 else False
    base_weight_dtype = weight_dtype
    if not disable_accelerator and is_quantized:
        if args.base_model_default_dtype == "fp32":
            base_weight_dtype = torch.float32
            enable_adamw_bf16 = False
        elif args.base_model_default_dtype == "bf16":
            base_weight_dtype = torch.bfloat16
            enable_adamw_bf16 = True
        if unet is not None and unet.dtype != base_weight_dtype:
            logger.info(
                f"Moving U-net from {unet.dtype} to {base_weight_dtype} precision"
            )
            unet.to("cpu", dtype=base_weight_dtype)
        elif transformer is not None and transformer.dtype != base_weight_dtype:
            logger.info(
                f"Moving transformer from {transformer.dtype} to {base_weight_dtype} precision"
            )
            transformer.to("cpu", dtype=base_weight_dtype)
        else:
            logger.info(f"Keeping some base model weights in {base_weight_dtype}.")

        if "quanto" in args.base_model_precision:
            try:
                from optimum.quanto import QTensor
            except ImportError as e:
                raise ImportError(
                    f"To use Quanto, please install the optimum library: `pip install optimum-quanto`: {e}"
                )
            from helpers.training.quantisation import quantoise

            # we'll quantise pretty much everything but the adapter, if we execute this here.
            quantoise(
                unet=unet,
                transformer=transformer,
                text_encoder_1=text_encoder_1,
                text_encoder_2=text_encoder_2,
                text_encoder_3=text_encoder_3,
                args=args,
            )

    if args.controlnet:
        if any(
            [
                args.pixart_sigma,
                args.sd3,
                args.kolors,
                StateTracker.is_sdxl_refiner(),
                "deepfloyd" in args.model_type,
            ]
        ):
            raise ValueError(
                f"ControlNet is not yet supported with {model_type_label} models. Please disable --controlnet, or switch model types."
            )
        logger.info("Creating the controlnet..")
        if args.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            controlnet = ControlNetModel.from_pretrained(
                args.controlnet_model_name_or_path
            )
        else:
            logger.info("Initializing controlnet weights from unet")
            controlnet = ControlNetModel.from_unet(unet)
    elif "lora" in args.model_type:
        if args.pixart_sigma:
            raise Exception(f"{model_type_label} does not support LoRA model training.")

        logger.info(f"Using LoRA training mode (rank={args.lora_rank})")
        if webhook_handler is not None:
            webhook_handler.send(message="Using LoRA training mode.")
        # now we will add new LoRA weights to the attention layers
        # Set correct lora layers
        if transformer is not None:
            transformer.requires_grad_(False)
        if unet is not None:
            unet.requires_grad_(False)
        lora_initialisation_style = True
        if hasattr(args, "lora_init_method") and args.lora_init_method is not None:
            lora_initialisation_style = args.lora_init_method
        lora_weight_init_type = (
            "gaussian"
            if torch.backends.mps.is_available()
            else lora_initialisation_style
        )
        if args.use_dora:
            logger.warning(
                "DoRA support is experimental and not very thoroughly tested."
            )
            lora_weight_init_type = "gaussian"
        if unet is not None:
            unet_lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                init_lora_weights=lora_weight_init_type,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                use_dora=args.use_dora,
            )
            logger.info("Adding LoRA adapter to the unet model..")
            unet.add_adapter(unet_lora_config)
        if transformer is not None:
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
            if args.flux and args.flux_lora_target == "all":
                target_modules = [
                    "to_k",
                    "to_q",
                    "to_v",
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_out.0",
                    "to_add_out.0",
                ]
            transformer_lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                init_lora_weights=lora_weight_init_type,
                target_modules=target_modules,
                use_dora=args.use_dora,
            )
            transformer.add_adapter(transformer_lora_config)
    if args.controlnet:
        # We freeze the base u-net for controlnet training.
        if unet is not None:
            unet.requires_grad_(False)
        if transformer is not None:
            transformer.requires_grad_(False)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    # Check if we have a valid gradient accumulation steps.
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1"
        )
    # We calculate the number of steps per epoch by dividing the number of images by the effective batch divisor.
    # Gradient accumulation steps mean that we only update the model weights every /n/ steps.
    collected_data_backend_str = list(StateTracker.get_data_backends().keys())
    if args.push_to_hub and accelerator.is_main_process:
        hub_manager.collected_data_backend_str = collected_data_backend_str
        hub_manager.set_validation_prompts(validation_prompts, validation_shortnames)
        logger.debug(f"Collected validation prompts: {validation_prompts}")
    logger.info(f"Collected the following data backends: {collected_data_backend_str}")
    if webhook_handler is not None:
        webhook_handler.send(
            message=f"Collected the following data backends: {collected_data_backend_str}"
        )
    total_num_batches = sum(
        [
            len(backend["metadata_backend"] if "metadata_backend" in backend else [])
            for _, backend in StateTracker.get_data_backends().items()
        ]
    )
    num_update_steps_per_epoch = math.ceil(
        total_num_batches / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None or args.max_train_steps == 0:
        if args.num_train_epochs is None or args.num_train_epochs == 0:
            raise ValueError(
                "You must specify either --max_train_steps or --num_train_epochs with a value > 0"
            )
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        logger.info(
            f"Calculated our maximum training steps at {args.max_train_steps} because we have"
            f" {args.num_train_epochs} epochs and {num_update_steps_per_epoch} steps per epoch."
        )
        overrode_max_train_steps = True
    logger.info(
        f"Loading {args.lr_scheduler} learning rate scheduler with {args.lr_warmup_steps} warmup steps"
    )
    if args.layer_freeze_strategy == "bitfit":
        from helpers.training.model_freeze import apply_bitfit_freezing

        if unet is not None:
            logger.info("Applying BitFit freezing strategy to the U-net.")
            unet = apply_bitfit_freezing(unet, args)
        if transformer is not None:
            logger.warning(
                "Training Diffusion transformer models with BitFit is not yet tested, and unexpected results may occur."
            )
            transformer = apply_bitfit_freezing(transformer, args)

    if args.gradient_checkpointing:
        if unet is not None:
            unet.enable_gradient_checkpointing()
        if transformer is not None and not args.smoldit:
            transformer.enable_gradient_checkpointing()
        if args.controlnet:
            controlnet.enable_gradient_checkpointing()
        if hasattr(args, "train_text_encoder") and args.train_text_encoder:
            text_encoder_1.gradient_checkpointing_enable()
            text_encoder_2.gradient_checkpointing_enable()
            # if text_encoder_3:
            #     text_encoder_3.gradient_checkpointing_enable()

    logger.info(f"Learning rate: {args.learning_rate}")
    extra_optimizer_args = {
        "weight_decay": args.adam_weight_decay,
        "eps": args.adam_epsilon,
    }
    use_deepspeed_optimizer = False
    use_deepspeed_scheduler = False
    if (
        hasattr(accelerator.state, "deepspeed_plugin")
        and accelerator.state.deepspeed_plugin is not None
    ):
        offload_param = accelerator.state.deepspeed_plugin.deepspeed_config[
            "zero_optimization"
        ]["offload_param"]
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"][
            "offload_param"
        ]["pin_memory"] = False
        if offload_param["device"] == "nvme":
            if offload_param["nvme_path"] == "none":
                if args.offload_param_path is None:
                    raise ValueError(
                        f"DeepSpeed is using {offload_param['device']} but nvme_path is not specified."
                    )
                else:
                    accelerator.state.deepspeed_plugin.deepspeed_config[
                        "zero_optimization"
                    ]["offload_param"]["nvme_path"] = args.offload_param_path
            logger.info(
                f"Using DeepSpeed NVMe offload at {accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']['offload_param']['nvme_path']}."
            )

        use_deepspeed_optimizer = True
        if "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config:
            accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"] = {
                "type": "AdamW",
                "params": {
                    "lr": args.learning_rate,
                    "betas": [args.adam_beta1, args.adam_beta2],
                    "eps": args.adam_epsilon,
                    "weight_decay": args.adam_weight_decay,
                },
            }

        use_deepspeed_scheduler = True
        if "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config:
            accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"] = {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": args.learning_rate,
                    "warmup_num_steps": args.lr_warmup_steps,
                },
            }

    # Initialize the optimizer
    if use_deepspeed_optimizer:
        logger.info("Using DeepSpeed optimizer.")
        optimizer_class = accelerate.utils.DummyOptim
        extra_optimizer_args["lr"] = float(args.learning_rate)
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["eps"] = args.adam_epsilon
        extra_optimizer_args["weight_decay"] = args.adam_weight_decay
    elif args.use_prodigy_optimizer:
        logger.info("Using Prodigy optimizer. Experimental.")
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warn(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        extra_optimizer_args["lr"] = float(args.learning_rate)
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["beta3"] = args.prodigy_beta3
        extra_optimizer_args["weight_decay"] = args.prodigy_weight_decay
        extra_optimizer_args["eps"] = args.prodigy_epsilon
        extra_optimizer_args["decouple"] = args.prodigy_decouple
        extra_optimizer_args["use_bias_correction"] = args.prodigy_use_bias_correction
        extra_optimizer_args["safeguard_warmup"] = args.prodigy_safeguard_warmup
        extra_optimizer_args["d_coef"] = args.prodigy_learning_rate
    elif args.adam_bfloat16:
        if is_quantized and not enable_adamw_bf16:
            logger.error(
                f"When --base_model_default_dtype=fp32, AdamWBF16 may not be used. Reverting to AdamW. You may use other optimizers, such as Adafactor."
            )
            optimizer_class = torch.optim.AdamW
        else:
            logger.info("Using bf16 AdamW optimizer with stochastic rounding.")
            from helpers.training import adam_bfloat16

            optimizer_class = adam_bfloat16.AdamWBF16
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["lr"] = float(args.learning_rate)
    elif args.use_8bit_adam:
        logger.info("Using 8bit AdamW optimizer.")
        try:
            import bitsandbytes as bnb  # type: ignore
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_class = bnb.optim.AdamW8bit
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["lr"] = float(args.learning_rate)
    elif hasattr(args, "use_dadapt_optimizer") and args.use_dadapt_optimizer:
        logger.info("Using D-Adaptation optimizer.")
        try:
            from dadaptation import DAdaptAdam
        except ImportError:
            raise ImportError(
                "Please install the dadaptation library to make use of DaDapt optimizer."
                "You can do so by running `pip install dadaptation`"
            )

        optimizer_class = DAdaptAdam
        if (
            hasattr(args, "dadaptation_learning_rate")
            and args.dadaptation_learning_rate is not None
        ):
            logger.debug(
                f"Overriding learning rate {args.learning_rate} with {args.dadaptation_learning_rate} for D-Adaptation optimizer."
            )
            args.learning_rate = args.dadaptation_learning_rate
            extra_optimizer_args["decouple"] = True
            extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
            extra_optimizer_args["lr"] = float(args.learning_rate)

    elif hasattr(args, "use_adafactor_optimizer") and args.use_adafactor_optimizer:
        logger.info("Using Adafactor optimizer.")
        try:
            from transformers.optimization import Adafactor, AdafactorSchedule
        except ImportError:
            raise ImportError(
                "Please install the latest transformers library to make use of Adafactor optimizer."
                "You can do so by running `pip install transformers`, or, `poetry install` from the SimpleTuner directory."
            )

        optimizer_class = Adafactor
        extra_optimizer_args = {}
        if args.adafactor_relative_step:
            extra_optimizer_args["lr"] = None
            extra_optimizer_args["relative_step"] = True
            extra_optimizer_args["scale_parameter"] = False
            extra_optimizer_args["warmup_init"] = True
        else:
            extra_optimizer_args["lr"] = float(args.learning_rate)
            extra_optimizer_args["relative_step"] = False
            extra_optimizer_args["scale_parameter"] = False
            extra_optimizer_args["warmup_init"] = False
    else:
        logger.info("Using AdamW optimizer.")
        optimizer_class = torch.optim.AdamW
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["lr"] = float(args.learning_rate)

    if args.model_type == "full":
        if args.controlnet:
            params_to_optimize = controlnet.parameters()
        elif unet is not None:
            params_to_optimize = list(
                filter(lambda p: p.requires_grad, unet.parameters())
            )
        elif transformer is not None:
            params_to_optimize = list(
                filter(lambda p: p.requires_grad, transformer.parameters())
            )
        if args.train_text_encoder:
            raise ValueError(
                "Full model tuning does not currently support text encoder training."
            )
    elif "lora" in args.model_type:
        if args.controlnet:
            raise ValueError(
                "SimpleTuner does not currently support training a ControlNet LoRA."
            )
        if unet is not None:
            params_to_optimize = list(
                filter(lambda p: p.requires_grad, unet.parameters())
            )
        if transformer is not None:
            params_to_optimize = list(
                filter(lambda p: p.requires_grad, transformer.parameters())
            )
        if args.train_text_encoder:
            if args.sd3 or args.pixart_sigma:
                raise ValueError(
                    f"{model_type_label} does not support finetuning the text encoders, as T5 does not benefit from it."
                )
            else:
                # add the first text encoder's parameters
                params_to_optimize = params_to_optimize + list(
                    filter(lambda p: p.requires_grad, text_encoder_1.parameters())
                )
                # if text_encoder_2 is not None, add its parameters
                if text_encoder_2 is None and not args.flux:
                    # but not flux. it has t5 as enc 2.
                    params_to_optimize = params_to_optimize + list(
                        filter(lambda p: p.requires_grad, text_encoder_2.parameters())
                    )

    if use_deepspeed_optimizer:
        optimizer = optimizer_class(params_to_optimize)
    else:
        logger.info(
            f"Optimizer arguments, weight_decay={args.adam_weight_decay} eps={args.adam_epsilon}, extra_arguments={extra_optimizer_args}"
        )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warn(
                f"Learning rates were provided both for the {'unet' if unet is not None else 'transformer'} and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = float(args.learning_rate)
            if text_encoder_2 is not None:
                params_to_optimize[2]["lr"] = float(args.learning_rate)

        optimizer = optimizer_class(
            params_to_optimize,
            **extra_optimizer_args,
        )
    from helpers.training.custom_schedule import get_lr_scheduler

    logger.info(
        f"Loading {args.lr_scheduler} learning rate scheduler with {args.lr_warmup_steps} warmup steps"
    )
    lr_scheduler = get_lr_scheduler(
        args, optimizer, accelerator, logger, use_deepspeed_scheduler=False
    )
    if hasattr(lr_scheduler, "num_update_steps_per_epoch"):
        lr_scheduler.num_update_steps_per_epoch = num_update_steps_per_epoch
    if hasattr(lr_scheduler, "last_step"):
        lr_scheduler.last_step = global_resume_step

    accelerator.wait_for_everyone()

    # Create EMA for the unet.
    ema_model = None
    if args.use_ema:
        if args.sd3:
            raise ValueError(
                "Using EMA is not currently supported for Stable Diffusion 3 training."
            )
        if "lora" in args.model_type:
            raise ValueError("Using EMA is not currently supported for LoRA training.")
        if accelerator.is_main_process:
            logger.info("Using EMA. Creating EMAModel.")

            ema_model = EMAModel(
                args,
                accelerator,
                unet.parameters() if unet is not None else transformer.parameters(),
                model_cls=(
                    UNet2DConditionModel
                    if unet is not None
                    else (
                        SD3Transformer2DModel
                        if args.sd3
                        else (PixArtTransformer2DModel if args.pixart_sigma else None)
                    )
                ),
                model_config=(
                    unet.config
                    if unet is not None
                    else transformer.config if transformer is not None else None
                ),
                decay=args.ema_decay,
                foreach=not args.ema_foreach_disable,
            )
            logger.info("EMA model creation complete.")
        accelerator.wait_for_everyone()

    from helpers.training.save_hooks import SaveHookManager

    model_hooks = SaveHookManager(
        args=args,
        unet=unet,
        transformer=transformer,
        ema_model=ema_model,
        accelerator=accelerator,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        use_deepspeed_optimizer=use_deepspeed_optimizer,
    )
    accelerator.register_save_state_pre_hook(model_hooks.save_model_hook)
    accelerator.register_load_state_pre_hook(model_hooks.load_model_hook)

    # Prepare everything with our `accelerator`.
    train_dataloaders = []
    for _, backend in StateTracker.get_data_backends().items():
        if "train_dataloader" not in backend:
            continue
        train_dataloaders.append(backend["train_dataloader"])
        break
    if len(train_dataloaders) == 0:
        logger.error("For some reason, no dataloaders were configured.")
        sys.exit(0)

    if not disable_accelerator:
        logger.info("Loading our accelerator...")
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        if webhook_handler is not None:
            webhook_handler.send(message="Moving weights to GPU...")
        primary_model = unet if unet is not None else transformer
        if args.controlnet:
            primary_model = controlnet
        results = accelerator.prepare(
            primary_model, lr_scheduler, optimizer, train_dataloaders[0]
        )
        if args.controlnet:
            controlnet = results[0]
        elif unet is not None:
            unet = results[0]
        elif transformer is not None:
            transformer = results[0]

        if args.unet_attention_slice:
            if torch.backends.mps.is_available():
                logger.warning(
                    "Using attention slicing when training SDXL on MPS can result in NaN errors on the first backward pass. If you run into issues, disable this option and reduce your batch size instead to reduce memory consumption."
                )
            if unet is not None:
                unet.set_attention_slice("auto")
            if transformer is not None:
                transformer.set_attention_slice("auto")
        lr_scheduler = results[1]
        optimizer = results[2]
        # The rest of the entries are dataloaders:
        train_dataloaders = [results[3:]]
        if args.use_ema and ema_model is not None:
            if args.ema_device == "accelerator":
                logger.info("Moving EMA model weights to accelerator...")
            ema_model.to(
                accelerator.device if args.ema_device == "accelerator" else "cpu",
                dtype=weight_dtype,
            )

            if args.ema_device == "cpu" and not args.ema_cpu_only:
                logger.info("Pinning EMA model weights to CPU...")
                try:
                    ema_model.pin_memory()
                except Exception as e:
                    logger.error(f"Failed to pin EMA model to CPU: {e}")

    idx_count = 0
    for _, backend in StateTracker.get_data_backends().items():
        if idx_count == 0 or "train_dataloader" not in backend:
            continue
        train_dataloaders.append(accelerator.prepare(backend["train_dataloader"]))
    idx_count = 0

    if "lora" in args.model_type and args.train_text_encoder:
        logger.info("Preparing text encoders for training.")
        if args.sd3:
            logger.info("NOTE: The third text encoder is not trained for SD3.")
        text_encoder_1, text_encoder_2 = accelerator.prepare(
            text_encoder_1, text_encoder_2
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    total_num_batches = sum(
        [
            len(backend["metadata_backend"] if "metadata_backend" in backend else [])
            for _, backend in StateTracker.get_data_backends().items()
        ]
    )
    num_update_steps_per_epoch = math.ceil(
        total_num_batches / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    if hasattr(lr_scheduler, "num_update_steps_per_epoch"):
        lr_scheduler.num_update_steps_per_epoch = num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    logger.info(
        "After removing any undesired samples and updating cache entries, we have settled on"
        f" {args.num_train_epochs} epochs and {num_update_steps_per_epoch} steps per epoch."
    )

    if not args.keep_vae_loaded and not args.vae_cache_ondemand:
        if torch.cuda.is_available():
            memory_before_unload = torch.cuda.memory_allocated() / 1024**3
        elif torch.backends.mps.is_available():
            memory_before_unload = torch.mps.current_allocated_memory() / 1024**3
        else:
            logger.warning(
                "CUDA, ROCm, or Apple MPS not detected here. We cannot report VRAM reductions."
            )
            memory_before_unload = 0

        vae = vae.to("cpu")
        del vae
        vae = None

        for _, backend in StateTracker.get_data_backends().items():
            if "vaecache" in backend:
                backend["vaecache"].vae = None

        reclaim_memory()
        if torch.cuda.is_available():
            memory_after_unload = torch.cuda.memory_allocated() / 1024**3
        elif torch.backends.mps.is_available():
            memory_after_unload = torch.mps.current_allocated_memory() / 1024**3
        else:
            memory_after_unload = 0
        memory_saved = memory_after_unload - memory_before_unload
        logger.info(
            f"After nuking the VAE from orbit, we freed {abs(round(memory_saved, 2)) * 1024} MB of VRAM."
        )

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    lr = 0.0
    # Global step represents the most recently *completed* optimization step, which means it
    #  takes into account the number of gradient_accumulation_steps. If we use 1 gradient_accumulation_step,
    #  then global_step and step will be the same throughout training. However, if we use
    #  2 gradient_accumulation_steps, then global_step will be twice as large as step, and so on.
    global_step = 0
    global_resume_step = 0
    StateTracker.set_global_step(global_step)

    # First_epoch represents the *currently training epoch*, as opposed to global_step, which represents
    #  the *last completed* optimization step.
    first_epoch = 1
    scheduler_kwargs = {}
    accelerator.wait_for_everyone()
    validation = Validation(
        accelerator=accelerator,
        prompt_handler=prompt_handler,
        unet=unet,
        transformer=transformer,
        args=args,
        validation_prompts=validation_prompts,
        validation_shortnames=validation_shortnames,
        text_encoder_1=text_encoder_1,
        tokenizer=tokenizer_1,
        vae_path=vae_path,
        weight_dtype=weight_dtype,
        embed_cache=StateTracker.get_default_text_embed_cache(),
        validation_negative_pooled_embeds=validation_negative_pooled_embeds,
        validation_negative_prompt_embeds=validation_negative_prompt_embeds,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        text_encoder_3=text_encoder_3,
        tokenizer_3=tokenizer_3,
        ema_model=ema_model,
        vae=vae,
        controlnet=controlnet if args.controlnet else None,
    )
    # validation.run_validations(validation_type="base_model", step=0)
    if not args.train_text_encoder:
        validation.clear_text_encoders()

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            for _, backend in StateTracker.get_data_backends().items():
                if "sampler" in backend:
                    backend["sampler"].load_states(
                        state_path=os.path.join(
                            args.output_dir, path, "training_state.json"
                        ),
                    )
            global_resume_step = global_step = StateTracker.get_global_step()
            StateTracker.set_global_resume_step(global_resume_step)
            logger.debug(
                f"Training state inside checkpoint: {StateTracker.get_training_state()}"
            )
            if hasattr(lr_scheduler, "last_step"):
                lr_scheduler.last_step = global_resume_step
            logger.info(f"Resuming from global_step {global_resume_step}.")

    # Log the current state of each data backend.
    for _, backend in StateTracker.get_data_backends().items():
        if "sampler" in backend:
            backend["sampler"].log_state()

    total_steps_remaining_at_start = args.max_train_steps
    # We store the number of dataset resets that have occurred inside the checkpoint.
    first_epoch = StateTracker.get_epoch()
    if first_epoch > 1 or global_resume_step > 1:
        total_steps_remaining_at_start -= global_resume_step
        logger.debug(
            f"Resuming from epoch {first_epoch}, which leaves us with {total_steps_remaining_at_start}."
        )
    current_epoch = first_epoch
    StateTracker.set_epoch(current_epoch)

    if current_epoch > args.num_train_epochs + 1:
        logger.info(
            f"Reached the end ({current_epoch} epochs) of our training run ({args.num_train_epochs} epochs). This run will do zero steps."
        )

    # if not use_deepspeed_scheduler:
    #     lr_scheduler = get_lr_scheduler(
    #         args, optimizer, accelerator, logger, use_deepspeed_scheduler=False
    #     )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # Copy args into public_args:
        public_args = copy.deepcopy(args)
        # Hash the contents of public_args to reflect a deterministic ID for a single set of params:
        public_args_hash = hashlib.md5(
            json.dumps(vars(public_args), sort_keys=True).encode("utf-8")
        ).hexdigest()
        project_name = args.tracker_project_name or "simpletuner-training"
        tracker_run_name = args.tracker_run_name or "simpletuner-training-run"
        accelerator.init_trackers(
            project_name,
            config=vars(public_args),
            init_kwargs={
                "wandb": {
                    "name": tracker_run_name,
                    "id": f"{public_args_hash}",
                    "resume": "allow",
                    "allow_val_change": True,
                }
            },
        )

    logger.info(
        f"Moving the {'U-net' if unet is not None else 'diffusion transformer'} to GPU in {weight_dtype if not is_quantized else args.base_model_precision} precision."
    )
    if unet is not None:
        if is_quantized:
            unet.to(accelerator.device)
        else:
            unet.to(accelerator.device, dtype=weight_dtype)
    if transformer is not None:
        if is_quantized:
            transformer.to(accelerator.device)
        else:
            transformer.to(accelerator.device, dtype=weight_dtype)
    if args.enable_xformers_memory_efficient_attention and not any(
        [args.sd3, args.pixart_sigma, args.flux, args.smoldit, args.kolors]
    ):
        logger.info("Enabling xformers memory-efficient attention.")
        if is_xformers_available():
            import xformers  # type: ignore # noqa

            if unet is not None:
                unet.enable_xformers_memory_efficient_attention()
            if transformer is not None:
                transformer.enable_xformers_memory_efficient_attention()
            if args.controlnet:
                controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
    elif args.enable_xformers_memory_efficient_attention:
        logger.warning(
            "xformers is not enabled, as it is incompatible with this model type."
        )

    if args.controlnet:
        controlnet.train()
        controlnet.to(device=accelerator.device, dtype=weight_dtype)
        if args.train_text_encoder:
            logger.warning(
                "Unknown results will occur when finetuning the text encoder alongside ControlNet."
            )

    initial_msg = "\n***** Running training *****"
    total_num_batches = sum(
        [
            len(backend["train_dataset"] if "train_dataset" in backend else [])
            for _, backend in StateTracker.get_data_backends().items()
        ]
    )
    initial_msg += f"\n-  Num batches = {total_num_batches}"

    initial_msg += f"\n-  Num Epochs = {args.num_train_epochs}"
    initial_msg += f"\n  - Current Epoch = {first_epoch}"
    initial_msg += f"\n-  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    initial_msg += (
        f"\n  - Instantaneous batch size per device = {args.train_batch_size}"
    )
    initial_msg += (
        f"\n  - Gradient Accumulation steps = {args.gradient_accumulation_steps}"
    )
    initial_msg += f"\n-  Total optimization steps = {args.max_train_steps}"
    if global_step > 1:
        initial_msg += f"\n  - Steps completed: {global_step}"
    initial_msg += f"\n-  Total optimization steps remaining = {max(0, total_steps_remaining_at_start)}"
    logger.info(initial_msg)
    if webhook_handler is not None:
        webhook_handler.send(message=initial_msg)

    # Only show the progress bar once on each machine.
    show_progress_bar = True
    if not accelerator.is_local_main_process:
        show_progress_bar = False
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        disable=not show_progress_bar,
        initial=global_step,
        desc=f"Epoch {first_epoch}/{args.num_train_epochs} Steps",
        ncols=125,
    )
    accelerator.wait_for_everyone()

    # Some values that are required to be initialised later.
    timesteps_buffer = []
    train_loss = 0.0
    grad_norm = None
    step = global_step
    training_luminance_values = []
    current_epoch_step = None
    global bf
    bf, fetch_thread = None, None
    iterator_fn = random_dataloader_iterator

    for epoch in range(first_epoch, args.num_train_epochs + 1):
        if current_epoch > args.num_train_epochs + 1:
            # This might immediately end training, but that's useful for simply exporting the model.
            logger.info(
                f"Training run is complete ({args.num_train_epochs}/{args.num_train_epochs} epochs, {global_step}/{args.max_train_steps} steps)."
            )
            break
        if first_epoch != epoch:
            logger.debug(
                f"Just completed epoch {current_epoch}. Beginning epoch {epoch}. Starting epoch was {first_epoch}. Final epoch will be {args.num_train_epochs}"
            )
            for backend_id, backend in StateTracker.get_data_backends().items():
                backend_config = StateTracker.get_data_backend_config(backend_id)
                if (
                    "crop" in backend_config
                    and backend_config["crop"] is True
                    and "crop_aspect" in backend_config
                    and backend_config["crop_aspect"] is not None
                    and backend_config["crop_aspect"] == "random"
                    and "metadata_backend" in backend
                    and not args.aspect_bucket_disable_rebuild
                ):
                    # when the aspect ratio is random, we need to shuffle the dataset on each epoch.
                    if accelerator.is_main_process:
                        # we only compute the aspect ratio indices on the main process.
                        # we have to set read_only to False since we're generating a new, un-split list.
                        # otherwise, we can't actually save the new cache to disk.
                        backend["metadata_backend"].read_only = False
                        # this will generate+save the new cache to the storage backend.
                        backend["metadata_backend"].compute_aspect_ratio_bucket_indices(
                            ignore_existing_cache=True
                        )
                    accelerator.wait_for_everyone()
                    logger.info(f"Reloading cache for backend {backend_id}")
                    backend["metadata_backend"].reload_cache(set_config=False)
                    logger.info("Waiting for other threads to finish..")
                    accelerator.wait_for_everyone()
                    # we'll have to split the buckets between GPUs again now, so that the VAE cache distributes properly.
                    logger.info("Splitting buckets across GPUs")
                    backend["metadata_backend"].split_buckets_between_processes(
                        gradient_accumulation_steps=args.gradient_accumulation_steps
                    )
                    # we have to rebuild the VAE cache if it exists.
                    if "vaecache" in backend:
                        logger.info("Rebuilding VAE cache..")
                        backend["vaecache"].rebuild_cache()
                    # no need to manually call metadata_backend.save_cache() here.
                elif (
                    "vae_cache_clear_each_epoch" in backend_config
                    and backend_config["vae_cache_clear_each_epoch"]
                    and "vaecache" in backend
                ):
                    # If the user has specified that this should happen,
                    # we will clear the cache and then rebuild it. This is useful for random crops.
                    logger.debug("VAE Cache rebuild is enabled. Rebuilding.")
                    logger.debug(f"Backend config: {backend_config}")
                    backend["vaecache"].rebuild_cache()
        current_epoch = epoch
        StateTracker.set_epoch(epoch)
        if args.lr_scheduler == "cosine_with_restarts":
            scheduler_kwargs["epoch"] = epoch

        if args.controlnet:
            controlnet.train()
            training_models = [controlnet]
        else:
            if unet is not None:
                unet.train()
                training_models = [unet]
            if transformer is not None:
                transformer.train()
                training_models = [transformer]
        if "lora" in args.model_type and args.train_text_encoder:
            text_encoder_1.train()
            text_encoder_2.train()
            training_models.append(text_encoder_1)
            training_models.append(text_encoder_2)

        if current_epoch_step is not None:
            # We are resetting to the next epoch, if it is not none.
            current_epoch_step = 0
        else:
            # If it's None, we need to calculate the current epoch step based on the current global step.
            current_epoch_step = global_step % num_update_steps_per_epoch
        train_backends = {}
        for backend_id, backend in StateTracker.get_data_backends().items():
            if (
                StateTracker.backend_status(backend_id)
                or "train_dataloader" not in backend
            ):
                # Exclude exhausted backends.
                logger.debug(
                    f"Excluding backend: {backend_id}, as it is exhausted? {StateTracker.backend_status(backend_id)} or not found {('train_dataloader' not in backend)}"
                )
                continue
            train_backends[backend_id] = backend["train_dataloader"]
        # Begin dataloader prefetch, if enabled.
        iterator_args = [train_backends]
        if args.dataloader_prefetch:
            iterator_args = []
            if bf is not None:
                bf.stop_fetching()
            bf = BatchFetcher(
                datasets=train_backends, max_size=args.dataloader_prefetch_qlen
            )
            if fetch_thread is not None:
                fetch_thread.join()
            fetch_thread = bf.start_fetching()
            iterator_fn = bf.next_response

        while True:
            step += 1
            batch = iterator_fn(step, *iterator_args)
            training_logger.debug(f"Iterator: {iterator_fn}")
            if args.lr_scheduler == "cosine_with_restarts":
                scheduler_kwargs["step"] = global_step

            if accelerator.is_main_process:
                progress_bar.set_description(
                    f"Epoch {current_epoch}/{args.num_train_epochs}, Steps"
                )

            # If we receive a False from the enumerator, we know we reached the next epoch.
            if batch is False:
                logger.debug(f"Reached the end of epoch {epoch}")
                break

            if batch is None:
                import traceback

                raise ValueError(
                    f"Received a None batch, which is not a good thing. Traceback: {traceback.format_exc()}"
                )

            # Add the current batch of training data's avg luminance to a list.
            if "batch_luminance" in batch:
                training_luminance_values.append(batch["batch_luminance"])

            with accelerator.accumulate(training_models):
                training_logger.debug("Sending latent batch to GPU.")
                latents = batch["latent_batch"].to(
                    accelerator.device, dtype=weight_dtype
                )

                # Sample noise that we'll add to the latents - args.noise_offset might need to be set to 0.1 by default.
                noise = torch.randn_like(latents)
                if not flow_matching:
                    if args.offset_noise:
                        if (
                            args.noise_offset_probability == 1.0
                            or random.random() < args.noise_offset_probability
                        ):
                            noise = noise + args.noise_offset * torch.randn(
                                latents.shape[0],
                                latents.shape[1],
                                1,
                                1,
                                device=latents.device,
                            )

                bsz = latents.shape[0]
                training_logger.debug(f"Working on batch size: {bsz}")
                if flow_matching:
                    # for weighting schemes where we sample timesteps non-uniformly
                    # thanks to @Slickytail who implemented this correctly via #8528
                    if args.weighting_scheme == "logit_normal":
                        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
                        u = torch.normal(
                            mean=args.logit_mean,
                            std=args.logit_std,
                            size=(bsz,),
                            device="cpu",
                        )
                        u = torch.nn.functional.sigmoid(u)
                    elif args.weighting_scheme == "mode":
                        u = torch.rand(size=(bsz,), device="cpu")
                        u = (
                            1
                            - u
                            - args.mode_scale
                            * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
                        )
                    else:
                        u = torch.rand(size=(bsz,), device="cpu")
                    if args.flux_fast_schedule:
                        # We need to train only timesteps [1, 0.75, 0.5, 0.25] based on SD3-Turbo paper
                        quarter_step = int(
                            noise_scheduler_copy.config.num_train_timesteps / 4
                        )
                        indices = ((u * 4).long() + 1) * quarter_step - 1
                        # indices = (u * 4).long() * quarter_step - 1
                    else:
                        indices = (
                            u * noise_scheduler_copy.config.num_train_timesteps
                        ).long()

                    indices = (
                        u * noise_scheduler_copy.config.num_train_timesteps
                    ).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(
                        device=latents.device
                    )
                else:
                    # Sample a random timestep for each image, potentially biased by the timestep weights.
                    # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                    weights = generate_timestep_weights(
                        args, noise_scheduler.config.num_train_timesteps
                    ).to(accelerator.device)
                    # Instead of uniformly sampling the timestep range, we'll split our weights and schedule into bsz number of segments.
                    # This enables more broad sampling and potentially more effective training.
                    if bsz > 1 and not args.disable_segmented_timestep_sampling:
                        timesteps = segmented_timestep_selection(
                            actual_num_timesteps=noise_scheduler.config.num_train_timesteps,
                            bsz=bsz,
                            weights=weights,
                            use_refiner_range=StateTracker.is_sdxl_refiner()
                            and not StateTracker.get_args().sdxl_refiner_uses_full_range,
                        ).to(accelerator.device)
                    else:
                        timesteps = torch.multinomial(
                            weights, bsz, replacement=True
                        ).long()

                # Prepare the data for the scatter plot
                for timestep in timesteps.tolist():
                    timesteps_buffer.append((global_step, timestep))

                if flow_matching:
                    # Add noise according to flow matching.
                    sigmas = get_sd3_sigmas(
                        accelerator,
                        noise_scheduler_copy,
                        timesteps,
                        n_dim=latents.ndim,
                        dtype=latents.dtype,
                    )
                    # print(f'shapes: {sigmas.shape}, {latents.shape}, {noise.shape}')
                    noisy_latents = (
                        1.0 - sigmas
                    ) * latents.float() + sigmas * noise.float()
                    # is equal to:
                    # zt = (1 - texp) * x + texp * z1
                else:
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(
                        latents.float(), noise.float(), timesteps
                    ).to(device=accelerator.device, dtype=weight_dtype)

                encoder_hidden_states = batch["prompt_embeds"].to(
                    dtype=weight_dtype, device=accelerator.device
                )
                training_logger.debug(
                    f"Encoder hidden states: {encoder_hidden_states.shape}"
                )

                add_text_embeds = batch["add_text_embeds"]
                training_logger.debug(
                    f"Pooled embeds: {add_text_embeds.shape if add_text_embeds is not None else None}"
                )
                # Get the target for loss depending on the prediction type
                if flow_matching:
                    # This is the flow-matching target for vanilla SD3.
                    # If flow_matching_loss == "diffusion", we will instead use v_prediction (see below)
                    if args.flow_matching_loss == "diffusers":
                        target = latents
                    elif args.flow_matching_loss == "compatible":
                        target = noise - latents
                elif noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction" or (
                    flow_matching and args.flow_matching_loss == "diffusion"
                ):
                    # When not using flow-matching, train on velocity prediction objective.
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    target = latents
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        "Supported types are 'epsilon', `sample`, and 'v_prediction'."
                    )

                # Predict the noise residual and compute loss
                if args.sd3:
                    # Even if we're using DDPM process, we don't add in extra kwargs, which are SDXL-specific.
                    added_cond_kwargs = None
                elif StateTracker.get_model_type() == "sdxl" or args.kolors:
                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds.to(
                            device=accelerator.device, dtype=weight_dtype
                        ),
                        "time_ids": batch["batch_time_ids"].to(
                            device=accelerator.device, dtype=weight_dtype
                        ),
                    }
                elif args.pixart_sigma or args.smoldit:
                    # pixart requires an input of {"resolution": .., "aspect_ratio": ..}
                    if "batch_time_ids" in batch:
                        added_cond_kwargs = batch["batch_time_ids"]
                    batch["encoder_attention_mask"] = batch[
                        "encoder_attention_mask"
                    ].to(device=accelerator.device, dtype=weight_dtype)

                training_logger.debug("Predicting noise residual.")

                if args.controlnet:
                    training_logger.debug(
                        f"Extra conditioning dtype: {batch['conditioning_pixel_values'].dtype}"
                    )
                if not os.environ.get("SIMPLETUNER_DISABLE_ACCELERATOR", False):
                    if args.controlnet:
                        # ControlNet conditioning.
                        controlnet_image = batch["conditioning_pixel_values"].to(
                            dtype=weight_dtype
                        )
                        training_logger.debug(f"Image shape: {controlnet_image.shape}")
                        down_block_res_samples, mid_block_res_sample = controlnet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            added_cond_kwargs=added_cond_kwargs,
                            controlnet_cond=controlnet_image,
                            return_dict=False,
                        )
                        # Predict the noise residual
                        if unet is not None:
                            model_pred = unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=encoder_hidden_states,
                                added_cond_kwargs=added_cond_kwargs,
                                down_block_additional_residuals=[
                                    sample.to(dtype=weight_dtype)
                                    for sample in down_block_res_samples
                                ],
                                mid_block_additional_residual=mid_block_res_sample.to(
                                    dtype=weight_dtype
                                ),
                                return_dict=False,
                            )[0]
                        if transformer is not None:
                            raise Exception(
                                "ControlNet predictions for transformer models are not yet implemented."
                            )
                    elif args.flux:
                        # handle guidance
                        packed_noisy_latents = pack_latents(
                            noisy_latents,
                            batch_size=latents.shape[0],
                            num_channels_latents=latents.shape[1],
                            height=latents.shape[2],
                            width=latents.shape[3],
                        )
                        if args.flux_guidance_mode == "constant":
                            guidance_scale = float(args.flux_guidance_value)
                        elif args.flux_guidance_mode == "random-range":
                            guidance_scale = random.uniform(
                                args.flux_guidance_min, args.flux_guidance_max
                            )
                        transformer_config = None
                        if hasattr(transformer, "module"):
                            transformer_config = transformer.module.config
                        elif hasattr(transformer, "config"):
                            transformer_config = transformer.config
                        if transformer_config is not None and getattr(
                            transformer_config, "guidance_embeds", False
                        ):
                            guidance = torch.tensor(
                                [guidance_scale], device=accelerator.device
                            )
                            guidance = guidance.expand(latents.shape[0])
                        else:
                            guidance = None
                        img_ids = prepare_latent_image_ids(
                            latents.shape[0],
                            latents.shape[2],
                            latents.shape[3],
                            accelerator.device,
                            weight_dtype,
                        )
                        timesteps = (
                            torch.tensor(timesteps)
                            .expand(noisy_latents.shape[0])
                            .to(device=accelerator.device)
                            / 1000
                        )

                        text_ids = torch.zeros(
                            packed_noisy_latents.shape[0],
                            batch["prompt_embeds"].shape[1],
                            3,
                        ).to(device=accelerator.device, dtype=base_weight_dtype)
                        model_pred = transformer(
                            hidden_states=packed_noisy_latents.to(
                                dtype=base_weight_dtype, device=accelerator.device
                            ),
                            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                            timestep=timesteps,
                            guidance=guidance,
                            pooled_projections=batch["add_text_embeds"].to(
                                device=accelerator.device, dtype=base_weight_dtype
                            ),
                            encoder_hidden_states=batch["prompt_embeds"].to(
                                device=accelerator.device, dtype=base_weight_dtype
                            ),
                            txt_ids=text_ids.to(
                                device=accelerator.device, dtype=base_weight_dtype
                            ),
                            img_ids=img_ids,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                        # print(f'actual prediction shape: {model_pred.shape}')

                    elif args.sd3:
                        # Stable Diffusion 3 uses a MM-DiT model where the VAE-produced
                        #  image embeds are passed in with the TE-produced text embeds.
                        model_pred = transformer(
                            hidden_states=noisy_latents,
                            timestep=timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            pooled_projections=add_text_embeds.to(
                                device=accelerator.device, dtype=weight_dtype
                            ),
                            return_dict=False,
                        )[0]
                    elif args.pixart_sigma:
                        model_pred = transformer(
                            noisy_latents,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=batch["encoder_attention_mask"],
                            timestep=timesteps,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                        model_pred = model_pred.chunk(2, dim=1)[0]
                    elif args.smoldit:
                        first_latent_shape = noisy_latents.shape
                        height = first_latent_shape[1] * 8
                        width = first_latent_shape[2] * 8
                        grid_height = height // 8 // transformer.config.patch_size
                        grid_width = width // 8 // transformer.config.patch_size
                        base_size = 512 // 8 // transformer.config.patch_size
                        grid_crops_coords = get_resize_crop_region_for_grid(
                            (grid_height, grid_width), base_size
                        )
                        inputs = {
                            "hidden_states": noisy_latents,
                            "timestep": timesteps,
                            "encoder_hidden_states": encoder_hidden_states,
                            "encoder_attention_mask": batch["encoder_attention_mask"],
                            "image_rotary_emb": get_2d_rotary_pos_embed(
                                transformer.inner_dim
                                // transformer.config.num_attention_heads,
                                grid_crops_coords,
                                (grid_height, grid_width),
                            ),
                        }
                        model_pred = transformer(**inputs).sample
                    elif unet is not None:
                        if args.legacy:
                            # SD 1.5 or 2.x
                            model_pred = unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states,
                            ).sample
                        else:
                            # SDXL, Kolors, other default unet prediction.
                            model_pred = unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states,
                                added_cond_kwargs=added_cond_kwargs,
                            ).sample
                    else:
                        raise Exception(
                            "Unknown error occurred, no prediction could be made."
                        )
                else:
                    # Dummy model prediction for debugging.
                    model_pred = torch.randn_like(noisy_latents)

                # if we're quantising with quanto, we need to dequantise the result
                if "quanto" in args.base_model_precision:
                    if hasattr(model_pred, "dequantize") and isinstance(
                        model_pred, QTensor
                    ):
                        # print(f"dequantizing the prediction: {model_pred.dtype}")
                        model_pred = model_pred.dequantize()
                        # print(f"new dtype: {model_pred.dtype}")

                if args.flux:
                    # print(f'unpack: {model_pred.shape}')
                    model_pred = unpack_latents(
                        model_pred,
                        height=latents.shape[2] * 8,
                        width=latents.shape[3] * 8,
                        vae_scale_factor=16,
                    )
                if flow_matching:
                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    # Preconditioning of the model outputs.
                    # print(f"preconditioning shape: {model_pred.shape}")
                    original_shape = model_pred.shape
                    if args.flow_matching_loss == "diffusers":
                        # print(f"post-preconditioning shape: {original_shape} * (-{sigmas}) + {noisy_latents.shape}")
                        model_pred = model_pred * (-sigmas) + noisy_latents
                    elif args.flow_matching_loss == "compatible":
                        # we shouldn't mess with the model prediction.
                        pass

                # x-prediction requires that we now subtract the noise residual from the prediction to get the target sample.
                if (
                    hasattr(noise_scheduler, "config")
                    and hasattr(noise_scheduler.config, "prediction_type")
                    and noise_scheduler.config.prediction_type == "sample"
                ):
                    model_pred = model_pred - noise

                if flow_matching:
                    # upstream TODO: weighting sceme needs to be experimented with :)
                    # these weighting schemes use a uniform timestep sampling
                    # and instead post-weight the loss
                    if args.weighting_scheme == "sigma_sqrt":
                        weighting = (sigmas**-2.0).float()
                    elif args.weighting_scheme == "cosmap":
                        bot = 1 - 2 * sigmas + 2 * sigmas**2
                        weighting = 2 / (math.pi * bot)
                    else:
                        weighting = torch.ones_like(sigmas)
                    loss = torch.mean(
                        (
                            weighting.float()
                            * (model_pred.float() - target.float()) ** 2
                        ).reshape(target.shape[0], -1),
                        1,
                    )
                    loss = loss.mean()

                elif args.snr_gamma is None or args.snr_gamma == 0:
                    training_logger.debug("Calculating loss")
                    loss = args.snr_weight * F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    training_logger.debug("Using min-SNR loss")
                    snr = compute_snr(timesteps, noise_scheduler)
                    snr_divisor = snr
                    if noise_scheduler.config.prediction_type == "v_prediction" or (
                        flow_matching and args.flow_matching_loss == "diffusion"
                    ):
                        snr_divisor = snr + 1

                    training_logger.debug(
                        "Calculating MSE loss weights using SNR as divisor"
                    )
                    mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr_divisor
                    )

                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    ).mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                if not os.environ.get("SIMPLETUNER_DISABLE_ACCELERATOR", False):
                    training_logger.debug("Backwards pass.")
                    # Check for NaNs
                    if (
                        torch.isnan(loss).any()
                        or torch.isnan(model_pred).any()
                        or torch.isnan(target).any()
                        or torch.isnan(avg_loss).any()
                    ):
                        raise ValueError(
                            f"NaNs detected. Loss: {loss}, Model prediction: {model_pred}, Target: {target}"
                        )
                    accelerator.backward(loss)

                    if args.gradient_precision == "fp32":
                        # After backward, convert gradients to fp32 for stable accumulation
                        for param in params_to_optimize:
                            if param.grad is not None:
                                param.grad.data = param.grad.data.to(torch.float32)

                    grad_norm = None
                    if (
                        accelerator.sync_gradients
                        and not args.use_adafactor_optimizer
                        and args.max_grad_norm > 0
                    ):
                        # Adafactor shouldn't have gradient clipping applied.
                        grad_norm = accelerator.clip_grad_norm_(
                            params_to_optimize, args.max_grad_norm
                        )
                    training_logger.debug("Stepping components forward.")
                    optimizer.step()
                    lr_scheduler.step(**scheduler_kwargs)
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                try:
                    if args.use_adafactor_optimizer:
                        lr = lr_scheduler.get_lr()[0]
                    else:
                        lr = lr_scheduler.get_last_lr()[0]
                except Exception as e:
                    logger.error(
                        f"Failed to get the last learning rate from the scheduler. Error: {e}"
                    )
                logs = {
                    "train_loss": train_loss,
                    "optimization_loss": loss,
                    "learning_rate": lr,
                    "epoch": epoch,
                }
                if grad_norm is not None:
                    logs["grad_norm"] = grad_norm
                progress_bar.update(1)
                global_step += 1
                current_epoch_step += 1
                StateTracker.set_global_step(global_step)

                ema_decay_value = "None (EMA not in use)"
                if args.use_ema:
                    if ema_model is not None:
                        training_logger.debug("Stepping EMA forward")
                        ema_model.step(
                            parameters=(
                                unet.parameters()
                                if unet is not None
                                else transformer.parameters()
                            ),
                            global_step=global_step,
                        )
                        logs["ema_decay_value"] = ema_model.get_decay()
                    accelerator.wait_for_everyone()

                # Log scatter plot to wandb
                if args.report_to == "wandb" and accelerator.is_main_process:
                    # Prepare the data for the scatter plot
                    data = [
                        [iteration, timestep]
                        for iteration, timestep in timesteps_buffer
                    ]
                    table = wandb.Table(data=data, columns=["global_step", "timestep"])
                    logs["timesteps_scatter"] = wandb.plot.scatter(
                        table,
                        "global_step",
                        "timestep",
                        title="Timestep distribution by step",
                    )

                # Clear buffers
                timesteps_buffer = []

                # Average out the luminance values of each batch, so that we can store that in this step.
                avg_training_data_luminance = sum(training_luminance_values) / len(
                    training_luminance_values
                )
                logs["train_luminance"] = avg_training_data_luminance

                logger.debug(
                    f"Step {global_step} of {args.max_train_steps}: loss {loss.item()}, lr {lr}, epoch {epoch}/{args.num_train_epochs}, ema_decay_value {ema_decay_value}, train_loss {train_loss}"
                )
                accelerator.log(
                    logs,
                    step=global_step,
                )
                if webhook_handler is not None:
                    webhook_pending_msg = f"Step {global_step} of {args.max_train_steps}: loss {round(loss.item(), 4)}, lr {lr}, epoch {epoch}/{args.num_train_epochs}, ema_decay_value {ema_decay_value}, train_loss {round(train_loss, 4)}"

                # Reset some values for the next go.
                training_luminance_values = []
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if webhook_handler is not None:
                        webhook_handler.send(
                            message=f"Checkpoint: `{webhook_pending_msg}`",
                            message_level="info",
                        )
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.debug(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.debug(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                    if accelerator.is_main_process or use_deepspeed_optimizer:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        print("\n")
                        accelerator.save_state(save_path)
                        for _, backend in StateTracker.get_data_backends().items():
                            if "sampler" in backend:
                                logger.debug(f"Backend: {backend}")
                                backend["sampler"].save_state(
                                    state_path=os.path.join(
                                        save_path, "training_state.json"
                                    ),
                                )

                if (
                    args.accelerator_cache_clear_interval is not None
                    and global_step % args.accelerator_cache_clear_interval == 0
                ):
                    reclaim_memory()

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr,
            }
            progress_bar.set_postfix(**logs)
            validation.run_validations(validation_type="intermediary", step=step)
            if (
                args.push_to_hub
                and args.push_checkpoints_to_hub
                and global_step % args.checkpointing_steps == 0
                and step % args.gradient_accumulation_steps == 0
                and global_step > global_resume_step
            ):
                if accelerator.is_main_process:
                    try:
                        hub_manager.upload_latest_checkpoint(
                            validation_images=validation.validation_images,
                            webhook_handler=webhook_handler,
                        )
                    except Exception as e:
                        logger.error(
                            f"Error uploading to hub: {e}, continuing training."
                        )
            accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps or epoch > args.num_train_epochs:
                logger.info(
                    f"Training has completed."
                    f"\n -> global_step = {global_step}, max_train_steps = {args.max_train_steps}, epoch = {epoch}, num_train_epochs = {args.num_train_epochs}",
                )
                break
        if global_step >= args.max_train_steps or epoch > args.num_train_epochs:
            logger.info(
                f"Exiting training loop. Beginning model unwind at epoch {epoch}, step {global_step}"
            )
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        validation_images = validation.run_validations(
            validation_type="final",
            step=global_step,
            force_evaluation=True,
            skip_execution=True,
        ).validation_images
        if unet is not None:
            unet = unwrap_model(accelerator, unet)
        if transformer is not None:
            transformer = unwrap_model(accelerator, transformer)
        if "lora" in args.model_type:
            if transformer is not None:
                transformer_lora_layers = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(transformer)
                )
            elif unet is not None:
                unet_lora_layers = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unet)
                )
            else:
                raise Exception(
                    "Couldn't locate the unet or transformer model for export."
                )

            if args.train_text_encoder:
                text_encoder_1 = accelerator.unwrap_model(text_encoder_1)
                text_encoder_lora_layers = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(text_encoder_1)
                )
                if text_encoder_2 is not None:
                    text_encoder_2 = accelerator.unwrap_model(text_encoder_2)
                    text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(text_encoder_2)
                    )
                    if text_encoder_3 is not None:
                        text_encoder_3 = accelerator.unwrap_model(text_encoder_3)
            else:
                text_encoder_lora_layers = None
                text_encoder_2_lora_layers = None
                # text_encoder_3_lora_layers = None

            if args.flux:
                from diffusers.pipelines import FluxPipeline

                FluxPipeline.save_lora_weights(
                    save_directory=args.output_dir,
                    transformer_lora_layers=transformer_lora_layers,
                    text_encoder_lora_layers=text_encoder_lora_layers,
                )
            elif args.sd3:
                StableDiffusion3Pipeline.save_lora_weights(
                    save_directory=args.output_dir,
                    transformer_lora_layers=transformer_lora_layers,
                    text_encoder_lora_layers=text_encoder_lora_layers,
                    text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                )
            else:
                StableDiffusionXLPipeline.save_lora_weights(
                    save_directory=args.output_dir,
                    unet_lora_layers=unet_lora_layers,
                    text_encoder_lora_layers=text_encoder_lora_layers,
                    text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                )

            del unet
            del transformer
            del text_encoder_lora_layers
            del text_encoder_2_lora_layers
            reclaim_memory()
        elif args.use_ema:
            if unet is not None:
                ema_model.copy_to(unet.parameters())
            if transformer is not None:
                ema_model.copy_to(transformer.parameters())

        if args.model_type == "full":
            # Now we build a full SDXL Pipeline to export the model with.
            if args.sd3:
                pipeline = StableDiffusion3Pipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    text_encoder=text_encoder_1
                    or (
                        text_encoder_cls_1.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="text_encoder",
                            revision=args.revision,
                            variant=args.variant,
                        )
                        if args.save_text_encoder
                        else None
                    ),
                    tokenizer=tokenizer_1,
                    text_encoder_2=text_encoder_2
                    or (
                        text_encoder_cls_2.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="text_encoder_2",
                            revision=args.revision,
                            variant=args.variant,
                        )
                        if args.save_text_encoder
                        else None
                    ),
                    tokenizer_2=tokenizer_2,
                    text_encoder_3=text_encoder_3
                    or (
                        text_encoder_cls_3.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="text_encoder_3",
                            revision=args.revision,
                            variant=args.variant,
                        )
                        if args.save_text_encoder
                        else None
                    ),
                    tokenizer_3=tokenizer_3,
                    vae=vae
                    or (
                        AutoencoderKL.from_pretrained(
                            vae_path,
                            subfolder=(
                                "vae"
                                if args.pretrained_vae_model_name_or_path is None
                                else None
                            ),
                            revision=args.revision,
                            variant=args.variant,
                            force_upcast=False,
                        )
                    ),
                    transformer=transformer,
                )
                if flow_matching and args.flow_matching_loss == "diffusion":
                    # Diffusion-based SD3 is currently fixed to a Euler v-prediction schedule.
                    pipeline.scheduler = SCHEDULER_NAME_MAP["euler"].from_pretrained(
                        args.pretrained_model_name_or_path,
                        revision=args.revision,
                        subfolder="scheduler",
                        prediction_type="v_prediction",
                        timestep_spacing=args.training_scheduler_timestep_spacing,
                        rescale_betas_zero_snr=args.rescale_betas_zero_snr,
                    )
                    logger.debug(
                        f"Setting scheduler to Euler for SD3. Config: {pipeline.scheduler.config}"
                    )
            elif args.legacy:
                from diffusers import StableDiffusionPipeline

                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    text_encoder=text_encoder_1
                    or (
                        text_encoder_cls_1.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="text_encoder",
                            revision=args.revision,
                            variant=args.variant,
                        )
                        if args.save_text_encoder
                        else None
                    ),
                    tokenizer=tokenizer_1,
                    vae=vae
                    or (
                        AutoencoderKL.from_pretrained(
                            vae_path,
                            subfolder=(
                                "vae"
                                if args.pretrained_vae_model_name_or_path is None
                                else None
                            ),
                            revision=args.revision,
                            variant=args.variant,
                            force_upcast=False,
                        )
                    ),
                    unet=unet,
                    torch_dtype=weight_dtype,
                )
            elif args.smoldit:
                from helpers.models.smoldit import SmolDiTPipeline

                pipeline = SmolDiTPipeline(
                    text_encoder=text_encoder_1
                    or (
                        text_encoder_cls_1.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="text_encoder",
                            revision=args.revision,
                            variant=args.variant,
                        )
                        if args.save_text_encoder
                        else None
                    ),
                    tokenizer=tokenizer_1,
                    vae=vae
                    or (
                        AutoencoderKL.from_pretrained(
                            vae_path,
                            subfolder=(
                                "vae"
                                if args.pretrained_vae_model_name_or_path is None
                                else None
                            ),
                            revision=args.revision,
                            variant=args.variant,
                            force_upcast=False,
                        )
                    ),
                    transformer=transformer,
                    scheduler=None,
                )

            else:
                sdxl_pipeline_cls = StableDiffusionXLPipeline
                if args.kolors:
                    from helpers.kolors.pipeline import KolorsPipeline

                    sdxl_pipeline_cls = KolorsPipeline
                pipeline = sdxl_pipeline_cls.from_pretrained(
                    args.pretrained_model_name_or_path,
                    text_encoder=(
                        text_encoder_cls_1.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="text_encoder",
                            revision=args.revision,
                            variant=args.variant,
                        )
                        if args.save_text_encoder
                        else None
                    ),
                    text_encoder_2=(
                        text_encoder_cls_2.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="text_encoder_2",
                            revision=args.revision,
                            variant=args.variant,
                        )
                        if args.save_text_encoder
                        else None
                    ),
                    tokenizer=tokenizer_1,
                    tokenizer_2=tokenizer_2,
                    vae=StateTracker.get_vae()
                    or AutoencoderKL.from_pretrained(
                        vae_path,
                        subfolder=(
                            "vae"
                            if args.pretrained_vae_model_name_or_path is None
                            else None
                        ),
                        revision=args.revision,
                        variant=args.variant,
                        force_upcast=False,
                    ),
                    unet=unet,
                    revision=args.revision,
                    add_watermarker=args.enable_watermark,
                    torch_dtype=weight_dtype,
                )
            if args.validation_noise_scheduler is not None:
                pipeline.scheduler = SCHEDULER_NAME_MAP[
                    args.validation_noise_scheduler
                ].from_pretrained(
                    args.pretrained_model_name_or_path,
                    revision=args.revision,
                    subfolder="scheduler",
                    prediction_type=args.prediction_type,
                    timestep_spacing=args.training_scheduler_timestep_spacing,
                    rescale_betas_zero_snr=args.rescale_betas_zero_snr,
                )
            pipeline.save_pretrained(
                os.path.join(args.output_dir, "pipeline"), safe_serialization=True
            )

        if args.push_to_hub and accelerator.is_main_process:
            hub_manager.upload_model(validation_images, webhook_handler)
    accelerator.end_training()
    # List any running child threads remaining:
    import threading

    logger.debug(f"Remaining threads: {threading.enumerate()}")


if __name__ == "__main__":
    global bf
    bf = None
    try:
        import multiprocessing

        multiprocessing.set_start_method("fork")
    except Exception as e:
        logger.error(
            "Failed to set the multiprocessing start method to 'fork'. Unexpected behaviour such as high memory overhead or poor performance may result."
            f"\nError: {e}"
        )
    try:
        main()
    except KeyboardInterrupt:
        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message="Training has been interrupted by user action (lost terminal, or ctrl+C)."
            )
    except Exception as e:
        import traceback

        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message=f"Training has failed. Please check the logs for more information: {e}"
            )
        print(e)
        print(traceback.format_exc())
    if bf is not None:
        bf.stop_fetching()
