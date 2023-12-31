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
from helpers import log_format

import shutil, hashlib, json, copy, random, logging, math, os

# Quiet down, you.
os.environ["ACCELERATE_LOG_LEVEL"] = "WARNING"

from pathlib import Path
from helpers.arguments import parse_args
from helpers.legacy.validation import prepare_validation_prompt_list, log_validations
from helpers.training.state_tracker import StateTracker
from helpers.training.deepspeed import deepspeed_zero_init_disabled_context_manager
from helpers.data_backend.factory import configure_multi_databackend
from helpers.data_backend.factory import random_dataloader_iterator
from helpers.caching.sdxl_embeds import TextEmbeddingCache
from helpers.training.custom_schedule import (
    get_polynomial_decay_schedule_with_warmup,
    generate_timestep_weights,
)
from helpers.training.min_snr_gamma import compute_snr
from helpers.prompts import PromptHandler
from accelerate.logging import get_logger

logger = get_logger("SimpleTuner")

filelock_logger = get_logger("filelock")
connection_logger = get_logger("urllib3.connectionpool")
training_logger = get_logger("training-loop")

# More important logs.
target_level = "INFO"
# Is env var set?
if os.environ.get("SIMPLETUNER_LOG_LEVEL"):
    target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL")
logger.setLevel(target_level)
training_logger_level = "WARNING"
if os.environ.get("SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL"):
    training_logger_level = os.environ.get("SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL")
training_logger.setLevel(training_logger_level)

# Less important logs.
filelock_logger.setLevel("WARNING")
connection_logger.setLevel("WARNING")
import accelerate
import datasets
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from transformers.utils import ContextManagers

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0")

logger = get_logger(__name__, log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

VAE_SCALING_FACTOR = 8
DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}
WANDB_TABLE_COL_NAMES = ["step", "image", "text"]
SCHEDULER_NAME_MAP = {
    "euler": EulerDiscreteScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "unipc": UniPCMultistepScheduler,
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
}


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
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
    else:
        raise ValueError(f"{model_class} is not supported.")


def compute_null_conditioning(
    tokenizers, text_encoders, tokenize_captions, accelerator
):
    null_conditioning_list = []
    for a_tokenizer, a_text_encoder in zip(tokenizers, text_encoders):
        null_conditioning_list.append(
            a_text_encoder(
                tokenize_captions([""], tokenizer=a_tokenizer).to(accelerator.device),
                output_hidden_states=True,
            ).hidden_states[-2]
        )
    return torch.concat(null_conditioning_list, dim=-1)


def main():
    args = parse_args()
    StateTracker.set_args(args)
    if not args.preserve_data_backend_cache:
        StateTracker.delete_cache_files()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    StateTracker.set_accelerator(accelerator)
    # Make one log on every process with the configuration for debugging.
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
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
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        logger.info(
            "Enabling tf32 precision boost for NVIDIA devices due to --allow_tf32."
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
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

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        logger.warning(
            f'Using "--fp16" with mixed precision training should be done with a custom VAE. Make sure you understand how this works.'
        )
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        logger.warning(
            f'Using "--bf16" with mixed precision training should be done with a custom VAE. Make sure you understand how this works.'
        )
    StateTracker.set_weight_dtype(weight_dtype)
    # Load scheduler, tokenizer and models.
    tokenizer_1 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )
    text_encoder_cls_1 = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_2 = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    betas_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        prediction_type=args.prediction_type,
        timestep_spacing=args.training_scheduler_timestep_spacing,
        rescale_betas_zero_snr=args.rescale_betas_zero_snr,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        prediction_type=args.prediction_type,
        timestep_spacing=args.training_scheduler_timestep_spacing,
        trained_betas=betas_scheduler.betas.numpy().tolist(),
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
        text_encoder_1 = text_encoder_cls_1.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
        )
        text_encoder_2 = text_encoder_cls_2.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=args.revision,
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            force_upcast=False,
        )

    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]

    # Freeze vae and text_encoders
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae_dtype = torch.float32
    if hasattr(args, "vae_dtype"):
        logger.info(
            f"Initialising VAE in {args.vae_dtype} precision, you may specify a different value if preferred: bf16, fp16, fp32, default"
        )
        # Let's use a case-switch for convenience: bf16, fp16, fp32, none/default
        if args.vae_dtype == "bf16":
            vae_dtype = torch.bfloat16
        elif args.vae_dtype == "fp16":
            vae_dtype = torch.float16
        elif args.vae_dtype == "fp32":
            vae_dtype = torch.float32
        elif args.vae_dtype == "none" or args.vae_dtype == "default":
            vae_dtype = torch.float32
    if args.pretrained_vae_model_name_or_path is not None:
        logger.debug(f"Initialising VAE with weight dtype {vae_dtype}")
        vae.to(accelerator.device, dtype=vae_dtype)
    else:
        logger.debug(f"Initialising VAE with custom dtype {vae_dtype}")
        vae.to(accelerator.device, dtype=vae_dtype)
    StateTracker.set_vae_dtype(vae_dtype)
    StateTracker.set_vae(vae)
    logger.info(f"Loaded VAE into VRAM.")

    # Create a DataBackend, so that we can access our dataset.
    try:
        configure_multi_databackend(args, accelerator)
    except Exception as e:
        logging.error(f"{e}")
        import sys

        sys.exit(0)

    prompt_handler = None
    if not args.disable_compel:
        prompt_handler = PromptHandler(
            args=args,
            text_encoders=[text_encoder_1, text_encoder_2],
            tokenizers=[tokenizer_1, tokenizer_2],
            accelerator=accelerator,
            model_type="sdxl",
        )
    logger.info("Initialise text embedding cache")
    embed_cache = TextEmbeddingCache(
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        accelerator=accelerator,
        prompt_handler=prompt_handler,
        model_type="sdxl",
    )
    StateTracker.set_embedcache(embed_cache)
    if (
        args.caption_dropout_probability is not None
        and args.caption_dropout_probability > 0
    ):
        logger.info("Pre-computing null embedding for caption dropout")
        with accelerator.main_process_first():
            embed_cache.compute_embeddings_for_sdxl_prompts([""], return_concat=False)
        accelerator.wait_for_everyone()
    else:
        logger.warning(
            f"Not using caption dropout will potentially lead to overfitting on captions."
        )

    if "text" not in args.skip_file_discovery:
        logger.info(f"Pre-computing text embeds / updating cache.")
        # Captions are extracted from datasets during `configure_multi_databackend(...)`
        all_captions = StateTracker.get_caption_files()
        if accelerator.is_main_process:
            embed_cache.compute_embeddings_for_sdxl_prompts(
                all_captions, return_concat=False
            )
        accelerator.wait_for_everyone()
        logger.info(f"Discovered {len(all_captions)} captions.")

    with accelerator.main_process_first():
        (
            validation_prompts,
            validation_shortnames,
            validation_negative_prompt_embeds,
            validation_negative_pooled_embeds,
        ) = prepare_validation_prompt_list(args=args, embed_cache=embed_cache)
    accelerator.wait_for_everyone()
    # Grab GPU memory used:
    if accelerator.is_main_process:
        logger.info(
            f"Moving text encoders back to CPU, to save VRAM. Currently, we cannot completely unload the text encoder."
        )
    if args.fully_unload_text_encoder:
        logger.warning(
            "Fully unloading the text encoder means we do not have functioning validations later (yet)."
        )
        import gc

        del text_encoder_1, text_encoder_2
        text_encoder_1 = None
        text_encoder_2 = None
        gc.collect()
        torch.cuda.empty_cache()
    else:
        memory_before_unload = torch.cuda.memory_allocated() / 1024**3
        text_encoder_1.to("cpu")
        text_encoder_2.to("cpu")
        memory_after_unload = torch.cuda.memory_allocated() / 1024**3
        memory_saved = memory_after_unload - memory_before_unload
        logger.info(
            f"After nuking text encoders from orbit, we freed {abs(round(memory_saved, 2))} GB of VRAM."
            " The real memories were the friends we trained a model on along the way."
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    # Check if we have a valid gradient accumulation steps.
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1"
        )
    # We calculate the number of steps per epoch by dividing the number of images by the effective batch divisor.
    # Gradient accumulation steps mean that we only update the model weights every /n/ steps.
    total_num_batches = sum(
        [
            len(backend["bucket_manager"])
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
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    if args.enable_xformers_memory_efficient_attention:
        logger.info("Enabling xformers memory-efficient attention.")
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.20"):
                logger.warn(
                    "SimpleTuner requires at least PyTorch 2.0.1, which in turn requires a new version (0.0.20) of Xformers."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if hasattr(args, "train_text_encoder") and args.train_text_encoder:
            text_encoder_1.gradient_checkpointing_enable()
            text_encoder_2.gradient_checkpointing_enable()

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
        extra_optimizer_args["lr"] = args.learning_rate
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["eps"] = args.adam_epsilon
        extra_optimizer_args["weight_decay"] = args.adam_weight_decay
    elif args.use_8bit_adam:
        logger.info("Using 8bit AdamW optimizer.")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_class = bnb.optim.AdamW8bit
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["lr"] = args.learning_rate
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
            extra_optimizer_args["lr"] = args.learning_rate

    elif hasattr(args, "use_adafactor_optimizer") and args.use_adafactor_optimizer:
        try:
            from transformers.optimization import Adafactor, AdafactorSchedule
        except ImportError:
            raise ImportError(
                "Please install the latest transformers library to make use of Adafactor optimizer."
                "You can do so by running `pip install transformers`, or, `poetry install` from the SimpleTuner directory."
            )

        optimizer_class = Adafactor
        extra_optimizer_args = {}
        extra_optimizer_args["lr"] = None
        extra_optimizer_args["relative_step"] = True
        extra_optimizer_args["scale_parameter"] = False
    else:
        logger.info("Using AdamW optimizer.")
        optimizer_class = torch.optim.AdamW
    if use_deepspeed_optimizer:
        optimizer = optimizer_class(unet.parameters())
    else:
        logger.info(
            f"Optimizer arguments, weight_decay={args.adam_weight_decay} eps={args.adam_epsilon}"
        )
        optimizer = optimizer_class(
            unet.parameters(),
            **extra_optimizer_args,
        )

    if use_deepspeed_scheduler:
        logger.info(f"Using DeepSpeed learning rate scheduler")
        lr_scheduler = accelerate.utils.DummyScheduler(
            optimizer,
            total_num_steps=args.max_train_steps,
            warmup_num_steps=args.lr_warmup_steps,
        )
    elif args.use_adafactor_optimizer:
        # Use the AdafactorScheduler.
        lr_scheduler = AdafactorSchedule(optimizer)
    elif args.lr_scheduler == "cosine_with_restarts":
        from helpers.training.custom_schedule import CosineAnnealingHardRestarts

        lr_scheduler = CosineAnnealingHardRestarts(
            optimizer=optimizer,
            T_0=int(args.lr_warmup_steps * accelerator.num_processes),
            T_mult=int(1),
            eta_min=float(args.lr_end),
            last_step=-1,
            verbose=os.environ.get("SIMPLETUNER_SCHEDULER_VERBOSE", "false").lower()
            == "true",
        )
    elif args.lr_scheduler == "cosine":
        from helpers.training.custom_schedule import Cosine

        lr_scheduler = Cosine(
            optimizer=optimizer,
            T_0=int(args.lr_warmup_steps * accelerator.num_processes),
            T_mult=int(1),
            eta_min=float(args.lr_end),
            last_step=-1,
            verbose=os.environ.get("SIMPLETUNER_SCHEDULER_VERBOSE", "false").lower()
            == "true",
        )
    elif args.lr_scheduler == "polynomial":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            lr_end=args.lr_end,
            power=args.lr_power,
            last_epoch=-1,
        )
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    accelerator.wait_for_everyone()

    # Create EMA for the unet.
    ema_unet = None
    if args.use_ema:
        logger.info("Using EMA. Creating EMAModel.")
        ema_unet = EMAModel(
            unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=unet.config,
            decay=args.ema_decay,
        )
        logger.info("EMA model creation complete.")

    from helpers.sdxl.save_hooks import SDXLSaveHook

    model_hooks = SDXLSaveHook(
        args=args,
        ema_unet=ema_unet,
        accelerator=accelerator,
    )
    accelerator.register_save_state_pre_hook(model_hooks.save_model_hook)
    accelerator.register_load_state_pre_hook(model_hooks.load_model_hook)

    # Prepare everything with our `accelerator`.
    disable_accelerator = os.environ.get("SIMPLETUNER_DISABLE_ACCELERATOR", False)
    train_dataloaders = []
    for _, backend in StateTracker.get_data_backends().items():
        train_dataloaders.append(backend["train_dataloader"])
        break

    if not disable_accelerator:
        logger.info(f"Loading our accelerator...")
        results = accelerator.prepare(
            unet, lr_scheduler, optimizer, train_dataloaders[0]
        )
        logger.debug(f"Accelerate prepare resules: {results}")
        unet = results[0]
        lr_scheduler = results[1]
        optimizer = results[2]
        # The rest of the entries are dataloaders:
        train_dataloaders = [results[3:]]
        if args.use_ema:
            logger.info("Moving EMA model weights to accelerator...")
            ema_unet.to(accelerator.device, dtype=weight_dtype)

    idx_count = 0
    for _, backend in StateTracker.get_data_backends().items():
        if idx_count == 0:
            continue
        train_dataloaders.append(accelerator.prepare(backend["train_dataloader"]))
    idx_count = 0

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    total_num_batches = sum(
        [
            len(backend["bucket_manager"])
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

    if not args.keep_vae_loaded:
        memory_before_unload = torch.cuda.memory_allocated() / 1024**3
        import gc

        del vae
        vae = None
        for _, backend in StateTracker.get_data_backends().items():
            backend["vaecache"].vae = None
        gc.collect()
        torch.cuda.empty_cache()
        memory_after_unload = torch.cuda.memory_allocated() / 1024**3
        memory_saved = memory_after_unload - memory_before_unload
        logger.info(
            f"After the VAE from orbit, we freed {abs(round(memory_saved, 2)) * 1024} MB of VRAM."
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
    resume_global_step = 0
    StateTracker.set_global_step(global_step)
    # First_epoch represents the *currently training epoch*, as opposed to global_step, which represents
    #  the *last completed* optimization step.
    first_epoch = 1
    scheduler_kwargs = {}
    accelerator.wait_for_everyone()

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
                backend["sampler"].load_states(
                    state_path=os.path.join(
                        args.output_dir, path, "training_state.json"
                    ),
                )
            resume_global_step = global_step = StateTracker.get_global_step()
            logger.debug(
                f"Training state inside checkpoint: {StateTracker.get_training_state()}"
            )

            # If we use a constant LR, we can update that now.
            if args.lr_scheduler == "constant":
                lr_scheduler = get_scheduler(
                    "constant",
                    optimizer=optimizer,
                    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                )
            if hasattr(lr_scheduler, "last_step"):
                lr_scheduler.last_step = resume_global_step
            logger.info(f"Resuming from global_step {resume_global_step}.")

    # Log the current state of each data backend.
    for _, backend in StateTracker.get_data_backends().items():
        backend["sampler"].log_state()

    total_steps_remaining_at_start = args.max_train_steps
    # We store the number of dataset resets that have occurred inside the checkpoint.
    first_epoch = StateTracker.get_epoch()
    if first_epoch > 1 or resume_global_step > 1:
        total_steps_remaining_at_start -= resume_global_step
        logger.debug(
            f"Resuming from epoch {first_epoch}, which leaves us with {total_steps_remaining_at_start}."
        )
    current_epoch = first_epoch
    if current_epoch >= args.num_train_epochs:
        logger.info(
            f"Reached the end ({current_epoch} epochs) of our training run ({args.num_train_epochs} epochs). This run will do zero steps."
        )
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
    logger.info("***** Running training *****")
    total_num_batches = sum(
        [
            len(backend["train_dataset"])
            for _, backend in StateTracker.get_data_backends().items()
        ]
    )
    logger.info(
        f" -> Num batches = {total_num_batches} ({total_num_batches * args.train_batch_size} total samples)"
    )
    logger.info(f" -> Num Epochs = {args.num_train_epochs}")
    logger.info(f" -> Current Epoch = {first_epoch}")
    logger.info(f" -> Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f" -> Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"   -> Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f" -> Total optimization steps = {args.max_train_steps}")
    if global_step > 1:
        logger.info(f" -> Steps completed: {global_step}")
    logger.info(
        f" -> Total optimization steps remaining = {total_steps_remaining_at_start}"
    )

    # Only show the progress bar once on each machine.
    show_progress_bar = True
    if not accelerator.is_local_main_process:
        show_progress_bar = False
    if (
        training_logger_level == "DEBUG"
        or os.environ.get("SIMPLETUNER_LOG_LEVEL") == "DEBUG"
    ):
        show_progress_bar = False
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        disable=not show_progress_bar,
        initial=global_step,
        desc="Steps",
    )
    accelerator.wait_for_everyone()

    # Some values that are required to be initialised later.
    timesteps_buffer = []
    train_loss = 0.0
    step = global_step
    training_luminance_values = []
    current_epoch_step = None

    for epoch in range(first_epoch, args.num_train_epochs):
        if current_epoch >= args.num_train_epochs:
            # This might immediately end training, but that's useful for simply exporting the model.
            logger.info(
                f"Reached the end ({current_epoch} epochs) of our training run ({args.num_train_epochs} epochs)."
            )
            break
        if first_epoch != epoch:
            logger.debug(
                f"Just completed epoch {current_epoch}. Beginning epoch {epoch}. Final epoch will be {args.num_train_epochs}"
            )
        current_epoch = epoch
        unet.train()
        if current_epoch_step is not None:
            # We are resetting to the next epoch, if it is not none.
            current_epoch_step = 0
        else:
            # If it's None, we need to calculate the current epoch step based on the current global step.
            current_epoch_step = global_step % num_update_steps_per_epoch
        train_backends = {}
        for backend_id, backend in StateTracker.get_data_backends().items():
            if StateTracker.backend_status(backend_id):
                # Exclude exhausted backends.
                continue
            train_backends[backend_id] = backend["train_dataloader"]

        for step, batch in random_dataloader_iterator(train_backends):
            if args.lr_scheduler == "cosine_with_restarts":
                scheduler_kwargs["step"] = global_step

            if accelerator.is_main_process:
                progress_bar.set_description(
                    f"Epoch {current_epoch}/{args.num_train_epochs}, Steps"
                )

            # If we receive a False from the enumerator, we know we reached the next epoch.
            if batch is False:
                logger.info(f"Reached the end of epoch {epoch}")
                break

            if batch is None:
                import traceback

                raise ValueError(
                    f"Received a None batch, which is not a good thing. Traceback: {traceback.format_exc()}"
                )

            # Add the current batch of training data's avg luminance to a list.
            if "batch_luminance" in batch:
                training_luminance_values.append(batch["batch_luminance"])
            with accelerator.accumulate(unet):
                training_logger.debug(
                    f"Sending latent batch from pinned memory to device."
                )
                latents = batch["latent_batch"].to(
                    accelerator.device, dtype=weight_dtype
                )

                # Sample noise that we'll add to the latents - args.noise_offset might need to be set to 0.1 by default.
                noise = torch.randn_like(latents)
                if args.offset_noise:
                    if (
                        args.noise_offset_probability == 1.0
                        or random.random() < args.noise_offset_probability
                    ):
                        noise = torch.randn_like(
                            latents
                        ) + args.noise_offset * torch.randn(
                            latents.shape[0],
                            latents.shape[1],
                            1,
                            1,
                            device=latents.device,
                        )
                else:
                    noise = torch.randn_like(latents)
                if args.input_perturbation:
                    if (
                        args.input_perturbation_probability == 1.0
                        or random.random() < args.input_perturbation_probability
                    ):
                        noise = noise + args.input_perturbation * torch.randn_like(
                            noise
                        )

                bsz = latents.shape[0]
                training_logger.debug(f"Working on batch size: {bsz}")
                # Sample a random timestep for each image, potentially biased by the timestep weights.
                # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                weights = generate_timestep_weights(
                    args, noise_scheduler.config.num_train_timesteps
                ).to(accelerator.device)
                timesteps = torch.multinomial(weights, bsz, replacement=True).long()

                # Prepare the data for the scatter plot
                for timestep in timesteps.tolist():
                    timesteps_buffer.append((step, timestep))

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(
                    accelerator.device
                )

                # SDXL additional inputs - probabilistic dropout
                encoder_hidden_states = batch["prompt_embeds"]
                training_logger.debug(
                    f"Encoder hidden states: {encoder_hidden_states.shape}"
                )

                add_text_embeds = batch["add_text_embeds"]
                training_logger.debug(f"Added text embeds: {add_text_embeds.shape}")
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
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
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds.to(accelerator.device),
                    "time_ids": batch["batch_time_ids"].to(accelerator.device),
                }
                training_logger.debug("Predicting noise residual.")
                training_logger.debug(
                    f"\n -> Latents device: {latents.device}"
                    f"\n -> Noise device: {noise.device}"
                    f"\n -> Timesteps device: {timesteps.device}"
                    f"\n -> Encoder hidden states device: {encoder_hidden_states.device}"
                    f"\n -> Added cond kwargs device: {added_cond_kwargs['text_embeds'].device}"
                    f"\n -> Time IDs device: {added_cond_kwargs['time_ids'].device}"
                )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # x-prediction requires that we now subtract the noise residual from the prediction to get the target sample.
                if noise_scheduler.config.prediction_type == "sample":
                    model_pred = model_pred - noise

                if args.snr_gamma is None:
                    training_logger.debug(f"Calculating loss")
                    loss = args.snr_weight * F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    training_logger.debug(f"Using min-SNR loss")
                    snr = compute_snr(timesteps, noise_scheduler)
                    snr_divisor = snr
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        snr_divisor = snr + 1

                    training_logger.debug(
                        f"Calculating MSE loss weights using SNR as divisor"
                    )
                    mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr_divisor
                    )

                    # For zero-terminal SNR, we have to handle the case where a sigma of Zero results in a Inf value.
                    # When we run this, the MSE loss weights for this timestep is set unconditionally to 1.
                    # If we do not run this, the loss value will go to NaN almost immediately, usually within one step.
                    mse_loss_weights[snr == 0] = 1.0

                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    training_logger.debug(
                        f"Calculating original MSE loss without reduction"
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    training_logger.debug(f"Calculating SNR-weighted MSE loss")
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    training_logger.debug(f"Reducing loss via mean")
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                training_logger.debug(f"Backwards pass.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                training_logger.debug(f"Stepping components forward.")
                optimizer.step()
                lr_scheduler.step(**scheduler_kwargs)
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                try:
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
                progress_bar.update(1)
                global_step += 1
                current_epoch_step += 1
                StateTracker.set_global_step(global_step)

                ema_decay_value = "None (EMA not in use)"
                if args.use_ema:
                    training_logger.debug(f"Stepping EMA unet forward")
                    ema_unet.step(unet.parameters())
                    # There seems to be an issue with EMAmodel not keeping proper track of itself.
                    ema_unet.optimization_step = global_step
                    ema_decay_value = ema_unet.get_decay(ema_unet.optimization_step)
                    logs["ema_decay_value"] = ema_decay_value
                    training_logger.debug(f"EMA decay value: {ema_decay_value}")
                logger.debug(
                    f"Step {global_step} of {args.max_train_steps}: loss {loss.item()}, lr {lr}, epoch {epoch}/{args.num_train_epochs}, ema_decay_value {ema_decay_value}, train_loss {train_loss}"
                )

                # Log scatter plot to wandb
                if args.report_to == "wandb":
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
                accelerator.log(
                    logs,
                    step=global_step,
                )
                # Reset some values for the next go.
                training_luminance_values = []
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
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

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        for _, backend in StateTracker.get_data_backends().items():
                            logger.debug(f"Backend: {backend}")
                            backend["sampler"].save_state(
                                state_path=os.path.join(
                                    save_path, "training_state.json"
                                ),
                            )
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr,
            }
            progress_bar.set_postfix(**logs)
            log_validations(
                accelerator,
                prompt_handler,
                unet,
                args,
                validation_prompts,
                validation_shortnames,
                global_step,
                resume_global_step,
                step,
                text_encoder_1,
                tokenizer=None,
                vae_path=vae_path,
                weight_dtype=weight_dtype,
                embed_cache=embed_cache,
                validation_negative_pooled_embeds=validation_negative_pooled_embeds,
                validation_negative_prompt_embeds=validation_negative_prompt_embeds,
                text_encoder_2=text_encoder_2,
                tokenizer_2=None,
                ema_unet=ema_unet,
                vae=vae,
                SCHEDULER_NAME_MAP=SCHEDULER_NAME_MAP,
            )
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
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        if StateTracker.get_vae() is None:
            StateTracker.set_vae(
                AutoencoderKL.from_pretrained(
                    vae_path,
                    subfolder="vae"
                    if args.pretrained_vae_model_name_or_path is None
                    else None,
                    revision=args.revision,
                    force_upcast=False,
                )
            )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            vae=StateTracker.get_vae(),
            unet=unet,
            revision=args.revision,
            add_watermarker=args.enable_watermark,
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline.scheduler = SCHEDULER_NAME_MAP[
            args.validation_noise_scheduler
        ].from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
            prediction_type=args.prediction_type,
            timestep_spacing=args.training_scheduler_timestep_spacing,
            rescale_betas_zero_snr=args.rescale_betas_zero_snr,
        )
        pipeline.save_pretrained(
            os.path.join(args.output_dir, "pipeline"), safe_serialization=True
        )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=os.path.join(args.output_dir, "pipeline"),
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        log_validations(
            accelerator,
            prompt_handler,
            unet,
            args,
            validation_prompts,
            validation_shortnames,
            global_step,
            resume_global_step,
            step,
            text_encoder_1,
            tokenizer=None,
            vae_path=vae_path,
            weight_dtype=weight_dtype,
            embed_cache=embed_cache,
            validation_negative_pooled_embeds=validation_negative_pooled_embeds,
            validation_negative_prompt_embeds=validation_negative_prompt_embeds,
            text_encoder_2=text_encoder_2,
            tokenizer_2=None,
            vae=vae,
            SCHEDULER_NAME_MAP=SCHEDULER_NAME_MAP,
            validation_type="finish",
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
