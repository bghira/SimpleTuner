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
import argparse
import logging
import math
import os

# Quiet down, you.
os.environ["ACCELERATE_LOG_LEVEL"] = "WARNING"
import shutil
import random
import gc
import warnings
from pathlib import Path
from urllib.parse import urlparse
from helpers.aspect_bucket import BalancedBucketSampler
from helpers.multiaspect_dataset import MultiAspectDataset
from helpers.multiaspect.bucket import BucketManager
from helpers.multiaspect.sampler import MultiAspectSampler
from helpers.state_tracker import StateTracker
from helpers.sdxl_embeds import TextEmbeddingCache
from helpers.image_tools import calculate_luminance, calculate_batch_luminance
from helpers.vae_cache import VAECache
from helpers.arguments import parse_args
from helpers.custom_schedule import get_polynomial_decay_schedule_with_warmup
from helpers.min_snr_gamma import compute_snr
from helpers.prompts import PromptHandler

logger = logging.getLogger()
filelock_logger = logging.getLogger("filelock")
connection_logger = logging.getLogger("urllib3.connectionpool")
training_logger = logging.getLogger("training-loop")

# More important logs.
target_level = 'INFO'
# Is env var set?
if os.environ.get('SIMPLETUNER_LOG_LEVEL'):
    target_level = os.environ.get('SIMPLETUNER_LOG_LEVEL')
logger.setLevel(target_level)
training_logger_level = 'WARNING'
if os.environ.get('SIMPLETUNER_LOG_LEVEL'):
    training_logger_level = os.environ.get('SIMPLETUNER_LOG_LEVEL')
training_logger.setLevel(training_logger_level)

# Less important logs.
filelock_logger.setLevel("WARNING")
connection_logger.setLevel("WARNING")

logger.info("Import accelerate")
import accelerate

logger.info("Import datasets")
import datasets

logger.info("Import numpy")
import numpy as np
import PIL

logger.info("Import pytorch")
import torch

logger.info("Import torch.nn")
import torch.nn as nn

logger.info("Import torch.nn.functional")
import torch.nn.functional as F

logger.info("Import torch.utils.checkpoint")
import torch.utils.checkpoint

logger.info("Import transformers")
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

logger.info("Import diffusers")
import diffusers

logger.info("Import pooplines.")
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from torchvision.transforms import ToTensor

# Convert PIL Image to PyTorch Tensor
to_tensor = ToTensor()


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0.dev0")

logger = get_logger(__name__, log_level="DEBUG")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}
WANDB_TABLE_COL_NAMES = ["step", "image", "text"]


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

def main():
    args = parse_args()
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
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
    # Make one log on every process with the configuration for debugging.
    logger.info(accelerator.state, main_process_only=False)
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
        level=logging.DEBUG,
    )
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

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
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        force_upcast=False,
    )
    vae.enable_slicing()
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Create EMA for the unet.
    ema_unet = None
    if args.use_ema:
        logger.info("Using EMA. Creating EMAModel.")
        ema_unet = EMAModel(
            unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config
        )
        logger.info("EMA model creation complete.")

    if args.enable_xformers_memory_efficient_attention:
        logger.info("Enabling xformers memory-efficient attention.")
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        from helpers.sdxl_save_hooks import SDXLSaveHook

        model_hooks = SDXLSaveHook(
            args=args,
            ema_unet=ema_unet,
            accelerator=accelerator,
        )
        accelerator.register_save_state_pre_hook(model_hooks.save_model_hook)
        accelerator.register_load_state_pre_hook(model_hooks.load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if hasattr(args, "train_text_encoder") and args.train_text_encoder:
            text_encoder_1.gradient_checkpointing_enable()
            text_encoder_2.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        logger.info("Enabling tf32 precision boost for NVIDIA devices.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        logging.warning(
            "If using an Ada or Ampere NVIDIA device, --allow_tf32 could add a bit more performance."
        )

    if args.scale_lr:
        logger.info(f"Scaling learning rate ({args.learning_rate}), due to --scale_lr")
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
    logger.info(f"Learning rate: {args.learning_rate}")
    extra_optimizer_args = {}
    # Initialize the optimizer
    if args.use_8bit_adam:
        logger.info("Using 8bit AdamW optimizer.")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif hasattr(args, "use_dadapt_optimizer") and args.use_dadapt_optimizer:
        logger.info("Using D-Adaptation optimizer.")
        try:
            from dadaptation import DAdaptAdam
        except ImportError:
            raise ImportError(
                "Please install the dadaptation library to make use of DaDapt optimizer."
                "You can do so by running `pip install dadaptation`"
            )

        optimizer_cls = DAdaptAdam
        if (
            hasattr(args, "dadaptation_learning_rate")
            and args.dadaptation_learning_rate is not None
        ):
            logger.debug(
                f"Overriding learning rate {args.learning_rate} with {args.dadaptation_learning_rate} for D-Adaptation optimizer."
            )
            args.learning_rate = args.dadaptation_learning_rate
            extra_optimizer_args["decouple"] = True
    else:
        logger.info("Using AdamW optimizer.")
        optimizer_cls = torch.optim.AdamW
    logger.info(
        f"Optimizer arguments, weight_decay={args.adam_weight_decay} eps={args.adam_epsilon}"
    )
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        **extra_optimizer_args,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    # Bucket manager. We keep the aspect config in the dataset so that switching datasets is simpler.
    bucket_manager = BucketManager(
        instance_data_root=args.instance_data_dir,
        cache_file=os.path.join(args.instance_data_dir, "aspect_ratio_bucket_indices.json")
    )
    bucket_manager.compute_aspect_ratio_bucket_indices()
    if len(bucket_manager) == 0:
        raise Exception("No images were discovered by the bucket manager in the dataset.")

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        logger.info(f"Loading Huggingface Hub dataset: {args.dataset_name}")
        from datasets import load_dataset        
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )

    # 6. Get the column names for input/target.
    if hasattr(args, "dataset_name") and args.dataset_name is not None:
        raise ValueError("Huggingface datasets are not currently supported.")
        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names
        dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
        if args.image_column is None:
            image_column = (
                dataset_columns[0] if dataset_columns is not None else column_names[0]
            )
        else:
            image_column = args.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if args.image_prompt_column is None:
            image_prompt_column = (
                dataset_columns[1] if dataset_columns is not None else column_names[1]
            )
        else:
            image_prompt_column = args.image_prompt_column
            if image_prompt_column not in column_names:
                raise ValueError(
                    f"--image_prompt_column' value '{args.image_prompt_column}' needs to be one of: {', '.join(column_names)}"
                )
    else:
        logging.info(
            "Using SimpleTuner dataset layout, instead of huggingface --dataset layout."
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
            f'Using "--fp16" with mixed precision training should be done with a custom VAE. Make sure you understand how this works.'
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions, tokenizer):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

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

    # We ALWAYS pre-compute the additional condition embeddings needed for SDXL
    # UNet as the model is already big and it uses two text encoders.
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]

    # Freeze vae and text_encoders
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    def compute_time_ids(width=None, height=None):
        if width is None:
            width = args.resolution
        if height is None:
            height = args.resolution
        crops_coords_top_left = (
            args.crops_coords_top_left_h,
            args.crops_coords_top_left_w,
        )
        original_size = target_size = (width, height)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype)
        return add_time_ids.to(accelerator.device).repeat(args.train_batch_size, 1)

    def collate_fn(examples):
        if not StateTracker.status_training():
            logger.debug(f"Not training, returning nothing from collate_fn")
            if len(examples) > 0:
                for example in examples:
                    if example is not None and 'instance_image' in example:
                        example['instance_image'].close()
            return
        training_logger.debug(f"Examples: {examples}")
        training_logger.debug(f"Computing luminance for input batch")
        batch_luminance = calculate_batch_luminance(
            [example["instance_images"] for example in examples]
        )
        # Initialize the VAE Cache if it doesn't exist
        global vaecache
        if "vaecache" not in globals():
            vaecache = VAECache(vae, accelerator)

        pixel_values = []
        filepaths = []  # we will store the file paths here
        for example in examples:
            image_data = example["instance_images"]
            width = image_data.width
            height = image_data.height
            pixel_values.append(
                to_tensor(image_data).to(
                    memory_format=torch.contiguous_format, dtype=vae_dtype
                )
            )
            filepaths.append(example["instance_images_path"])  # store the file path

        # Compute the VAE embeddings for individual images
        latents = [
            vaecache.encode_image(pv, fp) for pv, fp in zip(pixel_values, filepaths)
        ]
        pixel_values = torch.stack(latents)

        # Extract the captions from the examples.
        captions = [example["instance_prompt_text"] for example in examples]

        # Compute the embeddings using the captions.
        (
            prompt_embeds_all,
            add_text_embeds_all,
        ) = embed_cache.compute_embeddings_for_prompts(captions)
        prompt_embeds_all = torch.concat([prompt_embeds_all for _ in range(1)], dim=0)
        add_text_embeds_all = torch.concat(
            [add_text_embeds_all for _ in range(1)], dim=0
        )

        return {
            "pixel_values": pixel_values,
            "prompt_embeds": prompt_embeds_all,
            "add_text_embeds": add_text_embeds_all,
            "add_time_ids": compute_time_ids(width, height),
            "luminance": batch_luminance,
        }

    # Data loader
    logger.info("Creating dataset iterator object")
    train_dataset = MultiAspectDataset(
        bucket_manager=bucket_manager,
        instance_data_root=args.instance_data_dir,
        accelerator=accelerator,
        size=args.resolution,
        center_crop=args.center_crop,
        print_names=args.print_filenames or False,
        use_original_images=bool(args.use_original_images),
        prepend_instance_prompt=args.prepend_instance_prompt or False,
        use_captions=not args.only_instance_prompt or False,
        caption_dropout_interval=args.caption_dropout_interval,
        use_precomputed_token_ids=True,
        debug_dataset_loader=args.debug_dataset_loader,
        caption_strategy=args.caption_strategy,
    )
    logger.info("Creating aspect bucket sampler")
    custom_balanced_sampler = MultiAspectSampler(
        bucket_manager=bucket_manager,
        batch_size=args.train_batch_size,
        seen_images_path=args.seen_state_path,
        state_path=args.state_path,
        debug_aspect_buckets=args.debug_aspect_buckets,
        delete_unwanted_images=args.delete_unwanted_images,
        minimum_image_size=args.minimum_image_size
    )
    logger.info("Plugging sampler into dataloader")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,  # The sampler handles shuffling
        sampler=custom_balanced_sampler,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )
    logger.info("Initialise text embedding cache")
    embed_cache = TextEmbeddingCache(
        text_encoders=text_encoders, tokenizers=tokenizers, accelerator=accelerator
    )
    if args.caption_dropout_probability is not None and args.caption_dropout_probability > 0:
        logger.info("Pre-computing null embedding for caption dropout")
        embed_cache.precompute_embeddings_for_prompts([""])
    else:
        logger.warning(f'Not using caption dropout will potentially lead to overfitting on captions.')

    with accelerator.main_process_first():
        logger.info(f"Pre-computing text embeds / updating cache.")
        embed_cache.precompute_embeddings_for_prompts(
            PromptHandler.get_all_captions(
                instance_data_root=args.instance_data_dir,
                prepend_instance_prompt=args.prepend_instance_prompt or False,
                use_captions=args.only_instance_prompt or False
            )
        )

    validation_prompts = []
    validation_shortnames = []
    if args.validation_prompt_library:
        # Use the SimpleTuner prompts library for validation prompts.
        from helpers.prompts import prompts as prompt_library
        # Prompt format: { 'shortname': 'this is the prompt', ... }
        for shortname, prompt in prompt_library.items():
            logger.info(f'Precomputing validation prompt embeds: {shortname}')
            embed_cache.compute_embeddings_for_prompts([prompt])
            validation_prompts.append(prompt)
            validation_shortnames.append(shortname)
    elif args.validation_prompt is not None:
        # Use a single prompt for validation.
        validation_prompts = [args.validation_prompt]
        validation_shortnames = ['validation']
        (
            validation_prompt_embeds,
            validation_pooled_embeds,
        ) = embed_cache.compute_embeddings_for_prompts([args.validation_prompt])
    
    # Compute negative embed for validation prompts, if any are set.
    if validation_prompts:
        (
            validation_negative_prompt_embeds,
            validation_negative_pooled_embeds,
        ) = embed_cache.compute_embeddings_for_prompts(["blurry, cropped, ugly"])
    # Grab GPU memory used:
    if accelerator.is_main_process:
        logger.info(
            f"Moving text encoders back to CPU, to save VRAM. Currently, we cannot completely unload the text encoder."
        )
    memory_before_unload = torch.cuda.memory_allocated() / 1024**3
    text_encoder_1.to("cpu")
    text_encoder_2.to("cpu")
    memory_after_unload = torch.cuda.memory_allocated() / 1024**3
    memory_saved = memory_after_unload - memory_before_unload
    logger.info(
        f"After nuking text encoders from orbit, we freed {abs(round(memory_saved, 2))} GB of VRAM."
        "This number might be massively understated, because of how CUDA memory management works."
        "The real memories were the friends we trained a model on along the way."
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    logger.info(f"Loading noise scheduler...")
    if args.lr_scheduler != "polynomial":
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
    else:
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            lr_end=args.learning_rate_end,
            power=args.lr_power,
            last_epoch=-1,
        )
    accelerator.wait_for_everyone()
    # Prepare everything with our `accelerator`.
    logger.info(f"Loading our accelerator...")
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        logger.info("Moving EMA model weights to accelerator...")
        ema_unet.to(accelerator.device)

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
    logger.info(f"Loaded VAE into VRAM.")
    if accelerator.is_main_process:
        logger.info(f"Pre-computing VAE latent space.")
        vaecache = VAECache(vae, accelerator)
        vaecache.process_directory(args.instance_data_dir)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_run_name, config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    resume_step = 0

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
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            StateTracker.start_training()
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )
    else:
        StateTracker.start_training()

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        logger.debug(f"Starting into epoch: {epoch}")
        unet.train()
        train_loss = 0.0
        training_luminance_values = []
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                if step + 1 == resume_step:
                    # We want to trigger the batch to be properly generated when we start.
                    if not StateTracker.status_training():
                        logging.info(
                            f"Starting training, as resume_step has been reached."
                        )
                        StateTracker.start_training()
                logger.warning("Skipping step.")
                continue

            if batch is None:
                logging.warning(f"Burning a None size batch.")
                continue
            
            # Add the current batch of training data's avg luminance to a list.
            training_luminance_values.append(batch["luminance"])

            with accelerator.accumulate(unet):
                training_logger.debug(f"Beginning another step.")
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                training_logger.debug("Moved pixels to accelerator.")
                latents = pixel_values
                # Sample noise that we'll add to the latents
                training_logger.debug(f"Sampling random noise")
                # Sample noise that we'll add to the latents - args.noise_offset might need to be set to 0.1 by default.
                noise = None
                if args.offset_noise:
                    noise = torch.randn_like(latents) + args.noise_offset * torch.randn(
                        latents.shape[0], latents.shape[1], 1, 1, device=latents.device
                    )
                else:
                    noise = torch.randn_like(latents)
                if args.input_pertubation:
                    new_noise = noise + args.input_pertubation * torch.randn_like(noise)
                elif noise is None:
                    noise = torch.randn_like(latents)

                bsz = latents.shape[0]
                training_logger.debug(f"Working on batch size: {bsz}")
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                training_logger.debug(f"Generated and converted timesteps to float64.")

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_pertubation:
                    noisy_latents = noise_scheduler.add_noise(
                        latents, new_noise, timesteps
                    )
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                training_logger.debug(
                    f"Generated noisy latent frame from latents and noise."
                )
                # SDXL additional inputs - probabilistic dropout
                encoder_hidden_states = batch["prompt_embeds"]
                if args.caption_dropout_probability is not None and args.caption_dropout_probability > 0:
                    # When using caption dropout, we will use the null embed instead of prompt embeds.
                    # The chance of this happening is dictated by the caption_dropout_probability.
                    if random.random() < args.caption_dropout_probability:
                        training_logger.debug(f'Caption dropout triggered.')
                        (
                            batch["prompt_embeds_all"],
                            batch["add_text_embeds_all"],
                        ) = embed_cache.compute_embeddings_for_prompts([""])

                # Conditioning dropout not yet supported.
                add_text_embeds = batch["add_text_embeds"]
                training_logger.debug(
                    f"Encoder hidden states: {encoder_hidden_states.shape}"
                )
                training_logger.debug(f"Added text embeds: {add_text_embeds.shape}")
                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual and compute loss
                # training_logger.debug(f'add_text_embeds: {add_text_embeds.shape}, time_ids: {add_time_ids}')
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": batch["add_time_ids"],
                }
                training_logger.debug("Predicting noise residual.")
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                if args.snr_gamma is None:
                    training_logger.debug(f"Calculating loss")
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    training_logger.debug(f"Using min-SNR loss")
                    snr = compute_snr(timesteps, noise_scheduler)

                    if torch.any(torch.isnan(snr)):
                        training_logger.error("snr contains NaN values")
                    if torch.any(snr == 0):
                        training_logger.error("snr contains zero values")
                    training_logger.debug(f'Calculating MSE loss weights using SNR as divisor')
                    mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    # An experimental strategy for fixing min-SNR with zero terminal SNR is to set loss weighting to 1 when
                    #  any positional tensors have an SNR of zero. This is to preserve their loss values and also to hopefully
                    #  prevent the explosion of gradients or NaNs due to the presence of very small numbers.
                    mse_loss_weights[snr == 0] = 1.0
                    if torch.any(torch.isnan(mse_loss_weights)):
                        training_logger.error("mse_loss_weights contains NaN values")
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    training_logger.debug(f'Calculating original MSE loss without reduction')
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    training_logger.debug(f'Calculating SNR-weighted MSE loss')
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    training_logger.debug(f'Reducing loss via mean')
                    loss = loss.mean()
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                training_logger.debug(f"Backwards pass.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    training_logger.debug(f"Accelerator is syncing gradients")
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                training_logger.debug(f"Stepping components forward.")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    training_logger.debug(f"Stepping EMA unet forward")
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                # Average out the luminance values of each batch, so that we can store that in this step.
                avg_training_data_luminance = sum(training_luminance_values) / len(training_luminance_values)
                accelerator.log({"train_luminance": avg_training_data_luminance, "train_loss": train_loss, "learning_rate": lr_scheduler.get_last_lr()[0]}, step=global_step)
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
                        custom_balanced_sampler.save_state()
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            ### BEGIN: Perform validation every `validation_epochs` steps
            if accelerator.is_main_process:
                if validation_prompts and global_step % args.validation_steps == 0 and global_step > 1:
                    if (
                        args.validation_prompt is None
                        or args.num_validation_images is None
                        or args.num_validation_images <= 0
                    ):
                        logging.warning(
                            f"Not generating any validation images for this checkpoint. Live dangerously and prosper, pal!"
                        )
                        continue
                    if args.gradient_accumulation_steps > 0 and step % args.gradient_accumulation_steps != 0:
                        # We do not want to perform validation on a partial batch.
                        continue
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )
                    # create pipeline
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())

                    # The models need unwrapping because for compatibility in distributed training mode.
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        text_encoder=text_encoder_1,
                        text_encoder_2=text_encoder_2,
                        tokenizer=None,
                        tokenizer_2=None,
                        vae=vae,
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                    )
                    pipeline.scheduler = DDIMScheduler.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="scheduler",
                        prediction_type=args.prediction_type,
                        timestep_spacing=args.inference_scheduler_timestep_spacing,
                        rescale_betas_zero_snr=args.rescale_betas_zero_snr,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    # Save validation images
                    val_save_dir = os.path.join(args.output_dir, "validation_images")
                    if not os.path.exists(val_save_dir):
                        os.makedirs(val_save_dir)

                    with torch.autocast(
                        str(accelerator.device).replace(":0", ""),
                        enabled=(
                            accelerator.mixed_precision == "fp16"
                            or accelerator.mixed_precision == "bf16"
                        ),
                    ):
                        validation_images = []
                        pipeline = pipeline.to(accelerator.device)
                        with torch.autocast(str(accelerator.device).replace(":0", "")):
                            validation_generator = torch.Generator(
                                device=accelerator.device
                            ).manual_seed(args.seed or 0)
                            for validation_prompt in validation_prompts:
                                # Each validation prompt needs its own embed.
                                current_validation_prompt_embeds, current_validation_pooled_embeds = embed_cache.compute_embeddings_for_prompts(
                                    [validation_prompt]
                                )
                                logger.info(f'Generating validation image: {validation_prompt}')
                                validation_images.extend(pipeline(
                                    prompt_embeds=current_validation_prompt_embeds,
                                    pooled_prompt_embeds=current_validation_pooled_embeds,
                                    negative_prompt_embeds=validation_negative_prompt_embeds,
                                    negative_pooled_prompt_embeds=validation_negative_pooled_embeds,
                                    num_images_per_prompt=args.num_validation_images,
                                    num_inference_steps=30,
                                    guidance_scale=args.validation_guidance,
                                    guidance_rescale=args.validation_guidance_rescale,
                                    generator=validation_generator,
                                    height=args.validation_resolution,
                                    width=args.validation_resolution,
                                ).images)

                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                validation_document = {}
                                validation_luminance = []
                                for idx, validation_image in enumerate(validation_images):
                                    # Create a WandB entry containing each image.
                                    validation_document[validation_shortnames[idx]] = wandb.Image(validation_image)
                                    validation_luminance.append(
                                        calculate_luminance(validation_image)
                                    )
                                # Compute the mean luminance across all samples:
                                validation_luminance = torch.tensor(validation_luminance)
                                validation_document["validation_luminance"] = validation_luminance.mean()
                                del validation_luminance
                                tracker.log(validation_document, step=global_step)
                        val_img_idx = 0
                        for a_val_img in validation_images:
                            a_val_img.save(
                                os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_val_img_{val_img_idx}.png",
                                )
                            )
                            val_img_idx += 1

                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())

                    del pipeline
                    torch.cuda.empty_cache()
                ### END: Perform validation every `validation_epochs` steps

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.scheduler.config = noise_scheduler.config
        pipeline.save_pretrained("/notebooks/datasets/models/ptx0-xltest", safe_serialization=True)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        if validation_prompts:
            validation_images = []
            pipeline = pipeline.to(accelerator.device)
            with torch.autocast(str(accelerator.device).replace(":0", "")):
                validation_generator = torch.Generator(
                    device=accelerator.device
                ).manual_seed(args.seed or 0)
                for validation_prompt in validation_prompts:
                    # Each validation prompt needs its own embed.
                    current_validation_prompt_embeds, current_validation_pooled_embeds = embed_cache.compute_embeddings_for_prompts(
                        [validation_prompt]
                    )
                    validation_images.extend(pipeline(
                        prompt_embeds=current_validation_prompt_embeds,
                        pooled_prompt_embeds=current_validation_pooled_embeds,
                        negative_prompt_embeds=validation_negative_prompt_embeds,
                        negative_pooled_prompt_embeds=validation_negative_pooled_embeds,
                        num_images_per_prompt=args.num_validation_images,
                        num_inference_steps=30,
                        guidance_scale=args.validation_guidance,
                        guidance_rescale=args.validation_guidance_rescale,
                        generator=validation_generator,
                        height=args.validation_resolution,
                        width=args.validation_resolution,
                    ).images)

                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        validation_document = {}
                        validation_luminance = []
                        for idx, validation_image in enumerate(validation_images):
                            # Create a WandB entry containing each image.
                            validation_document[validation_shortnames[idx]] = wandb.Image(validation_image)
                            validation_luminance.append(
                                calculate_luminance(validation_image)
                            )
                        # Compute the mean luminance across all samples:
                        validation_luminance = torch.tensor(validation_luminance)
                        validation_document["validation_luminance"] = validation_luminance.mean()
                        del validation_luminance
                        tracker.log(validation_document, step=global_step)

    accelerator.end_training()


if __name__ == "__main__":
    main()
