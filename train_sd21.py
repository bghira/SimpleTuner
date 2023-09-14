#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
from helpers import log_format

import hashlib, random, itertools, logging, math, time, os, json, copy

os.environ["ACCELERATE_LOG_LEVEL"] = "WARNING"

from pathlib import Path
from helpers.arguments import parse_args
from helpers.training.state_tracker import StateTracker
from helpers.caching.sdxl_embeds import TextEmbeddingCache
from helpers.prompts import PromptHandler
from helpers.training.multi_process import rank_info
from helpers.legacy.sd_files import (
    import_model_class_from_model_name_or_path,
    register_file_hooks,
)
from helpers.multiaspect.bucket import BucketManager
from helpers.multiaspect.dataset import MultiAspectDataset
from helpers.multiaspect.sampler import MultiAspectSampler
from helpers.training.min_snr_gamma import compute_snr
from helpers.legacy.validation import prepare_validation_prompt_list, log_validations
from helpers.legacy.metadata import save_model_card
from helpers.training.custom_schedule import (
    enforce_zero_terminal_snr,
    get_polynomial_decay_schedule_with_warmup,
)
from helpers.training.model_freeze import freeze_entire_component, freeze_text_encoder
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from helpers.caching.vae import VAECache
from helpers.image_manipulation.brightness import (
    calculate_luminance,
    calculate_batch_luminance,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)

from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

tokenizer = None

torch.autograd.set_detect_anomaly(True)
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

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
if os.environ.get("SIMPLETUNER_LOG_LEVEL"):
    training_logger_level = os.environ.get("SIMPLETUNER_LOG_LEVEL")
training_logger.setLevel(training_logger_level)

# Less important logs.
filelock_logger.setLevel("WARNING")
connection_logger.setLevel("WARNING")

from torchvision.transforms import ToTensor

to_tensor = ToTensor()


def compute_ids(prompt: str):
    global tokenizer
    return tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed, args.seed_for_each_device)

    # Load the tokenizer
    global tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, revision=args.revision, use_fast=False
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    if not tokenizer:
        raise Exception("Failed to load tokenizer.")

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )

    # Load scheduler and models
    scheduler_model = args.pretrained_model_name_or_path
    temp_scheduler = DDIMScheduler.from_pretrained(
        scheduler_model,
        subfolder="scheduler",
        timestep_spacing="trailing",
        prediction_type="v_prediction",
        rescale_betas_zero_snr=True,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        scheduler_model,
        subfolder="scheduler",
        trained_betas=temp_scheduler.betas.clone().detach(),
        prediction_type="v_prediction",
    )
    text_encoder = freeze_text_encoder(
        args,
        text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
        ),
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    register_file_hooks(args, accelerator, unet, text_encoder, text_encoder_cls)

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.20"):
                logger.warn(
                    "SimpleTuner requires at least PyTorch 2.0.1, which in turn requires a new version (0.0.20) of Xformers."
                )
            unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if (
        args.train_text_encoder
        and accelerator.unwrap_model(text_encoder).dtype != torch.float32
    ):
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if args.seen_state_path is None:
        raise ValueError(
            'Please specify a path to a "seen" state dict via the --seen_state_path parameter.'
        )
    if args.state_path is None:
        raise ValueError(
            "Please specify a location of your training state status file via the --state_path parameter."
        )
    if args.caption_dropout_interval > 100:
        raise ValueError(
            "Please specify a caption dropout interval equal to or less than 100 via the --caption_dropout_interval parameter."
        )
    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters())
        if args.train_text_encoder
        else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # Create a DataBackend, so that we can access our dataset.
    if args.data_backend == "local":
        from helpers.data_backend.local import LocalDataBackend

        data_backend = LocalDataBackend()
    elif args.data_backend == "aws":
        from helpers.data_backend.aws import S3DataBackend

        data_backend = S3DataBackend(
            bucket_name=args.aws_bucket_name,
            region_name=args.aws_region_name,
            endpoint_url=args.aws_endpoint_url,
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
        )
    else:
        raise ValueError(f"Unsupported data backend: {args.data_backend}")
    logger.info(
        f"{rank_info(accelerator)} created {args.data_backend} data backend.",
        main_process_only=False,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    # Bucket manager. We keep the aspect config in the dataset so that switching datasets is simpler.

    logger.info(
        f"{rank_info(accelerator)} is creating a bucket manager.",
        main_process_only=False,
    )
    bucket_manager = BucketManager(
        instance_data_root=args.instance_data_dir,
        data_backend=data_backend,
        accelerator=accelerator,
        batch_size=args.train_batch_size,
        cache_file=os.path.join(
            args.instance_data_dir, "aspect_ratio_bucket_indices.json"
        ),
        apply_dataset_padding=args.apply_dataset_padding or False,
    )
    if accelerator.is_main_process:
        logger.info(
            f"{rank_info(accelerator)} is on its way to computing the indicies.",
            main_process_only=False,
        )
        bucket_manager.compute_aspect_ratio_bucket_indices()
        logger.info(
            f"{rank_info(accelerator)} is now refreshing the buckets..",
            main_process_only=False,
        )
        bucket_manager.refresh_buckets()
        logger.info(
            f"{rank_info(accelerator)} has completed its bucket manager tasks.",
            main_process_only=False,
        )
        logger.info(
            f"{rank_info(accelerator)} is now splitting the data.",
            main_process_only=False,
        )
    accelerator.wait_for_everyone()
    # Now split the contents of these buckets between all processes
    bucket_manager.split_buckets_between_processes()
    # Now, let's print the total of each bucket, along with the current rank, so that we might catch debug info:
    for bucket in bucket_manager.aspect_ratio_bucket_indices:
        print(
            f"{rank_info(accelerator)}: {len(bucket_manager.aspect_ratio_bucket_indices[bucket])} images in bucket {bucket}"
        )

    if len(bucket_manager) == 0:
        raise Exception(
            "No images were discovered by the bucket manager in the dataset."
        )
    logger.info("Creating dataset iterator object")

    train_dataset = MultiAspectDataset(
        bucket_manager=bucket_manager,
        data_backend=data_backend,
        instance_data_root=args.instance_data_dir,
        accelerator=accelerator,
        size=args.resolution,
        center_crop=args.center_crop,
        print_names=args.print_filenames or False,
        use_original_images=bool(args.use_original_images),
        prepend_instance_prompt=args.prepend_instance_prompt or False,
        use_captions=not args.only_instance_prompt or False,
        use_precomputed_token_ids=False,
        caption_dropout_interval=args.caption_dropout_interval,
        debug_dataset_loader=args.debug_dataset_loader,
        caption_strategy=args.caption_strategy,
        return_tensor=True,
    )
    logger.info("Creating aspect bucket sampler")

    custom_balanced_sampler = MultiAspectSampler(
        bucket_manager=bucket_manager,
        data_backend=data_backend,
        batch_size=args.train_batch_size,
        seen_images_path=args.seen_state_path,
        state_path=args.state_path,
        debug_aspect_buckets=args.debug_aspect_buckets,
        delete_unwanted_images=args.delete_unwanted_images,
        minimum_image_size=args.minimum_image_size,
        resolution=args.resolution,
    )

    def collate_fn(batch):
        if len(batch) != 1:
            raise ValueError(
                "This trainer is not designed to handle multiple batches in a single collate."
            )
        examples = batch[0]
        training_logger.debug(f"Examples: {examples}")
        training_logger.debug(f"Computing luminance for input batch")
        batch_luminance = calculate_batch_luminance(
            [example["instance_images"] for example in examples]
        )

        # Initialize the VAE Cache if it doesn't exist
        global vaecache
        if "vaecache" not in globals():
            vaecache = VAECache(
                vae=vae,
                accelerator=accelerator,
                data_backend=data_backend,
                resolution=args.resolution,
                delete_problematic_images=args.delete_problematic_images,
                vae_batch_size=args.vae_batch_size,
                write_batch_size=args.write_batch_size,
            )

        pixel_values = []
        filepaths = []  # we will store the file paths here
        for example in examples:
            image_data = example["instance_images"]
            # SDXL would grab the width/height here, but it's not needed for SD 2.x
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
        latent_batch = torch.stack(latents)

        # Extract the captions from the examples.
        captions = [example["instance_prompt_text"] for example in examples]

        # Compute the embeddings using the captions.
        prompt_embeds_all = embed_cache.compute_embeddings_for_legacy_prompts(captions)
        logger.debug(f'{len(prompt_embeds_all)} prompt_embeds_all: {prompt_embeds_all}')
        prompt_embeds_all = torch.concat(prompt_embeds_all, dim=0)

        return {
            "latent_batch": latent_batch,
            "prompt_embeds": prompt_embeds_all,
            "luminance": batch_luminance,
        }

    logger.info("Plugging sampler into dataloader")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,  # The sampler handles batching
        shuffle=False,  # The sampler handles shuffling
        sampler=custom_balanced_sampler,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    logger.info("Initialise text embedding cache")
    embed_cache = TextEmbeddingCache(
        text_encoders=[text_encoder, None],
        tokenizers=[tokenizer, None],
        accelerator=accelerator,
        model_type="legacy",
    )

    logger.info(f"Pre-computing text embeds / updating cache.")
    with accelerator.main_process_first():
        all_captions = PromptHandler.get_all_captions(
            data_backend=data_backend,
            instance_data_root=args.instance_data_dir,
            prepend_instance_prompt=args.prepend_instance_prompt or False,
            use_captions=not args.only_instance_prompt,
        )
        embed_cache.compute_embeddings_for_legacy_prompts(all_captions)
    (
        validation_prompts,
        validation_shortnames,
        validation_negative_prompt_embeds,
    ) = prepare_validation_prompt_list(args=args, embed_cache=embed_cache)

    logger.info("Configuring runtime step count and epoch limit")
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        logger.debug(
            f"Overriding max_train_steps to {args.max_train_steps} = {args.num_train_epochs} * {num_update_steps_per_epoch}"
        )
        overrode_max_train_steps = True

    if args.lr_scheduler != "polynomial":
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
    else:
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            lr_end=args.learning_rate_end,
            power=args.lr_power,
            last_epoch=-1,
        )

    # Create EMA for the unet.
    ema_unet = None
    if args.use_ema:
        logger.info("Using EMA. Creating EMAModel.")
        ema_unet = EMAModel(
            unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config
        )
        logger.info("EMA model creation complete.")

    logger.info("Preparing accelerator..")

    # Base components to prepare
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Conditionally prepare the text_encoder if required
    if args.train_text_encoder:
        text_encoder = accelerator.prepare(text_encoder)

    # Conditionally prepare the EMA model if required
    if args.use_ema:
        ema_model = accelerator.prepare(ema_model)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    logging.info("Moving VAE to GPU..")
    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        logging.info("Moving text encoder to GPU..")
        text_encoder.to(accelerator.device, dtype=weight_dtype)
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
    logger.info(f"Pre-computing VAE latent space.")
    vaecache = VAECache(
        vae=vae,
        accelerator=accelerator,
        data_backend=data_backend,
        delete_problematic_images=args.delete_problematic_images,
        resolution=args.resolution,
        vae_batch_size=args.vae_batch_size,
        write_batch_size=args.write_batch_size,
    )
    vaecache.split_cache_between_processes()
    vaecache.process_buckets(bucket_manager=bucket_manager)
    accelerator.wait_for_everyone()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    logging.info("Recalculating max step count.")
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
        # Copy args into public_args:
        import copy, json

        public_args = copy.deepcopy(args)
        # Remove the args that we don't want to track:
        del public_args.aws_access_key_id
        del public_args.aws_secret_access_key
        del public_args.aws_bucket_name
        del public_args.aws_region_name
        del public_args.aws_endpoint_url
        project_name = args.tracker_project_name or "simpletuner-training"
        tracker_run_name = args.tracker_run_name or "simpletuner-training-run"
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
    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataset) / total_batch_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    resume_global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logging.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            logging.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            custom_balanced_sampler.load_states(
                state_path=os.path.join(args.output_dir, path, "training_state.json"),
            )
            first_epoch = custom_balanced_sampler.current_epoch
            resume_global_step = global_step = int(path.split("-")[1])
    StateTracker.start_training()
    # We store the number of dataset resets that have occurred inside the checkpoint.
    first_epoch = custom_balanced_sampler.current_epoch
    if first_epoch > 1:
        logger.info(
            f"Resuming from epoch {first_epoch}, which is not the first epoch. This is a bit weird."
        )
        steps_to_remove = first_epoch * num_update_steps_per_epoch
        args.max_train_steps -= steps_to_remove

    current_epoch = first_epoch
    if current_epoch >= args.num_train_epochs:
        logger.info(
            f"Reached the end ({current_epoch} epochs) of our training run ({args.num_train_epochs} epochs). This run will do zero steps."
        )

    import time

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.update(global_step)
    progress_bar.set_description("Steps")
    current_percent_completion = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        if current_epoch >= args.num_train_epochs:
            # This might immediately end training, but that's useful for simply exporting the model.
            logger.info(
                f"Reached the end ({current_epoch} epochs) of our training run ({args.num_train_epochs} epochs)."
            )
            break
        logger.debug(
            f"Starting into epoch {epoch} out of {current_epoch}, final epoch will be {args.num_train_epochs}"
        )
        current_epoch = epoch
        if args.lr_scheduler == "cosine_annealing_warm_restarts":
            scheduler_kwargs["epoch"] = epoch

        unet.train()
        train_loss = 0.0
        training_luminance_values = []
        current_epoch_step = 0
        if args.train_text_encoder:
            logging.debug(f"Bumping text encoder.")
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            if accelerator.is_main_process:
                progress_bar.set_description(
                    f"Epoch {current_epoch}/{args.num_train_epochs}, Steps"
                )

            if batch is None:
                raise ValueError(
                    f"Trainer received invalid value for training examples"
                )

            with accelerator.accumulate(unet):
                logger.debug(f"Sending latent batch to device")
                latents = batch["latent_batch"].to(dtype=weight_dtype)

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
                logger.debug(f"Working on batch size: {bsz}")
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_pertubation:
                    noisy_latents = noise_scheduler.add_noise(
                        latents, new_noise, timesteps
                    )
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = batch["prompt_embeds"]
                training_logger.debug(
                    f"Encoder hidden states: {encoder_hidden_states.shape}"
                )
                if (
                    args.caption_dropout_probability is not None
                    and args.caption_dropout_probability > 0
                ):
                    # When using caption dropout, we will use the null embed instead of prompt embeds.
                    # The chance of this happening is dictated by the caption_dropout_probability.
                    if random.random() < args.caption_dropout_probability:
                        training_logger.debug(f"Caption dropout triggered.")
                        batch[
                            "prompt_embeds_all"
                        ] = embed_cache.compute_embeddings_for_legacy_prompts([""])

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        "Supported types are 'epsilon' and 'v_prediction'."
                    )

                # Predict the noise residual
                logging.debug(f"Running prediction")
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
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
                    training_logger.debug(
                        f"Calculating MSE loss weights using SNR as divisor"
                    )
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

                logging.debug(f"Backwards pass.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
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
                current_epoch_step += 1
                current_percent_completion = int(
                    progress_bar.n / progress_bar.total * 100
                )
                # Average out the luminance values of each batch, so that we can store that in this step.
                avg_training_data_luminance = sum(training_luminance_values) / min(1, len(
                    training_luminance_values
                ))
                logs = {
                    "train_luminance": avg_training_data_luminance,
                    "train_loss": train_loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(
                    **logs,
                    step=global_step,
                )
                # Reset some values for the next go.
                training_luminance_values = []
                train_loss = 0.0

                if (
                    args.freeze_encoder
                    and current_percent_completion > args.text_encoder_limit
                ):
                    # We want to stop training the text_encoder around 25% by default.
                    freeze_entire_component(text_encoder)
                    logging.warning(
                        f"Frozen text_encoder at {current_percent_completion}%!"
                    )
                    # This will help ensure we don't run this check every time from now on.
                    args.freeze_encoder = False
                    args.train_text_encoder = False

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
                        custom_balanced_sampler.save_state(
                            state_path=os.path.join(save_path, "training_state.json"),
                        )
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            log_validations(
                logger,
                accelerator,
                unet,
                args,
                validation_prompts,
                global_step,
                resume_global_step,
                step,
                progress_bar,
                text_encoder,
                tokenizer,
                vae_path=args.pretrained_model_name_or_path,
                weight_dtype=unet.dtype,
                embed_cache=embed_cache,
                validation_negative_pooled_embeds=None,
                text_encoder_2=None,
                tokenizer_2=None,
                ema_unet=ema_unet,
                vae=vae,
            )

            if global_step >= args.max_train_steps or epoch > args.num_train_epochs:
                logger.info(
                    f"Training has completed.",
                    f"\n -> global_step = {global_step}, max_train_steps = {args.max_train_steps}, epoch = {epoch}, num_train_epochs = {args.num_train_epochs}",
                )
                break
        if global_step >= args.max_train_steps or epoch > args.num_train_epochs:
            logger.info(
                f"Exiting training loop. Beginning model unwind at epoch {epoch}, step {global_step}"
            )
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.train_text_encoder:
            text_encoder = accelerator.unwrap_model(text_encoder)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        if vae is None:
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae"
                if args.pretrained_vae_model_name_or_path is None
                else None,
                revision=args.revision,
                force_upcast=False,
            )
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
            prediction_type=args.prediction_type,
            timestep_spacing=args.training_scheduler_timestep_spacing,
            rescale_betas_zero_snr=args.rescale_betas_zero_snr,
        )
        pipeline.save_pretrained(
            os.path.join(args.output_dir, args.hub_model_id or "pipeline"),
            safe_serialization=True,
        )

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                prompt=args.instance_prompt,
                repo_folder=args.output_dir,
            )
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
                for validation_prompt in tqdm(
                    validation_prompts, desc="Generating validation images"
                ):
                    # Each validation prompt needs its own embed.
                    current_validation_prompt_embeds = (
                        embed_cache.compute_embeddings_for_legacy_prompts(
                            [validation_prompt]
                        )
                    )
                    validation_images.extend(
                        pipeline(
                            prompt_embeds=current_validation_prompt_embeds,
                            negative_prompt_embeds=validation_negative_prompt_embeds,
                            num_images_per_prompt=args.num_validation_images,
                            num_inference_steps=30,
                            guidance_scale=args.validation_guidance,
                            guidance_rescale=args.validation_guidance_rescale,
                            generator=validation_generator,
                            height=args.validation_resolution,
                            width=args.validation_resolution,
                        ).images
                    )

                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        validation_document = {}
                        validation_luminance = []
                        for idx, validation_image in enumerate(validation_images):
                            # Create a WandB entry containing each image.
                            validation_document[
                                validation_shortnames[idx]
                            ] = wandb.Image(validation_image)
                            validation_luminance.append(
                                calculate_luminance(validation_image)
                            )
                        # Compute the mean luminance across all samples:
                        validation_luminance = torch.tensor(validation_luminance)
                        validation_document[
                            "validation_luminance"
                        ] = validation_luminance.mean()
                        del validation_luminance
                        tracker.log(validation_document, step=global_step)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
