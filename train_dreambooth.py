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

import hashlib
import itertools
import logging
import math
import time
import os
from pathlib import Path
from helpers.arguments import parse_args
from helpers.training.state_tracker import StateTracker
from helpers.legacy.sd_files import (
    import_model_class_from_model_name_or_path,
    register_file_hooks,
)
from helpers.legacy.aspect_bucket import BalancedBucketSampler
from helpers.training.min_snr_gamma import compute_snr
from helpers.legacy.validation import log_validation
from helpers.legacy.metadata import save_model_card
from helpers.training.custom_schedule import (
    enforce_zero_terminal_snr,
    get_polynomial_decay_schedule_with_warmup,
)
from helpers.training.model_freeze import freeze_entire_component, freeze_text_encoder
from helpers.legacy.dreambooth_dataset import DreamBoothDataset
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

import diffusers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)

from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

torch.autograd.set_detect_anomaly(True)
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

logger = get_logger("root")
from helpers import log_format
from torchvision.transforms import ToTensor

def collate_fn(examples):
    if not StateTracker.status_training():
        logging.debug("collate_fn: not training, returning examples.")
        return examples

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


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

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
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
        set_seed(args.seed)

    # Load the tokenizer
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
    )
    trained_betas = enforce_zero_terminal_snr(temp_scheduler.betas).numpy().tolist()
    noise_scheduler = DDPMScheduler.from_pretrained(
        scheduler_model,
        subfolder="scheduler",
        trained_betas=trained_betas,
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
            raise RuntimeError(
                "Please uninstall xformers. The memory efficient attention is now part of PyTorch 2."
            )

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

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        accelerator=accelerator,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        print_names=args.print_filenames or False,
        use_original_images=bool(args.use_original_images),
        prepend_instance_prompt=args.prepend_instance_prompt or False,
        use_captions=not args.only_instance_prompt or False,
        caption_dropout_interval=args.caption_dropout_interval,
    )
    custom_balanced_sampler = BalancedBucketSampler(
        train_dataset.aspect_ratio_bucket_indices,
        batch_size=args.train_batch_size,
        seen_images_path=args.seen_state_path,
        state_path=args.state_path,
        minimum_image_size=args.minimum_image_size,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,  # The sampler handles shuffling
        sampler=custom_balanced_sampler,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
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

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

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
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
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
            StateTracker.start_training()
        else:
            logging.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )
    else:
        StateTracker.start_training()

    import time

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    current_percent_completion = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            logging.debug(f"Bumping text encoder.")
            text_encoder.train()
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
                continue
            if type(batch) is list:
                logging.warning("Burning a step due to dummy data.")
                time.sleep(10)

                continue
            logging.debug(f"Accumulating...")
            with accelerator.accumulate(unet):
                logging.debug(f"Convert to latent space")
                # Convert images to latent space. This could run out of VRAM, so we'll try again.
                attempts = 5
                current_attempt = 0
                while current_attempt < attempts:
                    try:
                        latents = vae.encode(
                            batch["pixel_values"].to(dtype=weight_dtype)
                        ).latent_dist.sample()
                        break
                    except Exception as e:
                        import traceback

                        logging.error(
                            f"Error: {e}, traceback: {traceback.format_exc()}"
                        )
                        torch.clear_autocast_cache()
                        time.sleep(5)
                latents = latents * vae.config.scaling_factor

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
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                logging.debug(f"Calculate target for loss")
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                    logging.debug(f"Using Epsilon target.")
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    logging.debug(f"Using v-prediction target.")
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
                    logging.debug(f"Calculating loss")
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps, noise_scheduler)

                    if torch.any(torch.isnan(snr)):
                        print("snr contains NaN values")
                    if torch.any(snr == 0):
                        print("snr contains zero values")

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
                        print("mse_loss_weights contains NaN values")
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()
                logging.debug(f"Backwards pass.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    logging.debug(f"Syncing gradients")
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                logging.debug(f"Stepped")
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                logging.debug(f"Optimizer set grads.")
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                current_percent_completion = int(
                    progress_bar.n / progress_bar.total * 100
                )
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

                global_step += 1

                if accelerator.is_main_process:
                    images = []
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        custom_balanced_sampler.save_state()
                        logger.info(f"Saved state to {save_path}")

                    if (
                        args.validation_prompt is not None
                        and global_step % args.validation_steps == 0
                    ):
                        images = log_validation(
                            text_encoder,
                            tokenizer,
                            unet,
                            vae,
                            args,
                            accelerator,
                            weight_dtype,
                            epoch,
                        )
            logging.debug(f"Writing logs.")
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                logging.warn(f"Reached stopping point. Beginning to unwind.")
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
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

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
