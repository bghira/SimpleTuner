# SimpleTuner Training Script Options

## Overview

This guide provides a user-friendly breakdown of the command-line options available in SimpleTuner's `train_sdxl.py` script. These options offer a high degree of customization, allowing you to train your model to suit your specific requirements.

---

## 🌟 Core Model Configuration

### `--pretrained_model_name_or_path`

- **What**: Path to the pretrained model or its identifier from huggingface.co/models.
- **Why**: To specify the base model you'll start training from.

---

## 📂 Data Storage and Management

### `--instance_data_dir`

- **What**: Folder containing the training data.
- **Why**: Designates where your training images and other data are stored.

### `--data_backend`

- **What**: Specifies the data storage backend, either 'local' or 'aws'.
- **Why**: Allows for seamless switching between local and cloud storage.

---

## 🌈 Image and Text Processing

### `--resolution`

- **What**: Input image resolution.
- **Why**: All images in the dataset will be resized to this resolution for training.

### `--caption_strategy`

- **What**: Strategy for deriving image captions.
- **Why**: Determines how captions are generated for training images.

---

## 🎛 Training Parameters

### `--num_train_epochs`

- **What**: Number of training epochs.
- **Why**: Determines the duration of the training process.

### `--train_batch_size`

- **What**: Batch size for the training data loader.
- **Why**: Affects the model's performance and training speed.

---

## 🛠 Advanced Optimizations

### `--gradient_accumulation_steps`

- **What**: Number of update steps to accumulate before performing a backward/update pass.
- **Why**: Useful for handling larger models or datasets.

### `--learning_rate`

- **What**: Initial learning rate after potential warmup.
- **Why**: Affects the speed and quality of model training.

---

## 🔄 Checkpointing and Resumption

### `--checkpointing_steps`

- **What**: Interval at which training state checkpoints are saved.
- **Why**: Useful for resuming training and for inference.

### `--resume_from_checkpoint`

- **What**: Specifies if and from where to resume training.
- **Why**: Allows you to continue training from a saved state, either manually specified or the latest available.

---

## 📊 Logging and Monitoring

### `--logging_dir`

- **What**: Directory for TensorBoard logs.
- **Why**: Allows you to monitor training progress and performance metrics.

### `--report_to`

- **What**: Specifies the platform for reporting results and logs.
- **Why**: Enables integration with platforms like TensorBoard, wandb, or comet_ml for monitoring.

---

This is a basic overview meant to help you get started. For a complete list of options and more detailed explanations, please refer to the full specification:

```
usage: train_sdxl.py [-h] [--snr_gamma SNR_GAMMA] --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                     [--pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH] [--prediction_type {epsilon,v_prediction,sample}]
                     [--training_scheduler_timestep_spacing {leading,linspace,trailing}] [--inference_scheduler_timestep_spacing {leading,linspace,trailing}]
                     [--rescale_betas_zero_snr] [--vae_dtype VAE_DTYPE] [--revision REVISION] [--tokenizer_name TOKENIZER_NAME] --instance_data_dir
                     INSTANCE_DATA_DIR [--data_backend {local,aws}] [--apply_dataset_padding] [--aws_bucket_name AWS_BUCKET_NAME]
                     [--aws_endpoint_url AWS_ENDPOINT_URL] [--aws_region_name AWS_REGION_NAME] [--aws_access_key_id AWS_ACCESS_KEY_ID]
                     [--aws_secret_access_key AWS_SECRET_ACCESS_KEY] [--cache_dir CACHE_DIR] [--dataset_name DATASET_NAME]
                     [--dataset_config_name DATASET_CONFIG_NAME] [--image_column IMAGE_COLUMN] [--image_prompt_column IMAGE_PROMPT_COLUMN]
                     [--seen_state_path SEEN_STATE_PATH] [--state_path STATE_PATH] [--caption_strategy {filename,textfile,instance_prompt}]
                     [--instance_prompt INSTANCE_PROMPT] [--output_dir OUTPUT_DIR] [--seed SEED] [--seed_for_each_device] [--resolution RESOLUTION]
                     [--minimum_image_size MINIMUM_IMAGE_SIZE] [--crops_coords_top_left_h CROPS_COORDS_TOP_LEFT_H]
                     [--crops_coords_top_left_w CROPS_COORDS_TOP_LEFT_W] [--center_crop] [--random_flip] [--train_text_encoder]
                     [--train_batch_size TRAIN_BATCH_SIZE] [--num_train_epochs NUM_TRAIN_EPOCHS] [--max_train_samples MAX_TRAIN_SAMPLES]
                     [--max_train_steps MAX_TRAIN_STEPS] [--checkpointing_steps CHECKPOINTING_STEPS] [--checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT]
                     [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--gradient_checkpointing]
                     [--learning_rate LEARNING_RATE] [--scale_lr] [--lr_scheduler LR_SCHEDULER] [--lr_warmup_steps LR_WARMUP_STEPS]
                     [--lr_num_cycles LR_NUM_CYCLES] [--lr_power LR_POWER] [--use_ema] [--non_ema_revision NON_EMA_REVISION] [--use_8bit_adam]
                     [--use_adafactor_optimizer] [--use_dadapt_optimizer] [--dadaptation_learning_rate DADAPTATION_LEARNING_RATE]
                     [--dataloader_num_workers DATALOADER_NUM_WORKERS] [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                     [--adam_weight_decay ADAM_WEIGHT_DECAY] [--adam_epsilon ADAM_EPSILON] [--max_grad_norm MAX_GRAD_NORM] [--push_to_hub]
                     [--hub_token HUB_TOKEN] [--hub_model_id HUB_MODEL_ID] [--logging_dir LOGGING_DIR] [--allow_tf32] [--report_to REPORT_TO]
                     [--tracker_run_name TRACKER_RUN_NAME] [--tracker_project_name TRACKER_PROJECT_NAME] [--validation_prompt VALIDATION_PROMPT]
                     [--validation_prompt_library] [--user_prompt_library USER_PROMPT_LIBRARY] [--num_validation_images NUM_VALIDATION_IMAGES]
                     [--validation_steps VALIDATION_STEPS] [--validation_resolution VALIDATION_RESOLUTION] [--mixed_precision {no,fp16,bf16}]
                     [--local_rank LOCAL_RANK] [--enable_xformers_memory_efficient_attention] [--set_grads_to_none] [--noise_offset NOISE_OFFSET]
                     [--validation_epochs VALIDATION_EPOCHS] [--validation_guidance VALIDATION_GUIDANCE]
                     [--validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE] [--validation_randomize] [--validation_seed VALIDATION_SEED]
                     [--fully_unload_text_encoder] [--freeze_encoder_before FREEZE_ENCODER_BEFORE] [--freeze_encoder_after FREEZE_ENCODER_AFTER]
                     [--freeze_encoder_strategy FREEZE_ENCODER_STRATEGY] [--print_filenames] [--debug_aspect_buckets] [--debug_dataset_loader]
                     [--freeze_encoder] [--text_encoder_limit TEXT_ENCODER_LIMIT] [--prepend_instance_prompt] [--only_instance_prompt]
                     [--caption_dropout_interval CAPTION_DROPOUT_INTERVAL] [--conditioning_dropout_probability CONDITIONING_DROPOUT_PROBABILITY]
                     [--caption_dropout_probability CAPTION_DROPOUT_PROBABILITY] [--input_pertubation INPUT_PERTUBATION]
                     [--use_original_images USE_ORIGINAL_IMAGES] [--delete_unwanted_images] [--delete_problematic_images] [--offset_noise]
                     [--learning_rate_end LEARNING_RATE_END]

The following SimpleTuner command-line options are available:

options:
  -h, --help            show this help message and exit
  --snr_gamma SNR_GAMMA
                        SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from huggingface.co/models.
  --pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH
                        Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.
  --prediction_type {epsilon,v_prediction,sample}
                        The type of prediction to use for the VAE. Choose between ['epsilon', 'v_prediction', 'sample']. For SD 2.1-v, this is v_prediction.
                        For 2.1-base, it is epsilon. SDXL is generally epsilon. SD 1.5 is epsilon.
  --training_scheduler_timestep_spacing {leading,linspace,trailing}
                        Spacing timesteps can fundamentally alter the course of history. Er, I mean, your model weights. For all training, including terminal
                        SNR, it would seem that 'leading' is the right choice. However, for inference in terminal SNR models, 'trailing' is the correct choice.
  --inference_scheduler_timestep_spacing {leading,linspace,trailing}
                        The Bytedance paper on zero terminal SNR recommends inference using 'trailing'.
  --rescale_betas_zero_snr
                        If set, will rescale the betas to zero terminal SNR. This is recommended for training with v_prediction. For epsilon, this might help
                        with fine details, but will not result in contrast improvements.
  --vae_dtype VAE_DTYPE
                        The dtype of the VAE model. Choose between ['default', 'fp16', 'fp32', 'bf16'].The default VAE dtype is float32, due to NaN issues in
                        SDXL 1.0.
  --revision REVISION   Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be float32 precision.
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as model_name
  --instance_data_dir INSTANCE_DATA_DIR
                        A folder containing the training data. Folder contents must either follow the structure described in the SimpleTuner documentation
                        (https://github.com/bghira/SimpleTuner), or the structure described in https://huggingface.co/docs/datasets/image_dataset#imagefolder.
                        For 🤗 Datasets in particular, a `metadata.jsonl` file must exist to provide the captions for the images. For SimpleTuner layout, the
                        images can be in subfolders. No particular config is required. Ignored if `dataset_name` is specified.
  --data_backend {local,aws}
                        The data backend to use. Choose between ['local', 'aws']. Default: local. If using AWS, you must set the AWS_ACCESS_KEY_ID and
                        AWS_SECRET_ACCESS_KEY environment variables.
  --apply_dataset_padding
                        If set, will apply padding to the dataset to ensure that the number of images is divisible by the batch. This has some side-effects
                        (especially on smaller datasets) of over-sampling and overly repeating images.
  --aws_bucket_name AWS_BUCKET_NAME
                        The AWS bucket name to use.
  --aws_endpoint_url AWS_ENDPOINT_URL
                        The AWS server to use. If not specified, will use the default server for the region specified. For Wasabi, use
                        https://s3.wasabisys.com.
  --aws_region_name AWS_REGION_NAME
                        The AWS region to use. If not specified, will use the default region for the server specified. For example, if you specify
                        's3.amazonaws.com', the default region will be 'us-east-1'.
  --aws_access_key_id AWS_ACCESS_KEY_ID
                        The AWS access key ID.
  --aws_secret_access_key AWS_SECRET_ACCESS_KEY
                        The AWS secret access key.
  --cache_dir CACHE_DIR
                        The directory where the downloaded models and datasets will be stored.
  --dataset_name DATASET_NAME
                        The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private, dataset). It can also be a path
                        pointing to a local copy of a dataset in your filesystem, or to a folder containing files that 🤗 Datasets can understand.
  --dataset_config_name DATASET_CONFIG_NAME
                        The config of the Dataset, leave as None if there's only one config.
  --image_column IMAGE_COLUMN
                        The column of the dataset containing the original image on which edits where made.
  --image_prompt_column IMAGE_PROMPT_COLUMN
                        The column of the dataset containing the caption.
  --seen_state_path SEEN_STATE_PATH
                        Where the JSON document containing the state of the seen images is stored. This helps ensure we do not repeat images too many times.
  --state_path STATE_PATH
                        A JSON document containing the current state of training, will be placed here.
  --caption_strategy {filename,textfile,instance_prompt}
                        The default captioning strategy, 'filename', will use the filename as the caption, after stripping some characters like underscores.The
                        'textfile' strategy will use the contents of a text file with the same name as the image.
  --instance_prompt INSTANCE_PROMPT
                        This is unused. Filenames will be the captions instead.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written.
  --seed SEED           A seed for reproducible training.
  --seed_for_each_device
                        If provided, a unique seed will be used for each GPU. This is done deterministically, so that each GPU will receive the same seed
                        across invocations.
  --resolution RESOLUTION
                        The resolution for input images, all the images in the train/validation dataset will be resized to this resolution
  --minimum_image_size MINIMUM_IMAGE_SIZE
                        The minimum resolution for both sides of input images. If --delete_unwanted_images is set, images smaller than this will be DELETED.
  --crops_coords_top_left_h CROPS_COORDS_TOP_LEFT_H
                        Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet.
  --crops_coords_top_left_w CROPS_COORDS_TOP_LEFT_W
                        Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet.
  --center_crop         Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized
                        to the resolution first before cropping.
  --random_flip         whether to randomly flip images horizontally
  --train_text_encoder  Whether to train the text encoder. If set, the text encoder should be float32 precision.
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size (per device) for the training dataloader.
  --num_train_epochs NUM_TRAIN_EPOCHS
  --max_train_samples MAX_TRAIN_SAMPLES
                        For debugging purposes or quicker training, truncate the number of training examples to this value if set. Currently untested.
  --max_train_steps MAX_TRAIN_STEPS
                        Total number of training steps to perform. If provided, overrides num_train_epochs.
  --checkpointing_steps CHECKPOINTING_STEPS
                        Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`.
                        In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference.Using a
                        checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components.See
                        https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by
                        stepinstructions.
  --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
                        Max number of checkpoints to store.
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `"latest"` to
                        automatically select the last available checkpoint.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass.
  --gradient_checkpointing
                        Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
  --learning_rate LEARNING_RATE
                        Initial learning rate (after the potential warmup period) to use.
  --scale_lr            Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
  --lr_scheduler LR_SCHEDULER
                        The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                        "constant_with_warmup"]
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps for the warmup in the lr scheduler.
  --lr_num_cycles LR_NUM_CYCLES
                        Number of hard resets of the lr in cosine_with_restarts scheduler.
  --lr_power LR_POWER   Power factor of the polynomial scheduler.
  --use_ema             Whether to use EMA (exponential moving average) model.
  --non_ema_revision NON_EMA_REVISION
                        Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or remote repository specified
                        with --pretrained_model_name_or_path.
  --use_8bit_adam       Whether or not to use 8-bit Adam from bitsandbytes.
  --use_adafactor_optimizer
                        Whether or not to use the Adafactor optimizer.
  --use_dadapt_optimizer
                        Whether or not to use the discriminator adaptation optimizer.
  --dadaptation_learning_rate DADAPTATION_LEARNING_RATE
                        Learning rate for the discriminator adaptation. Default: 1.0
  --dataloader_num_workers DATALOADER_NUM_WORKERS
                        Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
  --adam_beta1 ADAM_BETA1
                        The beta1 parameter for the Adam optimizer.
  --adam_beta2 ADAM_BETA2
                        The beta2 parameter for the Adam optimizer.
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        Weight decay to use.
  --adam_epsilon ADAM_EPSILON
                        Epsilon value for the Adam optimizer
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --push_to_hub         Whether or not to push the model to the Hub.
  --hub_token HUB_TOKEN
                        The token to use to push to the Model Hub.
  --hub_model_id HUB_MODEL_ID
                        The name of the repository to keep in sync with the local `output_dir`.
  --logging_dir LOGGING_DIR
                        [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
  --allow_tf32          Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see
                        https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
  --report_to REPORT_TO
                        The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use
                        `"all"` to report to all integrations.
  --tracker_run_name TRACKER_RUN_NAME
                        The name of the run to track with the tracker.
  --tracker_project_name TRACKER_PROJECT_NAME
                        The name of the project for WandB or Tensorboard.
  --validation_prompt VALIDATION_PROMPT
                        A prompt that is used during validation to verify that the model is learning.
  --validation_prompt_library
                        If this is provided, the SimpleTuner prompt library will be used to generate multiple images.
  --user_prompt_library USER_PROMPT_LIBRARY
                        This should be a path to the JSON file containing your prompt library. See user_prompt_library.json.example.
  --num_validation_images NUM_VALIDATION_IMAGES
                        Number of images that should be generated during validation with `validation_prompt`.
  --validation_steps VALIDATION_STEPS
                        Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times:
                        `args.num_validation_images` and logging the images.
  --validation_resolution VALIDATION_RESOLUTION
                        Square resolution images will be output at this resolution (256x256).
  --mixed_precision {no,fp16,bf16}
                        Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.
                        Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this
                        argument to override the accelerate config.
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --enable_xformers_memory_efficient_attention
                        Whether or not to use xformers.
  --set_grads_to_none   Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain behaviors, so disable this
                        argument if it causes any problems. More info: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
  --noise_offset NOISE_OFFSET
                        The scale of noise offset. Default: 0.1
  --validation_epochs VALIDATION_EPOCHS
                        Run validation every X epochs.
  --validation_guidance VALIDATION_GUIDANCE
                        CFG value for validation images. Default: 7.5
  --validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE
                        CFG rescale value for validation images. Default: 0.0, max 1.0
  --validation_randomize
                        If supplied, validations will be random, ignoring any seeds.
  --validation_seed VALIDATION_SEED
                        If not supplied, the value for --seed will be used. If neither those nor --validation_randomize are supplied, a seed of zero is used.
  --fully_unload_text_encoder
                        If set, will fully unload the text_encoder from memory when not in use. This currently has the side effect of crashing validations, but
                        it is useful for initiating VAE caching on GPUs that would otherwise be too small.
  --freeze_encoder_before FREEZE_ENCODER_BEFORE
                        When using 'before' strategy, we will freeze layers earlier than this.
  --freeze_encoder_after FREEZE_ENCODER_AFTER
                        When using 'after' strategy, we will freeze layers later than this.
  --freeze_encoder_strategy FREEZE_ENCODER_STRATEGY
                        When freezing the text_encoder, we can use the 'before', 'between', or 'after' strategy. The 'between' strategy will freeze layers
                        between those two values, leaving the outer layers unfrozen. The default strategy is to freeze all layers from 17 up. This can be
                        helpful when fine-tuning Stable Diffusion 2.1 on a new style.
  --print_filenames     If any image files are stopping the process eg. due to corruption or truncation, this will help identify which is at fault.
  --debug_aspect_buckets
                        If set, will print excessive debugging for aspect bucket operations.
  --debug_dataset_loader
                        If set, will print excessive debugging for data loader operations.
  --freeze_encoder      Whether or not to freeze the text_encoder. The default is true.
  --text_encoder_limit TEXT_ENCODER_LIMIT
                        When training the text_encoder, we want to limit how long it trains for to avoid catastrophic loss.
  --prepend_instance_prompt
                        When determining the captions from the filename, prepend the instance prompt as an enforced keyword.
  --only_instance_prompt
                        Use the instance prompt instead of the caption from filename.
  --caption_dropout_interval CAPTION_DROPOUT_INTERVAL
                        Every X steps, we will drop the caption from the input to assist in classifier-free guidance training. When StabilityAI trained Stable
                        Diffusion, a value of 10 was used. Very high values might be useful to do some sort of enforced style training. Default value is zero,
                        maximum value is 100.
  --conditioning_dropout_probability CONDITIONING_DROPOUT_PROBABILITY
                        Conditioning dropout probability. Experimental. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.
  --caption_dropout_probability CAPTION_DROPOUT_PROBABILITY
                        Caption dropout probability. Same as caption_dropout_interval, but this is for SDXL.
  --input_pertubation INPUT_PERTUBATION
                        The scale of input pretubation. Recommended 0.1.
  --use_original_images USE_ORIGINAL_IMAGES
                        When this option is provided, image cropping and processing will be disabled. It is a good idea to use this with caution, for training
                        multiple aspect ratios.
  --delete_unwanted_images
                        If set, will delete images that are not of a minimum size to save on disk space for large training runs. Default behaviour: Unset,
                        remove images from bucket only.
  --delete_problematic_images
                        If set, any images that error out during load will be removed from the underlying storage medium. This is useful to prevent repeatedly
                        attempting to cache bad files on a cloud bucket.
  --offset_noise        Fine-tuning against a modified noise See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information.
  --learning_rate_end LEARNING_RATE_END
                        A polynomial learning rate will end up at this value after the specified number of warmup steps.
```