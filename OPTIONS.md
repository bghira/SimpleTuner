# SimpleTuner Training Script Options

## Overview

This guide provides a user-friendly breakdown of the command-line options available in SimpleTuner's `train_sdxl.py` script. These options offer a high degree of customization, allowing you to train your model to suit your specific requirements.

---

## 🌟 Core Model Configuration

### `--pretrained_model_name_or_path`

- **What**: Path to the pretrained model or its identifier from huggingface.co/models.
- **Why**: To specify the base model you'll start training from.

### `--hub_model_id`

- **What**: The name of the Huggingface Hub model and local results directory.
- **Why**: This value is used as the directory name under the location specified as `--output_dir`. If `--push_to_hub` is provided, this will become the name of the model on Huggingface Hub.
---

### `--push_to_hub`

- **What**: If provided, your model will be uploaded to [Huggingface Hub](https://huggingface.co) once training completes.

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

- **What**: Input image resolution. Can be expressed as pixels, or megapixels.
- **Why**: All images in the dataset will have their smaller edge resized to this resolution for training. If you use 1024px, the images may become very large and use an excessive amount of VRAM. The best mileage tends to be a 768 or 800 pixel base resolution, although 512px resolution training can really pay off with SDXL in particular.

### `--resolution_type`

- **What**: This tells SimpleTuner whether to use `area` size calculations or `pixel` edge calculations.
- **Why**: SimpleTuner's default `pixel` behaviour is to resize the image, keeping the aspect ratio. Setting the type to `area` instead uses the given megapixel value as the target size for the image, keeping the aspect ratio.

### `--validation_resolution`

- **What**: Output image resolution, measured in pixels.
- **Why**: All images generated during validation will be this resolution. Useful if the model is being trained with a different resolution.

### `--caption_strategy`

- **What**: Strategy for deriving image captions. __Choices__: `textfile`, `filename`
- **Why**: Determines how captions are generated for training images. `textfile` will use the contents of a `.txt` file with the same filename as the image, and `filename` will apply some cleanup to the filename before using it as the caption.

---

## 🎛 Training Parameters

### `--num_train_epochs`

- **What**: Number of training epochs (the number of times that all images are seen)
- **Why**: Determines the number of image repeats, which impacts the duration of the training process. More epochs tends to result in overfitting, but might be required to pick up the concepts you wish to train in. A reasonable value might be from 5 to 50.

### `--max_train_steps`

- **What**: Number of training steps to exit training after.
- **Why**: Useful for shortening the length of training.

### `--train_batch_size`

- **What**: Batch size for the training data loader.
- **Why**: Affects the model's memory consumption, convergence quality, and training speed. The higher the batch size, the better the results will be, but a very high batch size might result in overfitting or destabilized training, as well as increasing the duration of the training session unnecessarily. Experimentation is warranted, but in general, you want to try to max out your video memory while not decreasing the training speed.

---

## 🛠 Advanced Optimizations

### `--gradient_accumulation_steps`

- **What**: Number of update steps to accumulate before performing a backward/update pass.
- **Why**: Useful for handling larger models or datasets.

### `--learning_rate`

- **What**: Initial learning rate after potential warmup.
- **Why**: The learning rate behaves as a sort of "step size" for gradient updates - too high, and we overstep the solution. Too low, and we never reach the ideal solution. A minimal value might be as low as `4e-7` and a maximal value would likely be as high as `1e-5`. When a higher learning rate is used, it's advantageous to use an EMA network with a learning rate warmup - see `--use_ema`, `--lr_warmup_steps`, and `--lr_scheduler`.

### `--lr_scheduler`

- **What**: How to scale the learning rate over time.
- **Choices**: constant, constant_with_warmup, cosine, cosine_with_restarts, polynomial, linear
- **Why**: Models benefit from continual learning rate adjustments to further explore the loss landscape. A cosine schedule is used as the default; this allows the training to smoothly transition between two extremes. If using a constant learning rate, it is common to select a too-high or too-low value, causing divergence (too high) or getting stuck in a local minima (too low).

### `--snr_gamma`

- **What**: Utilising min-SNR weighted loss factor.
- **Why**: Minimum SNR gamma weights the loss factor of a timestep by its position in the schedule. Overly noisy timesteps have their contributions reduced, and less-noisy timesteps have it increased. Value recommended by the original paper is **5** but you can use values as low as **1** or as high as **20**, typically seen as the maximum value - beyond a value of 20, the math does not change things much. A value of **1** is the strongest.

---

## 🔄 Checkpointing and Resumption

### `--checkpointing_steps`

- **What**: Interval at which training state checkpoints are saved.
- **Why**: Useful for resuming training and for inference. Every _n_ iterations, a partial checkpoint will be saved in the `.safetensors` format, via the Diffusers filesystem layout.

### `--resume_from_checkpoint`

- **What**: Specifies if and from where to resume training.
- **Why**: Allows you to continue training from a saved state, either manually specified or the latest available. A checkpoint is composed of a `unet` and optionally, an `ema_unet`. The `unet` may be dropped into any Diffusers layout SDXL model, allowing it to be used as a normal model would.

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
usage: train_sdxl.py [-h] [--snr_gamma SNR_GAMMA]
                     --pretrained_model_name_or_path
                     PRETRAINED_MODEL_NAME_OR_PATH
                     [--pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH]
                     [--prediction_type {epsilon,v_prediction,sample}]
                     [--snr_weight SNR_WEIGHT]
                     [--training_scheduler_timestep_spacing {leading,linspace,trailing}]
                     [--inference_scheduler_timestep_spacing {leading,linspace,trailing}]
                     [--timestep_bias_strategy {earlier,later,range,none}]
                     [--timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER]
                     [--timestep_bias_begin TIMESTEP_BIAS_BEGIN]
                     [--timestep_bias_end TIMESTEP_BIAS_END]
                     [--timestep_bias_portion TIMESTEP_BIAS_PORTION]
                     [--rescale_betas_zero_snr] [--vae_dtype VAE_DTYPE]
                     [--vae_batch_size VAE_BATCH_SIZE] [--keep_vae_loaded]
                     [--skip_file_discovery SKIP_FILE_DISCOVERY]
                     [--revision REVISION] --instance_data_dir
                     INSTANCE_DATA_DIR [--preserve_data_backend_cache]
                     [--cache_dir_text CACHE_DIR_TEXT]
                     [--cache_dir_vae CACHE_DIR_VAE]
                     [--data_backend {local,aws}]
                     [--write_batch_size WRITE_BATCH_SIZE]
                     [--aws_config_file AWS_CONFIG_FILE]
                     [--aws_bucket_name AWS_BUCKET_NAME]
                     [--aws_bucket_image_prefix AWS_BUCKET_IMAGE_PREFIX]
                     [--aws_endpoint_url AWS_ENDPOINT_URL]
                     [--aws_region_name AWS_REGION_NAME]
                     [--aws_access_key_id AWS_ACCESS_KEY_ID]
                     [--aws_secret_access_key AWS_SECRET_ACCESS_KEY]
                     [--cache_dir CACHE_DIR]
                     [--cache_clear_validation_prompts]
                     [--seen_state_path SEEN_STATE_PATH]
                     [--state_path STATE_PATH]
                     [--caption_strategy {filename,textfile,instance_prompt}]
                     [--instance_prompt INSTANCE_PROMPT]
                     [--output_dir OUTPUT_DIR] [--seed SEED]
                     [--seed_for_each_device SEED_FOR_EACH_DEVICE]
                     [--resolution RESOLUTION]
                     [--resolution_type {pixel,area}]
                     [--minimum_image_size MINIMUM_IMAGE_SIZE] [--crop CROP]
                     [--crop_style {center,centre,corner,random}]
                     [--crop_aspect {square,preserve}] [--train_text_encoder]
                     [--train_batch_size TRAIN_BATCH_SIZE]
                     [--num_train_epochs NUM_TRAIN_EPOCHS]
                     [--max_train_steps MAX_TRAIN_STEPS]
                     [--checkpointing_steps CHECKPOINTING_STEPS]
                     [--checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT]
                     [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                     [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                     [--gradient_checkpointing]
                     [--learning_rate LEARNING_RATE] [--lr_scale]
                     [--lr_scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                     [--lr_warmup_steps LR_WARMUP_STEPS]
                     [--lr_num_cycles LR_NUM_CYCLES] [--lr_power LR_POWER]
                     [--use_ema] [--ema_decay EMA_DECAY]
                     [--non_ema_revision NON_EMA_REVISION]
                     [--offload_param_path OFFLOAD_PARAM_PATH]
                     [--use_8bit_adam] [--use_adafactor_optimizer]
                     [--use_dadapt_optimizer]
                     [--dadaptation_learning_rate DADAPTATION_LEARNING_RATE]
                     [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                     [--adam_weight_decay ADAM_WEIGHT_DECAY]
                     [--adam_epsilon ADAM_EPSILON]
                     [--max_grad_norm MAX_GRAD_NORM] [--push_to_hub]
                     [--hub_token HUB_TOKEN] [--hub_model_id HUB_MODEL_ID]
                     [--logging_dir LOGGING_DIR]
                     [--validation_torch_compile VALIDATION_TORCH_COMPILE]
                     [--validation_torch_compile_mode {reduce-overhead,default}]
                     [--allow_tf32] [--report_to REPORT_TO]
                     [--tracker_run_name TRACKER_RUN_NAME]
                     [--tracker_project_name TRACKER_PROJECT_NAME]
                     [--validation_prompt VALIDATION_PROMPT]
                     [--validation_prompt_library]
                     [--user_prompt_library USER_PROMPT_LIBRARY]
                     [--validation_negative_prompt VALIDATION_NEGATIVE_PROMPT]
                     [--num_validation_images NUM_VALIDATION_IMAGES]
                     [--validation_steps VALIDATION_STEPS]
                     [--validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS]
                     [--validation_resolution VALIDATION_RESOLUTION]
                     [--validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc}]
                     [--disable_compel] [--enable_watermark]
                     [--mixed_precision {no,fp16,bf16}]
                     [--local_rank LOCAL_RANK]
                     [--enable_xformers_memory_efficient_attention]
                     [--set_grads_to_none] [--noise_offset NOISE_OFFSET]
                     [--noise_offset_probability NOISE_OFFSET_PROBABILITY]
                     [--validation_guidance VALIDATION_GUIDANCE]
                     [--validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE]
                     [--validation_randomize]
                     [--validation_seed VALIDATION_SEED]
                     [--fully_unload_text_encoder]
                     [--freeze_encoder_before FREEZE_ENCODER_BEFORE]
                     [--freeze_encoder_after FREEZE_ENCODER_AFTER]
                     [--freeze_encoder_strategy FREEZE_ENCODER_STRATEGY]
                     [--print_filenames]
                     [--metadata_update_interval METADATA_UPDATE_INTERVAL]
                     [--debug_aspect_buckets] [--debug_dataset_loader]
                     [--freeze_encoder]
                     [--text_encoder_limit TEXT_ENCODER_LIMIT]
                     [--prepend_instance_prompt] [--only_instance_prompt]
                     [--caption_dropout_probability CAPTION_DROPOUT_PROBABILITY]
                     [--input_perturbation INPUT_PERTURBATION]
                     [--input_perturbation_probability INPUT_PERTURBATION_PROBABILITY]
                     [--delete_unwanted_images] [--delete_problematic_images]
                     [--offset_noise] [--lr_end LR_END]

The following SimpleTuner command-line options are available:

options:
  -h, --help            show this help message and exit
  --snr_gamma SNR_GAMMA
                        SNR weighting gamma to be used if rebalancing the
                        loss. Recommended value is 5.0. More details here:
                        https://arxiv.org/abs/2303.09556.
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from
                        huggingface.co/models.
  --pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH
                        Path to an improved VAE to stabilize training. For
                        more details check out:
                        https://github.com/huggingface/diffusers/pull/4038.
  --prediction_type {epsilon,v_prediction,sample}
                        The type of prediction to use for the u-net. Choose
                        between ['epsilon', 'v_prediction', 'sample']. For SD
                        2.1-v, this is v_prediction. For 2.1-base, it is
                        epsilon. SDXL is generally epsilon. SD 1.5 is epsilon.
  --snr_weight SNR_WEIGHT
                        When training a model using
                        `--prediction_type=sample`, one can supply an SNR
                        weight value to augment the loss with. If a value of
                        0.5 is provided here, the loss is taken half from the
                        SNR and half from the MSE.
  --training_scheduler_timestep_spacing {leading,linspace,trailing}
                        (SDXL Only) Spacing timesteps can fundamentally alter
                        the course of history. Er, I mean, your model weights.
                        For all training, including epsilon, it would seem
                        that 'trailing' is the right choice. SD 2.x always
                        uses 'trailing', but SDXL may do better in its default
                        state when using 'leading'.
  --inference_scheduler_timestep_spacing {leading,linspace,trailing}
                        (SDXL Only) The Bytedance paper on zero terminal SNR
                        recommends inference using 'trailing'. SD 2.x always
                        uses 'trailing', but SDXL may do better in its default
                        state when using 'leading'.
  --timestep_bias_strategy {earlier,later,range,none}
                        The timestep bias strategy, which may help direct the
                        model toward learning low or frequency details.
                        Choices: ['earlier', 'later', 'none']. The default is
                        'none', which means no bias is applied, and training
                        proceeds normally. The value of 'later' will prefer to
                        generate samples for later timesteps.
  --timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER
                        The multiplier for the bias. Defaults to 1.0, which
                        means no bias is applied. A value of 2.0 will double
                        the weight of the bias, and a value of 0.5 will halve
                        it.
  --timestep_bias_begin TIMESTEP_BIAS_BEGIN
                        When using `--timestep_bias_strategy=range`, the
                        beginning timestep to bias. Defaults to zero, which
                        equates to having no specific bias.
  --timestep_bias_end TIMESTEP_BIAS_END
                        When using `--timestep_bias_strategy=range`, the final
                        timestep to bias. Defaults to 1000, which is the
                        number of timesteps that SDXL Base and SD 2.x were
                        trained on.
  --timestep_bias_portion TIMESTEP_BIAS_PORTION
                        The portion of timesteps to bias. Defaults to 0.25,
                        which 25 percent of timesteps will be biased. A value
                        of 0.5 will bias one half of the timesteps. The value
                        provided for `--timestep_bias_strategy` determines
                        whether the biased portions are in the earlier or
                        later timesteps.
  --rescale_betas_zero_snr
                        If set, will rescale the betas to zero terminal SNR.
                        This is recommended for training with v_prediction.
                        For epsilon, this might help with fine details, but
                        will not result in contrast improvements.
  --vae_dtype VAE_DTYPE
                        The dtype of the VAE model. Choose between ['default',
                        'fp16', 'fp32', 'bf16'].The default VAE dtype is
                        float32, due to NaN issues in SDXL 1.0.
  --vae_batch_size VAE_BATCH_SIZE
                        When pre-caching latent vectors, this is the batch
                        size to use. Decreasing this may help with VRAM
                        issues, but if you are at that point of contention,
                        it's possible that your GPU has too little RAM.
                        Default: 4.
  --keep_vae_loaded     If set, will keep the VAE loaded in memory. This can
                        reduce disk churn, but consumes VRAM during the
                        forward pass.
  --skip_file_discovery SKIP_FILE_DISCOVERY
                        Comma-separated values of which stages to skip
                        discovery for. Skipping any stage will speed up
                        resumption, but will increase the risk of errors, as
                        missing images or incorrectly bucketed images may not
                        be caught. 'vae' will skip the VAE cache process,
                        'aspect' will not build any aspect buckets, and 'text'
                        will avoid text embed management. Valid options:
                        aspect, vae, text, metadata.
  --revision REVISION   Revision of pretrained model identifier from
                        huggingface.co/models. Trainable model components
                        should be float32 precision.
  --instance_data_dir INSTANCE_DATA_DIR
                        A folder containing the training data. Folder contents
                        must either follow the structure described in the
                        SimpleTuner documentation
                        (https://github.com/bghira/SimpleTuner), or the
                        structure described in https://huggingface.co/docs/dat
                        asets/image_dataset#imagefolder. For 🤗 Datasets in
                        particular, a `metadata.jsonl` file must exist to
                        provide the captions for the images. For SimpleTuner
                        layout, the images can be in subfolders. No particular
                        config is required.
  --preserve_data_backend_cache
                        For very large cloud storage buckets that will never
                        change, enabling this option will prevent the trainer
                        from scanning it at startup, by preserving the cache
                        files that we generate. Be careful when using this,
                        as, switching datasets can result in the preserved
                        cache being used, which would be problematic.
                        Currently, cache is not stored in the dataset itself
                        but rather, locally. This may change in a future
                        release.
  --cache_dir_text CACHE_DIR_TEXT
                        This is the path to a local directory that will
                        contain your text embed cache.
  --cache_dir_vae CACHE_DIR_VAE
                        This is the path to a local directory that will
                        contain your VAE outputs. Unlike the text embed cache,
                        your VAE latents will be stored in the AWS data
                        backend. If the AWS backend is in use, this will be a
                        prefix for the bucket's VAE cache entries.
  --data_backend {local,aws}
                        The data backend to use. Choose between ['local',
                        'aws']. Default: local. If using AWS, you must set the
                        AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
                        environment variables.
  --write_batch_size WRITE_BATCH_SIZE
                        When using certain storage backends, it is better to
                        batch smaller writes rather than continuous
                        dispatching. In SimpleTuner, write batching is
                        currently applied during VAE caching, when many small
                        objects are written. This mostly applies to S3, but
                        some shared server filesystems may benefit as well,
                        eg. Ceph. Default: 64.
  --aws_config_file AWS_CONFIG_FILE
                        Path to the AWS configuration file in JSON format.
                        Config key names are the same as SimpleTuner option
                        counterparts.
  --aws_bucket_name AWS_BUCKET_NAME
                        The AWS bucket name to use.
  --aws_bucket_image_prefix AWS_BUCKET_IMAGE_PREFIX
                        Instead of using --instance_data_dir, AWS S3 relies on
                        aws_bucket_*_prefix parameters. When provided, this
                        parameter will be prepended to the image path.
  --aws_endpoint_url AWS_ENDPOINT_URL
                        The AWS server to use. If not specified, will use the
                        default server for the region specified. For Wasabi,
                        use https://s3.wasabisys.com.
  --aws_region_name AWS_REGION_NAME
                        The AWS region to use. If not specified, will use the
                        default region for the server specified. For example,
                        if you specify 's3.amazonaws.com', the default region
                        will be 'us-east-1'.
  --aws_access_key_id AWS_ACCESS_KEY_ID
                        The AWS access key ID.
  --aws_secret_access_key AWS_SECRET_ACCESS_KEY
                        The AWS secret access key.
  --cache_dir CACHE_DIR
                        The directory where the downloaded models and datasets
                        will be stored.
  --cache_clear_validation_prompts
                        When provided, any validation prompt entries in the
                        text embed cache will be recreated. This is useful if
                        you've modified any of the existing prompts, or,
                        disabled/enabled Compel, via `--disable_compel`
  --seen_state_path SEEN_STATE_PATH
                        Where the JSON document containing the state of the
                        seen images is stored. This helps ensure we do not
                        repeat images too many times.
  --state_path STATE_PATH
                        A JSON document containing the current state of
                        training, will be placed here.
  --caption_strategy {filename,textfile,instance_prompt}
                        The default captioning strategy, 'filename', will use
                        the filename as the caption, after stripping some
                        characters like underscores.The 'textfile' strategy
                        will use the contents of a text file with the same
                        name as the image.
  --instance_prompt INSTANCE_PROMPT
                        This is unused. Filenames will be the captions
                        instead.
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
  --seed SEED           A seed for reproducible training.
  --seed_for_each_device SEED_FOR_EACH_DEVICE
                        By default, a unique seed will be used for each GPU.
                        This is done deterministically, so that each GPU will
                        receive the same seed across invocations. If
                        --seed_for_each_device=false is provided, then we will
                        use the same seed across all GPUs, which will almost
                        certainly result in the over-sampling of inputs on
                        larger datasets.
  --resolution RESOLUTION
                        The resolution for input images, all the images in the
                        train/validation dataset will be resized to this
                        resolution. If using --resolution_type=area, this
                        float value represents megapixels.
  --resolution_type {pixel,area}
                        Resizing images maintains aspect ratio. This defines
                        the resizing strategy. If 'pixel', the images will be
                        resized to the resolution by pixel edge. If 'area',
                        the images will be resized so the pixel area is this
                        many megapixels.
  --minimum_image_size MINIMUM_IMAGE_SIZE
                        The minimum resolution for both sides of input images.
                        If --delete_unwanted_images is set, images smaller
                        than this will be DELETED. The default value is None,
                        which means no minimum resolution is enforced. If this
                        option is not provided, it is possible that images
                        will be destructively upsampled, harming model
                        performance.
  --crop CROP           Whether to crop the input images to the resolution. If
                        not set, the images will be downsampled instead. When
                        cropping is enabled, the images will not be resized
                        before cropping. If training SDXL, the VAE cache and
                        aspect bucket cache will need to be (re)built so they
                        include crop coordinates.
  --crop_style {center,centre,corner,random}
                        When --crop is provided, a crop style may be defined
                        that designates which part of an image to crop to. The
                        old behaviour was to crop to the lower right corner,
                        but this isn't always ideal for training. The default
                        is 'random', which will locate a random segment of the
                        image matching the given resolution.
  --crop_aspect {square,preserve}
                        When --crop is supplied, the default behaviour is to
                        crop to square images, which greatly simplifies aspect
                        bucketing. However, --crop_aspect may be set to
                        'preserve', which will crop based on the
                        --resolution_type value. If --resolution_type=area,
                        the crop will be equal to the target pixel area. If
                        --resolution_type=pixel, the crop will have the
                        smaller edge equal to the value of --resolution.
  --train_text_encoder  (SD 2.x only) Whether to train the text encoder. If
                        set, the text encoder should be float32 precision.
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size (per device) for the training dataloader.
  --num_train_epochs NUM_TRAIN_EPOCHS
  --max_train_steps MAX_TRAIN_STEPS
                        Total number of training steps to perform. If
                        provided, overrides num_train_epochs.
  --checkpointing_steps CHECKPOINTING_STEPS
                        Save a checkpoint of the training state every X
                        updates. Checkpoints can be used for resuming training
                        via `--resume_from_checkpoint`. In the case that the
                        checkpoint is better than the final trained model, the
                        checkpoint can also be used for inference.Using a
                        checkpoint for inference requires separate loading of
                        the original pipeline and the individual checkpointed
                        model components.See https://huggingface.co/docs/diffu
                        sers/main/en/training/dreambooth#performing-inference-
                        using-a-saved-checkpoint for step by stepinstructions.
  --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
                        Max number of checkpoints to store.
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Whether training should be resumed from a previous
                        checkpoint. Use a path saved by
                        `--checkpointing_steps`, or `"latest"` to
                        automatically select the last available checkpoint.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --gradient_checkpointing
                        Whether or not to use gradient checkpointing to save
                        memory at the expense of slower backward pass.
  --learning_rate LEARNING_RATE
                        Initial learning rate (after the potential warmup
                        period) to use.
  --lr_scale            Scale the learning rate by the number of GPUs,
                        gradient accumulation steps, and batch size.
  --lr_scheduler {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        The scheduler type to use. Default: cosine
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps for the warmup in the lr scheduler.
  --lr_num_cycles LR_NUM_CYCLES
                        Number of hard resets of the lr in
                        cosine_with_restarts scheduler.
  --lr_power LR_POWER   Power factor of the polynomial scheduler.
  --use_ema             Whether to use EMA (exponential moving average) model.
  --ema_decay EMA_DECAY
                        The closer to 0.9999 this gets, the less updates will
                        occur over time. Setting it to a lower value, such as
                        0.990, will allow greater influence of later updates.
  --non_ema_revision NON_EMA_REVISION
                        Revision of pretrained non-ema model identifier. Must
                        be a branch, tag or git identifier of the local or
                        remote repository specified with
                        --pretrained_model_name_or_path.
  --offload_param_path OFFLOAD_PARAM_PATH
                        When using DeepSpeed ZeRo stage 2 or 3 with NVMe
                        offload, this may be specified to provide a path for
                        the offload.
  --use_8bit_adam       Whether or not to use 8-bit Adam from bitsandbytes.
  --use_adafactor_optimizer
                        Whether or not to use the Adafactor optimizer.
  --use_dadapt_optimizer
                        Whether or not to use the discriminator adaptation
                        optimizer.
  --dadaptation_learning_rate DADAPTATION_LEARNING_RATE
                        Learning rate for the discriminator adaptation.
                        Default: 1.0
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
                        The name of the repository to keep in sync with the
                        local `output_dir`.
  --logging_dir LOGGING_DIR
                        [TensorBoard](https://www.tensorflow.org/tensorboard)
                        log directory. Will default to
                        *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
  --validation_torch_compile VALIDATION_TORCH_COMPILE
                        Supply `--validation_torch_compile=true` to enable the
                        use of torch.compile() on the validation pipeline. For
                        some setups, torch.compile() may error out. This is
                        dependent on PyTorch version, phase of the moon, but
                        if it works, you should leave it enabled for a great
                        speed-up.
  --validation_torch_compile_mode {reduce-overhead,default}
                        PyTorch provides different modes for the Torch
                        Inductor when compiling graphs. reduce-overhead, the
                        default mode, provides the most benefit.
  --allow_tf32          Whether or not to allow TF32 on Ampere GPUs. Can be
                        used to speed up training. For more information, see h
                        ttps://pytorch.org/docs/stable/notes/cuda.html#tensorf
                        loat-32-tf32-on-ampere-devices
  --report_to REPORT_TO
                        The integration to report the results and logs to.
                        Supported platforms are `"tensorboard"` (default),
                        `"wandb"` and `"comet_ml"`. Use `"all"` to report to
                        all integrations.
  --tracker_run_name TRACKER_RUN_NAME
                        The name of the run to track with the tracker.
  --tracker_project_name TRACKER_PROJECT_NAME
                        The name of the project for WandB or Tensorboard.
  --validation_prompt VALIDATION_PROMPT
                        A prompt that is used during validation to verify that
                        the model is learning.
  --validation_prompt_library
                        If this is provided, the SimpleTuner prompt library
                        will be used to generate multiple images.
  --user_prompt_library USER_PROMPT_LIBRARY
                        This should be a path to the JSON file containing your
                        prompt library. See user_prompt_library.json.example.
  --validation_negative_prompt VALIDATION_NEGATIVE_PROMPT
                        When validating images, a negative prompt may be used
                        to guide the model away from certain features. When
                        this value is set to --validation_negative_prompt='',
                        no negative guidance will be applied. Default: blurry,
                        cropped, ugly
  --num_validation_images NUM_VALIDATION_IMAGES
                        Number of images that should be generated during
                        validation with `validation_prompt`.
  --validation_steps VALIDATION_STEPS
                        Run validation every X steps. Validation consists of
                        running the prompt `args.validation_prompt` multiple
                        times: `args.num_validation_images` and logging the
                        images.
  --validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS
                        The default scheduler, DDIM, benefits from more steps.
                        UniPC can do well with just 10-15. For more speed
                        during validations, reduce this value. For better
                        quality, increase it. For model distilation, you will
                        likely want to keep this low.
  --validation_resolution VALIDATION_RESOLUTION
                        Square resolution images will be output at this
                        resolution (256x256).
  --validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc}
                        When validating the model at inference time, a
                        different scheduler may be chosen. UniPC can offer
                        better speed, and Euler A can put up with
                        instabilities a bit better. For zero-terminal SNR
                        models, DDIM is the best choice. Choices: ['ddim',
                        'ddpm', 'euler', 'euler-a', 'unipc'], Default: ddim
  --disable_compel      If provided, prompts will be handled using the typical
                        prompt encoding strategy. Otherwise, the default
                        behaviour is to use Compel for prompt embed
                        generation.
  --enable_watermark    The SDXL 0.9 and 1.0 licenses both require a watermark
                        be used to identify any images created to be shared.
                        Since the images created during validation typically
                        are not shared, and we want the most accurate results,
                        this watermarker is disabled by default. If you are
                        sharing the validation images, it is up to you to
                        ensure that you are complying with the license,
                        whether that is through this watermarker, or another.
  --mixed_precision {no,fp16,bf16}
                        Whether to use mixed precision. Choose between fp16
                        and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and
                        an Nvidia Ampere GPU. Default to the value of
                        accelerate config of the current system or the flag
                        passed with the `accelerate.launch` command. Use this
                        argument to override the accelerate config.
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --enable_xformers_memory_efficient_attention
                        Whether or not to use xformers.
  --set_grads_to_none   Save more memory by using setting grads to None
                        instead of zero. Be aware, that this changes certain
                        behaviors, so disable this argument if it causes any
                        problems. More info: https://pytorch.org/docs/stable/g
                        enerated/torch.optim.Optimizer.zero_grad.html
  --noise_offset NOISE_OFFSET
                        The scale of noise offset. Default: 0.1
  --noise_offset_probability NOISE_OFFSET_PROBABILITY
                        When training with --offset_noise, the value of
                        --noise_offset will only be applied probabilistically.
                        The default behaviour is for offset noise (if enabled)
                        to be applied 25 percent of the time.
  --validation_guidance VALIDATION_GUIDANCE
                        CFG value for validation images. Default: 7.5
  --validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE
                        CFG rescale value for validation images. Default: 0.0,
                        max 1.0
  --validation_randomize
                        If supplied, validations will be random, ignoring any
                        seeds.
  --validation_seed VALIDATION_SEED
                        If not supplied, the value for --seed will be used. If
                        neither those nor --validation_randomize are supplied,
                        a seed of zero is used.
  --fully_unload_text_encoder
                        If set, will fully unload the text_encoder from memory
                        when not in use. This currently has the side effect of
                        crashing validations, but it is useful for initiating
                        VAE caching on GPUs that would otherwise be too small.
  --freeze_encoder_before FREEZE_ENCODER_BEFORE
                        When using 'before' strategy, we will freeze layers
                        earlier than this.
  --freeze_encoder_after FREEZE_ENCODER_AFTER
                        When using 'after' strategy, we will freeze layers
                        later than this.
  --freeze_encoder_strategy FREEZE_ENCODER_STRATEGY
                        When freezing the text_encoder, we can use the
                        'before', 'between', or 'after' strategy. The
                        'between' strategy will freeze layers between those
                        two values, leaving the outer layers unfrozen. The
                        default strategy is to freeze all layers from 17 up.
                        This can be helpful when fine-tuning Stable Diffusion
                        2.1 on a new style.
  --print_filenames     If any image files are stopping the process eg. due to
                        corruption or truncation, this will help identify
                        which is at fault.
  --metadata_update_interval METADATA_UPDATE_INTERVAL
                        When generating the aspect bucket indicies, we want to
                        save it every X seconds. The default is to save it
                        every 1 hour, such that progress is not lost on
                        clusters where runtime is limited to 6-hour increments
                        (e.g. the JUWELS Supercomputer). The minimum value is
                        60 seconds.
  --debug_aspect_buckets
                        If set, will print excessive debugging for aspect
                        bucket operations.
  --debug_dataset_loader
                        If set, will print excessive debugging for data loader
                        operations.
  --freeze_encoder      Whether or not to freeze the text_encoder. The default
                        is true.
  --text_encoder_limit TEXT_ENCODER_LIMIT
                        When training the text_encoder, we want to limit how
                        long it trains for to avoid catastrophic loss.
  --prepend_instance_prompt
                        When determining the captions from the filename,
                        prepend the instance prompt as an enforced keyword.
  --only_instance_prompt
                        Use the instance prompt instead of the caption from
                        filename.
  --caption_dropout_probability CAPTION_DROPOUT_PROBABILITY
                        Caption dropout will randomly drop captions and, for
                        SDXL, size conditioning inputs based on this
                        probability. When set to a value of 0.1, it will drop
                        approximately 10 percent of the inputs. Maximum
                        recommended value is probably less than 0.5, or 50
                        percent of the inputs. Maximum technical value is 1.0.
                        The default is to use zero caption dropout, though for
                        better generalisation, a value of 0.1 is recommended.
  --input_perturbation INPUT_PERTURBATION
                        The scale of input pretubation. Recommended 0.1.
  --input_perturbation_probability INPUT_PERTURBATION_PROBABILITY
                        While input perturbation can help with training
                        convergence, having it applied all the time is likely
                        damaging. When this value is less than 1.0, any
                        perturbed noise will be applied probabilistically.
                        Default: 0.25
  --delete_unwanted_images
                        If set, will delete images that are not of a minimum
                        size to save on disk space for large training runs.
                        Default behaviour: Unset, remove images from bucket
                        only.
  --delete_problematic_images
                        If set, any images that error out during load will be
                        removed from the underlying storage medium. This is
                        useful to prevent repeatedly attempting to cache bad
                        files on a cloud bucket.
  --offset_noise        Fine-tuning against a modified noise See:
                        https://www.crosslabs.org//blog/diffusion-with-offset-
                        noise for more information.
  --lr_end LR_END       A polynomial learning rate will end up at this value
                        after the specified number of warmup steps.
```