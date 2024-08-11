# SimpleTuner Training Script Options

## Overview

This guide provides a user-friendly breakdown of the command-line options available in SimpleTuner's `train.py` script. These options offer a high degree of customization, allowing you to train your model to suit your specific requirements.

---

## 🌟 Core Model Configuration

### `--model_type`

- **What**: Choices: lora, full, deepfloyd, deepfloyd-lora, deepfloyd-stage2, deepfloyd-stage2-lora. Default: lora
- **Why**: Select whether a LoRA or full fine-tune are created. LoRA only supported for SDXL.

## `--flux`

- **What**: Enable Flux training style.
- **Why**: Flux is an enormous model and uses flow-matching. We must take careful considerations when handling its text embeds and validations.

### `--sd3`

- **What**: Enable Stable Diffusion 3 training quirks/overrides.
- **Why**: SD3 has three text encoders, it's pretty hefty and needs specific validation-time options considered. The equivalent option for this in the `config/config.env` environment file is `STABLE_DIFFUSION_3`.

### `--pixart_sigma`

- **What**: Enable PixArt Sigma training quirks/overrides.
- **Why**: PixArt is similar to SD3 and DeepFloyd in one way or another, and needs special treatment at validation, training, and inference time. Use this option to enable PixArt training support. PixArt does not support ControlNet, LoRA, or `--validation_using_datasets`

### `--pretrained_model_name_or_path`

- **What**: Path to the pretrained model or its identifier from huggingface.co/models.
- **Why**: To specify the base model you'll start training from. Use `--revision` and `--variant` to specify specific versions from a repository.

### `--pretrained_t5_model_name_or_path`

- **What**: Path to the pretrained T5 model or its identifier from huggingface.co/models.
- **Why**: When training PixArt, you might want to use a specific source for your T5 weights so that you can avoid downloading them multiple times when switching the base model you train from.

### `--hub_model_id`

- **What**: The name of the Huggingface Hub model and local results directory.
- **Why**: This value is used as the directory name under the location specified as `--output_dir`. If `--push_to_hub` is provided, this will become the name of the model on Huggingface Hub.

---

### `--push_to_hub`

- **What**: If provided, your model will be uploaded to [Huggingface Hub](https://huggingface.co) once training completes. Using `--push_checkpoints_to_hub` will additionally push every intermediary checkpoint.

### `--refiner_training`

- **What**: Enables training a custom mixture-of-experts model series. See [Mixture-of-Experts](/documentation/MIXTURE_OF_EXPERTS.md) for more information on these options.

## 📂 Data Storage and Management

### `--data_backend_config`

- **What**: Path to your SimpleTuner dataset configuration.
- **Why**: Multiple datasets on different storage medium may be combined into a single training session.
- **Example**: See (multidatabackend.json.example)[/multidatabackend.json.example] for an example configuration, and [this document](/documentation/DATALOADER.md) for more information on configuring the data loader.

### `--override_dataset_config`

- **What**: When provided, will allow SimpleTuner to ignore differences between the cached config inside the dataset and the current values.
- **Why**: When SimplerTuner is run for the first time on a dataset, it will create a cache document containing information about everything in that dataset. This includes the dataset config, including its "crop" and "resolution" related configuration values. Changing these arbitrarily or by accident could result in your training jobs crashing randomly, so it's highly recommended to not use this parameter, and instead resolve the differences you'd like to apply in your dataset some other way.

### `--vae_cache_scan_behaviour`

- **What**: Configure the behaviour of the integrity scan check.
- **Why**: A dataset could have incorrect settings applied at multiple points of training, eg. if you accidentally delete the `.json` cache files from your dataset and switch the data backend config to use square images rather than aspect-crops. This will result in an inconsistent data cache, which can be corrected by setting `scan_for_errors` to `true` in your `multidatabackend.json` configuration file. When this scan runs, it relies on the setting of `--vae_cache_scan_behaviour` to determine how to resolve the inconsistency: `recreate` (the default) will remove the offending cache entry so that it can be recreated, and `sync` will update the bucket metadata to reflect the reality of the real training sample. Recommended value: `recreate`.

### `--dataloader_prefetch`

- **What**: Retrieve batches ahead-of-time.
- **Why**: Especially when using large batch sizes, training will "pause" while samples are retrieved from disk (even NVMe), impacting GPU utilisation metrics. Enabling dataloader prefetch will keep a buffer full of entire batches, so that they can be loaded instantly.

### `--dataloader_prefetch_qlen`

- **What**: Increase or reduce the number of batches held in memory.
- **Why**: When using dataloader prefetch, a default of 10 entries are kept in memory per GPU/process. This may be too much or too little. This value can be adjusted to increase the number of batches prepared in advance.

### `--compress_disk_cache`

- **What**: Compress the VAE and text embed caches on-disk.
- **Why**: The T5 encoder used by DeepFloyd, SD3, and PixArt, produces very-large text embeds that end up being mostly empty space for shorter or redundant captions. Enabling `--compress_disk_cache` can reduce space consumed by up to 75%, with average savings of 40%.

> ⚠️ You will need to manually remove the existing cache directories so they can be recreated with compression by the trainer.

---

## 🌈 Image and Text Processing

A lot of settings are instead set through the [dataloader config](/documentation/DATALOADER.md), but these will apply globally.

### `--resolution`

- **What**: Input image resolution. Can be expressed as pixels, or megapixels.
- **Why**: All images in the dataset will have their smaller edge resized to this resolution for training. It is recommended use a value of 1.0 if also using `--resolution_type=area`. When using `--resolution_type=pixel` and `--resolution=1024px`, the images may become very large and use an excessive amount of VRAM. The recommended configuration is to combine `--resolution_type=area` with `--resolution=1` (or lower - .25 would be a 512px model with data bucketing).

### `--resolution_type`

- **What**: This tells SimpleTuner whether to use `area` size calculations or `pixel` edge calculations.
- **Why**: SimpleTuner's default `pixel` behaviour is to resize the image, keeping the aspect ratio. Setting the type to `area` instead uses the given megapixel value as the target size for the image, keeping the aspect ratio.

### `--validation_resolution`

- **What**: Output image resolution, measured in pixels.
- **Why**: All images generated during validation will be this resolution. Useful if the model is being trained with a different resolution.

### `--caption_strategy`

- **What**: Strategy for deriving image captions. **Choices**: `textfile`, `filename`, `parquet`, `instanceprompt`
- **Why**: Determines how captions are generated for training images.
  - `textfile` will use the contents of a `.txt` file with the same filename as the image
  - `filename` will apply some cleanup to the filename before using it as the caption.
  - `parquet` requires a parquet file to be present in the dataset, and will use the `caption` column as the caption unless `parquet_caption_column` is provided. All captions must be present unless a `parquet_fallback_caption_column` is provided.
  - `instanceprompt` will use the value for `instance_prompt` in the dataset config as the prompt for every image in the dataset.

### `--crop`

- **What**: When `--crop=true` is supplied, SimpleTuner will crop all (new) images in the training dataset. It will not re-process old images.
- **Why**: Training on cropped images seems to result in better fine detail learning, especially on SDXL models.

### `--crop_style`

- **What**: When `--crop=true`, the trainer may be instructed to crop in different ways.
- **Why**: The `crop_style` option can be set to `center` (or `centre`) for a classic centre-crop, `corner` to elect for the lowest-right corner, `face` to detect and centre upon the largest subject face, and `random` for a random image slice. Default: random.

### `--crop_aspect`

- **What**: When using `--crop=true`, the `--crop_aspect` option may be supplied with a value of `square` or `preserve`.
- **Why**: The default crop behaviour is to crop all images to a square aspect ratio, but when `--crop_aspect=preserve` is supplied, the trainer will crop images to a size matching their original aspect ratio. This may help to keep multi-resolution support, but it may also harm training quality. Your mileage may vary.

---

## 🎛 Training Parameters

### `--num_train_epochs`

- **What**: Number of training epochs (the number of times that all images are seen). Setting this to 0 will allow `--max_train_steps` to take precedence.
- **Why**: Determines the number of image repeats, which impacts the duration of the training process. More epochs tends to result in overfitting, but might be required to pick up the concepts you wish to train in. A reasonable value might be from 5 to 50.

### `--max_train_steps`

- **What**: Number of training steps to exit training after. If set to 0, will allow `--num_train_epochs` to take priority.
- **Why**: Useful for shortening the length of training.

### `--train_batch_size`

- **What**: Batch size for the training data loader.
- **Why**: Affects the model's memory consumption, convergence quality, and training speed. The higher the batch size, the better the results will be, but a very high batch size might result in overfitting or destabilized training, as well as increasing the duration of the training session unnecessarily. Experimentation is warranted, but in general, you want to try to max out your video memory while not decreasing the training speed.

---

## 🛠 Advanced Optimizations

### `--use_ema`

- **What**: Keeping an exponential moving average of your weights over the models' training lifetime is like periodically back-merging the model into itself.
- **Why**: It can improve training stability at the cost of more system resources, and a slight increase in training runtime.

## `--ema_device`

- **Choices**: `cpu`, `accelerator`, default: `cpu`
- **What**: Place the EMA weights on the accelerator instead of CPU.
- **Why**: The default location of CPU for EMA weights might result in a substantial slowdown on some systems. However, `--ema_cpu_only` will override this value if provided.

### `--ema_cpu_only`

- **What**: Keeps EMA weights on the CPU. The default behaviour is to move the EMA weights to the GPU before updating them.
- **Why**: Moving the EMA weights to the GPU is unnecessary, as the update on CPU can be nearly just as quick. However, some systems may experience a substantial slowdown, so EMA weights will remain on GPU by default.

### `--ema_update_interval`

- **What**: Reduce the update interval of your EMA shadow parameters.
- **Why**: Updating the EMA weights on every step could be an unnecessary waste of resources. Providing `--ema_update_interval=100` will update the EMA weights only once every 100 optimizer steps.

### `--gradient_accumulation_steps`

- **What**: Number of update steps to accumulate before performing a backward/update pass, essentially splitting the work over multiple batches to save memory at the cost of a higher training runtime.
- **Why**: Useful for handling larger models or datasets.

### `--learning_rate`

- **What**: Initial learning rate after potential warmup.
- **Why**: The learning rate behaves as a sort of "step size" for gradient updates - too high, and we overstep the solution. Too low, and we never reach the ideal solution. A minimal value for a `full` tune might be as low as `1e-7` to a max of `1e-6` while for `lora` tuning a minimal value might be `1e-5` with a maximal value as high as `1e-3`. When a higher learning rate is used, it's advantageous to use an EMA network with a learning rate warmup - see `--use_ema`, `--lr_warmup_steps`, and `--lr_scheduler`.

### `--lr_scheduler`

- **What**: How to scale the learning rate over time.
- **Choices**: constant, constant_with_warmup, cosine, cosine_with_restarts, **polynomial** (recommended), linear
- **Why**: Models benefit from continual learning rate adjustments to further explore the loss landscape. A cosine schedule is used as the default; this allows the training to smoothly transition between two extremes. If using a constant learning rate, it is common to select a too-high or too-low value, causing divergence (too high) or getting stuck in a local minima (too low). A polynomial schedule is best paired with a warmup, where it will gradually approach the `learning_rate` value before then slowing down and approaching `--lr_end` by the end.

### `--snr_gamma`

- **What**: Utilising min-SNR weighted loss factor.
- **Why**: Minimum SNR gamma weights the loss factor of a timestep by its position in the schedule. Overly noisy timesteps have their contributions reduced, and less-noisy timesteps have it increased. Value recommended by the original paper is **5** but you can use values as low as **1** or as high as **20**, typically seen as the maximum value - beyond a value of 20, the math does not change things much. A value of **1** is the strongest.

### `--use_soft_min_snr`

- **What**: Train a model using a more gradual weighting on the loss landscape.
- **Why**: When training pixel diffusion models, they will simply degrade without using a specific loss weighting schedule. This is the case with DeepFloyd, where soft-min-snr-gamma was found to essentially be mandatory for good results. You may find success with latent diffusion model training, but in small experiments, it was found to potentially produce blurry results.

---

## 🔄 Checkpointing and Resumption

### `--checkpointing_steps`

- **What**: Interval at which training state checkpoints are saved.
- **Why**: Useful for resuming training and for inference. Every _n_ iterations, a partial checkpoint will be saved in the `.safetensors` format, via the Diffusers filesystem layout.

### `--resume_from_checkpoint`

- **What**: Specifies if and from where to resume training.
- **Why**: Allows you to continue training from a saved state, either manually specified or the latest available. A checkpoint is composed of a `unet` and optionally, a `unet_ema` subfolder. The `unet` may be dropped into any Diffusers layout SDXL model, allowing it to be used as a normal model would.

> ℹ️ Transformer models such as PixArt, SD3, or Hunyuan, use the `transformer` and `transformer_ema` subfolder names.

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
usage: train.py [-h] [--snr_gamma SNR_GAMMA] [--use_soft_min_snr]
                [--soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA]
                [--model_type {full,lora,deepfloyd-full,deepfloyd-lora,deepfloyd-stage2,deepfloyd-stage2-lora}]
                [--legacy] [--kolors] [--flux]
                [--flux_lora_target {mmdit,context,all,all+ffs}]
                [--flow_matching_sigmoid_scale FLOW_MATCHING_SIGMOID_SCALE]
                [--flux_fast_schedule]
                [--flux_guidance_mode {constant,random-range}]
                [--flux_guidance_value FLUX_GUIDANCE_VALUE]
                [--flux_guidance_min FLUX_GUIDANCE_MIN]
                [--flux_guidance_max FLUX_GUIDANCE_MAX] [--smoldit]
                [--smoldit_config {smoldit-small,smoldit-swiglu,smoldit-base,smoldit-large,smoldit-huge}]
                [--flow_matching_loss {diffusers,compatible,diffusion}]
                [--pixart_sigma] [--sd3]
                [--sd3_t5_mask_behaviour {do-nothing,mask}]
                [--lora_type {Standard}]
                [--lora_init_type {default,gaussian,loftq,olora,pissa}]
                [--lora_rank LORA_RANK] [--lora_alpha LORA_ALPHA]
                [--lora_dropout LORA_DROPOUT] [--controlnet]
                [--controlnet_model_name_or_path]
                --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                [--pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH]
                [--pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH]
                [--prediction_type {epsilon,v_prediction,sample}]
                [--snr_weight SNR_WEIGHT]
                [--training_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--inference_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--refiner_training] [--refiner_training_invert_schedule]
                [--refiner_training_strength REFINER_TRAINING_STRENGTH]
                [--timestep_bias_strategy {earlier,later,range,none}]
                [--timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER]
                [--timestep_bias_begin TIMESTEP_BIAS_BEGIN]
                [--timestep_bias_end TIMESTEP_BIAS_END]
                [--timestep_bias_portion TIMESTEP_BIAS_PORTION]
                [--disable_segmented_timestep_sampling]
                [--rescale_betas_zero_snr]
                [--vae_dtype {default,fp16,fp32,bf16}]
                [--vae_batch_size VAE_BATCH_SIZE]
                [--vae_cache_scan_behaviour {recreate,sync}]
                [--vae_cache_preprocess] [--vae_cache_ondemand]
                [--compress_disk_cache] [--aspect_bucket_disable_rebuild]
                [--keep_vae_loaded]
                [--skip_file_discovery SKIP_FILE_DISCOVERY]
                [--revision REVISION] [--variant VARIANT]
                [--preserve_data_backend_cache] [--use_dora]
                [--override_dataset_config] [--cache_dir_text CACHE_DIR_TEXT]
                [--cache_dir_vae CACHE_DIR_VAE] --data_backend_config
                DATA_BACKEND_CONFIG [--write_batch_size WRITE_BATCH_SIZE]
                [--read_batch_size READ_BATCH_SIZE]
                [--image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE]
                [--enable_multiprocessing] [--max_workers MAX_WORKERS]
                [--aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS]
                [--torch_num_threads TORCH_NUM_THREADS]
                [--dataloader_prefetch]
                [--dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN]
                [--aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT]
                [--cache_dir CACHE_DIR] [--cache_clear_validation_prompts]
                [--caption_strategy {filename,textfile,instance_prompt,parquet}]
                [--parquet_caption_column PARQUET_CAPTION_COLUMN]
                [--parquet_filename_column PARQUET_FILENAME_COLUMN]
                [--instance_prompt INSTANCE_PROMPT] [--output_dir OUTPUT_DIR]
                [--seed SEED] [--seed_for_each_device SEED_FOR_EACH_DEVICE]
                [--resolution RESOLUTION] [--resolution_type {pixel,area}]
                [--aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}]
                [--aspect_bucket_alignment {8,64}]
                [--minimum_image_size MINIMUM_IMAGE_SIZE]
                [--maximum_image_size MAXIMUM_IMAGE_SIZE]
                [--target_downsample_size TARGET_DOWNSAMPLE_SIZE]
                [--train_text_encoder]
                [--tokenizer_max_length TOKENIZER_MAX_LENGTH]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--num_train_epochs NUM_TRAIN_EPOCHS]
                [--max_train_steps MAX_TRAIN_STEPS]
                [--checkpointing_steps CHECKPOINTING_STEPS]
                [--checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT]
                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--gradient_checkpointing] [--learning_rate LEARNING_RATE]
                [--text_encoder_lr TEXT_ENCODER_LR] [--lr_scale]
                [--lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                [--lr_warmup_steps LR_WARMUP_STEPS]
                [--lr_num_cycles LR_NUM_CYCLES] [--lr_power LR_POWER]
                [--use_ema] [--ema_device {cpu,accelerator}] [--ema_cpu_only]
                [--ema_foreach_disable]
                [--ema_update_interval EMA_UPDATE_INTERVAL]
                [--ema_decay EMA_DECAY] [--non_ema_revision NON_EMA_REVISION]
                [--offload_param_path OFFLOAD_PARAM_PATH] [--use_8bit_adam]
                [--use_adafactor_optimizer]
                [--adafactor_relative_step ADAFACTOR_RELATIVE_STEP]
                [--use_prodigy_optimizer] [--prodigy_beta3 PRODIGY_BETA3]
                [--prodigy_decouple PRODIGY_DECOUPLE]
                [--prodigy_use_bias_correction PRODIGY_USE_BIAS_CORRECTION]
                [--prodigy_safeguard_warmup PRODIGY_SAFEGUARD_WARMUP]
                [--prodigy_learning_rate PRODIGY_LEARNING_RATE]
                [--prodigy_weight_decay PRODIGY_WEIGHT_DECAY]
                [--prodigy_epsilon PRODIGY_EPSILON] [--use_dadapt_optimizer]
                [--dadaptation_learning_rate DADAPTATION_LEARNING_RATE]
                [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                [--adam_weight_decay ADAM_WEIGHT_DECAY]
                [--adam_epsilon ADAM_EPSILON] [--adam_bfloat16]
                [--max_grad_norm MAX_GRAD_NORM] [--push_to_hub]
                [--push_checkpoints_to_hub] [--hub_model_id HUB_MODEL_ID]
                [--model_card_note MODEL_CARD_NOTE]
                [--logging_dir LOGGING_DIR]
                [--validation_seed_source {gpu,cpu}]
                [--validation_torch_compile VALIDATION_TORCH_COMPILE]
                [--validation_torch_compile_mode {max-autotune,reduce-overhead,default}]
                [--allow_tf32] [--validation_using_datasets]
                [--webhook_config WEBHOOK_CONFIG] [--report_to REPORT_TO]
                [--tracker_run_name TRACKER_RUN_NAME]
                [--tracker_project_name TRACKER_PROJECT_NAME]
                [--validation_prompt VALIDATION_PROMPT]
                [--validation_prompt_library]
                [--user_prompt_library USER_PROMPT_LIBRARY]
                [--validation_negative_prompt VALIDATION_NEGATIVE_PROMPT]
                [--num_validation_images NUM_VALIDATION_IMAGES]
                [--validation_steps VALIDATION_STEPS]
                [--num_eval_images NUM_EVAL_IMAGES]
                [--eval_dataset_id EVAL_DATASET_ID]
                [--validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS]
                [--validation_resolution VALIDATION_RESOLUTION]
                [--validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc}]
                [--validation_disable_unconditional] [--disable_compel]
                [--enable_watermark] [--mixed_precision {bf16,no}]
                [--gradient_precision {unmodified,fp32}]
                [--base_model_precision {no_change,fp8-quanto,int8-quanto,int4-quanto,int2-quanto}]
                [--base_model_default_dtype {bf16,fp32}]
                [--text_encoder_1_precision {no_change,fp8-quanto,int8-quanto,int4-quanto,int2-quanto}]
                [--text_encoder_2_precision {no_change,fp8-quanto,int8-quanto,int4-quanto,int2-quanto}]
                [--text_encoder_3_precision {no_change,fp8-quanto,int8-quanto,int4-quanto,int2-quanto}]
                [--local_rank LOCAL_RANK]
                [--enable_xformers_memory_efficient_attention]
                [--set_grads_to_none] [--noise_offset NOISE_OFFSET]
                [--noise_offset_probability NOISE_OFFSET_PROBABILITY]
                [--validation_guidance VALIDATION_GUIDANCE]
                [--validation_guidance_real VALIDATION_GUIDANCE_REAL]
                [--validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP]
                [--validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE]
                [--validation_randomize] [--validation_seed VALIDATION_SEED]
                [--fully_unload_text_encoder]
                [--freeze_encoder_before FREEZE_ENCODER_BEFORE]
                [--freeze_encoder_after FREEZE_ENCODER_AFTER]
                [--freeze_encoder_strategy FREEZE_ENCODER_STRATEGY]
                [--layer_freeze_strategy {none,bitfit}]
                [--unet_attention_slice] [--print_filenames]
                [--print_sampler_statistics]
                [--metadata_update_interval METADATA_UPDATE_INTERVAL]
                [--debug_aspect_buckets] [--debug_dataset_loader]
                [--freeze_encoder FREEZE_ENCODER] [--save_text_encoder]
                [--text_encoder_limit TEXT_ENCODER_LIMIT]
                [--prepend_instance_prompt] [--only_instance_prompt]
                [--data_aesthetic_score DATA_AESTHETIC_SCORE]
                [--sdxl_refiner_uses_full_range]
                [--caption_dropout_probability CAPTION_DROPOUT_PROBABILITY]
                [--delete_unwanted_images] [--delete_problematic_images]
                [--offset_noise] [--lr_end LR_END] [--i_know_what_i_am_doing]
                [--accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL]

The following SimpleTuner command-line options are available:

options:
  -h, --help            show this help message and exit
  --snr_gamma SNR_GAMMA
                        SNR weighting gamma to be used if rebalancing the
                        loss. Recommended value is 5.0. More details here:
                        https://arxiv.org/abs/2303.09556.
  --use_soft_min_snr    If set, will use the soft min SNR calculation method.
                        This method uses the sigma_data parameter. If not
                        provided, the method will raise an error.
  --soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA
                        The standard deviation of the data used in the soft
                        min weighting method. This is required when using the
                        soft min SNR calculation method.
  --model_type {full,lora,deepfloyd-full,deepfloyd-lora,deepfloyd-stage2,deepfloyd-stage2-lora}
                        The training type to use. 'full' will train the full
                        model, while 'lora' will train the LoRA model. LoRA is
                        a smaller model that can be used for faster training.
  --legacy              This option must be provided when training a Stable
                        Diffusion 1.x or 2.x model.
  --kolors              This option must be provided when training a Kolors
                        model.
  --flux                This option must be provided when training a Flux
                        model.
  --flux_lora_target {mmdit,context,all,all+ffs}
                        Flux has single and joint attention blocks. Only the
                        multimodal 'dual stream' attention blocks are trained
                        by default. If 'mmdit' is provided, the text input
                        layers will not be trained. If 'context' is provided,
                        the mmdit layers will not be trained. If 'all' is
                        provided, all layers will be trained, minus feed-
                        forward and norms. If 'all+ffs' is provided, all
                        layers will be trained including feed-forward and
                        norms.
  --flow_matching_sigmoid_scale FLOW_MATCHING_SIGMOID_SCALE
                        Scale factor for sigmoid timestep sampling for flow-
                        matching models..
  --flux_fast_schedule  An experimental feature to train Flux.1S using a noise
                        schedule closer to what it was trained with, which has
                        improved results in short experiments. Thanks to
                        @mhirki for the contribution.
  --flux_guidance_mode {constant,random-range}
                        Flux has a 'guidance' value used during training time
                        that reflects the CFG range of your training samples.
                        The default mode 'constant' will use a single value
                        for every sample. The mode 'random-range' will
                        randomly select a value from the range of the CFG for
                        each sample. Set the range using --flux_guidance_min
                        and --flux_guidance_max.
  --flux_guidance_value FLUX_GUIDANCE_VALUE
                        When using --flux_guidance_mode=constant, this value
                        will be used for every input sample. Using a value of
                        1.0 seems to preserve the CFG distillation for the Dev
                        model, and using any other value will result in the
                        resulting LoRA requiring CFG at inference time.
  --flux_guidance_min FLUX_GUIDANCE_MIN
  --flux_guidance_max FLUX_GUIDANCE_MAX
  --smoldit             Use the experimental SmolDiT model architecture.
  --smoldit_config {smoldit-small,smoldit-swiglu,smoldit-base,smoldit-large,smoldit-huge}
                        The SmolDiT configuration to use. This is a list of
                        pre-configured models. The default is 'smoldit-base'.
  --flow_matching_loss {diffusers,compatible,diffusion}
                        A discrepancy exists between the Diffusers
                        implementation of flow matching and the minimal
                        implementation provided by StabilityAI. This
                        experimental option allows switching loss calculations
                        to be compatible with those. Additionally, 'diffusion'
                        is offered as an option to reparameterise a model to
                        v_prediction loss.
  --pixart_sigma        This must be set when training a PixArt Sigma model.
  --sd3                 This option must be provided when training a Stable
                        Diffusion 3 model.
  --sd3_t5_mask_behaviour {do-nothing,mask}
                        StabilityAI did not correctly implement their
                        attention masking on T5 inputs for SD3 Medium. This
                        option enables you to switch between their broken
                        implementation or the corrected mask implementation.
                        Although, the corrected masking is still applied via
                        hackish workaround, manually applying the mask to the
                        prompt embeds so that the padded positions are zero.
                        This improves the results for short captions, but does
                        not change the behaviour for long captions. It is
                        important to note that this limitation currently
                        prevents expansion of SD3 Medium's prompt length, as
                        it will unnecessarily attend to every token in the
                        prompt embed, even masked positions.
  --lora_type {Standard}
                        When training using --model_type=lora, you may specify
                        a different type of LoRA to train here. Currently,
                        only 'Standard' type is supported. This option exists
                        for compatibility with Kohya configuration files.
  --lora_init_type {default,gaussian,loftq,olora,pissa}
                        The initialization type for the LoRA model. 'default'
                        will use Microsoft's initialization method, 'gaussian'
                        will use a Gaussian scaled distribution, and 'loftq'
                        will use LoftQ initialization. In short experiments,
                        'default' produced accurate results earlier in
                        training, 'gaussian' had slightly more creative
                        outputs, and LoftQ produces an entirely different
                        result with worse quality at first, taking potentially
                        longer to converge than the other methods.
  --lora_rank LORA_RANK
                        The dimension of the LoRA update matrices.
  --lora_alpha LORA_ALPHA
                        The alpha value for the LoRA model. This is the
                        learning rate for the LoRA update matrices.
  --lora_dropout LORA_DROPOUT
                        LoRA dropout randomly ignores neurons during training.
                        This can help prevent overfitting.
  --controlnet          If set, ControlNet style training will be used, where
                        a conditioning input image is required alongside the
                        training data.
  --controlnet_model_name_or_path
                        When provided alongside --controlnet, this will
                        specify ControlNet model weights to preload from the
                        hub.
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from
                        huggingface.co/models.
  --pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH
                        Path to an improved VAE to stabilize training. For
                        more details check out:
                        https://github.com/huggingface/diffusers/pull/4038.
  --pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH
                        T5-XXL is a huge model, and starting from many
                        different models will download a separate one each
                        time. This option allows you to specify a specific
                        location to retrieve T5-XXL v1.1 from, so that it only
                        downloads once..
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
  --refiner_training    When training or adapting a model into a mixture-of-
                        experts 2nd stage / refiner model, this option should
                        be set. This will slice the timestep schedule defined
                        by --refiner_training_strength proportion value
                        (default 0.2)
  --refiner_training_invert_schedule
                        While the refiner training strength is applied to the
                        end of the schedule, this option will invert the
                        result for training a **base** model, eg. the first
                        model in a mixture-of-experts series. A
                        --refiner_training_strength of 0.35 will result in the
                        refiner learning timesteps 349-0. Setting
                        --refiner_training_invert_schedule then would result
                        in the base model learning timesteps 999-350.
  --refiner_training_strength REFINER_TRAINING_STRENGTH
                        When training a refiner / 2nd stage mixture of experts
                        model, the refiner training strength indicates how
                        much of the *end* of the schedule it will be trained
                        on. A value of 0.2 means timesteps 199-0 will be the
                        focus of this model, and 0.3 would be 299-0 and so on.
                        The default value is 0.2, in line with the SDXL
                        refiner pretraining.
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
  --disable_segmented_timestep_sampling
                        By default, the timestep schedule is divided into
                        roughly `train_batch_size` number of segments, and
                        then each of those are sampled from separately. This
                        improves the selection distribution, but may not be
                        desired in certain training scenarios, eg. when
                        limiting the timestep selection range.
  --rescale_betas_zero_snr
                        If set, will rescale the betas to zero terminal SNR.
                        This is recommended for training with v_prediction.
                        For epsilon, this might help with fine details, but
                        will not result in contrast improvements.
  --vae_dtype {default,fp16,fp32,bf16}
                        The dtype of the VAE model. Choose between ['default',
                        'fp16', 'fp32', 'bf16']. The default VAE dtype is
                        bfloat16, due to NaN issues in SDXL 1.0. Using fp16 is
                        not recommended.
  --vae_batch_size VAE_BATCH_SIZE
                        When pre-caching latent vectors, this is the batch
                        size to use. Decreasing this may help with VRAM
                        issues, but if you are at that point of contention,
                        it's possible that your GPU has too little RAM.
                        Default: 4.
  --vae_cache_scan_behaviour {recreate,sync}
                        When a mismatched latent vector is detected, a scan
                        will be initiated to locate inconsistencies and
                        resolve them. The default setting 'recreate' will
                        delete any inconsistent cache entries and rebuild it.
                        Alternatively, 'sync' will update the bucket
                        configuration so that the image is in a bucket that
                        matches its latent size. The recommended behaviour is
                        to use the default value and allow the cache to be
                        recreated.
  --vae_cache_preprocess
                        This option is deprecated and will be removed in a
                        future release. Use --vae_cache_ondemand instead.
  --vae_cache_ondemand  By default, will batch-encode images before training.
                        For some situations, ondemand may be desired, but it
                        greatly slows training and increases memory pressure.
  --compress_disk_cache
                        If set, will gzip-compress the disk cache for Pytorch
                        files. This will save substantial disk space, but may
                        slow down the training process.
  --aspect_bucket_disable_rebuild
                        When using a randomised aspect bucket list, the VAE
                        and aspect cache are rebuilt on each epoch. With a
                        large and diverse enough dataset, rebuilding the
                        aspect list may take a long time, and this may be
                        undesirable. This option will not override
                        vae_cache_clear_each_epoch. If both options are
                        provided, only the VAE cache will be rebuilt.
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
                        should be at least bfloat16 precision.
  --variant VARIANT     Variant of pretrained model identifier from
                        huggingface.co/models. Trainable model components
                        should be at least bfloat16 precision.
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
  --use_dora            If set, will use the DoRA-enhanced LoRA training. This
                        is an experimental feature, may slow down training,
                        and is not recommended for general use.
  --override_dataset_config
                        When provided, the dataset's config will not be
                        checked against the live backend config. This is
                        useful if you want to simply update the behaviour of
                        an existing dataset, but the recommendation is to not
                        change the dataset configuration after caching has
                        begun, as most options cannot be changed without
                        unexpected behaviour later on. Additionally, it
                        prevents accidentally loading an SDXL configuration on
                        a SD 2.x model and vice versa.
  --cache_dir_text CACHE_DIR_TEXT
                        This is the path to a local directory that will
                        contain your text embed cache.
  --cache_dir_vae CACHE_DIR_VAE
                        This is the path to a local directory that will
                        contain your VAE outputs. Unlike the text embed cache,
                        your VAE latents will be stored in the AWS data
                        backend. Each backend can have its own value, but if
                        that is not provided, this will be the default value.
  --data_backend_config DATA_BACKEND_CONFIG
                        The relative or fully-qualified path for your data
                        backend config. See multidatabackend.json.example for
                        an example.
  --write_batch_size WRITE_BATCH_SIZE
                        When using certain storage backends, it is better to
                        batch smaller writes rather than continuous
                        dispatching. In SimpleTuner, write batching is
                        currently applied during VAE caching, when many small
                        objects are written. This mostly applies to S3, but
                        some shared server filesystems may benefit as well,
                        eg. Ceph. Default: 64.
  --read_batch_size READ_BATCH_SIZE
                        Used by the VAE cache to prefetch image data. This is
                        the number of images to read ahead.
  --image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE
                        When resizing and cropping images, we do it in
                        parallel using processes or threads. This defines how
                        many images will be read into the queue before they
                        are processed.
  --enable_multiprocessing
                        If set, will use processes instead of threads during
                        metadata caching operations. For some systems,
                        multiprocessing may be faster than threading, but will
                        consume a lot more memory. Use this option with
                        caution, and monitor your system's memory usage.
  --max_workers MAX_WORKERS
                        How many active threads or processes to run during VAE
                        caching.
  --aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS
                        When using AWS backends, the maximum number of
                        connections to keep open to the S3 bucket at a single
                        time. This should be greater or equal to the
                        max_workers and aspect bucket worker count values.
  --torch_num_threads TORCH_NUM_THREADS
                        The number of threads to use for PyTorch operations.
                        This is not the same as the number of workers.
                        Default: 8.
  --dataloader_prefetch
                        When provided, the dataloader will read-ahead and
                        attempt to retrieve latents, text embeds, and other
                        metadata ahead of the time when the batch is required,
                        so that it can be immediately available.
  --dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN
                        Set the number of prefetched batches.
  --aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT
                        The number of workers to use for aspect bucketing.
                        This is a CPU-bound task, so the number of workers
                        should be set to the number of CPU threads available.
                        If you use an I/O bound backend, an even higher value
                        may make sense. Default: 12.
  --cache_dir CACHE_DIR
                        The directory where the downloaded models and datasets
                        will be stored.
  --cache_clear_validation_prompts
                        When provided, any validation prompt entries in the
                        text embed cache will be recreated. This is useful if
                        you've modified any of the existing prompts, or,
                        disabled/enabled Compel, via `--disable_compel`
  --caption_strategy {filename,textfile,instance_prompt,parquet}
                        The default captioning strategy, 'filename', will use
                        the filename as the caption, after stripping some
                        characters like underscores. The 'textfile' strategy
                        will use the contents of a text file with the same
                        name as the image. The 'parquet' strategy requires a
                        parquet file with the same name as the image,
                        containing a 'caption' column.
  --parquet_caption_column PARQUET_CAPTION_COLUMN
                        When using caption_strategy=parquet, this option will
                        allow you to globally set the default caption field
                        across all datasets that do not have an override set.
  --parquet_filename_column PARQUET_FILENAME_COLUMN
                        When using caption_strategy=parquet, this option will
                        allow you to globally set the default filename field
                        across all datasets that do not have an override set.
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
  --aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}
                        The number of decimal places to round the aspect ratio
                        to. This is used to create buckets for aspect ratios.
                        For higher precision, ensure the image sizes remain
                        compatible. Higher precision levels result in a
                        greater number of buckets, which may not be a
                        desirable outcome.
  --aspect_bucket_alignment {8,64}
                        When training diffusion models, the image sizes
                        generally must align to a 64 pixel interval. This is
                        an exception when training models like DeepFloyd that
                        use a base resolution of 64 pixels, as aligning to 64
                        pixels would result in a 1:1 or 2:1 aspect ratio,
                        overly distorting images. For DeepFloyd, this value is
                        set to 8, but all other training defaults to 64. You
                        may experiment with this value, but it is not
                        recommended.
  --minimum_image_size MINIMUM_IMAGE_SIZE
                        The minimum resolution for both sides of input images.
                        If --delete_unwanted_images is set, images smaller
                        than this will be DELETED. The default value is None,
                        which means no minimum resolution is enforced. If this
                        option is not provided, it is possible that images
                        will be destructively upsampled, harming model
                        performance.
  --maximum_image_size MAXIMUM_IMAGE_SIZE
                        When cropping images that are excessively large, the
                        entire scene context may be lost, eg. the crop might
                        just end up being a portion of the background. To
                        avoid this, a maximum image size may be provided,
                        which will result in very-large images being
                        downsampled before cropping them. This value uses
                        --resolution_type to determine whether it is a pixel
                        edge or megapixel value.
  --target_downsample_size TARGET_DOWNSAMPLE_SIZE
                        When using --maximum_image_size, very-large images
                        exceeding that value will be downsampled to this
                        target size before cropping. If --resolution_type=area
                        and --maximum_image_size=4.0,
                        --target_downsample_size=2.0 would result in a 4
                        megapixel image being resized to 2 megapixel before
                        cropping to 1 megapixel.
  --train_text_encoder  (SD 2.x only) Whether to train the text encoder. If
                        set, the text encoder should be float32 precision.
  --tokenizer_max_length TOKENIZER_MAX_LENGTH
                        The maximum length of the tokenizer. If not set, will
                        default to the tokenizer's max length.
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
                        period) to use. When using a cosine or sine schedule,
                        --learning_rate defines the maximum learning rate.
  --text_encoder_lr TEXT_ENCODER_LR
                        Learning rate for the text encoder. If not provided,
                        the value of --learning_rate will be used.
  --lr_scale            Scale the learning rate by the number of GPUs,
                        gradient accumulation steps, and batch size.
  --lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        The scheduler type to use. Default: sine
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps for the warmup in the lr scheduler.
  --lr_num_cycles LR_NUM_CYCLES
                        Number of hard resets of the lr in
                        cosine_with_restarts scheduler.
  --lr_power LR_POWER   Power factor of the polynomial scheduler.
  --use_ema             Whether to use EMA (exponential moving average) model.
  --ema_device {cpu,accelerator}
                        The device to use for the EMA model. If set to
                        'accelerator', the EMA model will be placed on the
                        accelerator. This provides the fastest EMA update
                        times, but is not ultimately necessary for EMA to
                        function.
  --ema_cpu_only        When using EMA, the shadow model is moved to the
                        accelerator before we update its parameters. When
                        provided, this option will disable the moving of the
                        EMA model to the accelerator. This will save a lot of
                        VRAM at the cost of a lot of time for updates. It is
                        recommended to also supply --ema_update_interval to
                        reduce the number of updates to eg. every 100 steps.
  --ema_foreach_disable
                        By default, we use torch._foreach functions for
                        updating the shadow parameters, which should be fast.
                        When provided, this option will disable the foreach
                        methods and use vanilla EMA updates.
  --ema_update_interval EMA_UPDATE_INTERVAL
                        The number of optimization steps between EMA updates.
                        If not provided, EMA network will update on every
                        step.
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
  --adafactor_relative_step ADAFACTOR_RELATIVE_STEP
                        When set, will use the experimental Adafactor mode for
                        relative step computations instead of the value set by
                        --learning_rate. This is an experimental feature, and
                        you are on your own for support.
  --use_prodigy_optimizer
                        Whether or not to use the Prodigy optimizer.
  --prodigy_beta3 PRODIGY_BETA3
                        coefficients for computing the Prodidy stepsize using
                        running averages. If set to None, uses the value of
                        square root of beta2. Ignored if optimizer is adamW
  --prodigy_decouple PRODIGY_DECOUPLE
                        Use AdamW style decoupled weight decay
  --prodigy_use_bias_correction PRODIGY_USE_BIAS_CORRECTION
                        Turn on Adam's bias correction. True by default.
                        Ignored if optimizer is adamW
  --prodigy_safeguard_warmup PRODIGY_SAFEGUARD_WARMUP
                        Remove lr from the denominator of D estimate to avoid
                        issues during warm-up stage. True by default. Ignored
                        if optimizer is adamW
  --prodigy_learning_rate PRODIGY_LEARNING_RATE
                        Though this is called the prodigy learning rate, it
                        corresponds to the d_coef parameter in the Prodigy
                        optimizer. This acts as a coefficient in the
                        expression for the estimate of d. Default for this
                        trainer is 0.5, but the Prodigy default is 1.0, which
                        ends up over-cooking models.
  --prodigy_weight_decay PRODIGY_WEIGHT_DECAY
                        Weight decay to use. Prodigy default is 0, but
                        SimpleTuner uses 1e-2.
  --prodigy_epsilon PRODIGY_EPSILON
                        Epsilon value for the Adam optimizer
  --use_dadapt_optimizer
                        Whether or not to use the discriminator adaptation
                        optimizer.
  --dadaptation_learning_rate DADAPTATION_LEARNING_RATE
                        Learning rate for the discriminator adaptation.
                        Default: 1.0
  --adam_beta1 ADAM_BETA1
                        The beta1 parameter for the Adam and other optimizers.
  --adam_beta2 ADAM_BETA2
                        The beta2 parameter for the Adam and other optimizers.
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        Weight decay to use.
  --adam_epsilon ADAM_EPSILON
                        Epsilon value for the Adam optimizer
  --adam_bfloat16       Whether or not to use stochastic bf16 in Adam.
                        Currently the only supported optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Clipping the max gradient norm can help prevent
                        exploding gradients, but may also harm training by
                        introducing artifacts or making it hard to train
                        artifacts away.
  --push_to_hub         Whether or not to push the model to the Hub.
  --push_checkpoints_to_hub
                        When set along with --push_to_hub, all intermediary
                        checkpoints will be pushed to the hub as if they were
                        a final checkpoint.
  --hub_model_id HUB_MODEL_ID
                        The name of the repository to keep in sync with the
                        local `output_dir`.
  --model_card_note MODEL_CARD_NOTE
                        Add a string to the top of your model card to provide
                        users with some additional context.
  --logging_dir LOGGING_DIR
                        [TensorBoard](https://www.tensorflow.org/tensorboard)
                        log directory. Will default to
                        *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
  --validation_seed_source {gpu,cpu}
                        Some systems may benefit from using CPU-based seeds
                        for reproducibility. On other systems, this may cause
                        a TypeError. Setting this option to 'cpu' may cause
                        validation errors. If so, please set
                        SIMPLETUNER_LOG_LEVEL=DEBUG and submit debug.log to a
                        new Github issue report.
  --validation_torch_compile VALIDATION_TORCH_COMPILE
                        Supply `--validation_torch_compile=true` to enable the
                        use of torch.compile() on the validation pipeline. For
                        some setups, torch.compile() may error out. This is
                        dependent on PyTorch version, phase of the moon, but
                        if it works, you should leave it enabled for a great
                        speed-up.
  --validation_torch_compile_mode {max-autotune,reduce-overhead,default}
                        PyTorch provides different modes for the Torch
                        Inductor when compiling graphs. max-autotune, the
                        default mode, provides the most benefit.
  --allow_tf32          Whether or not to allow TF32 on Ampere GPUs. Can be
                        used to speed up training. For more information, see h
                        ttps://pytorch.org/docs/stable/notes/cuda.html#tensorf
                        loat-32-tf32-on-ampere-devices
  --validation_using_datasets
                        When set, validation will use images sampled randomly
                        from each dataset for validation. Be mindful of
                        privacy issues when publishing training data to the
                        internet.
  --webhook_config WEBHOOK_CONFIG
                        The path to the webhook configuration file. This file
                        should be a JSON file with the following format:
                        {"url": "https://your.webhook.url", "webhook_type":
                        "discord"}}
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
  --num_eval_images NUM_EVAL_IMAGES
                        If possible, this many eval images will be selected
                        from each dataset. This is used when training super-
                        resolution models such as DeepFloyd Stage II, which
                        will upscale input images from the training set.
  --eval_dataset_id EVAL_DATASET_ID
                        When provided, only this dataset's images will be used
                        as the eval set, to keep the training and eval images
                        split.
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
                        'ddpm', 'euler', 'euler-a', 'unipc'], Default: None
                        (use the model default)
  --validation_disable_unconditional
                        When set, the validation pipeline will not generate
                        unconditional samples. This is useful to speed up
                        validations with a single prompt on slower systems, or
                        if you are not interested in unconditional space
                        generations.
  --disable_compel      This option does nothing. It is deprecated and will be
                        removed in a future release.
  --enable_watermark    The SDXL 0.9 and 1.0 licenses both require a watermark
                        be used to identify any images created to be shared.
                        Since the images created during validation typically
                        are not shared, and we want the most accurate results,
                        this watermarker is disabled by default. If you are
                        sharing the validation images, it is up to you to
                        ensure that you are complying with the license,
                        whether that is through this watermarker, or another.
  --mixed_precision {bf16,no}
                        SimpleTuner only supports bf16 training. Bf16 requires
                        PyTorch >= 1.10. on an Nvidia Ampere or later GPU, and
                        PyTorch 2.3 or newer for Apple Silicon. Default to the
                        value of accelerate config of the current system or
                        the flag passed with the `accelerate.launch` command.
                        Use this argument to override the accelerate config.
  --gradient_precision {unmodified,fp32}
                        One of the hallmark discoveries of the Llama 3.1 paper
                        is numeric instability when calculating gradients in
                        bf16 precision. The default behaviour when gradient
                        accumulation steps are enabled is now to use fp32
                        gradients, which is slower, but provides more accurate
                        updates.
  --base_model_precision {no_change,fp8-quanto,int8-quanto,int4-quanto,int2-quanto}
                        When training a LoRA, you might want to quantise the
                        base model to a lower precision to save more VRAM. The
                        default value, 'no_change', does not quantise any
                        weights. Using 'fp4-bnb' or 'fp8-bnb' will require
                        Bits n Bytes for quantisation (NVIDIA, maybe AMD).
                        Using 'fp8-quanto' will require Quanto for
                        quantisation (Apple Silicon, NVIDIA, AMD).
  --base_model_default_dtype {bf16,fp32}
                        Unlike --mixed_precision, this value applies
                        specifically for the default weights of your quantised
                        base model. When quantised, not every parameter can or
                        should be quantised down to the target precision. By
                        default, we use bf16 weights for the base model - but
                        this can be changed to fp32 to enable the use of other
                        optimizers than adamw_bf16. However, this uses
                        marginally more memory, and may not be necessary for
                        your use case.
  --text_encoder_1_precision {no_change,fp8-quanto,int8-quanto,int4-quanto,int2-quanto}
                        When training a LoRA, you might want to quantise text
                        encoder 1 to a lower precision to save more VRAM. The
                        default value is to follow base_model_precision
                        (no_change). Using 'fp4-bnb' or 'fp8-bnb' will require
                        Bits n Bytes for quantisation (NVIDIA, maybe AMD).
                        Using 'fp8-quanto' will require Quanto for
                        quantisation (Apple Silicon, NVIDIA, AMD).
  --text_encoder_2_precision {no_change,fp8-quanto,int8-quanto,int4-quanto,int2-quanto}
                        When training a LoRA, you might want to quantise text
                        encoder 2 to a lower precision to save more VRAM. The
                        default value is to follow base_model_precision
                        (no_change). Using 'fp4-bnb' or 'fp8-bnb' will require
                        Bits n Bytes for quantisation (NVIDIA, maybe AMD).
                        Using 'fp8-quanto' will require Quanto for
                        quantisation (Apple Silicon, NVIDIA, AMD).
  --text_encoder_3_precision {no_change,fp8-quanto,int8-quanto,int4-quanto,int2-quanto}
                        When training a LoRA, you might want to quantise text
                        encoder 3 to a lower precision to save more VRAM. The
                        default value is to follow base_model_precision
                        (no_change). Using 'fp4-bnb' or 'fp8-bnb' will require
                        Bits n Bytes for quantisation (NVIDIA, maybe AMD).
                        Using 'fp8-quanto' will require Quanto for
                        quantisation (Apple Silicon, NVIDIA, AMD).
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
  --validation_guidance_real VALIDATION_GUIDANCE_REAL
                        Use real CFG sampling for Flux validation images.
                        Default: 1.0
  --validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP
                        When using real CFG sampling for Flux validation
                        images, skip doing CFG on these timesteps. Default: 2
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
  --layer_freeze_strategy {none,bitfit}
                        When freezing parameters, we can use the 'none' or
                        'bitfit' strategy. The 'bitfit' strategy will freeze
                        all weights, and leave bias in a trainable state. The
                        default strategy is to leave all parameters in a
                        trainable state. Freezing the weights can improve
                        convergence for finetuning. Using bitfit only
                        moderately reduces VRAM consumption, but substantially
                        reduces the count of trainable parameters.
  --unet_attention_slice
                        If set, will use attention slicing for the SDXL UNet.
                        This is an experimental feature and is not recommended
                        for general use. SD 2.x makes use of attention slicing
                        on Apple MPS platform to avoid a NDArray size crash,
                        but SDXL does not seem to require attention slicing on
                        MPS. If memory constrained, try enabling it anyway.
  --print_filenames     If any image files are stopping the process eg. due to
                        corruption or truncation, this will help identify
                        which is at fault.
  --print_sampler_statistics
                        If provided, will print statistics about the dataset
                        sampler. This is useful for debugging. The default
                        behaviour is to not print sampler statistics.
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
  --freeze_encoder FREEZE_ENCODER
                        Whether or not to freeze the text_encoder. The default
                        is true.
  --save_text_encoder   If set, will save the text_encoder after training.
                        This is useful if you're using --push_to_hub so that
                        the final pipeline contains all necessary components
                        to run.
  --text_encoder_limit TEXT_ENCODER_LIMIT
                        When training the text_encoder, we want to limit how
                        long it trains for to avoid catastrophic loss.
  --prepend_instance_prompt
                        When determining the captions from the filename,
                        prepend the instance prompt as an enforced keyword.
  --only_instance_prompt
                        Use the instance prompt instead of the caption from
                        filename.
  --data_aesthetic_score DATA_AESTHETIC_SCORE
                        Since currently we do not calculate aesthetic scores
                        for data, we will statically set it to one value. This
                        is only used by the SDXL Refiner.
  --sdxl_refiner_uses_full_range
                        If set, the SDXL Refiner will use the full range of
                        the model, rather than the design value of 20 percent.
                        This is useful for training models that will be used
                        for inference from end-to-end of the noise schedule.
                        You may use this for example, to turn the SDXL refiner
                        into a full text-to-image model.
  --caption_dropout_probability CAPTION_DROPOUT_PROBABILITY
                        Caption dropout will randomly drop captions and, for
                        SDXL, size conditioning inputs based on this
                        probability. When set to a value of 0.1, it will drop
                        approximately 10 percent of the inputs. Maximum
                        recommended value is probably less than 0.5, or 50
                        percent of the inputs. Maximum technical value is 1.0.
                        The default is to use zero caption dropout, though for
                        better generalisation, a value of 0.1 is recommended.
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
                        after the specified number of warmup steps. A sine or
                        cosine wave will use this value as its lower bound for
                        the learning rate.
  --i_know_what_i_am_doing
                        If you are using an optimizer other than AdamW, you
                        must set this flag to continue. This is a safety
                        feature to prevent accidental use of an unsupported
                        optimizer, as weights are stored in bfloat16.
  --accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL
                        Clear the cache from VRAM every X steps. This can help
                        prevent memory leaks, but may slow down training.
```