# SimpleTuner Training Script Options

## Overview

This guide provides a user-friendly breakdown of the command-line options available in SimpleTuner's `train.py` script. These options offer a high degree of customization, allowing you to train your model to suit your specific requirements.

### JSON Configuration file format

The JSON filename expected is `config.json` and the key names are the same as the below `--arguments`. The leading `--` is not required for the JSON file, but it can be left in as well.

### Easy configure script (***RECOMMENDED***)

The script `configure.py` in the project root can be used via `python configure.py` to set up a `config.json` file with mostly-ideal default settings.

> ⚠️ For users located in countries where Hugging Face Hub is not readily accessible, you should add `HF_ENDPOINT=https://hf-mirror.com` to your `~/.bashrc` or `~/.zshrc` depending on which `$SHELL` your system uses.

---

## 🌟 Core Model Configuration

### `--model_type`

- **What**: Select whether a LoRA or full fine-tune are created.
- **Choices**: lora, full.
- **Default**: lora
  - If lora is used, `--lora_type` dictates whether PEFT or LyCORIS are in use. Some models (PixArt) work only with LyCORIS adapters.

### `--model_family`

- **What**: Determines which model architecture is being trained.
- **Choices**: pixart_sigma, flux, sd3, sdxl, kolors, legacy

### `--fused_qkv_projections`

- **What**: Fuses the QKV projections in the model's attention blocks to make more efficient use of hardware.
- **Note**: Only available with NVIDIA H100 or H200 with Flash Attention 3 installed manually.

### `--offload_during_startup`

- **What**: Offloads text encoder weights to CPU when VAE caching is going.
- **Why**: This is useful for large models like HiDream and Wan 2.1, which can OOM when loading the VAE cache. This option does not impact quality of training, but for very large text encoders or slow CPUs, it can extend startup time substantially with many datasets. This is disabled by default due to this reason.

### `--pretrained_model_name_or_path`

- **What**: Path to the pretrained model or its identifier from https://huggingface.co/models.
- **Why**: To specify the base model you'll start training from. Use `--revision` and `--variant` to specify specific versions from a repository. This also supports single-file `.safetensors` paths for SDXL, Flux, and SD3.x.

### `--pretrained_t5_model_name_or_path`

- **What**: Path to the pretrained T5 model or its identifier from https://huggingface.co/models.
- **Why**: When training PixArt, you might want to use a specific source for your T5 weights so that you can avoid downloading them multiple times when switching the base model you train from.

### `--gradient_checkpointing`

- **What**: During training, gradients will be calculated layerwise and accumulated to save on peak VRAM requirements at the cost of slower training.

### `--gradient_checkpointing_interval`

- **What**: Checkpoint only every _n_ blocks, where _n_ is a value greater than zero. A value of 1 is effectively the same as just leaving `--gradient_checkpointing` enabled, and a value of 2 will checkpoint every other block.
- **Note**: SDXL and Flux are currently the only models supporting this option. SDXL uses a hackish implementation.

### `--refiner_training`

- **What**: Enables training a custom mixture-of-experts model series. See [Mixture-of-Experts](/documentation/MIXTURE_OF_EXPERTS.md) for more information on these options.

## Precision

### `--quantize_via`

- **Choices**: `cpu`, `accelerator`
  - On `accelerator`, it may work moderately faster at the risk of possibly OOM'ing on 24G cards for a model as large as Flux.
  - On `cpu`, quantisation takes about 30 seconds. (**Default**)


### `--base_model_precision`

- **What**: Reduce model precision and train using less memory. There are currently two supported quantisation backends: quanto and torchao.

#### Optimum Quanto

Provided by Hugging Face, the optimum-quanto library has robust support across all supported platforms.

- `int8-quanto` is the most broadly compatible and probably produces the best results
  - fastest training for RTX4090 and probably other GPUs
  - uses hardware-accelerated matmul on CUDA devices for int8, int4
    - int4 is still abysmally slow
  - works with `TRAINING_DYNAMO_BACKEND=inductor` (`torch.compile()`)
- `fp8uz-quanto` is an experimental fp8 variant for CUDA and ROCm devices.
  - better-supported on AMD silicon such as Instinct or newer architecture
  - can be slightly faster than `int8-quanto` on a 4090 for training, but not inference (1 second slower)
  - works with `TRAINING_DYNAMO_BACKEND=inductor` (`torch.compile()`)
- `fp8-quanto` will not (currently) use fp8 matmul, does not work on Apple systems.
  - does not have hardware fp8 matmul yet on CUDA or ROCm devices, so it will possibly be noticeably slower than int8
    - uses MARLIN kernel for fp8 GEMM
  - incompatible with dynamo, will automatically disable dynamo if the combination is attempted.
  
#### TorchAO

A newer library from Pytorch, AO allows us to replace the linears and 2D convolutions (eg. unet style models) with quantised counterparts.
<!-- Additionally, it provides an experimental CPU offload optimiser that essentially provides a simpler reimplementation of DeepSpeed. -->

- `int8-torchao` will reduce memory consumption to the same level as any of Quanto's precision levels
  - at the time of writing, runs slightly slower (11s/iter) than Quanto does (9s/iter) on Apple MPS
  - When not using `torch.compile`, same speed and memory use as `int8-quanto` on CUDA devices, unknown speed profile on ROCm
  - When using `torch.compile`, slower than `int8-quanto`
- `fp8-torchao` is only available for Hopper (H100, H200) or newer (Blackwell B200) accelerators

##### Optimisers

TorchAO includes generally-available 4bit and 8bit optimisers: `ao-adamw8bit`, `ao-adamw4bit`

It also provides two optimisers that are directed toward Hopper (H100 or better) users: `ao-adamfp8`, and `ao-adamwfp8`

#### Torch Dynamo

To enable `torch.compile()`, add the following line to `config/config.env`:
```bash
TRAINING_DYNAMO_BACKEND=inductor
```

If you wish to use added features like max-autotune, run the following:

```bash
accelerate config
```

Carefully answer the questions and use bf16 mixed precision training when prompted. Say **yes** to using Dynamo, **no** to fullgraph, and **yes** to max-autotune.

Note that the first several steps of training will be slower than usual because of compilation occuring in the background.

### `--attention_mechanism`

Alternative attention mechanisms are supported, with varying levels of compatibility or other trade-offs;

- `diffusers` uses the native Pytorch SDPA functions and is the default attention mechanism
- `xformers` allows the use of Meta's [xformers](https://github.com/facebook/xformers) attention implementation which supports both training and inference fully
- `sageattention` is an inference-focused attention mechanism which does not fully support being used for training ([SageAttention](https://github.com/thu-ml/SageAttention) project page)
  - In simplest terms, SageAttention reduces compute requirement for inference

Using `--sageattention_usage` to enable training with SageAttention should be enabled with care, as it does not track or propagate gradients from its custom CUDA implementations for the QKV linears.
  - This results in these layers being completely untrained, which might cause model collapse or, slight improvements in short training runs.

---

## 📰 Publishing

### `--push_to_hub`

- **What**: If provided, your model will be uploaded to [Huggingface Hub](https://huggingface.co) once training completes. Using `--push_checkpoints_to_hub` will additionally push every intermediary checkpoint.

### `--hub_model_id`

- **What**: The name of the Huggingface Hub model and local results directory.
- **Why**: This value is used as the directory name under the location specified as `--output_dir`. If `--push_to_hub` is provided, this will become the name of the model on Huggingface Hub.

### `--disable_benchmark`

- **What**: Disable the startup validation/benchmark that occurs at step 0 on the base model. These outputs are stitchd to the left side of your trained model validation images.

## 📂 Data Storage and Management

### `--data_backend_config`

- **What**: Path to your SimpleTuner dataset configuration.
- **Why**: Multiple datasets on different storage medium may be combined into a single training session.
- **Example**: See (multidatabackend.json.example)[/multidatabackend.json.example] for an example configuration, and [this document](/documentation/DATALOADER.md) for more information on configuring the data loader.

### `--override_dataset_config`

- **What**: When provided, will allow SimpleTuner to ignore differences between the cached config inside the dataset and the current values.
- **Why**: When SimplerTuner is run for the first time on a dataset, it will create a cache document containing information about everything in that dataset. This includes the dataset config, including its "crop" and "resolution" related configuration values. Changing these arbitrarily or by accident could result in your training jobs crashing randomly, so it's highly recommended to not use this parameter, and instead resolve the differences you'd like to apply in your dataset some other way.

### `--data_backend_sampling`

- **What**: When using multiple data backends, sampling can be done using different strategies.
- **Options**:
  - `uniform` - the previous behaviour from v0.9.8.1 and earlier where dataset length was not considered, only manual probability weightings.
  - `auto-weighting` - the default behaviour where dataset length is used to equally sample all datasets, maintaining a uniform sampling of the entire data distribution.
    - This is required if you have differently-sized datasets that you want the model to learn equally.
    - But adjusting `repeats` manually is **required** to properly sample Dreambooth images against your regularisation set

### `--vae_cache_scan_behaviour`

- **What**: Configure the behaviour of the integrity scan check.
- **Why**: A dataset could have incorrect settings applied at multiple points of training, eg. if you accidentally delete the `.json` cache files from your dataset and switch the data backend config to use square images rather than aspect-crops. This will result in an inconsistent data cache, which can be corrected by setting `scan_for_errors` to `true` in your `multidatabackend.json` configuration file. When this scan runs, it relies on the setting of `--vae_cache_scan_behaviour` to determine how to resolve the inconsistency: `recreate` (the default) will remove the offending cache entry so that it can be recreated, and `sync` will update the bucket metadata to reflect the reality of the real training sample. Recommended value: `recreate`.

### `--dataloader_prefetch`

- **What**: Retrieve batches ahead-of-time.
- **Why**: Especially when using large batch sizes, training will "pause" while samples are retrieved from disk (even NVMe), impacting GPU utilisation metrics. Enabling dataloader prefetch will keep a buffer full of entire batches, so that they can be loaded instantly.

> ⚠️ This is really only relevant for H100 or better at a low resolution where I/O becomes the bottleneck. For most other use cases, it is an unnecessary complexity.

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

### `--resolution_type`

- **What**: This tells SimpleTuner whether to use `area` size calculations or `pixel` edge calculations. A hybrid approach of `pixel_area` is also supported, which allows using pixel instead of megapixel for `area` measurements.
- **Options**: 
  - `resolution_type=pixel_area`
    - A `resolution` value of 1024 will be internally mapped to an accurate area measurement for efficient aspect bucketing.
    - Example resulting sizes for `1024`: 1024x1024, 1216x832, 832x1216
  - `resolution_type=pixel`
    - All images in the dataset will have their smaller edge resized to this resolution for training, which could result in a lot of VRAM use due to the size of the resulting images.
    - Example resulting sizes for `1024`: 1024x1024, 1766x1024, 1024x1766
  - `resolution_type=area`
    - **Deprecated**. Use `pixel_area` instead.

### `--resolution`

- **What**: Input image resolution expressed in pixel edge length
- **Default**: 1024
- **Note**: This is the global default, if a dataset does not have a resolution set.

### `--validation_resolution`

- **What**: Output image resolution, measured in pixels, or, formatted as: `widthxheight`, as in `1024x1024`. Multiple resolutions can be defined, separated by commas.
- **Why**: All images generated during validation will be this resolution. Useful if the model is being trained with a different resolution.

### `--evaluation_type`

- **What**: Enable CLIP evaluation of generated images during validations.
- **Why**: CLIP scores calculate the distance of the generated image features to the provided validation prompt. This can give an idea of whether prompt adherence is improving, though it requires a large number of validation prompts to have any meaningful value.
- **Options**: "none" or "clip"

### `--crop`

- **What**: When `--crop=true` is supplied, SimpleTuner will crop all (new) images in the training dataset. It will not re-process old images.
- **Why**: Training on cropped images seems to result in better fine detail learning, especially on SDXL models.

### `--crop_style`

- **What**: When `--crop=true`, the trainer may be instructed to crop in different ways.
- **Why**: The `crop_style` option can be set to `center` (or `centre`) for a classic centre-crop, `corner` to elect for the lowest-right corner, `face` to detect and centre upon the largest subject face, and `random` for a random image slice. Default: random.

### `--crop_aspect`

- **What**: When using `--crop=true`, the `--crop_aspect` option may be supplied with a value of `square` or `preserve`.
- **Options**: If cropping is enabled, default behaviour is to crop all images to a square aspect ratio.
  - `crop_aspect=preserve` will crop images to a size matching their original aspect ratio.
  - `crop_aspect=closest` will use the closest value from `crop_aspect_buckets`
  - `crop_aspect=random` will use a random aspect value from `crop_aspect_buckets` without going too far - it will use square crops if your aspects are incompatible
  - `crop_aspect=square` will use the standard square crop style

### `--caption_strategy`

- **What**: Strategy for deriving image captions. **Choices**: `textfile`, `filename`, `parquet`, `instanceprompt`
- **Why**: Determines how captions are generated for training images.
  - `textfile` will use the contents of a `.txt` file with the same filename as the image
  - `filename` will apply some cleanup to the filename before using it as the caption.
  - `parquet` requires a parquet file to be present in the dataset, and will use the `caption` column as the caption unless `parquet_caption_column` is provided. All captions must be present unless a `parquet_fallback_caption_column` is provided.
  - `instanceprompt` will use the value for `instance_prompt` in the dataset config as the prompt for every image in the dataset.

---

## 🎛 Training Parameters

### `--num_train_epochs`

- **What**: Number of training epochs (the number of times that all images are seen). Setting this to 0 will allow `--max_train_steps` to take precedence.
- **Why**: Determines the number of image repeats, which impacts the duration of the training process. More epochs tends to result in overfitting, but might be required to pick up the concepts you wish to train in. A reasonable value might be from 5 to 50.

### `--max_train_steps`

- **What**: Number of training steps to exit training after. If set to 0, will allow `--num_train_epochs` to take priority.
- **Why**: Useful for shortening the length of training.

### `--ignore_final_epochs`

- **What**: Ignore the final counted epochs in favour of `--max_train_steps`.
- **Why**: When changing the dataloader length, training may end earlier than you want because the epoch calculation changes. This option will ignore the final epochs and instead continue to train until `--max_train_steps` is reached.

### `--learning_rate`

- **What**: Initial learning rate after potential warmup.
- **Why**: The learning rate behaves as a sort of "step size" for gradient updates - too high, and we overstep the solution. Too low, and we never reach the ideal solution. A minimal value for a `full` tune might be as low as `1e-7` to a max of `1e-6` while for `lora` tuning a minimal value might be `1e-5` with a maximal value as high as `1e-3`. When a higher learning rate is used, it's advantageous to use an EMA network with a learning rate warmup - see `--use_ema`, `--lr_warmup_steps`, and `--lr_scheduler`.

### `--lr_scheduler`

- **What**: How to scale the learning rate over time.
- **Choices**: constant, constant_with_warmup, cosine, cosine_with_restarts, **polynomial** (recommended), linear
- **Why**: Models benefit from continual learning rate adjustments to further explore the loss landscape. A cosine schedule is used as the default; this allows the training to smoothly transition between two extremes. If using a constant learning rate, it is common to select a too-high or too-low value, causing divergence (too high) or getting stuck in a local minima (too low). A polynomial schedule is best paired with a warmup, where it will gradually approach the `learning_rate` value before then slowing down and approaching `--lr_end` by the end.

### `--optimizer`

- **What**: The optimizer to use for training.
- **Choices**: adamw_bf16, ao-adamw8bit, ao-adamw4bit, ao-adamfp8, ao-adamwfp8, adamw_schedulefree, adamw_schedulefree+aggressive, adamw_schedulefree+no_kahan, optimi-stableadamw, optimi-adamw, optimi-lion, optimi-radam, optimi-ranger, optimi-adan, optimi-adam, optimi-sgd, soap, bnb-adagrad, bnb-adagrad8bit, bnb-adam, bnb-adam8bit, bnb-adamw, bnb-adamw8bit, bnb-adamw-paged, bnb-adamw8bit-paged, bnb-lion, bnb-lion8bit, bnb-lion-paged, bnb-lion8bit-paged, bnb-ademamix, bnb-ademamix8bit, bnb-ademamix-paged, bnb-ademamix8bit-paged, prodigy

> Note: Some optimisers may not be available on non-NVIDIA hardware.

### `--optimizer_config`

- **What**: Tweak optimizer settings.
- **Why**: Because optimizers have so many different settings, it's not feasible to provide a command-line argument for each one. Instead, you can provide a comma-separated list of values to override any of the default settings.
- **Example**: You may wish to set the `d_coef` for the **prodigy** optimizer: `--optimizer_config=d_coef=0.1`

> Note: Optimizer betas are overridden using dedicated parameters, `--optimizer_beta1`, `--optimizer_beta2`.

### `--train_batch_size`

- **What**: Batch size for the training data loader.
- **Why**: Affects the model's memory consumption, convergence quality, and training speed. The higher the batch size, the better the results will be, but a very high batch size might result in overfitting or destabilized training, as well as increasing the duration of the training session unnecessarily. Experimentation is warranted, but in general, you want to try to max out your video memory while not decreasing the training speed.

### `--gradient_accumulation_steps`

- **What**: Number of update steps to accumulate before performing a backward/update pass, essentially splitting the work over multiple batches to save memory at the cost of a higher training runtime.
- **Why**: Useful for handling larger models or datasets.

> Note: Do not enable fused backward pass for any optimizers when using gradient accumulation steps.

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
- **Why**: Enables integration with platforms like TensorBoard, wandb, or comet_ml for monitoring. Use multiple values separated by a comma to report to multiple trackers;
- **Choices**: wandb, tensorboard, comet_ml

# Environment configuration variables

The above options apply for the most part, to `config.json` - but some entries must be set inside `config.env` instead.

- `TRAINING_NUM_PROCESSES` should be set to the number of GPUs in the system. For most use-cases, this is enough to enable DistributedDataParallel (DDP) training
- `TRAINING_DYNAMO_BACKEND` defaults to `no` but can be set to `inductor` for substantial speed improvements on NVIDIA hardware
- `SIMPLETUNER_LOG_LEVEL` defaults to `INFO` but can be set to `DEBUG` to add more information for issue reports into `debug.log`
- `VENV_PATH` can be set to the location of your python virtual env, if it is not in the typical `.venv` location
- `ACCELERATE_EXTRA_ARGS` can be left unset, or, contain extra arguments to add like `--multi_gpu` or FSDP-specific flags

---

This is a basic overview meant to help you get started. For a complete list of options and more detailed explanations, please refer to the full specification:

```
usage: train.py [-h] [--snr_gamma SNR_GAMMA] [--use_soft_min_snr]
                [--soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA]
                --model_family
                {sd1x,sd2x,sd3,deepfloyd,sana,sdxl,kolors,flux,wan,ltxvideo,pixart_sigma,omnigen,hidream,auraflow,lumina2,cosmos2image}
                [--model_flavour {1.5,1.4,dreamshaper,realvis,digitaldiffusion,pseudoflex-v2,pseudojourney,2.1,2.0,medium,large,i-medium-400m,i-large-900m,i-xlarge-4.3b,ii-medium-450m,ii-large-1.2b,sana1.5-4.8b-1024,sana1.5-1.6b-1024,sana1.0-1.6b-2048,sana1.0-1.6b-1024,sana1.0-600m-1024,sana1.0-600m-512,base-1.0,refiner-1.0,base-0.9,refiner-0.9,1.0,dev,schnell,kontext,t2v-480p-1.3b-2.1,t2v-480p-14b-2.1,0.9.5,0.9.0,900M-1024-v0.6,900M-1024-v0.7-stage1,900M-1024-v0.7-stage2,600M-512,600M-1024,600M-2048,v1,dev,full,fast,v0.3,v0.2,v0.1,2.0,2b,14b}]
                [--model_type {full,lora}] [--loss_type {l2,huber,smooth_l1}]
                [--huber_schedule {snr,exponential,constant}]
                [--huber_c HUBER_C] [--hidream_use_load_balancing_loss]
                [--hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT]
                [--flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,all+ffs+embedder,all+ffs+embedder+controlnet}]
                [--flow_sigmoid_scale FLOW_SIGMOID_SCALE]
                [--flux_fast_schedule] [--flow_use_uniform_schedule]
                [--flow_use_beta_schedule]
                [--flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA]
                [--flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA]
                [--flow_schedule_shift FLOW_SCHEDULE_SHIFT]
                [--flow_schedule_auto_shift]
                [--flux_guidance_mode {constant,random-range}]
                [--flux_guidance_value FLUX_GUIDANCE_VALUE]
                [--flux_guidance_min FLUX_GUIDANCE_MIN]
                [--flux_guidance_max FLUX_GUIDANCE_MAX]
                [--flux_attention_masked_training]
                [--ltx_train_mode {t2v,i2v}] [--ltx_i2v_prob LTX_I2V_PROB]
                [--ltx_protect_first_frame]
                [--ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION]
                [--t5_padding {zero,unmodified}]
                [--sd3_clip_uncond_behaviour {empty_string,zero}]
                [--sd3_t5_uncond_behaviour {empty_string,zero}]
                [--lora_type {standard,lycoris}]
                [--peft_lora_mode {standard,singlora}]
                [--singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS]
                [--lora_init_type {default,gaussian,loftq,olora,pissa}]
                [--init_lora INIT_LORA] [--lora_rank LORA_RANK]
                [--lora_alpha LORA_ALPHA] [--lora_dropout LORA_DROPOUT]
                [--lycoris_config LYCORIS_CONFIG]
                [--init_lokr_norm INIT_LOKR_NORM]
                [--conditioning_multidataset_sampling {combined,random}]
                [--control] [--controlnet]
                [--controlnet_custom_config CONTROLNET_CUSTOM_CONFIG]
                [--controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH]
                [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                [--pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH]
                [--pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER]
                [--pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH]
                [--pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER]
                [--pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH]
                [--pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH]
                [--prediction_type {epsilon,v_prediction,sample,flow_matching}]
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
                [--vae_batch_size VAE_BATCH_SIZE] [--vae_enable_tiling]
                [--vae_enable_slicing]
                [--vae_cache_scan_behaviour {recreate,sync}]
                [--vae_cache_ondemand] [--compress_disk_cache]
                [--aspect_bucket_disable_rebuild] [--keep_vae_loaded]
                [--skip_file_discovery SKIP_FILE_DISCOVERY]
                [--revision REVISION] [--variant VARIANT]
                [--preserve_data_backend_cache] [--use_dora]
                [--override_dataset_config] [--cache_dir_text CACHE_DIR_TEXT]
                [--cache_dir_vae CACHE_DIR_VAE]
                [--data_backend_config DATA_BACKEND_CONFIG]
                [--data_backend_sampling {uniform,auto-weighting}]
                [--ignore_missing_files] [--write_batch_size WRITE_BATCH_SIZE]
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
                [--framerate FRAMERATE] [--resolution RESOLUTION]
                [--resolution_type {pixel,area,pixel_area}]
                [--aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}]
                [--aspect_bucket_alignment {8,64}]
                [--minimum_image_size MINIMUM_IMAGE_SIZE]
                [--maximum_image_size MAXIMUM_IMAGE_SIZE]
                [--target_downsample_size TARGET_DOWNSAMPLE_SIZE]
                [--train_text_encoder]
                [--tokenizer_max_length TOKENIZER_MAX_LENGTH]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--num_train_epochs NUM_TRAIN_EPOCHS]
                [--max_train_steps MAX_TRAIN_STEPS] [--ignore_final_epochs]
                [--checkpointing_steps CHECKPOINTING_STEPS]
                [--checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS]
                [--checkpointing_use_tempdir]
                [--checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT]
                [--checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT]
                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--gradient_checkpointing]
                [--gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL]
                [--learning_rate LEARNING_RATE]
                [--text_encoder_lr TEXT_ENCODER_LR] [--lr_scale]
                [--lr_scale_sqrt]
                [--lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                [--lr_warmup_steps LR_WARMUP_STEPS]
                [--lr_num_cycles LR_NUM_CYCLES] [--lr_power LR_POWER]
                [--distillation_method {lcm,dcm}]
                [--distillation_config DISTILLATION_CONFIG] [--use_ema]
                [--ema_device {cpu,accelerator}]
                [--ema_validation {none,ema_only,comparison}] [--ema_cpu_only]
                [--ema_foreach_disable]
                [--ema_update_interval EMA_UPDATE_INTERVAL]
                [--ema_decay EMA_DECAY] [--non_ema_revision NON_EMA_REVISION]
                [--offload_during_startup]
                [--offload_param_path OFFLOAD_PARAM_PATH] --optimizer
                {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,bnb-adagrad,bnb-adagrad8bit,bnb-adam,bnb-adam8bit,bnb-adamw,bnb-adamw8bit,bnb-adamw-paged,bnb-adamw8bit-paged,bnb-lion,bnb-lion8bit,bnb-lion-paged,bnb-lion8bit-paged,bnb-ademamix,bnb-ademamix8bit,bnb-ademamix-paged,bnb-ademamix8bit-paged,prodigy}
                [--optimizer_config OPTIMIZER_CONFIG]
                [--optimizer_cpu_offload_method {none}]
                [--optimizer_offload_gradients] [--fuse_optimizer]
                [--optimizer_beta1 OPTIMIZER_BETA1]
                [--optimizer_beta2 OPTIMIZER_BETA2]
                [--optimizer_release_gradients] [--adam_beta1 ADAM_BETA1]
                [--adam_beta2 ADAM_BETA2]
                [--adam_weight_decay ADAM_WEIGHT_DECAY]
                [--adam_epsilon ADAM_EPSILON] [--prodigy_steps PRODIGY_STEPS]
                [--max_grad_norm MAX_GRAD_NORM]
                [--grad_clip_method {value,norm}] [--push_to_hub]
                [--push_checkpoints_to_hub] [--hub_model_id HUB_MODEL_ID]
                [--model_card_note MODEL_CARD_NOTE]
                [--model_card_safe_for_work] [--logging_dir LOGGING_DIR]
                [--disable_benchmark] [--evaluation_type {clip,none}]
                [--eval_dataset_pooling]
                [--pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH]
                [--validation_on_startup] [--validation_seed_source {gpu,cpu}]
                [--validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH]
                [--validation_torch_compile]
                [--validation_torch_compile_mode {max-autotune,reduce-overhead,default}]
                [--validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS]
                [--validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START]
                [--validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP]
                [--validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE]
                [--sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION]
                [--disable_tf32] [--validation_using_datasets]
                [--webhook_config WEBHOOK_CONFIG]
                [--webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL]
                [--report_to REPORT_TO] [--tracker_run_name TRACKER_RUN_NAME]
                [--tracker_project_name TRACKER_PROJECT_NAME]
                [--tracker_image_layout {gallery,table}]
                [--validation_prompt VALIDATION_PROMPT]
                [--validation_prompt_library]
                [--user_prompt_library USER_PROMPT_LIBRARY]
                [--validation_negative_prompt VALIDATION_NEGATIVE_PROMPT]
                [--num_validation_images NUM_VALIDATION_IMAGES]
                [--validation_disable] [--validation_steps VALIDATION_STEPS]
                [--validation_stitch_input_location {left,right}]
                [--eval_steps_interval EVAL_STEPS_INTERVAL]
                [--eval_timesteps EVAL_TIMESTEPS]
                [--num_eval_images NUM_EVAL_IMAGES]
                [--eval_dataset_id EVAL_DATASET_ID]
                [--validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS]
                [--validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES]
                [--validation_resolution VALIDATION_RESOLUTION]
                [--validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++}]
                [--validation_disable_unconditional] [--enable_watermark]
                [--mixed_precision {bf16,fp16,fp8,no}]
                [--gradient_precision {unmodified,fp32}]
                [--quantize_via {cpu,accelerator}]
                [--base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--quantize_activations]
                [--base_model_default_dtype {bf16,fp32}]
                [--text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--local_rank LOCAL_RANK] [--fuse_qkv_projections]
                [--attention_mechanism {diffusers,xformers,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}]
                [--sageattention_usage {training,inference,training+inference}]
                [--set_grads_to_none] [--noise_offset NOISE_OFFSET]
                [--noise_offset_probability NOISE_OFFSET_PROBABILITY]
                [--masked_loss_probability MASKED_LOSS_PROBABILITY]
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
                [--disable_bucket_pruning] [--offset_noise]
                [--input_perturbation INPUT_PERTURBATION]
                [--input_perturbation_steps INPUT_PERTURBATION_STEPS]
                [--lr_end LR_END] [--i_know_what_i_am_doing]
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
  --model_family {sd1x,sd2x,sd3,deepfloyd,sana,sdxl,kolors,flux,wan,ltxvideo,pixart_sigma,omnigen,hidream,auraflow,lumina2,cosmos2image}
                        The model family to train. This option is required.
  --model_flavour {1.5,1.4,dreamshaper,realvis,digitaldiffusion,pseudoflex-v2,pseudojourney,2.1,2.0,medium,large,i-medium-400m,i-large-900m,i-xlarge-4.3b,ii-medium-450m,ii-large-1.2b,sana1.5-4.8b-1024,sana1.5-1.6b-1024,sana1.0-1.6b-2048,sana1.0-1.6b-1024,sana1.0-600m-1024,sana1.0-600m-512,base-1.0,refiner-1.0,base-0.9,refiner-0.9,1.0,dev,schnell,kontext,t2v-480p-1.3b-2.1,t2v-480p-14b-2.1,0.9.5,0.9.0,900M-1024-v0.6,900M-1024-v0.7-stage1,900M-1024-v0.7-stage2,600M-512,600M-1024,600M-2048,v1,dev,full,fast,v0.3,v0.2,v0.1,2.0,2b,14b}
                        Certain models require designating a given flavour to
                        reference configurations from. The value for this
                        depends on the model that is selected. Currently
                        supported values: sd1x: ['1.5', '1.4', 'dreamshaper',
                        'realvis'] sd2x: ['digitaldiffusion', 'pseudoflex-v2',
                        'pseudojourney', '2.1', '2.0'] sd3: ['medium',
                        'large'] deepfloyd: ['i-medium-400m', 'i-large-900m',
                        'i-xlarge-4.3b', 'ii-medium-450m', 'ii-large-1.2b']
                        sana: ['sana1.5-4.8b-1024', 'sana1.5-1.6b-1024',
                        'sana1.0-1.6b-2048', 'sana1.0-1.6b-1024',
                        'sana1.0-600m-1024', 'sana1.0-600m-512'] sdxl:
                        ['base-1.0', 'refiner-1.0', 'base-0.9', 'refiner-0.9']
                        kolors: ['1.0'] flux: ['dev', 'schnell', 'kontext']
                        wan: ['t2v-480p-1.3b-2.1', 't2v-480p-14b-2.1']
                        ltxvideo: ['0.9.5', '0.9.0'] pixart_sigma:
                        ['900M-1024-v0.6', '900M-1024-v0.7-stage1',
                        '900M-1024-v0.7-stage2', '600M-512', '600M-1024',
                        '600M-2048'] omnigen: ['v1'] hidream: ['dev', 'full',
                        'fast'] auraflow: ['v0.3', 'v0.2', 'v0.1'] lumina2:
                        ['2.0'] cosmos2image: ['2b', '14b']
  --model_type {full,lora}
                        The training type to use. 'full' will train the full
                        model, while 'lora' will train the LoRA model. LoRA is
                        a smaller model that can be used for faster training.
  --loss_type {l2,huber,smooth_l1}
                        The loss function to use during training. 'l2' is the
                        default, but 'huber' and 'smooth_l1' are also
                        available. Huber loss is less sensitive to outliers
                        than L2 loss, and smooth L1 is a combination of L1 and
                        L2 loss. When using Huber loss, it will be scheduled
                        via --huber_schedule and --huber_c. NOTE: When
                        training flow-matching models, L2 loss will always be
                        in use.
  --huber_schedule {snr,exponential,constant}
                        constant: Uses a fixed huber_c value. exponential:
                        Exponentially decays huber_c based on timestep snr:
                        Adjusts huber_c based on signal-to-noise ratio.
                        default: snr.
  --huber_c HUBER_C     The huber_c value to use for Huber loss. This is the
                        threshold at which the loss function transitions from
                        L2 to L1. A lower value will make the loss function
                        more sensitive to outliers, while a higher value will
                        make it less sensitive. The default value is 0.1,
                        which is a good starting point for most models.
  --hidream_use_load_balancing_loss
                        When set, will use the load balancing loss for HiDream
                        training. This is an experimental implementation.
  --hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT
                        When set, will use augment the load balancing loss for
                        HiDream training. This is an experimental
                        implementation.
  --flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,all+ffs+embedder,all+ffs+embedder+controlnet}
                        This option only applies to Standard LoRA, not
                        Lycoris. Flux has single and joint attention blocks.
                        By default, all attention layers are trained, but not
                        the feed-forward layers If 'mmdit' is provided, the
                        text input layers will not be trained. If 'context' is
                        provided, then ONLY the text attention layers are
                        trained If 'context+ffs' is provided, then text
                        attention and text feed-forward layers are trained.
                        This is somewhat similar to text-encoder-only training
                        in earlier SD versions. If 'all' is provided, all
                        layers will be trained, minus feed-forward. If
                        'all+ffs' is provided, all layers will be trained
                        including feed-forward. If 'ai-toolkit' is provided,
                        all layers will be trained including feed-forward and
                        norms (based on ostris/ai-toolkit). If 'tiny' is
                        provided, only two layers will be trained. If 'nano'
                        is provided, only one layers will be trained.
  --flow_sigmoid_scale FLOW_SIGMOID_SCALE
                        Scale factor for sigmoid timestep sampling for flow-
                        matching models.
  --flux_fast_schedule  An experimental feature to train Flux.1S using a noise
                        schedule closer to what it was trained with, which has
                        improved results in short experiments. Thanks to
                        @mhirki for the contribution.
  --flow_use_uniform_schedule
                        Whether or not to use a uniform schedule instead of
                        sigmoid for flow-matching noise schedule. Using
                        uniform sampling may cause a bias toward dark images,
                        and should be used with caution.
  --flow_use_beta_schedule
                        Whether or not to use a beta schedule instead of
                        sigmoid for flow-matching. The default values of alpha
                        and beta approximate a sigmoid.
  --flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA
                        The alpha value of the flow-matching beta schedule.
                        Default is 2.0
  --flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA
                        The beta value of the flow-matching beta schedule.
                        Default is 2.0
  --flow_schedule_shift FLOW_SCHEDULE_SHIFT
                        Shift the noise schedule. This is a value between 0
                        and ~4.0, where 0 disables the timestep-dependent
                        shift, and anything greater than 0 will shift the
                        timestep sampling accordingly. Sana and SD3 were
                        trained with a shift value of 3. This value can change
                        how contrast/brightness are learnt by the model, and
                        whether fine details are ignored or accentuated. A
                        higher value will focus more on large compositional
                        features, and a lower value will focus on the high
                        frequency fine details.
  --flow_schedule_auto_shift
                        Shift the noise schedule depending on image
                        resolution. The shift value calculation is taken from
                        the official Flux inference code. Shift value is
                        math.exp(1.15) = 3.1581 for a pixel count of 1024px *
                        1024px. The shift value grows exponentially with
                        higher pixel counts. It is a good idea to train on a
                        mix of different resolutions when this option is
                        enabled. You may need to lower your learning rate with
                        this enabled.
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
  --flux_attention_masked_training
                        Use attention masking while training flux. This can be
                        a destructive operation, unless finetuning a model
                        which was already trained with it.
  --ltx_train_mode {t2v,i2v}
                        This value will be the default for all video datasets
                        that do not have their own i2v settings defined. By
                        default, we enable i2v mode, but it can be switched to
                        t2v for your convenience.
  --ltx_i2v_prob LTX_I2V_PROB
                        Probability in [0,1] of applying i2v (image-to-video)
                        style training. If random.random() < i2v_prob during
                        training, partial or complete first-frame protection
                        will be triggered (depending on
                        --ltx_protect_first_frame). If set to 0.0, no i2v
                        logic is applied (pure t2v). Default: 0.1 (from
                        finetuners project)
  --ltx_protect_first_frame
                        If specified, fully protect the first frame whenever
                        i2v logic is triggered (see --ltx_i2v_prob). This
                        means the first frame is never noised or denoised,
                        effectively pinned to the original content.
  --ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION
                        Maximum fraction of noise to introduce into the first
                        frame when i2v is triggered and the first frame is not
                        fully protected. For instance, a value of 0.05 means
                        the first frame can have up to 5 percent random noise
                        mixed in, preserving 95 percent of the original
                        content. Ignored if --ltx_protect_first_frame is set.
  --t5_padding {zero,unmodified}
                        The padding behaviour for Flux and SD3. 'zero' will
                        pad the input with zeros. The default is 'unmodified',
                        which will not pad the input.
  --sd3_clip_uncond_behaviour {empty_string,zero}
                        SD3 can be trained using zeroed prompt embeds during
                        unconditional dropout, or an encoded empty string may
                        be used instead (the default). Changing this value may
                        stabilise or destabilise training. The default is
                        'empty_string'.
  --sd3_t5_uncond_behaviour {empty_string,zero}
                        Override the value of unconditional prompts from T5
                        embeds. The default is to follow the value of
                        --sd3_clip_uncond_behaviour.
  --lora_type {standard,lycoris}
                        When training using --model_type=lora, you may specify
                        a different type of LoRA to train here. standard
                        refers to training a vanilla LoRA via PEFT, lycoris
                        refers to training with KohakuBlueleaf's library of
                        the same name.
  --peft_lora_mode {standard,singlora}
                        When training using --model_type=lora, you may specify
                        a different type of LoRA to train here. standard
                        refers to training a vanilla LoRA via PEFT, singlora
                        refers to training with SingLoRA, a more efficient
                        representation.
  --singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS
                        When using SingLoRA, this specifies the number of
                        ramp-up steps. For diffusion models, it seems that
                        ramp-up steps are harmful to training. (default: 0)
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
  --init_lora INIT_LORA
                        Specify an existing LoRA or LyCORIS safetensors file
                        to initialize the adapter and continue training, if a
                        full checkpoint is not available.
  --lora_rank LORA_RANK
                        The dimension of the LoRA update matrices.
  --lora_alpha LORA_ALPHA
                        The alpha value for the LoRA model. This is the
                        learning rate for the LoRA update matrices.
  --lora_dropout LORA_DROPOUT
                        LoRA dropout randomly ignores neurons during training.
                        This can help prevent overfitting.
  --lycoris_config LYCORIS_CONFIG
                        The location for the JSON file of the Lycoris
                        configuration.
  --init_lokr_norm INIT_LOKR_NORM
                        Setting this turns on perturbed normal initialization
                        of the LyCORIS LoKr PEFT layers. A good value is
                        between 1e-4 and 1e-2.
  --conditioning_multidataset_sampling {combined,random}
                        How to sample from multiple conditioning datasets: -
                        'combined': Use all conditioning images from all
                        datasets, increases VRAM requirements a lot. -
                        'random': Randomly select one conditioning dataset per
                        training sample (default) Random mode uses
                        deterministic selection based on image path and epoch.
  --control             If set, channel-wise control style training will be
                        used, where a conditioning input image is required
                        alongside the training data.
  --controlnet          If set, ControlNet style training will be used, where
                        a conditioning input image is required alongside the
                        training data.
  --controlnet_custom_config CONTROLNET_CUSTOM_CONFIG
                        When training certain ControlNet models (eg. HiDream)
                        you may set a config containing keys like num_layers
                        or num_single_layers to adjust the resulting
                        ControlNet size. This is not supported by most models,
                        and may be ignored if the model does not support it.
  --controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH
                        When provided alongside --controlnet, this will
                        specify ControlNet model weights to preload from the
                        hub.
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from
                        huggingface.co/models. Some model architectures
                        support loading single-file .safetensors directly.
                        Note that when using single-file safetensors, the
                        tokeniser and noise schedule configs will be used from
                        the vanilla upstream Huggingface repository, which
                        requires network access. If you are training on a
                        machine without network access, you should pre-
                        download the entire Huggingface model repository
                        instead of using single-file loader.
  --pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH
                        Path to pretrained transformer model or model
                        identifier from huggingface.co/models.
  --pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER
                        The subfolder to load the transformer model from. Use
                        'none' for a flat directory.
  --pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH
                        Path to pretrained unet model or model identifier from
                        huggingface.co/models.
  --pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER
                        The subfolder to load the unet model from. Use 'none'
                        for a flat directory.
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
  --prediction_type {epsilon,v_prediction,sample,flow_matching}
                        For models which support it, you can supply this value
                        to override the prediction type. Choose between
                        ['epsilon', 'v_prediction', 'sample',
                        'flow_matching']. This may be needed for some SDXL
                        derivatives that are trained using v_prediction or
                        flow_matching.
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
                        trained on. Just to throw a wrench into the works,
                        Kolors was trained on 1100 timesteps.
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
  --vae_enable_tiling   If set, will enable tiling for VAE caching. This is
                        useful for very large images when VRAM is limited.
                        This may be required for 2048px VAE caching on 24G
                        accelerators, in addition to reducing
                        --vae_batch_size.
  --vae_enable_slicing  If set, will enable slicing for VAE caching. This is
                        useful for video models.
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
  --data_backend_sampling {uniform,auto-weighting}
                        When using multiple data backends, the sampling
                        weighting can be set to 'uniform' or 'auto-weighting'.
                        The default value is 'auto-weighting', which will
                        automatically adjust the sampling weights based on the
                        number of images in each backend. 'uniform' will
                        sample from each backend equally.
  --ignore_missing_files
                        This option will disable the check for files that have
                        been deleted or removed from your data directory. This
                        would allow training on large datasets without keeping
                        the associated images on disk, though it's not
                        recommended and is not a supported feature. Use with
                        caution, as it mostly exists for experimentation.
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
                        you've modified any of the existing prompts.
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
  --framerate FRAMERATE
                        By default, SimpleTuner will use a framerate of 25 for
                        training and inference on video models. You are on
                        your own if you modify this value, but it is provided
                        for your convenience.
  --resolution RESOLUTION
                        The resolution for input images, all the images in the
                        train/validation dataset will be resized to this
                        resolution. If using --resolution_type=area, this
                        float value represents megapixels.
  --resolution_type {pixel,area,pixel_area}
                        Resizing images maintains aspect ratio. This defines
                        the resizing strategy. If 'pixel', the images will be
                        resized to the resolution by the shortest pixel edge,
                        if the target size does not match the current size. If
                        'area', the images will be resized so the pixel area
                        is this many megapixels. Common rounded values such as
                        `0.5` and `1.0` will be implicitly adjusted to their
                        squared size equivalents. If 'pixel_area', the pixel
                        value (eg. 1024) will be converted to the proper value
                        for 'area', and then calculate everything the same as
                        'area' would.
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
                        set to 32, but all other training defaults to 64. You
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
                        The maximum sequence length of the tokenizer output,
                        which defines the sequence length of text embed
                        outputs. If not set, will default to the tokenizer's
                        max length. Unfortunately, this option only applies to
                        T5 models, and due to the biases inducted by sequence
                        length, changing it will result in potentially
                        catastrophic model collapse. This option causes poor
                        training results. This is normal, and can be expected
                        from changing this value.
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size (per device) for the training dataloader.
  --num_train_epochs NUM_TRAIN_EPOCHS
  --max_train_steps MAX_TRAIN_STEPS
                        Total number of training steps to perform. If
                        provided, overrides num_train_epochs.
  --ignore_final_epochs
                        When provided, the max epoch counter will not
                        determine the end of the training run. Instead, it
                        will end when it hits --max_train_steps.
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
  --checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS
                        Functions similarly to --checkpointing_steps, but only
                        a single rolling checkpoint is ever saved and this
                        checkpoint does not count towards the value of
                        --checkpoints_total_limit. Useful for when the
                        underlying runtime environment may be prone to
                        spontaneous interruption (e.g. spot instances,
                        unreliable hardware, etc) and saving state more
                        frequently is beneficial. This allows one to save
                        normal checkpoints at a reasonable cadence but save a
                        rolling checkpoint more frequently so as to avoid
                        losing progress.
  --checkpointing_use_tempdir
                        Write saved checkpoint directories to a temporary name
                        and atomically rename after successfully writing all
                        state. This ensures that a given checkpoint will never
                        be considered for resuming if it wasn't fully written
                        out - as the state cannot be guaranteed. Useful for
                        when the underlying runtime environment may be prone
                        to spontaneous interruption (e.g. spot instances,
                        unreliable hardware, etc).
  --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
                        Max number of checkpoints to store.
  --checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT
                        Max number of rolling checkpoints to store. One almost
                        always wants this to be 1, but there could be a
                        usecase where one desires to run a shorter window of
                        more frequent checkpoints alongside a normal
                        checkpointing interval done at less frequent steps.
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
  --gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL
                        Some models (Flux, SDXL, SD1.x/2.x, SD3) can have
                        their gradient checkpointing limited to every nth
                        block. This can speed up training but will use more
                        memory with larger intervals.
  --learning_rate LEARNING_RATE
                        Initial learning rate (after the potential warmup
                        period) to use. When using a cosine or sine schedule,
                        --learning_rate defines the maximum learning rate.
  --text_encoder_lr TEXT_ENCODER_LR
                        Learning rate for the text encoder. If not provided,
                        the value of --learning_rate will be used.
  --lr_scale            Scale the learning rate by the number of GPUs,
                        gradient accumulation steps, and batch size.
  --lr_scale_sqrt       If using --lr-scale, use the square root of (number of
                        GPUs * gradient accumulation steps * batch size).
  --lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        The scheduler type to use. Default: sine
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps for the warmup in the lr scheduler.
  --lr_num_cycles LR_NUM_CYCLES
                        Number of hard resets of the lr in
                        cosine_with_restarts scheduler.
  --lr_power LR_POWER   Power factor of the polynomial scheduler.
  --distillation_method {lcm,dcm}
                        The distillation method to use. Currently, LCM and DCM
                        are supported via LoRA. This will apply the selected
                        distillation method to the model.
  --distillation_config DISTILLATION_CONFIG
                        The config for your selected distillation method. If
                        passing it via config.json, simply provide the JSON
                        object directly.
  --use_ema             Whether to use EMA (exponential moving average) model.
                        Works with LoRA, Lycoris, and full training.
  --ema_device {cpu,accelerator}
                        The device to use for the EMA model. If set to
                        'accelerator', the EMA model will be placed on the
                        accelerator. This provides the fastest EMA update
                        times, but is not ultimately necessary for EMA to
                        function.
  --ema_validation {none,ema_only,comparison}
                        When 'none' is set, no EMA validation will be done.
                        When using 'ema_only', the validations will rely
                        mostly on the EMA weights. When using 'comparison'
                        (default) mode, the validations will first run on the
                        checkpoint before also running for the EMA weights. In
                        comparison mode, the resulting images will be provided
                        side-by-side.
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
  --offload_during_startup
                        When set, text encoders, the VAE, or other models will
                        be moved to and from the CPU as needed, which can slow
                        down startup, but saves VRAM. This is useful for video
                        models or high-resolution pre-caching of latent
                        embeds.
  --offload_param_path OFFLOAD_PARAM_PATH
                        When using DeepSpeed ZeRo stage 2 or 3 with NVMe
                        offload, this may be specified to provide a path for
                        the offload.
  --optimizer {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,bnb-adagrad,bnb-adagrad8bit,bnb-adam,bnb-adam8bit,bnb-adamw,bnb-adamw8bit,bnb-adamw-paged,bnb-adamw8bit-paged,bnb-lion,bnb-lion8bit,bnb-lion-paged,bnb-lion8bit-paged,bnb-ademamix,bnb-ademamix8bit,bnb-ademamix-paged,bnb-ademamix8bit-paged,prodigy}
  --optimizer_config OPTIMIZER_CONFIG
                        When setting a given optimizer, this allows a comma-
                        separated list of key-value pairs to be provided that
                        will override the optimizer defaults. For example, `--
                        optimizer_config=decouple_lr=True,weight_decay=0.01`.
  --optimizer_cpu_offload_method {none}
                        This option is a placeholder. In the future, it will
                        allow for the selection of different CPU offload
                        methods.
  --optimizer_offload_gradients
                        When creating a CPU-offloaded optimiser, the gradients
                        can be offloaded to the CPU to save more memory.
  --fuse_optimizer      When creating a CPU-offloaded optimiser, the fused
                        optimiser could be used to save on memory, while
                        running slightly slower.
  --optimizer_beta1 OPTIMIZER_BETA1
                        The value to use for the first beta value in the
                        optimiser, which is used for the first moment
                        estimate. A range of 0.8-0.9 is common.
  --optimizer_beta2 OPTIMIZER_BETA2
                        The value to use for the second beta value in the
                        optimiser, which is used for the second moment
                        estimate. A range of 0.999-0.9999 is common.
  --optimizer_release_gradients
                        When using Optimi optimizers, this option will release
                        the gradients after the optimizer step. This can save
                        memory, but may slow down training. With Quanto, there
                        may be no benefit.
  --adam_beta1 ADAM_BETA1
                        The beta1 parameter for the Adam and other optimizers.
  --adam_beta2 ADAM_BETA2
                        The beta2 parameter for the Adam and other optimizers.
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        Weight decay to use.
  --adam_epsilon ADAM_EPSILON
                        Epsilon value for the Adam optimizer
  --prodigy_steps PRODIGY_STEPS
                        When training with Prodigy, this defines how many
                        steps it should be adjusting its learning rate for. It
                        seems to be that Diffusion models benefit from a
                        capping off of the adjustments after 25 percent of the
                        training run (dependent on batch size, repeats, and
                        epochs). It this value is not supplied, it will be
                        calculated at 25 percent of your training steps.
  --max_grad_norm MAX_GRAD_NORM
                        Clipping the max gradient norm can help prevent
                        exploding gradients, but may also harm training by
                        introducing artifacts or making it hard to train
                        artifacts away.
  --grad_clip_method {value,norm}
                        When applying --max_grad_norm, the method to use for
                        clipping the gradients. The previous default option
                        'norm' will scale ALL gradient values when any
                        outliers in the gradient are encountered, which can
                        reduce training precision. The new default option
                        'value' will clip individual gradient values using
                        this value as a maximum, which may preserve precision
                        while avoiding outliers, enhancing convergence. In
                        simple terms, the default will help the model learn
                        faster without blowing up (SD3.5 Medium was the main
                        test model). Use 'norm' to return to the old
                        behaviour.
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
  --model_card_safe_for_work
                        Hugging Face Hub requires a warning to be added to
                        models that may generate NSFW content. This is done by
                        default in SimpleTuner for safety purposes, but can be
                        disabled with this option. Additionally, removing the
                        not-for-all-audiences tag from the README.md in the
                        repo will also disable this warning on previously-
                        uploaded models.
  --logging_dir LOGGING_DIR
                        [TensorBoard](https://www.tensorflow.org/tensorboard)
                        log directory. Will default to
                        *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
  --disable_benchmark   By default, the model will be benchmarked on the first
                        batch of the first epoch. This can be disabled with
                        this option.
  --evaluation_type {clip,none}
                        Validations must be enabled for model evaluation to
                        function. The default is to use no evaluator, and
                        'clip' will use a CLIP model to evaluate the resulting
                        model's performance during validations.
  --eval_dataset_pooling
                        When provided, only the pooled evaluation results will
                        be returned in a single chart from all eval sets.
                        Without this option, all eval sets will have separate
                        charts.
  --pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH
                        Optionally provide a custom model to use for ViT
                        evaluations. The default is currently clip-vit-large-
                        patch14-336, allowing for lower patch sizes (greater
                        accuracy) and an input resolution of 336x336.
  --validation_on_startup
                        When training begins, the starting model will have
                        validation prompts run through it, for later
                        comparison.
  --validation_seed_source {gpu,cpu}
                        Some systems may benefit from using CPU-based seeds
                        for reproducibility. On other systems, this may cause
                        a TypeError. Setting this option to 'cpu' may cause
                        validation errors. If so, please set
                        SIMPLETUNER_LOG_LEVEL=DEBUG and submit debug.log to a
                        new Github issue report.
  --validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH
                        When inferencing for validations, the Lycoris model
                        will by default be run at its training strength, 1.0.
                        However, this value can be increased to a value of
                        around 1.3 or 1.5 to get a stronger effect from the
                        model.
  --validation_torch_compile
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
  --validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS
                        StabilityAI recommends a value of [7, 8, 9] for Stable
                        Diffusion 3.5 Medium. For Wan 2.1, a value of [9],
                        [10], or, [9, 10] was found to work well.
  --validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START
                        StabilityAI recommends a value of 0.01 for SLG start.
  --validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP
                        StabilityAI recommends a value of 0.2 for SLG stop.
  --validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE
                        StabilityAI recommends a value of 2.8 for SLG guidance
                        skip scaling. When adding more layers, you must
                        increase the scale, eg. adding one more layer requires
                        doubling the value given.
  --sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION
                        When generating embeds for Sana, a complex human
                        instruction will be attached to your prompt by
                        default. This is required for the Gemma model to
                        produce meaningful image caption embeds.
  --disable_tf32        Previous defaults were to disable TF32 on Ampere GPUs.
                        This option is provided to explicitly disable TF32,
                        after default configuration was updated to enable TF32
                        on Ampere GPUs.
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
  --webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL
                        When using 'raw' webhooks that receive structured
                        data, you can specify a reporting interval here for
                        training progress updates to be sent at. This does not
                        impact 'discord' webhook types.
  --report_to REPORT_TO
                        The integration to report the results and logs to.
                        Supported platforms are `"tensorboard"` (default),
                        `"wandb"` and `"comet_ml"`. Use `"all"` to report to
                        all integrations, or `"none"` to disable logging.
  --tracker_run_name TRACKER_RUN_NAME
                        The name of the run to track with the tracker.
  --tracker_project_name TRACKER_PROJECT_NAME
                        The name of the project for WandB or Tensorboard.
  --tracker_image_layout {gallery,table}
                        When running validations with multiple images, you may
                        want them all placed together in a table, row-wise.
                        Gallery mode, the default, will allow use of a slider
                        to view the historical images easily.
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
  --validation_disable  Enable to completely disable the generation of
                        validation images.
  --validation_steps VALIDATION_STEPS
                        Run validation every X steps. Validation consists of
                        running the prompt `args.validation_prompt` multiple
                        times: `args.num_validation_images` and logging the
                        images.
  --validation_stitch_input_location {left,right}
                        When set, the input image will be stitched to the left
                        of the generated image during validation. This is
                        useful for img2img models, such as DeepFloyd Stage II,
                        where the input image is used as a reference.
  --eval_steps_interval EVAL_STEPS_INTERVAL
                        When set, the model will be evaluated every X steps.
                        This is useful for monitoring the model's progress
                        during training, but it requires an eval set
                        configured in your dataloader.
  --eval_timesteps EVAL_TIMESTEPS
                        Defines how many timesteps to sample during eval. You
                        can emulate inference by setting this to the value of
                        --validation_num_inference_steps.
  --num_eval_images NUM_EVAL_IMAGES
                        If possible, this many eval images will be selected
                        from each dataset. This is used when training super-
                        resolution models such as DeepFloyd Stage II, which
                        will upscale input images from the training set during
                        validation. If using --eval_steps_interval, this will
                        be the number of batches sampled for loss
                        calculations.
  --eval_dataset_id EVAL_DATASET_ID
                        When provided, only this dataset's images will be used
                        as the eval set, to keep the training and eval images
                        split. This option only applies for img2img
                        validations, not validation loss calculations.
  --validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS
                        The default scheduler, DDIM, benefits from more steps.
                        UniPC can do well with just 10-15. For more speed
                        during validations, reduce this value. For better
                        quality, increase it. For model distilation, you will
                        likely want to keep this low.
  --validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES
                        When this is set, you can reduce the number of frames
                        from the default model value (but not go beyond that).
  --validation_resolution VALIDATION_RESOLUTION
                        Square resolution images will be output at this
                        resolution (256x256).
  --validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++}
                        When validating the model at inference time, a
                        different scheduler may be chosen. UniPC can offer
                        better speed, and Euler A can put up with
                        instabilities a bit better. For zero-terminal SNR
                        models, DDIM is the best choice. Choices: ['ddim',
                        'ddpm', 'euler', 'euler-a', 'unipc', 'dpm++'],
                        Default: None (use the model default)
  --validation_disable_unconditional
                        When set, the validation pipeline will not generate
                        unconditional samples. This is useful to speed up
                        validations with a single prompt on slower systems, or
                        if you are not interested in unconditional space
                        generations.
  --enable_watermark    The SDXL 0.9 and 1.0 licenses both require a watermark
                        be used to identify any images created to be shared.
                        Since the images created during validation typically
                        are not shared, and we want the most accurate results,
                        this watermarker is disabled by default. If you are
                        sharing the validation images, it is up to you to
                        ensure that you are complying with the license,
                        whether that is through this watermarker, or another.
  --mixed_precision {bf16,fp16,fp8,no}
                        SimpleTuner only supports bf16 training. Bf16 requires
                        PyTorch >= 1.10. on an Nvidia Ampere or later GPU, and
                        PyTorch 2.3 or newer for Apple Silicon. Default to the
                        value of accelerate config of the current system or
                        the flag passed with the `accelerate.launch` command.
                        Use this argument to override the accelerate config.
                        fp16 is offered as an experimental option, but is not
                        recommended as it is less-tested and you will likely
                        encounter errors. fp8 is another experimental option
                        that relies in TorchAO for mixed precision ops.
  --gradient_precision {unmodified,fp32}
                        One of the hallmark discoveries of the Llama 3.1 paper
                        is numeric instability when calculating gradients in
                        bf16 precision. The default behaviour when gradient
                        accumulation steps are enabled is now to use fp32
                        gradients, which is slower, but provides more accurate
                        updates.
  --quantize_via {cpu,accelerator}
                        When quantising the model, the quantisation process
                        can be done on the CPU or the accelerator. When done
                        on the accelerator (default), slightly more VRAM is
                        required, but the process completes in milliseconds.
                        When done on the CPU, the process may take upwards of
                        60 seconds, but can complete without OOM on 16G cards.
  --base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        When training a LoRA, you might want to quantise the
                        base model to a lower precision to save more VRAM. The
                        default value, 'no_change', does not quantise any
                        weights. Using 'fp4-bnb' or 'fp8-bnb' will require
                        Bits n Bytes for quantisation (NVIDIA, maybe AMD).
                        Using 'fp8-quanto' will require Quanto for
                        quantisation (Apple Silicon, NVIDIA, AMD).
  --quantize_activations
                        (EXPERIMENTAL) This option is currently unsupported,
                        and exists solely for development purposes.
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
  --text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        When training a LoRA, you might want to quantise text
                        encoder 1 to a lower precision to save more VRAM. The
                        default value is to follow base_model_precision
                        (no_change). Using 'fp4-bnb' or 'fp8-bnb' will require
                        Bits n Bytes for quantisation (NVIDIA, maybe AMD).
                        Using 'fp8-quanto' will require Quanto for
                        quantisation (Apple Silicon, NVIDIA, AMD).
  --text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        When training a LoRA, you might want to quantise text
                        encoder 2 to a lower precision to save more VRAM. The
                        default value is to follow base_model_precision
                        (no_change). Using 'fp4-bnb' or 'fp8-bnb' will require
                        Bits n Bytes for quantisation (NVIDIA, maybe AMD).
                        Using 'fp8-quanto' will require Quanto for
                        quantisation (Apple Silicon, NVIDIA, AMD).
  --text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        When training a LoRA, you might want to quantise text
                        encoder 3 to a lower precision to save more VRAM. The
                        default value is to follow base_model_precision
                        (no_change). Using 'fp4-bnb' or 'fp8-bnb' will require
                        Bits n Bytes for quantisation (NVIDIA, maybe AMD).
                        Using 'fp8-quanto' will require Quanto for
                        quantisation (Apple Silicon, NVIDIA, AMD).
  --text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        When training a LoRA, you might want to quantise text
                        encoder 4 to a lower precision to save more VRAM. The
                        default value is to follow base_model_precision
                        (no_change). Using 'fp4-bnb' or 'fp8-bnb' will require
                        Bits n Bytes for quantisation (NVIDIA, maybe AMD).
                        Using 'fp8-quanto' will require Quanto for
                        quantisation (Apple Silicon, NVIDIA, AMD).
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --fuse_qkv_projections
                        QKV projections can be fused into a single linear
                        layer. This can save memory and speed up training, but
                        may not work with all models. If you encounter issues,
                        disable this option. It is considered experimental.
  --attention_mechanism {diffusers,xformers,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}
                        On NVIDIA CUDA devices, alternative flash attention
                        implementations are offered, with the default being
                        native pytorch SDPA. SageAttention has multiple
                        backends to select from. The recommended value,
                        'sageattention', guesses what would be the 'best'
                        option for SageAttention on your hardware (usually
                        this is the int8-fp16-cuda backend). However, manually
                        setting this value to int8-fp16-triton may provide
                        better averages for per-step training and inference
                        performance while the cuda backend may provide the
                        highest maximum speed (with also a lower minimum
                        speed). NOTE: SageAttention training quality has not
                        been validated.
  --sageattention_usage {training,inference,training+inference}
                        SageAttention breaks gradient tracking through the
                        backward pass, leading to untrained QKV layers. This
                        can result in substantial problems for training, so it
                        is recommended to use SageAttention only for inference
                        (default behaviour). If you are confident in your
                        training setup or do not wish to train QKV layers, you
                        may use 'training' to enable SageAttention for
                        training.
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
  --masked_loss_probability MASKED_LOSS_PROBABILITY
  --validation_guidance VALIDATION_GUIDANCE
                        CFG value for validation images. Default: 7.5
  --validation_guidance_real VALIDATION_GUIDANCE_REAL
                        Use real CFG sampling for distilled models. Default:
                        1.0 (no CFG)
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
  --disable_bucket_pruning
                        When training on very small datasets, you might not
                        care that the batch sizes will outpace your image
                        count. Setting this option will prevent SimpleTuner
                        from deleting your bucket lists that do not meet the
                        minimum image count requirements. Use at your own
                        risk, it may end up throwing off your statistics or
                        epoch tracking.
  --offset_noise        Fine-tuning against a modified noise See:
                        https://www.crosslabs.org//blog/diffusion-with-offset-
                        noise for more information.
  --input_perturbation INPUT_PERTURBATION
                        Add additional noise only to the inputs fed to the
                        model during training. This will make the training
                        converge faster. A value of 0.1 is suggested if you
                        want to enable this. Input perturbation seems to also
                        work with flow-matching (e.g. SD3 and Flux).
  --input_perturbation_steps INPUT_PERTURBATION_STEPS
                        Only apply input perturbation over the first N steps
                        with linear decay. This should prevent artifacts from
                        showing up in longer training runs.
  --lr_end LR_END       A polynomial learning rate will end up at this value
                        after the specified number of warmup steps. A sine or
                        cosine wave will use this value as its lower bound for
                        the learning rate.
  --i_know_what_i_am_doing
                        This flag allows you to override some safety checks.
                        It's not recommended to use this unless you are
                        developing the platform. Generally speaking, issue
                        reports submitted with this flag enabled will go to
                        the bottom of the queue.
  --accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL
                        Clear the cache from VRAM every X steps. This can help
                        prevent memory leaks, but may slow down training.
```
