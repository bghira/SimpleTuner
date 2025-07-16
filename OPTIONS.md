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
[fill in]
```