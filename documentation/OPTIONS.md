# SimpleTuner Training Script Options

## Overview

This guide provides a user-friendly breakdown of the command-line options available in SimpleTuner's `train.py` script. These options offer a high degree of customization, allowing you to train your model to suit your specific requirements.

### JSON Configuration file format

The JSON filename expected is `config.json` and the key names are the same as the below `--arguments`. The leading `--` is not required for the JSON file, but it can be left in as well.

Looking for ready-to-run examples? See the curated presets in [simpletuner/examples/README.md](/simpletuner/examples/README.md).

### Easy configure script (***RECOMMENDED***)

The `simpletuner configure` command can be used to set up a `config.json` file with mostly-ideal default settings.

#### Modifying existing configurations

The `configure` command is capable of accepting a single argument, a compatible `config.json`, allowing interactive modification of your training setup:

```bash
simpletuner configure config/foo/config.json
```

Where `foo` is your config environment - or just use `config/config.json` if you're not using config environments.

<img width="1484" height="560" alt="image" src="https://github.com/user-attachments/assets/67dec8d8-3e41-42df-96e6-f95892d2814c" />

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

### `--lora_format`

- **What**: Select the LoRA checkpoint key format for load/save.
- **Choices**: `diffusers` (default), `comfyui`
- **Notes**:
  - `diffusers` is the standard PEFT/Diffusers layout.
  - `comfyui` converts to/from ComfyUI-style keys (`diffusion_model.*` with `lora_A/lora_B` and `.alpha` tensors). Flux, Flux2, Lumina2, and Z-Image will auto-detect ComfyUI inputs even if this is left at `diffusers`, but set it to `comfyui` to force ComfyUI output when saving.

### `--fuse_qkv_projections`

- **What**: Fuses the QKV projections in the model's attention blocks to make more efficient use of hardware.
- **Note**: Only available with NVIDIA H100 or H200 with Flash Attention 3 installed manually.

### `--offload_during_startup`

- **What**: Offloads text encoder weights to CPU when VAE caching is going.
- **Why**: This is useful for large models like HiDream and Wan 2.1, which can OOM when loading the VAE cache. This option does not impact quality of training, but for very large text encoders or slow CPUs, it can extend startup time substantially with many datasets. This is disabled by default due to this reason.
- **Tip**: Complements the group offloading feature below for especially memory-constrained systems.

### `--offload_during_save`

- **What**: Temporarily move the entire pipeline to CPU while `save_hooks.py` prepares checkpoints so that all FP8/quantized weights are written off-device.
- **Why**: Saving fp8-quanto weights can spike VRAM usage (for example, during `state_dict()` serialization). This option keeps the model on the accelerator for training but offloads it briefly when a save is triggered to avoid CUDA OOMs.
- **Tip**: Enable this only when saving fails with OOM errors; the loader moves the model back afterward so training resumes seamlessly.

### `--delete_model_after_load`

- **What**: Deletes model files from the HuggingFace cache after they are loaded into memory.
- **Why**: Reduces disk usage for budget-constrained setups that bill by gigabyte used. After models are loaded into VRAM/RAM, the on-disk cache is no longer needed until the next run. This shifts burden from storage to network bandwidth on subsequent runs.
- **Notes**:
  - The VAE is **not** deleted if validation is enabled, as it's needed for generating validation images.
  - Text encoders are deleted after the data backend factory completes startup (after embed caching).
  - Transformer/UNet models are deleted immediately after loading.
  - On multi-node setups, only local-rank 0 on each node performs the deletion. Deletion failures are silently ignored to handle race conditions on shared network storage.
  - This does **not** affect saved training checkpoints — only the pre-trained base model cache.

### `--enable_group_offload`

- **What**: Enables diffusers' grouped module offloading so model blocks can be staged on CPU (or disk) between forward passes.
- **Why**: Dramatically reduces peak VRAM usage on large transformers (Flux, Wan, Auraflow, LTXVideo, Cosmos2Image) with minimal performance impact when used with CUDA streams.
- **Notes**:
  - Mutually exclusive with `--enable_model_cpu_offload` — pick one strategy per run.
  - Requires diffusers **v0.33.0** or newer.

### `--group_offload_type`

- **Choices**: `block_level` (default), `leaf_level`
- **What**: Controls how layers are grouped. `block_level` balances VRAM savings with throughput, while `leaf_level` maximises savings at the cost of more CPU transfers.

### `--group_offload_blocks_per_group`

- **What**: When using `block_level`, the number of transformer blocks to bundle into a single offload group.
- **Default**: `1`
- **Why**: Increasing this number reduces transfer frequency (faster) but keeps more parameters resident on the accelerator (uses more VRAM).

### `--group_offload_use_stream`

- **What**: Uses a dedicated CUDA stream to overlap host/device transfers with compute.
- **Default**: `False`
- **Notes**:
  - Automatically falls back to CPU-style transfers on non-CUDA backends (Apple MPS, ROCm, CPU).
  - Recommended when training on NVIDIA GPUs with spare copy engine capacity.

### `--group_offload_to_disk_path`

- **What**: Directory path used to spill grouped parameters to disk instead of RAM.
- **Why**: Useful for extremely tight CPU RAM budgets (e.g., workstation with large NVMe drive).
- **Tip**: Use a fast local SSD; network filesystems will significantly slow training.

### `--musubi_blocks_to_swap`

- **What**: Musubi block swap for LongCat-Video, Wan, LTXVideo, Kandinsky5-Video, Qwen-Image, Flux, Flux.2, Cosmos2Image, and HunyuanVideo — keep the last N transformer blocks on CPU and stream weights per block during forward.
- **Default**: `0` (disabled)
- **Notes**: Musubi-style weight offload; reduces VRAM at a throughput cost and is skipped when gradients are enabled.

### `--musubi_block_swap_device`

- **What**: Device string for storing swapped transformer blocks (e.g., `cpu`, `cuda:0`).
- **Default**: `cpu`
- **Notes**: Only used when `--musubi_blocks_to_swap` > 0.

### `--ramtorch`

- **What**: Replaces `nn.Linear` layers with RamTorch CPU-streamed implementations.
- **Why**: Shares Linear weights in CPU memory and streams them to the accelerator to reduce VRAM pressure.
- **Notes**:
  - Requires CUDA or ROCm (not supported on Apple/MPS).
  - Mutually exclusive with `--enable_group_offload`.
  - Automatically enables `--set_grads_to_none`.

### `--ramtorch_target_modules`

- **What**: Comma-separated glob patterns limiting which Linear modules are converted to RamTorch.
- **Default**: All Linear layers are converted when no pattern is provided.
- **Notes**: Matches fully qualified module names or class names (wildcards allowed).

### `--ramtorch_text_encoder`

- **What**: Applies RamTorch replacements to all text encoder Linear layers.
- **Default**: `False`

### `--ramtorch_vae`

- **What**: Experimental RamTorch conversion for the VAE mid-block Linear layers only.
- **Default**: `False`
- **Notes**: VAE up/down convolutional blocks are left unchanged.

### `--ramtorch_controlnet`

- **What**: Applies RamTorch replacements to ControlNet Linear layers when training a ControlNet.
- **Default**: `False`

### `--pretrained_model_name_or_path`

- **What**: Path to the pretrained model or its identifier from <https://huggingface.co/models>.
- **Why**: To specify the base model you'll start training from. Use `--revision` and `--variant` to specify specific versions from a repository. This also supports single-file `.safetensors` paths for SDXL, Flux, and SD3.x.

### `--pretrained_t5_model_name_or_path`

- **What**: Path to the pretrained T5 model or its identifier from <https://huggingface.co/models>.
- **Why**: When training PixArt, you might want to use a specific source for your T5 weights so that you can avoid downloading them multiple times when switching the base model you train from.

### `--gradient_checkpointing`

- **What**: During training, gradients will be calculated layerwise and accumulated to save on peak VRAM requirements at the cost of slower training.

### `--gradient_checkpointing_interval`

- **What**: Checkpoint only every *n* blocks, where *n* is a value greater than zero. A value of 1 is effectively the same as just leaving `--gradient_checkpointing` enabled, and a value of 2 will checkpoint every other block.
- **Note**: SDXL and Flux are currently the only models supporting this option. SDXL uses a hackish implementation.

### `--refiner_training`

- **What**: Enables training a custom mixture-of-experts model series. See [Mixture-of-Experts](/documentation/MIXTURE_OF_EXPERTS.md) for more information on these options.

## Precision

### `--quantize_via`

- **Choices**: `cpu`, `accelerator`, `pipeline`
  - On `accelerator`, it may work moderately faster at the risk of possibly OOM'ing on 24G cards for a model as large as Flux.
  - On `cpu`, quantisation takes about 30 seconds. (**Default**)
  - `pipeline` delegates quantization to Diffusers using `--quantization_config` or pipeline-capable presets (e.g., `nf4-bnb`, `int4-torchao`, or `.gguf` checkpoints). Manual Quanto/TorchAO presets are not supported when this mode is enabled.

### `--base_model_precision`

- **What**: Reduce model precision and train using less memory. There are three supported quantisation backends: BitsAndBytes (pipeline), TorchAO (pipeline or manual), and Optimum Quanto (manual).

#### Diffusers pipeline presets

- `nf4-bnb` loads through Diffusers with a 4-bit NF4 BitsAndBytes config (CUDA only). Requires `bitsandbytes` and a diffusers build with BnB support.
- `int4-torchao` uses a TorchAoConfig with `Int4WeightOnlyConfig` via Diffusers (CUDA). Requires `torchao` and `transformers>=4.39`.
- `.gguf` checkpoints are auto-detected and loaded with `GGUFQuantizationConfig` when available. Install recent diffusers/transformers for GGUF support.

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

### `--quantization_config`

- **What**: JSON object or file path describing Diffusers `quantization_config` overrides when using `--quantize_via=pipeline`.
- **How**: Accepts inline JSON (or a file) with per-component entries. Keys may include `unet`, `transformer`, `text_encoder`, or `default`.
- **Examples**:

```json
{
  "unet": {"load_in_4bit": true, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16"},
  "text_encoder": {"quant_type": {"group_size": 128}}
}
```

This example enables 4-bit NF4 BnB on the UNet and TorchAO int4 on the text encoder.

#### Torch Dynamo

Enable `torch.compile()` from the WebUI by visiting **Hardware → Accelerate (advanced)** and setting **Torch Dynamo Backend** to your preferred compiler (for example, *inductor*). Additional toggles let you pick an optimisation **mode**, enable **dynamic shape** guards, or opt into **regional compilation** to speed up cold starts on very deep transformer models.

The same configuration can be expressed in `config/config.env`:

```bash
TRAINING_DYNAMO_BACKEND=inductor
```

You can optionally pair this with `--dynamo_mode=max-autotune` or the other Dynamo flags exposed in the UI for finer control.

Note that the first several steps of training will be slower than usual because of compilation occuring in the background.

To persist the settings in `config.json`, add the equivalent keys:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_mode": "max-autotune",
  "dynamo_fullgraph": false,
  "dynamo_dynamic": false,
  "dynamo_use_regional_compilation": true
}
```

Omit any entries you want to inherit from Accelerate’s defaults (for example, leave out `dynamo_mode` to use automatic selection).

### `--attention_mechanism`

Alternative attention mechanisms are supported, with varying levels of compatibility or other trade-offs:

- `diffusers` uses PyTorch’s native SDPA kernels and is the default.
- `xformers` enables Meta’s [xformers](https://github.com/facebookresearch/xformers) attention kernel (training + inference) when the underlying model exposes `enable_xformers_memory_efficient_attention`.
- `flash-attn`, `flash-attn-2`, `flash-attn-3`, and `flash-attn-3-varlen` hook into Diffusers’ new `attention_backend` helper to route attention through FlashAttention v1/2/3 kernels. Install the corresponding `flash-attn` / `flash-attn-interface` wheels and note that FA3 currently requires Hopper GPUs.
- `flex` selects PyTorch 2.5’s FlexAttention backend (FP16/BF16 on CUDA). You must compile/install the Flex kernels separately — see [documentation/attention/FLEX.md](attention/FLEX.md).
- `cudnn`, `native-efficient`, `native-flash`, `native-math`, `native-npu`, and `native-xla` select the matching SDPA backend exposed by `torch.nn.attention.sdpa_kernel`. These are handy when you want determinism (`native-math`), the CuDNN SDPA kernel, or the vendor-native accelerators (NPU/XLA).
- `sla` enables [Sparse–Linear Attention (SLA)](https://github.com/thu-ml/SLA), providing a fine-tunable sparse/linear hybrid kernel that can be used for both training and validation without additional gating.
  - Install the SLA package (for example via `pip install -e ~/src/SLA`) before selecting this backend.
  - SimpleTuner saves SLA’s learned projection weights into `sla_attention.pt` inside every checkpoint; keep this file with the rest of the checkpoint so resumes and inference retain the trained SLA state.
  - Because the backbone is tuned around SLA’s mixed sparse/linear behaviour, SLA will be required at inference time as well. See `documentation/attention/SLA.md` for a focused guide.
  - Use `--sla_config '{"topk":0.15,"blkq":32,"tie_feature_map_qk":false}'` (JSON or Python dict syntax) to override SLA runtime defaults if you need to experiment.
- `sageattention`, `sageattention-int8-fp16-triton`, `sageattention-int8-fp16-cuda`, and `sageattention-int8-fp8-cuda` wrap the corresponding [SageAttention](https://github.com/thu-ml/SageAttention) kernels. These are inference-oriented and must be used with `--sageattention_usage` to guard against accidental training.
  - In simplest terms, SageAttention reduces compute requirement for inference

> ℹ️ The Flash/Flex/PyTorch backend selectors rely on Diffusers’ `attention_backend` dispatcher, so they currently benefit transformer-style models that already opt into that code path (Flux, Wan 2.x, LTXVideo, QwenImage, etc.). Classic SD/SDXL UNets still use PyTorch SDPA directly.

Using `--sageattention_usage` to enable training with SageAttention should be enabled with care, as it does not track or propagate gradients from its custom CUDA implementations for the QKV linears.

- This results in these layers being completely untrained, which might cause model collapse or slight improvements in short training runs.

---

## 📰 Publishing

### `--push_to_hub`

- **What**: If provided, your model will be uploaded to [Huggingface Hub](https://huggingface.co) once training completes. Using `--push_checkpoints_to_hub` will additionally push every intermediary checkpoint.

### `--push_to_hub_background`

- **What**: Uploads to Hugging Face Hub from a background worker so checkpoint pushes do not pause the training loop.
- **Why**: Keeps training and validation running while Hub uploads proceed asynchronously. Final uploads are still awaited before the run exits so failures surface.

### `--webhook_config`

- **What**: Configuration for webhook targets (e.g., Discord, custom endpoints) to receive real-time training events.
- **Why**: Allows you to monitor training runs with external tools and dashboards, receiving notifications at key training stages.
- **Notes**: The `job_id` field in webhook payloads can be populated by setting the `SIMPLETUNER_JOB_ID` environment variable before training:
  ```bash
  export SIMPLETUNER_JOB_ID="my-training-run-name"
  python train.py
  ```
This is useful for monitoring tools receiving webhooks from multiple training runs to identify which config sent each event. If SIMPLETUNER_JOB_ID is not set, job_id will be null in webhook payloads.

### `--publishing_config`

- **What**: Optional JSON/dict/file path describing non-Hugging Face publishing targets (S3-compatible storage, Backblaze B2, Azure Blob Storage, Dropbox).
- **Why**: Mirrors `--webhook_config` parsing so you can fan out artifacts beyond the Hub. Publishing runs on the main process after validation using the current `output_dir`.
- **Notes**: Providers are additive to `--push_to_hub`. Install provider SDKs (e.g., `boto3`, `azure-storage-blob`, `dropbox`) inside your `.venv` when you enable them. See `documentation/publishing/README.md` for complete examples.

### `--hub_model_id`

- **What**: The name of the Huggingface Hub model and local results directory.
- **Why**: This value is used as the directory name under the location specified as `--output_dir`. If `--push_to_hub` is provided, this will become the name of the model on Huggingface Hub.

### `--disable_benchmark`

- **What**: Disable the startup validation/benchmark that occurs at step 0 on the base model. These outputs are stitchd to the left side of your trained model validation images.

## 📂 Data Storage and Management

### `--data_backend_config`

- **What**: Path to your SimpleTuner dataset configuration.
- **Why**: Multiple datasets on different storage medium may be combined into a single training session.
- **Example**: See [multidatabackend.json.example](/multidatabackend.json.example) for an example configuration, and [this document](/documentation/DATALOADER.md) for more information on configuring the data loader.

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

### `--validation_method`

- **What**: Choose how validation runs are executed.
- **Options**: `simpletuner-local` (default) runs the built-in pipeline; `external-script` runs a user-provided executable instead.
- **Why**: Lets you hand off validation to an external system without pausing training for local pipeline work.

### `--validation_external_script`

- **What**: Executable to run when `--validation_method=external-script`. Uses shell-style splitting, so quote the command string accordingly.
- **Placeholders**: You can embed these tokens (formatted with `.format`) to pass training context. Missing values are replaced with an empty string unless noted:
  - `{local_checkpoint_path}` → latest checkpoint directory under `output_dir` (requires at least one checkpoint).
  - `{global_step}` → current global step.
  - `{tracker_run_name}` → value of `--tracker_run_name`.
  - `{tracker_project_name}` → value of `--tracker_project_name`.
  - `{model_family}` → value of `--model_family`.
  - `{model_type}` / `{lora_type}` → model type and LoRA flavour.
  - `{huggingface_path}` → value of `--hub_model_id` (if set).
  - `{remote_checkpoint_path}` → remote URL of your last upload (empty for validation hook).
  - Any `validation_*` config value (e.g., `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`).
- **Example**: `--validation_external_script="/opt/tools/validate.sh {local_checkpoint_path} {global_step}"`

### `--validation_external_background`

- **What**: When set, `--validation_external_script` is launched in the background (fire-and-forget).
- **Why**: Keep training moving without waiting for the external script; exit codes are not checked in this mode.

### `--post_upload_script`

- **What**: Optional executable run after each publishing provider and Hugging Face Hub upload finishes (final model and checkpoint uploads). Runs asynchronously so training doesn't block.
- **Placeholders**: Same replacements as `--validation_external_script`, plus `{remote_checkpoint_path}` (URI returned by the provider) so you can forward the published URL to downstream systems.
- **Notes**:
  - Scripts run per provider/upload; errors are logged but do not halt training.
  - Scripts are also invoked when no remote upload occurs, so you can use them for local automation (e.g., running inference on another GPU).
  - SimpleTuner does not ingest results from your script—log to your tracker directly if you want metrics or images recorded.
- **Example**:
  ```bash
  --post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'
  ```
  Where `/opt/hooks/notify.sh` might post to your tracking system:
  ```bash
  #!/usr/bin/env bash
  REMOTE="$1"
  PROJECT="$2"
  RUN="$3"
  curl -X POST "https://tracker.internal/api/runs/${PROJECT}/${RUN}/artifacts" \
       -H "Content-Type: application/json" \
       -d "{\"remote_uri\":\"${REMOTE}\"}"
  ```
- **Working samples**:
  - `simpletuner/examples/external-validation/replicate_post_upload.py` shows a Replicate hook that consumes `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}`, and `{huggingface_path}` to trigger inference after uploads.
  - `simpletuner/examples/external-validation/wavespeed_post_upload.py` shows a WaveSpeed hook using the same placeholders plus WaveSpeed's async polling.
  - `simpletuner/examples/external-validation/fal_post_upload.py` shows a fal.ai Flux LoRA hook (requires `FAL_KEY`).
  - `simpletuner/examples/external-validation/use_second_gpu.py` runs Flux LoRA inference on a secondary GPU and works even without remote uploads.

### `--post_checkpoint_script`

- **What**: Executable to run immediately after each checkpoint directory is written to disk (before any uploads kick off). Runs asynchronously on the main process.
- **Placeholders**: Same replacements as `--validation_external_script`, including `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{model_type}`, `{lora_type}`, `{huggingface_path}`, and any `validation_*` config value. `{remote_checkpoint_path}` resolves to empty for this hook.
- **Notes**:
  - Fires for scheduled, manual, and rolling checkpoints as soon as they finish saving locally.
  - Useful for triggering local automation (copying to another volume, running eval jobs) without waiting for uploads to finish.
- **Example**:
  ```bash
  --post_checkpoint_script='/opt/hooks/run_eval.sh {local_checkpoint_path} {global_step}'
  ```


### `--validation_adapter_path`

- **What**: Temporarily loads a single LoRA adapter when running scheduled validations.
- **Formats**:
  - Hugging Face repo: `org/repo` or `org/repo:weight_name.safetensors` (defaults to `pytorch_lora_weights.safetensors`).
  - Local file or directory path pointing to a safetensors adapter.
- **Notes**:
  - Mutually exclusive with `--validation_adapter_config`; supplying both raises an error.
  - The adapter is only attached for validation runs (baseline training weights remain untouched).

### `--validation_adapter_name`

- **What**: Optional identifier to apply to the temporary adapter loaded via `--validation_adapter_path`.
- **Why**: Controls how the adapter run is labelled in logs/web UI and ensures predictable adapter names when multiple adapters are tested sequentially.

### `--validation_adapter_strength`

- **What**: Strength multiplier applied when enabling the temporary adapter (defaults to `1.0`).
- **Why**: Lets you sweep lighter/heavier LoRA scaling during validation without altering training state; accepts any value greater than zero.

### `--validation_adapter_mode`

- **Choices**: `adapter_only`, `comparison`, `none`
- **What**:
  - `adapter_only`: run validations only with the temporary adapter attached.
  - `comparison`: generate both base-model and adapter-enabled samples for side-by-side review.
  - `none`: skip attaching the adapter (useful for disabling the feature without deleting CLI flags).

### `--validation_adapter_config`

- **What**: JSON file or inline JSON that describes multiple validation adapter combinations to iterate over.
- **Format**: Either an array of entries or an object with a `runs` array. Each entry may include:
  - `label`: Friendly name shown in logs/UI.
  - `path`: Hugging Face repo ID or local path (same formats as `--validation_adapter_path`).
  - `adapter_name`: Optional identifier per adapter.
  - `strength`: Optional scalar override.
  - `adapters`/`paths`: Array of objects/strings to load multiple adapters in a single run.
- **Notes**:
  - When provided, the single-adapter options (`--validation_adapter_path`, `--validation_adapter_name`, `--validation_adapter_strength`, `--validation_adapter_mode`) are ignored/disabled in the UI.
  - Each run is loaded one at a time and fully detached before the next run begins.

### `--validation_preview`

- **What**: Stream intermediate validation previews during diffusion sampling using Tiny AutoEncoders
- **Default**: False
- **Why**: Enables real-time preview of validation images as they're being generated, decoded via lightweight Tiny AutoEncoder models and sent through webhook callbacks. This allows you to monitor the progression of validation samples step-by-step rather than waiting for the complete generation.
- **Notes**:
  - Only available on model families with Tiny AutoEncoder support (e.g., Flux, SDXL, SD3)
  - Requires webhook configuration to receive preview images
  - Use `--validation_preview_steps` to control how frequently previews are decoded

### `--validation_preview_steps`

- **What**: Interval for decoding and streaming validation previews
- **Default**: 1
- **Why**: Controls how often intermediate latents are decoded during validation sampling. Setting this to a higher value (e.g., 3) reduces the overhead of running the Tiny AutoEncoder by only decoding every N sampling steps.
- **Example**: With `--validation_num_inference_steps=20` and `--validation_preview_steps=5`, you'll receive 4 preview images during the generation process (at steps 5, 10, 15, 20).

### `--evaluation_type`

- **What**: Enable CLIP evaluation of generated images during validations.
- **Why**: CLIP scores calculate the distance of the generated image features to the provided validation prompt. This can give an idea of whether prompt adherence is improving, though it requires a large number of validation prompts to have any meaningful value.
- **Options**: "none" or "clip"
- **Scheduling**: Use `--eval_steps_interval` for step-based scheduling or `--eval_epoch_interval` for epoch-based scheduling (fractions like `0.5` run multiple times per epoch). If both are set, the trainer logs a warning and runs both schedules.

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

### `--allow_dataset_oversubscription`

- **What**: Automatically adjusts dataset `repeats` when the dataset is smaller than the effective batch size.
- **Why**: Prevents training failures when your dataset size doesn't meet the minimum requirements for your multi-GPU configuration.
- **How it works**:
  - Calculates the **effective batch size**: `train_batch_size × num_gpus × gradient_accumulation_steps`
  - If any aspect bucket has fewer samples than the effective batch size, automatically increases `repeats`
  - Only applies when `repeats` is not explicitly configured in your dataset config
  - Logs a warning showing the adjustment and reasoning
- **Use cases**:
  - Small datasets (< 100 images) with multiple GPUs
  - Experimenting with different batch sizes without reconfiguring datasets
  - Prototyping before collecting a full dataset
- **Example**: With 25 images, 8 GPUs, and `train_batch_size=4`, the effective batch size is 32. This flag would automatically set `repeats=1` to provide 50 samples (25 × 2).
- **Note**: This will **not** override manually-set `repeats` values in your dataloader configuration. Similar to `--disable_bucket_pruning`, this flag provides convenience without surprising behavior.

See the [DATALOADER.md](DATALOADER.md#automatic-dataset-oversubscription) guide for more details on dataset sizing for multi-GPU training.

---

## 🛠 Advanced Optimizations

### `--use_ema`

- **What**: Keeping an exponential moving average of your weights over the models' training lifetime is like periodically back-merging the model into itself.
- **Why**: It can improve training stability at the cost of more system resources, and a slight increase in training runtime.

### `--ema_device`

- **Choices**: `cpu`, `accelerator`; default: `cpu`
- **What**: Chooses where the EMA weights live between updates.
- **Why**: Keeping the EMA on the accelerator gives the fastest updates but costs VRAM. Keeping it on CPU reduces memory pressure but requires shuttling weights unless `--ema_cpu_only` is set.

### `--ema_cpu_only`

- **What**: Prevents the EMA weights from being moved back to the accelerator for updates when `--ema_device=cpu`.
- **Why**: Saves the host-to-device transfer time and VRAM usage for large EMAs. Has no effect if `--ema_device=accelerator` because the weights already reside on the accelerator.

### `--ema_foreach_disable`

- **What**: Disables the use of `torch._foreach_*` kernels for EMA updates.
- **Why**: Some back-ends or hardware combinations have issues with foreach ops. Disabling them falls back to the scalar implementation at the cost of slightly slower updates.

### `--ema_update_interval`

- **What**: Reduces how often the EMA shadow parameters are updated.
- **Why**: Updating every step is unnecessary for many workflows. For example, `--ema_update_interval=100` only performs an EMA update once every 100 optimizer steps, reducing overhead when `--ema_device=cpu` or `--ema_cpu_only` is enabled.

### `--ema_decay`

- **What**: Controls the smoothing factor used when applying EMA updates.
- **Why**: Higher values (e.g. `0.999`) make the EMA respond slowly but produce very stable weights. Lower values (e.g. `0.99`) adapt faster to new training signals.

### `--snr_gamma`

- **What**: Utilising min-SNR weighted loss factor.
- **Why**: Minimum SNR gamma weights the loss factor of a timestep by its position in the schedule. Overly noisy timesteps have their contributions reduced, and less-noisy timesteps have it increased. Value recommended by the original paper is **5** but you can use values as low as **1** or as high as **20**, typically seen as the maximum value - beyond a value of 20, the math does not change things much. A value of **1** is the strongest.

### `--use_soft_min_snr`

- **What**: Train a model using a more gradual weighting on the loss landscape.
- **Why**: When training pixel diffusion models, they will simply degrade without using a specific loss weighting schedule. This is the case with DeepFloyd, where soft-min-snr-gamma was found to essentially be mandatory for good results. You may find success with latent diffusion model training, but in small experiments, it was found to potentially produce blurry results.

### `--diff2flow_enabled`

- **What**: Enable the Diffusion-to-Flow bridge for epsilon or v-prediction models.
- **Why**: Allows models trained with standard diffusion objectives to use flow-matching targets (noise - latents) without changing the model architecture.
- **Note**: Experimental feature.

### `--diff2flow_loss`

- **What**: Train with Flow Matching loss instead of the native prediction loss.
- **Why**: When enabled alongside `--diff2flow_enabled`, this calculates the loss against the flow target (noise - latents) instead of the model's native target (epsilon or velocity).
- **Note**: Requires `--diff2flow_enabled`.

### `--scheduled_sampling_max_step_offset`

- **What**: Maximum number of steps to "roll out" during training.
- **Why**: Enables Scheduled Sampling (Rollout), where the model generates its own inputs for a few steps during training. This helps the model learn to correct its own errors and reduces exposure bias.
- **Default**: 0 (disabled). Set to a positive integer (e.g., 5 or 10) to enable.

### `--scheduled_sampling_strategy`

- **What**: Strategy for choosing the rollout offset.
- **Choices**: `uniform`, `biased_early`, `biased_late`.
- **Default**: `uniform`.
- **Why**: Controls the distribution of rollout lengths. `uniform` samples evenly; `biased_early` favors shorter rollouts; `biased_late` favors longer rollouts.

### `--scheduled_sampling_probability`

- **What**: Probability of applying a non-zero rollout offset for a given sample.
- **Default**: 0.0.
- **Why**: Controls how often scheduled sampling is applied. A value of 0.0 disables it even if `max_step_offset` is > 0. A value of 1.0 applies it to every sample.

### `--scheduled_sampling_prob_start`

- **What**: Initial probability for scheduled sampling at the start of the ramp.
- **Default**: 0.0.

### `--scheduled_sampling_prob_end`

- **What**: Final probability for scheduled sampling at the end of the ramp.
- **Default**: 0.5.

### `--scheduled_sampling_ramp_steps`

- **What**: Number of steps to ramp the probability from `prob_start` to `prob_end`.
- **Default**: 0 (no ramp).

### `--scheduled_sampling_start_step`

- **What**: Global step to start the scheduled sampling ramp.
- **Default**: 0.0.

### `--scheduled_sampling_ramp_shape`

- **What**: Shape of the probability ramp.
- **Choices**: `linear`, `cosine`.
- **Default**: `linear`.

### `--scheduled_sampling_sampler`

- **What**: The solver used for the rollout generation steps.
- **Choices**: `unipc`, `euler`, `dpm`, `rk4`.
- **Default**: `unipc`.

### `--scheduled_sampling_order`

- **What**: The order of the solver used for rollout.
- **Default**: 2.

### `--scheduled_sampling_reflexflow`

- **What**: Enable ReflexFlow-style enhancements (anti-drift + frequency-compensated weighting) during scheduled sampling for flow-matching models.
- **Why**: Reduces exposure bias when rolling out flow-matching models by adding directional regularization and bias-aware loss weighting.
- **Default**: Auto-enable for flow-matching models when `--scheduled_sampling_max_step_offset` > 0; override with `--scheduled_sampling_reflexflow=false`.

### `--scheduled_sampling_reflexflow_alpha`

- **What**: Scaling factor for the frequency-compensation weight derived from exposure bias.
- **Default**: 1.0.
- **Why**: Higher values up-weight regions with larger exposure bias during rollout for flow-matching models.

### `--scheduled_sampling_reflexflow_beta1`

- **What**: Weight for the ReflexFlow anti-drift (directional) regularizer.
- **Default**: 10.0.
- **Why**: Controls how strongly the model is encouraged to align its predicted direction with the target clean sample when using scheduled sampling on flow-matching models.

### `--scheduled_sampling_reflexflow_beta2`

- **What**: Weight for the ReflexFlow frequency-compensation (loss reweighting) term.
- **Default**: 1.0.
- **Why**: Scales the reweighted flow-matching loss, matching the β₂ knob described in the ReflexFlow paper.

---

## 🎯 CREPA (Cross-frame Representation Alignment)

CREPA is a regularization technique for fine-tuning video diffusion models that improves temporal consistency by aligning hidden states with pretrained visual features from adjacent frames. Based on the paper ["Cross-Frame Representation Alignment for Fine-Tuning Video Diffusion Models"](https://arxiv.org/abs/2506.09229).

### `--crepa_enabled`

- **What**: Enable CREPA regularization during training.
- **Why**: Improves semantic consistency across video frames by aligning DiT hidden states with DINOv2 features from neighboring frames.
- **Default**: `false`
- **Note**: Only applies to video models (Wan, LTXVideo, SanaVideo, Kandinsky5).

### `--crepa_block_index`

- **What**: Which transformer block's hidden states to use for alignment.
- **Why**: The paper recommends block 8 for CogVideoX and block 10 for Hunyuan Video. Earlier blocks tend to work better as they act as the "encoder" portion of the DiT.
- **Required**: Yes, when CREPA is enabled.

### `--crepa_lambda`

- **What**: Weight of the CREPA alignment loss relative to the main training loss.
- **Why**: Controls how strongly the alignment regularization influences training. The paper uses 0.5 for CogVideoX and 1.0 for Hunyuan Video.
- **Default**: `0.5`

### `--crepa_adjacent_distance`

- **What**: Distance `d` for neighbor frame alignment.
- **Why**: Per the paper's Equation 6, $K = \{f-d, f+d\}$ defines which neighboring frames to align with. With `d=1`, each frame aligns with its immediate neighbors.
- **Default**: `1`

### `--crepa_adjacent_tau`

- **What**: Temperature coefficient for the exponential distance weighting.
- **Why**: Controls how quickly alignment weight decays with frame distance via $e^{-|k-f|/\tau}$. Lower values focus more strongly on immediate neighbors.
- **Default**: `1.0`

### `--crepa_cumulative_neighbors`

- **What**: Use cumulative mode instead of adjacent mode.
- **Why**:
  - **Adjacent mode (default)**: Only aligns with frames at exact distance `d` (matches paper's $K = \{f-d, f+d\}$)
  - **Cumulative mode**: Aligns with all frames from distance 1 to `d`, providing smoother gradients
- **Default**: `false`

### `--crepa_normalize_by_frames`

- **What**: Normalize the alignment loss by the number of frames.
- **Why**: Ensures consistent loss scale regardless of video length. Disable to give longer videos stronger alignment signal.
- **Default**: `true`

### `--crepa_spatial_align`

- **What**: Use spatial interpolation when token counts differ between DiT and encoder.
- **Why**: DiT hidden states and DINOv2 features may have different spatial resolutions. When enabled, bilinear interpolation aligns them spatially. When disabled, falls back to global pooling.
- **Default**: `true`

### `--crepa_model`

- **What**: Which pretrained encoder to use for feature extraction.
- **Why**: The paper uses DINOv2-g (ViT-Giant). Smaller variants like `dinov2_vitb14` use less memory.
- **Default**: `dinov2_vitg14`
- **Choices**: `dinov2_vitg14`, `dinov2_vitb14`, `dinov2_vits14`

### `--crepa_use_backbone_features`

- **What**: Skip the external encoder and align a student block to a teacher block inside the diffusion model.
- **Why**: Avoids loading DINOv2 when the backbone already has a stronger semantic layer to supervise from.
- **Default**: `false`

### `--crepa_teacher_block_index`

- **What**: Teacher block index when using backbone features.
- **Why**: Lets you align an earlier student block to a later teacher block without an external encoder. Falls back to the student block when unset.
- **Default**: Uses `crepa_block_index` if not provided.

### `--crepa_encoder_image_size`

- **What**: Input resolution for the encoder.
- **Why**: DINOv2 models work best at their training resolution. The giant model uses 518x518.
- **Default**: `518`

### Example Configuration

```toml
# Enable CREPA for video fine-tuning
crepa_enabled = true
crepa_block_index = 8          # Adjust based on your model
crepa_lambda = 0.5
crepa_adjacent_distance = 1
crepa_adjacent_tau = 1.0
crepa_cumulative_neighbors = false
crepa_normalize_by_frames = true
crepa_spatial_align = true
crepa_model = "dinov2_vitg14"
crepa_use_backbone_features = false
# crepa_teacher_block_index = 16
crepa_encoder_image_size = 518
```

---

## 🔄 Checkpointing and Resumption

### `--checkpoint_step_interval` (alias: `--checkpointing_steps`)

- **What**: Interval at which training state checkpoints are saved (in steps).
- **Why**: Useful for resuming training and for inference. Every *n* iterations, a partial checkpoint will be saved in the `.safetensors` format, via the Diffusers filesystem layout.

---

## 🔁 LayerSync (Hidden State Self-Alignment)

LayerSync encourages a "student" layer to match a stronger "teacher" layer inside the same transformer, using cosine similarity over hidden tokens.

### `--layersync_enabled`

- **What**: Enable LayerSync hidden-state alignment between two transformer blocks inside the same model.
- **Notes**: Allocates a hidden-state buffer; raises at startup if required flags are missing.
- **Default**: `false`

### `--layersync_student_block`

- **What**: Transformer block index to treat as the student anchor.
- **Indexing**: Accepts LayerSync paper-style 1-based depths or 0-based layer ids; the implementation tries `idx-1` first, then `idx`.
- **Required**: Yes when LayerSync is enabled.

### `--layersync_teacher_block`

- **What**: Transformer block index to treat as the teacher target (can be deeper than the student).
- **Indexing**: Same 1-based-first, then 0-based fallback as the student block.
- **Default**: Uses the student block when omitted so the loss becomes self-similarity.

### `--layersync_lambda`

- **What**: Weight for the LayerSync cosine alignment loss between the student and teacher hidden states (negative cosine similarity).
- **Effect**: Scales the auxiliary regularizer added on top of the base loss; higher values push the student tokens to align more strongly with the teacher tokens.
- **Upstream name**: `--reg-weight` in the original LayerSync codebase.
- **Required**: Must be > 0 when LayerSync is enabled (otherwise training aborts).
- **Default**: `0.2` when LayerSync is enabled (matches the reference repo), `0.0` otherwise.

Upstream option mapping (LayerSync → SimpleTuner):
- `--encoder-depth` → `--layersync_student_block` (accepts 1-based depth as in upstream, or 0-based layer index)
- `--gt-encoder-depth` → `--layersync_teacher_block` (1-based preferred; defaults to student when omitted)
- `--reg-weight` → `--layersync_lambda`

> Notes: LayerSync always detaches the teacher hidden state before similarity, matching the reference implementation. It relies on models that expose transformer hidden states (most transformer backbones in SimpleTuner) and adds per-step memory for the hidden-state buffer; disable if VRAM is tight.

### `--checkpoint_epoch_interval`

- **What**: Run checkpointing every N completed epochs.
- **Why**: Complements step-based checkpoints by ensuring you always capture the state at epoch boundaries, even when step counts vary with multi-dataset sampling.

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

## Environment configuration variables

The above options apply for the most part, to `config.json` - but some entries must be set inside `config.env` instead.

- `TRAINING_NUM_PROCESSES` should be set to the number of GPUs in the system. For most use-cases, this is enough to enable DistributedDataParallel (DDP) training. Use `num_processes` inside `config.json` if you prefer to not use `config.env`.
- `TRAINING_DYNAMO_BACKEND` defaults to `no` but can be set to any supported torch.compile backend (e.g. `inductor`, `aot_eager`, `cudagraphs`) and combined with `--dynamo_mode`, `--dynamo_fullgraph`, or `--dynamo_use_regional_compilation` for finer tuning
- `SIMPLETUNER_LOG_LEVEL` defaults to `INFO` but can be set to `DEBUG` to add more information for issue reports into `debug.log`
- `VENV_PATH` can be set to the location of your python virtual env, if it is not in the typical `.venv` location
- `ACCELERATE_EXTRA_ARGS` can be left unset, or, contain extra arguments to add like `--multi_gpu` or FSDP-specific flags

---

This is a basic overview meant to help you get started. For a complete list of options and more detailed explanations, please refer to the full specification:

```
usage: train.py [-h] --model_family
                {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo}
                [--model_flavour MODEL_FLAVOUR] [--controlnet [CONTROLNET]]
                [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                --output_dir OUTPUT_DIR [--logging_dir LOGGING_DIR]
                --model_type {full,lora} [--seed SEED]
                [--resolution RESOLUTION]
                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                [--prediction_type {epsilon,v_prediction,sample,flow_matching}]
                [--pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH]
                [--vae_dtype {default,fp32,fp16,bf16}]
                [--vae_cache_ondemand [VAE_CACHE_ONDEMAND]]
                [--accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL]
                [--aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}]
                [--base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL]
                [--offload_during_startup [OFFLOAD_DURING_STARTUP]]
                [--quantize_via {cpu,accelerator,pipeline}]
                [--quantization_config QUANTIZATION_CONFIG]
                [--fuse_qkv_projections [FUSE_QKV_PROJECTIONS]]
                [--control [CONTROL]]
                [--controlnet_custom_config CONTROLNET_CUSTOM_CONFIG]
                [--controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH]
                [--tread_config TREAD_CONFIG]
                [--pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH]
                [--pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER]
                [--pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH]
                [--pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER]
                [--pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH]
                [--revision REVISION] [--variant VARIANT]
                [--base_model_default_dtype {bf16,fp32}]
                [--unet_attention_slice [UNET_ATTENTION_SLICE]]
                [--num_train_epochs NUM_TRAIN_EPOCHS]
                [--max_train_steps MAX_TRAIN_STEPS]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--learning_rate LEARNING_RATE] --optimizer
                {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,prodigy}
                [--optimizer_config OPTIMIZER_CONFIG]
                [--lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--lr_warmup_steps LR_WARMUP_STEPS]
                [--checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT]
                [--gradient_checkpointing [GRADIENT_CHECKPOINTING]]
                [--train_text_encoder [TRAIN_TEXT_ENCODER]]
                [--text_encoder_lr TEXT_ENCODER_LR]
                [--lr_num_cycles LR_NUM_CYCLES] [--lr_power LR_POWER]
                [--use_soft_min_snr [USE_SOFT_MIN_SNR]] [--use_ema [USE_EMA]]
                [--ema_device {accelerator,cpu}]
                [--ema_cpu_only [EMA_CPU_ONLY]]
                [--ema_update_interval EMA_UPDATE_INTERVAL]
                [--ema_foreach_disable [EMA_FOREACH_DISABLE]]
                [--ema_decay EMA_DECAY] [--lora_rank LORA_RANK]
                [--lora_alpha LORA_ALPHA] [--lora_type {standard,lycoris}]
                [--lora_dropout LORA_DROPOUT]
                [--lora_init_type {default,gaussian,loftq,olora,pissa}]
                [--peft_lora_mode {standard,singlora}]
                [--peft_lora_target_modules PEFT_LORA_TARGET_MODULES]
                [--singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS]
                [--init_lora INIT_LORA] [--lycoris_config LYCORIS_CONFIG]
                [--init_lokr_norm INIT_LOKR_NORM]
                [--flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,controlnet,all+ffs+embedder,all+ffs+embedder+controlnet}]
                [--use_dora [USE_DORA]]
                [--resolution_type {pixel,area,pixel_area}]
                --data_backend_config DATA_BACKEND_CONFIG
                [--caption_strategy {filename,textfile,instance_prompt,parquet}]
                [--conditioning_multidataset_sampling {combined,random}]
                [--instance_prompt INSTANCE_PROMPT]
                [--parquet_caption_column PARQUET_CAPTION_COLUMN]
                [--parquet_filename_column PARQUET_FILENAME_COLUMN]
                [--ignore_missing_files [IGNORE_MISSING_FILES]]
                [--vae_cache_scan_behaviour {recreate,sync}]
                [--vae_enable_slicing [VAE_ENABLE_SLICING]]
                [--vae_enable_tiling [VAE_ENABLE_TILING]]
                [--vae_enable_patch_conv [VAE_ENABLE_PATCH_CONV]]
                [--vae_batch_size VAE_BATCH_SIZE]
                [--caption_dropout_probability CAPTION_DROPOUT_PROBABILITY]
                [--tokenizer_max_length TOKENIZER_MAX_LENGTH]
                [--validation_step_interval VALIDATION_STEP_INTERVAL]
                [--validation_epoch_interval VALIDATION_EPOCH_INTERVAL]
                [--disable_benchmark [DISABLE_BENCHMARK]]
                [--validation_prompt VALIDATION_PROMPT]
                [--num_validation_images NUM_VALIDATION_IMAGES]
                [--num_eval_images NUM_EVAL_IMAGES]
                [--eval_steps_interval EVAL_STEPS_INTERVAL]
                [--eval_epoch_interval EVAL_EPOCH_INTERVAL]
                [--eval_timesteps EVAL_TIMESTEPS]
                [--eval_dataset_pooling [EVAL_DATASET_POOLING]]
                [--evaluation_type {none,clip}]
                [--pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH]
                [--validation_guidance VALIDATION_GUIDANCE]
                [--validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS]
                [--validation_on_startup [VALIDATION_ON_STARTUP]]
                [--validation_using_datasets [VALIDATION_USING_DATASETS]]
                [--validation_torch_compile [VALIDATION_TORCH_COMPILE]]
                [--validation_guidance_real VALIDATION_GUIDANCE_REAL]
                [--validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP]
                [--validation_negative_prompt VALIDATION_NEGATIVE_PROMPT]
                [--validation_randomize [VALIDATION_RANDOMIZE]]
                [--validation_seed VALIDATION_SEED]
                [--validation_disable [VALIDATION_DISABLE]]
                [--validation_prompt_library [VALIDATION_PROMPT_LIBRARY]]
                [--user_prompt_library USER_PROMPT_LIBRARY]
                [--eval_dataset_id EVAL_DATASET_ID]
                [--validation_stitch_input_location {left,right}]
                [--validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE]
                [--validation_disable_unconditional [VALIDATION_DISABLE_UNCONDITIONAL]]
                [--validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS]
                [--validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START]
                [--validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP]
                [--validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE]
                [--validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH]
                [--validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++,perflow}]
                [--validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES]
                [--validation_resolution VALIDATION_RESOLUTION]
                [--validation_seed_source {cpu,gpu}]
                [--i_know_what_i_am_doing [I_KNOW_WHAT_I_AM_DOING]]
                [--flow_sigmoid_scale FLOW_SIGMOID_SCALE]
                [--flux_fast_schedule [FLUX_FAST_SCHEDULE]]
                [--flow_use_uniform_schedule [FLOW_USE_UNIFORM_SCHEDULE]]
                [--flow_use_beta_schedule [FLOW_USE_BETA_SCHEDULE]]
                [--flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA]
                [--flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA]
                [--flow_schedule_shift FLOW_SCHEDULE_SHIFT]
                [--flow_schedule_auto_shift [FLOW_SCHEDULE_AUTO_SHIFT]]
                [--flux_guidance_mode {constant,random-range}]
                [--flux_attention_masked_training [FLUX_ATTENTION_MASKED_TRAINING]]
                [--flux_guidance_value FLUX_GUIDANCE_VALUE]
                [--flux_guidance_min FLUX_GUIDANCE_MIN]
                [--flux_guidance_max FLUX_GUIDANCE_MAX]
                [--t5_padding {zero,unmodified}]
                [--sd3_clip_uncond_behaviour {empty_string,zero}]
                [--sd3_t5_uncond_behaviour {empty_string,zero}]
                [--soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA]
                [--mixed_precision {no,fp16,bf16,fp8}]
                [--attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}]
                [--sageattention_usage {training,inference,training+inference}]
                [--disable_tf32 [DISABLE_TF32]]
                [--set_grads_to_none [SET_GRADS_TO_NONE]]
                [--noise_offset NOISE_OFFSET]
                [--noise_offset_probability NOISE_OFFSET_PROBABILITY]
                [--input_perturbation INPUT_PERTURBATION]
                [--input_perturbation_steps INPUT_PERTURBATION_STEPS]
                [--lr_end LR_END] [--lr_scale [LR_SCALE]]
                [--lr_scale_sqrt [LR_SCALE_SQRT]]
                [--ignore_final_epochs [IGNORE_FINAL_EPOCHS]]
                [--freeze_encoder_before FREEZE_ENCODER_BEFORE]
                [--freeze_encoder_after FREEZE_ENCODER_AFTER]
                [--freeze_encoder_strategy {before,between,after}]
                [--layer_freeze_strategy {none,bitfit}]
                [--fully_unload_text_encoder [FULLY_UNLOAD_TEXT_ENCODER]]
                [--save_text_encoder [SAVE_TEXT_ENCODER]]
                [--text_encoder_limit TEXT_ENCODER_LIMIT]
                [--prepend_instance_prompt [PREPEND_INSTANCE_PROMPT]]
                [--only_instance_prompt [ONLY_INSTANCE_PROMPT]]
                [--data_aesthetic_score DATA_AESTHETIC_SCORE]
                [--delete_unwanted_images [DELETE_UNWANTED_IMAGES]]
                [--delete_problematic_images [DELETE_PROBLEMATIC_IMAGES]]
                [--disable_bucket_pruning [DISABLE_BUCKET_PRUNING]]
                [--disable_segmented_timestep_sampling [DISABLE_SEGMENTED_TIMESTEP_SAMPLING]]
                [--preserve_data_backend_cache [PRESERVE_DATA_BACKEND_CACHE]]
                [--override_dataset_config [OVERRIDE_DATASET_CONFIG]]
                [--cache_dir CACHE_DIR] [--cache_dir_text CACHE_DIR_TEXT]
                [--cache_dir_vae CACHE_DIR_VAE]
                [--compress_disk_cache [COMPRESS_DISK_CACHE]]
                [--aspect_bucket_disable_rebuild [ASPECT_BUCKET_DISABLE_REBUILD]]
                [--keep_vae_loaded [KEEP_VAE_LOADED]]
                [--skip_file_discovery SKIP_FILE_DISCOVERY]
                [--data_backend_sampling {uniform,auto-weighting}]
                [--image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE]
                [--write_batch_size WRITE_BATCH_SIZE]
                [--read_batch_size READ_BATCH_SIZE]
                [--enable_multiprocessing [ENABLE_MULTIPROCESSING]]
                [--max_workers MAX_WORKERS]
                [--aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS]
                [--torch_num_threads TORCH_NUM_THREADS]
                [--dataloader_prefetch [DATALOADER_PREFETCH]]
                [--dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN]
                [--aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT]
                [--aspect_bucket_alignment {8,16,24,32,64}]
                [--minimum_image_size MINIMUM_IMAGE_SIZE]
                [--maximum_image_size MAXIMUM_IMAGE_SIZE]
                [--target_downsample_size TARGET_DOWNSAMPLE_SIZE]
                [--max_upscale_threshold MAX_UPSCALE_THRESHOLD]
                [--metadata_update_interval METADATA_UPDATE_INTERVAL]
                [--debug_aspect_buckets [DEBUG_ASPECT_BUCKETS]]
                [--debug_dataset_loader [DEBUG_DATASET_LOADER]]
                [--print_filenames [PRINT_FILENAMES]]
                [--print_sampler_statistics [PRINT_SAMPLER_STATISTICS]]
                [--timestep_bias_strategy {earlier,later,range,none}]
                [--timestep_bias_begin TIMESTEP_BIAS_BEGIN]
                [--timestep_bias_end TIMESTEP_BIAS_END]
                [--timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER]
                [--timestep_bias_portion TIMESTEP_BIAS_PORTION]
                [--training_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--inference_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--loss_type {l2,huber,smooth_l1}]
                [--huber_schedule {snr,exponential,constant}]
                [--huber_c HUBER_C] [--snr_gamma SNR_GAMMA]
                [--masked_loss_probability MASKED_LOSS_PROBABILITY]
                [--hidream_use_load_balancing_loss [HIDREAM_USE_LOAD_BALANCING_LOSS]]
                [--hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT]
                [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                [--optimizer_beta1 OPTIMIZER_BETA1]
                [--optimizer_beta2 OPTIMIZER_BETA2]
                [--optimizer_cpu_offload_method {none}]
                [--gradient_precision {unmodified,fp32}]
                [--adam_weight_decay ADAM_WEIGHT_DECAY]
                [--adam_epsilon ADAM_EPSILON] [--prodigy_steps PRODIGY_STEPS]
                [--max_grad_norm MAX_GRAD_NORM]
                [--grad_clip_method {value,norm}]
                [--optimizer_offload_gradients [OPTIMIZER_OFFLOAD_GRADIENTS]]
                [--fuse_optimizer [FUSE_OPTIMIZER]]
                [--optimizer_release_gradients [OPTIMIZER_RELEASE_GRADIENTS]]
                [--push_to_hub [PUSH_TO_HUB]]
                [--push_to_hub_background [PUSH_TO_HUB_BACKGROUND]]
                [--push_checkpoints_to_hub [PUSH_CHECKPOINTS_TO_HUB]]
                [--publishing_config PUBLISHING_CONFIG]
                [--hub_model_id HUB_MODEL_ID]
                [--model_card_private [MODEL_CARD_PRIVATE]]
                [--model_card_safe_for_work [MODEL_CARD_SAFE_FOR_WORK]]
                [--model_card_note MODEL_CARD_NOTE]
                [--report_to {tensorboard,wandb,comet_ml,all,none}]
                [--checkpoint_step_interval CHECKPOINT_STEP_INTERVAL]
                [--checkpoint_epoch_interval CHECKPOINT_EPOCH_INTERVAL]
                [--checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS]
                [--checkpointing_use_tempdir [CHECKPOINTING_USE_TEMPDIR]]
                [--checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT]
                [--tracker_run_name TRACKER_RUN_NAME]
                [--tracker_project_name TRACKER_PROJECT_NAME]
                [--tracker_image_layout {gallery,table}]
                [--enable_watermark [ENABLE_WATERMARK]]
                [--framerate FRAMERATE]
                [--seed_for_each_device [SEED_FOR_EACH_DEVICE]]
                [--snr_weight SNR_WEIGHT]
                [--rescale_betas_zero_snr [RESCALE_BETAS_ZERO_SNR]]
                [--webhook_config WEBHOOK_CONFIG]
                [--webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL]
                [--distillation_method {lcm,dcm,dmd,perflow}]
                [--distillation_config DISTILLATION_CONFIG]
                [--ema_validation {none,ema_only,comparison}]
                [--local_rank LOCAL_RANK] [--ltx_train_mode {t2v,i2v}]
                [--ltx_i2v_prob LTX_I2V_PROB]
                [--ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION]
                [--ltx_protect_first_frame [LTX_PROTECT_FIRST_FRAME]]
                [--offload_param_path OFFLOAD_PARAM_PATH]
                [--offset_noise [OFFSET_NOISE]]
                [--quantize_activations [QUANTIZE_ACTIVATIONS]]
                [--refiner_training [REFINER_TRAINING]]
                [--refiner_training_invert_schedule [REFINER_TRAINING_INVERT_SCHEDULE]]
                [--refiner_training_strength REFINER_TRAINING_STRENGTH]
                [--sdxl_refiner_uses_full_range [SDXL_REFINER_USES_FULL_RANGE]]
                [--sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION]

The following SimpleTuner command-line options are available:

options:
  -h, --help            show this help message and exit
  --model_family {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo}
                        The base model architecture family to train
  --model_flavour MODEL_FLAVOUR
                        Specific variant of the selected model family
  --controlnet [CONTROLNET]
                        Train ControlNet (full or LoRA) branches alongside the
                        primary network.
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        Optional override of the model checkpoint. Leave blank
                        to use the default path for the selected model
                        flavour.
  --output_dir OUTPUT_DIR
                        Directory where model checkpoints and logs will be
                        saved
  --logging_dir LOGGING_DIR
                        Directory for TensorBoard logs
  --model_type {full,lora}
                        Choose between full model training or LoRA adapter
                        training
  --seed SEED           Seed used for deterministic training behaviour
  --resolution RESOLUTION
                        Resolution for training images
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Select checkpoint to resume training from
  --prediction_type {epsilon,v_prediction,sample,flow_matching}
                        The parameterization type for the diffusion model
  --pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH
                        Path to pretrained VAE model
  --vae_dtype {default,fp32,fp16,bf16}
                        Precision for VAE encoding/decoding. Lower precision
                        saves memory.
  --vae_cache_ondemand [VAE_CACHE_ONDEMAND]
                        Process VAE latents during training instead of
                        precomputing them
  --vae_cache_disable [VAE_CACHE_DISABLE]
                        Implicitly enables on-demand caching and disables
                        writing embeddings to disk.
  --accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL
                        Clear the cache from VRAM every X steps to prevent
                        memory leaks
  --aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}
                        Number of decimal places to round aspect ratios to for
                        bucket creation
  --base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for loading the base model. Lower precision
                        saves memory.
  --text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL
                        Checkpoint every N transformer blocks
  --offload_during_startup [OFFLOAD_DURING_STARTUP]
                        Offload text encoders to CPU during VAE caching
  --quantize_via {cpu,accelerator,pipeline}
                        Where to perform model quantization
  --quantization_config QUANTIZATION_CONFIG
                        JSON or file path describing Diffusers quantization
                        config for pipeline quantization
  --fuse_qkv_projections [FUSE_QKV_PROJECTIONS]
                        Enables Flash Attention 3 when supported; otherwise
                        falls back to PyTorch SDPA.
  --control [CONTROL]   Enable channel-wise control style training
  --controlnet_custom_config CONTROLNET_CUSTOM_CONFIG
                        Custom configuration for ControlNet models
  --controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH
                        Path to ControlNet model weights to preload
  --tread_config TREAD_CONFIG
                        Configuration for TREAD training method
  --pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH
                        Path to pretrained transformer model
  --pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER
                        Subfolder containing transformer model weights
  --pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH
                        Path to pretrained UNet model
  --pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER
                        Subfolder containing UNet model weights
  --pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH
                        Path to pretrained T5 model
  --revision REVISION   Git branch/tag/commit for model version
  --variant VARIANT     Model variant (e.g., fp16, bf16)
  --base_model_default_dtype {bf16,fp32}
                        Default precision for quantized base model weights
  --unet_attention_slice [UNET_ATTENTION_SLICE]
                        Enable attention slicing for SDXL UNet
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of times to iterate through the entire dataset
  --max_train_steps MAX_TRAIN_STEPS
                        Maximum number of training steps (0 = use epochs
                        instead)
  --train_batch_size TRAIN_BATCH_SIZE
                        Number of samples processed per forward/backward pass
                        (per device).
  --learning_rate LEARNING_RATE
                        Base learning rate for training
  --optimizer {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,prodigy}
                        Optimization algorithm for training
  --optimizer_config OPTIMIZER_CONFIG
                        Comma-separated key=value pairs forwarded to the
                        selected optimizer
  --lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        How learning rate changes during training
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of steps to accumulate gradients
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps to gradually increase LR from 0
  --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
                        Maximum number of checkpoints to keep on disk
  --gradient_checkpointing [GRADIENT_CHECKPOINTING]
                        Trade compute for memory during training
  --train_text_encoder [TRAIN_TEXT_ENCODER]
                        Also train the text encoder (CLIP) model
  --text_encoder_lr TEXT_ENCODER_LR
                        Separate learning rate for text encoder
  --lr_num_cycles LR_NUM_CYCLES
                        Number of cosine annealing cycles
  --lr_power LR_POWER   Power for polynomial decay scheduler
  --use_soft_min_snr [USE_SOFT_MIN_SNR]
                        Use soft clamping instead of hard clamping for Min-SNR
  --use_ema [USE_EMA]   Maintain an exponential moving average copy of the
                        model during training.
  --ema_device {accelerator,cpu}
                        Where to keep the EMA weights in-between updates.
  --ema_cpu_only [EMA_CPU_ONLY]
                        Keep EMA weights exclusively on CPU even when
                        ema_device would normally move them.
  --ema_update_interval EMA_UPDATE_INTERVAL
                        Update EMA weights every N optimizer steps
  --ema_foreach_disable [EMA_FOREACH_DISABLE]
                        Fallback to standard tensor ops instead of
                        torch.foreach updates.
  --ema_decay EMA_DECAY
                        Smoothing factor for EMA updates (closer to 1.0 =
                        slower drift).
  --lora_rank LORA_RANK
                        Dimension of LoRA update matrices
  --lora_alpha LORA_ALPHA
                        Scaling factor for LoRA updates
  --lora_type {standard,lycoris}
                        LoRA implementation type
  --lora_dropout LORA_DROPOUT
                        LoRA dropout randomly ignores neurons during training.
                        This can help prevent overfitting.
  --lora_init_type {default,gaussian,loftq,olora,pissa}
                        The initialization type for the LoRA model
  --peft_lora_mode {standard,singlora}
                        PEFT LoRA training mode
  --peft_lora_target_modules PEFT_LORA_TARGET_MODULES
                        JSON array (or path to a JSON file) listing PEFT
                        LoRA target module names. Overrides preset targets.
  --singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS
                        Number of ramp-up steps for SingLoRA
  --slider_lora_target [SLIDER_LORA_TARGET]
                        Route LoRA training to slider-friendly targets
                        (self-attn + conv/time embeddings). Only affects
                        standard PEFT LoRA.
  --init_lora INIT_LORA
                        Specify an existing LoRA or LyCORIS safetensors file
                        to initialize the adapter
  --lycoris_config LYCORIS_CONFIG
                        Path to LyCORIS configuration JSON file
  --init_lokr_norm INIT_LOKR_NORM
                        Perturbed normal initialization for LyCORIS LoKr
                        layers
  --flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,controlnet,all+ffs+embedder,all+ffs+embedder+controlnet}
                        Which layers to train in Flux models
  --use_dora [USE_DORA]
                        Enable DoRA (Weight-Decomposed LoRA)
  --resolution_type {pixel,area,pixel_area}
                        How to interpret the resolution value
  --data_backend_config DATA_BACKEND_CONFIG
                        Select a saved dataset configuration (managed in
                        Datasets & Environments tabs)
  --caption_strategy {filename,textfile,instance_prompt,parquet}
                        How to load captions for images
  --conditioning_multidataset_sampling {combined,random}
                        How to sample from multiple conditioning datasets
  --instance_prompt INSTANCE_PROMPT
                        Instance prompt for training
  --parquet_caption_column PARQUET_CAPTION_COLUMN
                        Column name containing captions in parquet files
  --parquet_filename_column PARQUET_FILENAME_COLUMN
                        Column name containing image paths in parquet files
  --ignore_missing_files [IGNORE_MISSING_FILES]
                        Continue training even if some files are missing
  --vae_cache_scan_behaviour {recreate,sync}
                        How to scan VAE cache for missing files
  --vae_enable_slicing [VAE_ENABLE_SLICING]
                        Enable VAE attention slicing for memory efficiency
  --vae_enable_tiling [VAE_ENABLE_TILING]
                        Enable VAE tiling for large images
  --vae_enable_patch_conv [VAE_ENABLE_PATCH_CONV]
                        Enable patch-based 3D conv for HunyuanVideo VAE to
                        reduce peak VRAM (slight slowdown)
  --vae_batch_size VAE_BATCH_SIZE
                        Batch size for VAE encoding during caching
  --caption_dropout_probability CAPTION_DROPOUT_PROBABILITY
                        Caption dropout will randomly drop captions and, for
                        SDXL, size conditioning inputs based on this
                        probability
  --tokenizer_max_length TOKENIZER_MAX_LENGTH
                        Override the tokenizer sequence length (advanced).
  --validation_step_interval VALIDATION_STEP_INTERVAL
                        Run validation every N training steps (deprecated alias: --validation_steps)
  --validation_epoch_interval VALIDATION_EPOCH_INTERVAL
                        Run validation every N training epochs
  --disable_benchmark [DISABLE_BENCHMARK]
                        Skip generating baseline comparison images before
                        training starts
  --validation_prompt VALIDATION_PROMPT
                        Prompt to use for validation images
  --num_validation_images NUM_VALIDATION_IMAGES
                        Number of images to generate per validation
  --num_eval_images NUM_EVAL_IMAGES
                        Number of images to generate for evaluation metrics
  --eval_steps_interval EVAL_STEPS_INTERVAL
                        Run evaluation every N training steps
  --eval_epoch_interval EVAL_EPOCH_INTERVAL
                        Run evaluation every N training epochs (decimals run
                        multiple times per epoch)
  --eval_timesteps EVAL_TIMESTEPS
                        Number of timesteps for evaluation
  --eval_dataset_pooling [EVAL_DATASET_POOLING]
                        Combine evaluation metrics from all datasets into a
                        single chart
  --evaluation_type {none,clip}
                        Type of evaluation metrics to compute
  --pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH
                        Path to pretrained model for evaluation metrics
  --validation_guidance VALIDATION_GUIDANCE
                        CFG guidance scale for validation images
  --validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS
                        Number of diffusion steps for validation renders
  --validation_on_startup [VALIDATION_ON_STARTUP]
                        Run validation on the base model before training
                        starts
  --validation_using_datasets [VALIDATION_USING_DATASETS]
                        Use random images from training datasets for
                        validation
  --validation_torch_compile [VALIDATION_TORCH_COMPILE]
                        Use torch.compile() on validation pipeline for speed
  --validation_guidance_real VALIDATION_GUIDANCE_REAL
                        CFG value for distilled models (e.g., FLUX schnell)
  --validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP
                        Skip CFG for initial timesteps (Flux only)
  --validation_negative_prompt VALIDATION_NEGATIVE_PROMPT
                        Negative prompt for validation images
  --validation_randomize [VALIDATION_RANDOMIZE]
                        Use random seeds for each validation
  --validation_seed VALIDATION_SEED
                        Fixed seed for reproducible validation images
  --validation_disable [VALIDATION_DISABLE]
                        Completely disable validation image generation
  --validation_prompt_library [VALIDATION_PROMPT_LIBRARY]
                        Use SimpleTuner's built-in prompt library
  --user_prompt_library USER_PROMPT_LIBRARY
                        Path to custom JSON prompt library
  --eval_dataset_id EVAL_DATASET_ID
                        Specific dataset to use for evaluation metrics
  --validation_stitch_input_location {left,right}
                        Where to place input image in img2img validations
  --validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE
                        CFG rescale value for validation
  --validation_disable_unconditional [VALIDATION_DISABLE_UNCONDITIONAL]
                        Disable unconditional image generation during
                        validation
  --validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS
                        JSON list of transformer layers to skip during
                        classifier-free guidance
  --validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START
                        Starting layer index to skip guidance
  --validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP
                        Ending layer index to skip guidance
  --validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE
                        Scale guidance strength when applying layer skipping
  --validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH
                        Strength multiplier for LyCORIS validation
  --validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++,perflow}
                        Noise scheduler for validation
  --validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES
                        Number of frames for video validation
  --validation_resolution VALIDATION_RESOLUTION
                        Override resolution for validation images (pixels or
                        megapixels)
  --validation_seed_source {cpu,gpu}
                        Source device used to generate validation seeds
  --i_know_what_i_am_doing [I_KNOW_WHAT_I_AM_DOING]
                        Unlock experimental overrides and bypass built-in
                        safety limits.
  --flow_sigmoid_scale FLOW_SIGMOID_SCALE
                        Scale factor for sigmoid timestep sampling for flow-
                        matching models.
  --flux_fast_schedule [FLUX_FAST_SCHEDULE]
                        Use experimental fast schedule for Flux training
  --flow_use_uniform_schedule [FLOW_USE_UNIFORM_SCHEDULE]
                        Use uniform schedule instead of sigmoid for flow-
                        matching
  --flow_use_beta_schedule [FLOW_USE_BETA_SCHEDULE]
                        Use beta schedule instead of sigmoid for flow-matching
  --flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA
                        Alpha value for beta schedule (default: 2.0)
  --flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA
                        Beta value for beta schedule (default: 2.0)
  --flow_schedule_shift FLOW_SCHEDULE_SHIFT
                        Shift the noise schedule for flow-matching models
  --flow_schedule_auto_shift [FLOW_SCHEDULE_AUTO_SHIFT]
                        Auto-adjust schedule shift based on image resolution
  --flux_guidance_mode {constant,random-range}
                        Guidance mode for Flux training
  --flux_attention_masked_training [FLUX_ATTENTION_MASKED_TRAINING]
                        Enable attention masked training for Flux models
  --flux_guidance_value FLUX_GUIDANCE_VALUE
                        Guidance value for constant mode
  --flux_guidance_min FLUX_GUIDANCE_MIN
                        Minimum guidance value for random-range mode
  --flux_guidance_max FLUX_GUIDANCE_MAX
                        Maximum guidance value for random-range mode
  --t5_padding {zero,unmodified}
                        Padding behavior for T5 text encoder
  --sd3_clip_uncond_behaviour {empty_string,zero}
                        How SD3 handles unconditional prompts
  --sd3_t5_uncond_behaviour {empty_string,zero}
                        How SD3 T5 handles unconditional prompts
  --soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA
                        Sigma data for soft min SNR weighting
  --mixed_precision {no,fp16,bf16,fp8}
                        Precision for training computations
  --attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}
                        Attention computation backend
  --sageattention_usage {training,inference,training+inference}
                        When to use SageAttention
  --disable_tf32 [DISABLE_TF32]
                        Force IEEE FP32 precision (disables TF32) using
                        PyTorch's fp32_precision controls when available
  --set_grads_to_none [SET_GRADS_TO_NONE]
                        Set gradients to None instead of zero
  --noise_offset NOISE_OFFSET
                        Add noise offset to training
  --noise_offset_probability NOISE_OFFSET_PROBABILITY
                        Probability of applying noise offset
  --input_perturbation INPUT_PERTURBATION
                        Add additional noise only to the inputs fed to the
                        model during training
  --input_perturbation_steps INPUT_PERTURBATION_STEPS
                        Only apply input perturbation over the first N steps
                        with linear decay
  --lr_end LR_END       A polynomial learning rate will end up at this value
                        after the specified number of warmup steps
  --lr_scale [LR_SCALE]
                        Scale the learning rate by the number of GPUs,
                        gradient accumulation steps, and batch size
  --lr_scale_sqrt [LR_SCALE_SQRT]
                        If using --lr_scale, use the square root of (number of
                        GPUs * gradient accumulation steps * batch size)
  --ignore_final_epochs [IGNORE_FINAL_EPOCHS]
                        When provided, the max epoch counter will not
                        determine the end of the training run
  --freeze_encoder_before FREEZE_ENCODER_BEFORE
                        When using 'before' strategy, we will freeze layers
                        earlier than this
  --freeze_encoder_after FREEZE_ENCODER_AFTER
                        When using 'after' strategy, we will freeze layers
                        later than this
  --freeze_encoder_strategy {before,between,after}
                        When freezing the text encoder, we can use the
                        'before', 'between', or 'after' strategy
  --layer_freeze_strategy {none,bitfit}
                        When freezing parameters, we can use the 'none' or
                        'bitfit' strategy
  --fully_unload_text_encoder [FULLY_UNLOAD_TEXT_ENCODER]
                        If set, will fully unload the text_encoder from memory
                        when not in use
  --save_text_encoder [SAVE_TEXT_ENCODER]
                        If set, will save the text encoder after training
  --text_encoder_limit TEXT_ENCODER_LIMIT
                        When training the text encoder, we want to limit how
                        long it trains for to avoid catastrophic loss
  --prepend_instance_prompt [PREPEND_INSTANCE_PROMPT]
                        When determining the captions from the filename,
                        prepend the instance prompt as an enforced keyword
  --only_instance_prompt [ONLY_INSTANCE_PROMPT]
                        Use the instance prompt instead of the caption from
                        filename
  --data_aesthetic_score DATA_AESTHETIC_SCORE
                        Since currently we do not calculate aesthetic scores
                        for data, we will statically set it to one value. This
                        is only used by the SDXL Refiner
  --delete_unwanted_images [DELETE_UNWANTED_IMAGES]
                        If set, will delete images that are not of a minimum
                        size to save on disk space for large training runs
  --delete_problematic_images [DELETE_PROBLEMATIC_IMAGES]
                        If set, any images that error out during load will be
                        removed from the underlying storage medium
  --disable_bucket_pruning [DISABLE_BUCKET_PRUNING]
                        When training on very small datasets, you might not
                        care that the batch sizes will outpace your image
                        count. Setting this option will prevent SimpleTuner
                        from deleting your bucket lists that do not meet the
                        minimum image count requirements. Use at your own
                        risk, it may end up throwing off your statistics or
                        epoch tracking
  --disable_segmented_timestep_sampling [DISABLE_SEGMENTED_TIMESTEP_SAMPLING]
                        By default, the timestep schedule is divided into
                        roughly `train_batch_size` number of segments, and
                        then each of those are sampled from separately. This
                        improves the selection distribution, but may not be
                        desired in certain training scenarios, eg. when
                        limiting the timestep selection range
  --preserve_data_backend_cache [PRESERVE_DATA_BACKEND_CACHE]
                        For very large cloud storage buckets that will never
                        change, enabling this option will prevent the trainer
                        from scanning it at startup, by preserving the cache
                        files that we generate. Be careful when using this,
                        as, switching datasets can result in the preserved
                        cache being used, which would be problematic.
                        Currently, cache is not stored in the dataset itself
                        but rather, locally. This may change in a future
                        release
  --override_dataset_config [OVERRIDE_DATASET_CONFIG]
                        When provided, the dataset's config will not be
                        checked against the live backend config
  --cache_dir CACHE_DIR
                        The directory where the downloaded models and datasets
                        will be stored
  --cache_dir_text CACHE_DIR_TEXT
                        This is the path to a local directory that will
                        contain your text embed cache
  --cache_dir_vae CACHE_DIR_VAE
                        This is the path to a local directory that will
                        contain your VAE outputs
  --compress_disk_cache [COMPRESS_DISK_CACHE]
                        If set, will gzip-compress the disk cache for Pytorch
                        files. This will save substantial disk space, but may
                        slow down the training process
  --aspect_bucket_disable_rebuild [ASPECT_BUCKET_DISABLE_REBUILD]
                        When using a randomised aspect bucket list, the VAE
                        and aspect cache are rebuilt on each epoch. With a
                        large and diverse enough dataset, rebuilding the
                        aspect list may take a long time, and this may be
                        undesirable. This option will not override
                        vae_cache_clear_each_epoch. If both options are
                        provided, only the VAE cache will be rebuilt
  --keep_vae_loaded [KEEP_VAE_LOADED]
                        If set, will keep the VAE loaded in memory. This can
                        reduce disk churn, but consumes VRAM during the
                        forward pass
  --skip_file_discovery SKIP_FILE_DISCOVERY
                        Comma-separated values of which stages to skip
                        discovery for. Skipping any stage will speed up
                        resumption, but will increase the risk of errors, as
                        missing images or incorrectly bucketed images may not
                        be caught. Valid options: aspect, vae, text, metadata
  --data_backend_sampling {uniform,auto-weighting}
                        When using multiple data backends, the sampling
                        weighting can be set to 'uniform' or 'auto-weighting'
  --image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE
                        When resizing and cropping images, we do it in
                        parallel using processes or threads. This defines how
                        many images will be read into the queue before they
                        are processed
  --write_batch_size WRITE_BATCH_SIZE
                        When using certain storage backends, it is better to
                        batch smaller writes rather than continuous
                        dispatching. In SimpleTuner, write batching is
                        currently applied during VAE caching, when many small
                        objects are written. This mostly applies to S3, but
                        some shared server filesystems may benefit as well.
                        Default: 64
  --read_batch_size READ_BATCH_SIZE
                        Used by the VAE cache to prefetch image data. This is
                        the number of images to read ahead
  --enable_multiprocessing [ENABLE_MULTIPROCESSING]
                        If set, will use processes instead of threads during
                        metadata caching operations
  --max_workers MAX_WORKERS
                        How many active threads or processes to run during VAE
                        caching
  --aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS
                        When using AWS backends, the maximum number of
                        connections to keep open to the S3 bucket at a single
                        time
  --torch_num_threads TORCH_NUM_THREADS
                        The number of threads to use for PyTorch operations.
                        This is not the same as the number of workers
  --dataloader_prefetch [DATALOADER_PREFETCH]
                        When provided, the dataloader will read-ahead and
                        attempt to retrieve latents, text embeds, and other
                        metadata ahead of the time when the batch is required,
                        so that it can be immediately available
  --dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN
                        Set the number of prefetched batches
  --aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT
                        The number of workers to use for aspect bucketing.
                        This is a CPU-bound task, so the number of workers
                        should be set to the number of CPU threads available.
                        If you use an I/O bound backend, an even higher value
                        may make sense. Default: 12
  --aspect_bucket_alignment {8,16,24,32,64}
                        When training diffusion models, the image sizes
                        generally must align to a 64 pixel interval
  --minimum_image_size MINIMUM_IMAGE_SIZE
                        The minimum resolution for both sides of input images
  --maximum_image_size MAXIMUM_IMAGE_SIZE
                        When cropping images that are excessively large, the
                        entire scene context may be lost, eg. the crop might
                        just end up being a portion of the background. To
                        avoid this, a maximum image size may be provided,
                        which will result in very-large images being
                        downsampled before cropping them. This value uses
                        --resolution_type to determine whether it is a pixel
                        edge or megapixel value
  --target_downsample_size TARGET_DOWNSAMPLE_SIZE
                        When using --maximum_image_size, very-large images
                        exceeding that value will be downsampled to this
                        target size before cropping
  --max_upscale_threshold MAX_UPSCALE_THRESHOLD
                        Limit upscaling of small images to prevent quality
                        degradation (opt-in). When set, filters out aspect
                        buckets requiring upscaling beyond this threshold.
                        For example, 0.2 allows up to 20% upscaling. Default
                        (None) allows unlimited upscaling. Must be between 0
                        and 1.
  --metadata_update_interval METADATA_UPDATE_INTERVAL
                        When generating the aspect bucket indicies, we want to
                        save it every X seconds
  --debug_aspect_buckets [DEBUG_ASPECT_BUCKETS]
                        If set, will print excessive debugging for aspect
                        bucket operations
  --debug_dataset_loader [DEBUG_DATASET_LOADER]
                        If set, will print excessive debugging for data loader
                        operations
  --print_filenames [PRINT_FILENAMES]
                        If any image files are stopping the process eg. due to
                        corruption or truncation, this will help identify
                        which is at fault
  --print_sampler_statistics [PRINT_SAMPLER_STATISTICS]
                        If provided, will print statistics about the dataset
                        sampler. This is useful for debugging
  --timestep_bias_strategy {earlier,later,range,none}
                        Strategy for biasing timestep sampling
  --timestep_bias_begin TIMESTEP_BIAS_BEGIN
                        Beginning of timestep bias range
  --timestep_bias_end TIMESTEP_BIAS_END
                        End of timestep bias range
  --timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER
                        Multiplier for timestep bias probability
  --timestep_bias_portion TIMESTEP_BIAS_PORTION
                        Portion of training steps to apply timestep bias
  --training_scheduler_timestep_spacing {leading,linspace,trailing}
                        Timestep spacing for training scheduler
  --inference_scheduler_timestep_spacing {leading,linspace,trailing}
                        Timestep spacing for inference scheduler
  --loss_type {l2,huber,smooth_l1}
                        Loss function for training
  --huber_schedule {snr,exponential,constant}
                        Schedule for Huber loss transition threshold
  --huber_c HUBER_C     Transition point between L2 and L1 regions for Huber
                        loss
  --snr_gamma SNR_GAMMA
                        SNR weighting gamma value (0 = disabled)
  --masked_loss_probability MASKED_LOSS_PROBABILITY
                        Probability of applying masked loss weighting per
                        batch
  --hidream_use_load_balancing_loss [HIDREAM_USE_LOAD_BALANCING_LOSS]
                        Apply experimental load balancing loss when training
                        HiDream models.
  --hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT
                        Strength multiplier for HiDream load balancing loss.
  --adam_beta1 ADAM_BETA1
                        First moment decay rate for Adam optimizers
  --adam_beta2 ADAM_BETA2
                        Second moment decay rate for Adam optimizers
  --optimizer_beta1 OPTIMIZER_BETA1
                        First moment decay rate for optimizers
  --optimizer_beta2 OPTIMIZER_BETA2
                        Second moment decay rate for optimizers
  --optimizer_cpu_offload_method {none}
                        Method for CPU offloading optimizer states
  --gradient_precision {unmodified,fp32}
                        Precision for gradient computation
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        L2 regularisation strength for Adam-family optimizers.
  --adam_epsilon ADAM_EPSILON
                        Small constant added for numerical stability.
  --prodigy_steps PRODIGY_STEPS
                        Number of steps Prodigy should spend adapting its
                        learning rate.
  --max_grad_norm MAX_GRAD_NORM
                        Gradient clipping threshold to prevent exploding
                        gradients.
  --grad_clip_method {value,norm}
                        Strategy for applying max_grad_norm during clipping.
  --optimizer_offload_gradients [OPTIMIZER_OFFLOAD_GRADIENTS]
                        Move optimizer gradients to CPU to save GPU memory.
  --fuse_optimizer [FUSE_OPTIMIZER]
                        Enable fused kernels when offloading to reduce memory
                        overhead.
  --optimizer_release_gradients [OPTIMIZER_RELEASE_GRADIENTS]
                        Free gradient tensors immediately after optimizer step
                        when using Optimi optimizers.
  --push_to_hub [PUSH_TO_HUB]
                        Automatically upload the trained model to your Hugging
                        Face Hub repository.
  --push_to_hub_background [PUSH_TO_HUB_BACKGROUND]
                        Run Hub uploads in a background worker so training is
                        not blocked while pushing.
  --push_checkpoints_to_hub [PUSH_CHECKPOINTS_TO_HUB]
                        Upload intermediate checkpoints to the same Hugging
                        Face repository during training.
  --publishing_config PUBLISHING_CONFIG
                        Optional JSON/file path describing additional
                        publishing targets (S3/Backblaze B2/Azure Blob/Dropbox).
  --hub_model_id HUB_MODEL_ID
                        If left blank, SimpleTuner derives a name from the
                        project settings when pushing to Hub.
  --model_card_private [MODEL_CARD_PRIVATE]
                        Create the Hugging Face repository as private instead
                        of public.
  --model_card_safe_for_work [MODEL_CARD_SAFE_FOR_WORK]
                        Remove the default NSFW warning from the generated
                        model card on Hugging Face Hub.
  --model_card_note MODEL_CARD_NOTE
                        Optional note that appears at the top of the generated
                        model card.
  --report_to {tensorboard,wandb,comet_ml,all,none}
                        Where to log training metrics
  --checkpoint_step_interval CHECKPOINT_STEP_INTERVAL
                        Save model checkpoint every N steps (deprecated alias: --checkpointing_steps)
  --checkpoint_epoch_interval CHECKPOINT_EPOCH_INTERVAL
                        Save model checkpoint every N epochs
  --checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS
                        Rolling checkpoint window size for continuous
                        checkpointing
  --checkpointing_use_tempdir [CHECKPOINTING_USE_TEMPDIR]
                        Use temporary directory for checkpoint files before
                        final save
  --checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT
                        Maximum number of rolling checkpoints to keep
  --tracker_run_name TRACKER_RUN_NAME
                        Name for this training run in tracking platforms
  --tracker_project_name TRACKER_PROJECT_NAME
                        Project name in tracking platforms
  --tracker_image_layout {gallery,table}
                        How validation images are displayed in trackers
  --enable_watermark [ENABLE_WATERMARK]
                        Add invisible watermark to generated images
  --framerate FRAMERATE
                        Framerate for video model training
  --seed_for_each_device [SEED_FOR_EACH_DEVICE]
                        Use a unique deterministic seed per GPU instead of
                        sharing one seed across devices.
  --snr_weight SNR_WEIGHT
                        Weight factor for SNR-based loss scaling
  --rescale_betas_zero_snr [RESCALE_BETAS_ZERO_SNR]
                        Rescale betas for zero terminal SNR
  --webhook_config WEBHOOK_CONFIG
                        Path to webhook configuration file
  --webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL
                        Interval for webhook reports (seconds)
  --distillation_method {lcm,dcm,dmd,perflow}
                        Method for model distillation
  --distillation_config DISTILLATION_CONFIG
                        Path to distillation configuration file
  --ema_validation {none,ema_only,comparison}
                        Control how EMA weights are used during validation
                        runs.
  --local_rank LOCAL_RANK
                        Local rank for distributed training
  --ltx_train_mode {t2v,i2v}
                        Training mode for LTX models
  --ltx_i2v_prob LTX_I2V_PROB
                        Probability of using image-to-video training for LTX
  --ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION
                        Fraction of noise to add for LTX partial training
  --ltx_protect_first_frame [LTX_PROTECT_FIRST_FRAME]
                        Protect the first frame from noise in LTX training
  --offload_param_path OFFLOAD_PARAM_PATH
                        Path to offloaded parameter files
  --offset_noise [OFFSET_NOISE]
                        Enable offset-noise training
  --quantize_activations [QUANTIZE_ACTIVATIONS]
                        Quantize model activations during training
  --refiner_training [REFINER_TRAINING]
                        Enable refiner model training mode
  --refiner_training_invert_schedule [REFINER_TRAINING_INVERT_SCHEDULE]
                        Invert the noise schedule for refiner training
  --refiner_training_strength REFINER_TRAINING_STRENGTH
                        Strength of refiner training
  --sdxl_refiner_uses_full_range [SDXL_REFINER_USES_FULL_RANGE]
                        Use full timestep range for SDXL refiner
  --sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION
                        Complex human instruction for Sana model training
```
