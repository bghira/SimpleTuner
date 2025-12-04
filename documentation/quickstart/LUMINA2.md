## Lumina2 Quickstart

In this example, we'll be training a Lumina2 LoRA or full model fine-tune.

### Hardware requirements

Lumina2 is a 2B parameter model, making it much more accessible than larger models like Flux or SD3. The model's smaller size means:

When training a rank-16 LoRA, it uses:
- Approximately 12-14GB VRAM for LoRA training
- Approximately 16-20GB VRAM for full model fine-tuning
- About 20-30GB of system RAM during startup

You'll need:
- **Minimum**: A single RTX 3060 12GB or RTX 4060 Ti 16GB
- **Recommended**: RTX 3090, RTX 4090, or A100 for faster training
- **System RAM**: At least 32GB recommended

### Prerequisites

Make sure that you have python installed; SimpleTuner does well with 3.10 through 3.12.

You can check this by running:

```bash
python --version
```

If you don't have python 3.12 installed on Ubuntu, you can try the following:

```bash
apt -y install python3.12 python3.12-venv
```

#### Container image dependencies

For Vast, RunPod, and TensorDock (among others), the following will work on a CUDA 12.2-12.8 image:

```bash
apt -y install nvidia-cuda-toolkit
```

### Installation

Install SimpleTuner via pip:

```bash
pip install simpletuner[cuda]
```

For manual installation or development setup, see the [installation documentation](/documentation/INSTALL.md).

### Setting up the environment

To run SimpleTuner, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

#### Configuration file

Copy `config/config.json.example` to `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

There, you will need to modify the following variables:

- `model_type` - Set this to `lora` for LoRA training or `full` for full fine-tuning.
- `model_family` - Set this to `lumina2`.
- `output_dir` - Set this to the directory where you want to store your checkpoints and validation images. It's recommended to use a full path here.
- `train_batch_size` - Can be 1-4 depending on your GPU memory and dataset size.
- `validation_resolution` - Lumina2 supports multiple resolutions. Common options: `1024x1024`, `512x512`, `768x768`.
- `validation_guidance` - Lumina2 uses classifier-free guidance. Values of 3.5-7.0 work well.
- `validation_num_inference_steps` - 20-30 steps work well for Lumina2.
- `gradient_accumulation_steps` - Can be used to simulate larger batch sizes. A value of 2-4 works well.
- `optimizer` - `adamw_bf16` is recommended. `lion` and `optimi-stableadamw` also work well.
- `mixed_precision` - Keep this as `bf16` for best results.
- `gradient_checkpointing` - Set to `true` to save VRAM.
- `learning_rate` - For LoRA: `1e-4` to `5e-5`. For full fine-tuning: `1e-5` to `1e-6`.

#### Lumina2 example configuration

This goes into `config.json`

```json
{
    "base_model_precision": "int8-torchao",
    "checkpoint_step_interval": 50,
    "data_backend_config": "config/lumina2/multidatabackend.json",
    "disable_bucket_pruning": true,
    "eval_steps_interval": 50,
    "evaluation_type": "clip",
    "flow_schedule_auto_shift": true,
    "gradient_checkpointing": true,
    "hub_model_id": "lumina2-lora",
    "learning_rate": 1e-4,
    "lora_alpha": 16,
    "lora_rank": 16,
    "lora_type": "standard",
    "lr_scheduler": "constant",
    "max_train_steps": 400000,
    "model_family": "lumina2",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/lumina2",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "quantize_via": "cpu",
    "report_to": "wandb",
    "seed": 42,
    "tracker_project_name": "lumina2-training",
    "tracker_run_name": "lumina2-lora",
    "train_batch_size": 4,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_disable_unconditional": true,
    "validation_guidance": 4.0,
    "validation_guidance_rescale": 0.0,
    "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
    "validation_num_inference_steps": 40,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_prompt_library": false,
    "validation_resolution": "1024x1024",
    "validation_seed": 42,
    "validation_step_interval": 50
}
```

For Lycoris training, switch `lora_type` to `lycoris`

### Advanced Experimental Features

SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> ⚠️ These features increase the computational overhead of training.

#### Validation prompts

Inside `config/config.json` is the "primary validation prompt". Additionally, create a prompt library file:

```json
{
  "portrait": "a high-quality portrait photograph with natural lighting",
  "landscape": "a breathtaking landscape photograph with dramatic lighting",
  "artistic": "an artistic rendering with vibrant colors and creative composition",
  "detailed": "a highly detailed image with sharp focus and rich textures",
  "stylized": "a stylized illustration with unique artistic flair"
}
```

Add to your config:
```json
{
  "--user_prompt_library": "config/user_prompt_library.json"
}
```

#### Dataset considerations

Lumina2 benefits from high-quality training data. Create a `--data_backend_config` (`config/multidatabackend.json`):

> 💡 **Tip:** For large datasets where disk space is a concern, you can use `--vae_cache_disable` to perform online VAE encoding without caching the results to disk.

```json
[
  {
    "id": "lumina2-training",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 2048,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/lumina2/training",
    "instance_data_dir": "/datasets/training",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/lumina2",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

Create your dataset directory. Be sure to update this path with your actual location.

```bash
mkdir -p /datasets/training
# Place your images and caption files in /datasets/training/
```

Caption files should have the same name as the image with a `.txt` extension.

#### Login to WandB

SimpleTuner has **optional** tracker support, primarily focused on Weights & Biases. You can disable this with `report_to=none`.

To enable wandb, run the following commands:

```bash
wandb login
```

#### Login to Huggingface Hub

To push checkpoints to Huggingface Hub, ensure
```bash
huggingface-cli login
```

### Executing the training run

From the SimpleTuner directory, you have several options to start training:

**Option 1 (Recommended - pip install):**
```bash
pip install simpletuner[cuda]
simpletuner train
```

**Option 2 (Git clone method):**
```bash
simpletuner train
```

**Option 3 (Legacy method - still works):**
```bash
./train.sh
```

This will begin the text embed and VAE output caching to disk.

## Training tips for Lumina2

### Learning rates

#### LoRA Training
- Start with `1e-4` and adjust based on results
- Lumina2 trains quickly, so monitor early iterations closely
- Ranks 8-32 work well for most use cases, 64-128 may require closer monitoring, and 256-512 may be useful for training new tasks into the model

#### Full Fine-tuning
- Use lower learning rates: `1e-5` to `5e-6`
- Consider using EMA (Exponential Moving Average) for stability
- Gradient clipping (`max_grad_norm`) of 1.0 is recommended

### Resolution considerations

Lumina2 supports flexible resolutions:
- Training at 1024x1024 provides best quality
- Mixed resolution training (512px, 768px, 1024px) has not been field-tested for quality impact
- Aspect ratio bucketing works well with Lumina2

### Training duration

Due to Lumina2's efficient 2B parameter size:
- LoRA training often converges in 500-2000 steps
- Full fine-tuning may need 2000-5000 steps
- Monitor validation images frequently as the model trains quickly

### Common issues and solutions

1. **Model converging too quickly**: Lower the learning rate, switch from Lion optimiser to AdamW
2. **Artifacts in generated images**: Ensure high-quality training data and consider reducing learning rate
3. **Out of memory**: Enable gradient checkpointing and reduce batch size
4. **Easily overfitting**: Use regularisation datasets

## Inference tips

### Using your trained model

Lumina2 models can be used with:
- Diffusers library directly
- ComfyUI with appropriate nodes
- Other inference frameworks supporting Gemma2-based models

### Optimal inference settings

- Guidance scale: 4.0-6.0
- Inference steps: 20-50
- Use the same resolution you trained on for best results

## Notes

### Advantages of Lumina2

- Fast training due to 2B parameter size
- Good quality-to-size ratio
- Supports various training modes (LoRA, LyCORIS, full)
- Efficient memory usage

### Current limitations

- No ControlNet support yet
- Limited to text-to-image generation
- Requires caption quality to be high for best results

### Memory optimization

Unlike larger models, Lumina2 typically doesn't require:
- Model quantization
- Extreme memory optimization techniques
- Complex mixed precision strategies
