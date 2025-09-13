## Qwen Image Quickstart

In this example, we'll be training a LoRA for Qwen Image, a 20B parameter vision-language model. Due to its size, we'll need aggressive memory optimization techniques.

A 24GB GPU is the absolute minimum, and even then you'll need extensive quantization and careful configuration. 40GB+ is strongly recommended for a smoother experience.

When training on 24G, validations will run out of memory unless you use lower resolution or aggressive quant level beyond int8.

### Hardware requirements

Qwen Image is a 20B parameter model with a sophisticated text encoder that alone consumes ~16GB VRAM before quantization. The model uses a custom VAE with 16 latent channels.

**Important limitations:**
- **Not supported on AMD ROCm or MacOS** due to lack of efficient flash attention
- Batch size > 1 is not currently working correctly; use gradient accumulation instead
- TREAD (Text-Representation Enhanced Adversarial Diffusion) is not yet supported

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
apt -y install nvidia-cuda-toolkit libgl1-mesa-glx
```

If `libgl1-mesa-glx` is not found, you might need to use `libgl1-mesa-dri` instead. Your mileage may vary.

### Installation

Clone the SimpleTuner repository and set up the python venv:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

# if python --version shows 3.12 you can just also use the 'python' command here.
python3.12 -m venv .venv

source .venv/bin/activate

```

**Note:** We're currently installing the `release` branch here; the `main` branch may contain experimental features that might have better results or lower memory use.

Depending on your system, you will run one of 3 commands:

```bash
# Linux
pip install -e .
```

### Setting up the environment

To run SimpleTuner, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

#### Configuration file

An experimental script, `configure.py`, may allow you to entirely skip this section through an interactive step-by-step configuration. It contains some safety features that help avoid common pitfalls.

**Note:** This doesn't configure your dataloader. You will still have to do that manually, later.

To run it:

```bash
simpletuner configure
```

> ⚠️ For users located in countries where Hugging Face Hub is not readily accessible, you should add `HF_ENDPOINT=https://hf-mirror.com` to your `~/.bashrc` or `~/.zshrc` depending on which `$SHELL` your system uses.

If you prefer to manually configure:

Copy `config/config.json.example` to `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

There, you will possibly need to modify the following variables:

- `model_type` - Set this to `lora`.
- `lora_type` - Set this to `standard` for PEFT LoRA or `lycoris` for LoKr.
- `model_family` - Set this to `qwen_image`.
- `model_flavour` - Set this to `v1.0`.
- `output_dir` - Set this to the directory where you want to store your checkpoints and validation images. It's recommended to use a full path here.
- `train_batch_size` - Must be set to 1 (batch size > 1 is not currently working).
- `gradient_accumulation_steps` - Set this to 2-8 to simulate larger batch sizes.
- `validation_resolution` - You should set this to `1024x1024` or lower for memory constraints.
  - 24G cannot handle 1024x1024 validations currently - you'll need to reduce the size
  - Other resolutions may be specified using commas to separate them: `1024x1024,768x768,512x512`
- `validation_guidance` - Use a value around 3.0-4.0 for good results.
- `validation_num_inference_steps` - Use somewhere around 30.
- `use_ema` - Setting this to `true` will help obtain smoother results but uses more memory.

- `optimizer` - Use `optimi-lion` for good results, or `adamw-bf16` if you have memory to spare.
- `mixed_precision` - Must be set to `bf16` for Qwen Image.
- `gradient_checkpointing` - **Required** to be enabled (`true`) for reasonable memory usage.
- `base_model_precision` - **Strongly recommended** to set to `int8-quanto` or `nf4-bnb` for 24GB cards.
- `quantize_via` - Set to `cpu` to avoid OOM during quantization on smaller GPUs.
- `quantize_activations` - Keep this `false` to maintain training quality.

Memory optimization settings for 24GB GPUs:
- `lora_rank` - Use 8 or lower.
- `lora_alpha` - Match this to your lora_rank value.
- `flow_schedule_shift` - Set to 1.73 (or experiment between 1.0-3.0).

Your config.json will look something like this for a minimal setup:

```json
{
    "model_type": "lora",
    "model_family": "qwen_image",
    "model_flavour": "v1.0",
    "lora_type": "standard",
    "lora_rank": 8,
    "lora_alpha": 8,
    "output_dir": "output/models-qwen_image",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "validation_resolution": "1024x1024",
    "validation_guidance": 4.0,
    "validation_num_inference_steps": 30,
    "validation_seed": 42,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_steps": 100,
    "vae_batch_size": 1,
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 1000,
    "max_grad_norm": 0.01,
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "learning_rate": "1e-4",
    "gradient_checkpointing": "true",
    "base_model_precision": "int2-quanto",
    "quantize_via": "cpu",
    "quantize_activations": false,
    "flow_schedule_shift": 1.73,
    "disable_benchmark": false,
    "data_backend_config": "config/qwen_image/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpointing_steps": 500,
    "caption_dropout_probability": 0.0,
    "aspect_bucket_rounding": 2
}
```

> ℹ️ Multi-GPU users can reference [this document](/documentation/OPTIONS.md#environment-configuration-variables) for information on configuring the number of GPUs to use.

> ⚠️ **Critical for 24GB GPUs**: The text encoder alone uses ~16GB VRAM. With `int2-quanto` or `nf4-bnb` quantization, this can be reduced significantly.

For a quick sanity check with a known working configuration:

**Option 1 (Recommended - pip install):**
```bash
pip install simpletuner
simpletuner train example=qwen_image.peft-lora
```

**Option 2 (Git clone method):**
```bash
simpletuner train env=examples/qwen_image.peft-lora
```

**Option 3 (Legacy method - still works):**
```bash
ENV=examples/qwen_image.peft-lora ./train.sh
```

#### Validation prompts

Inside `config/config.json` is the "primary validation prompt", which is typically the main instance_prompt you are training on for your single subject or style. Additionally, a JSON file may be created that contains extra prompts to run through during validations.

The example config file `config/user_prompt_library.json.example` contains the following format:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

The nicknames are the filename for the validation, so keep them short and compatible with your filesystem.

To point the trainer to this prompt library, add it to your config.json:
```json
  "validation_prompt_library": "config/user_prompt_library.json",
```

A set of diverse prompts will help determine whether the model is learning properly:

```json
{
    "anime_style": "a breathtaking anime-style portrait with vibrant colors and expressive features",
    "chef_cooking": "a high-quality, detailed photograph of a sous-chef immersed in culinary creation",
    "portrait": "a lifelike and intimate portrait showcasing unique personality and charm",
    "cinematic": "a cinematic, visually stunning photo with dramatic and captivating presence",
    "elegant": "an elegant and timeless portrait exuding grace and sophistication",
    "adventurous": "a dynamic and adventurous photo captured in an exciting moment",
    "mysterious": "a mysterious and enigmatic portrait shrouded in shadows and intrigue",
    "vintage": "a vintage-style portrait evoking the charm and nostalgia of a bygone era",
    "artistic": "an artistic and abstract representation blending creativity with visual storytelling",
    "futuristic": "a futuristic and cutting-edge portrayal set against advanced technology"
}
```

#### CLIP score tracking

If you wish to enable evaluations to score the model's performance, see [this document](/documentation/evaluation/CLIP_SCORES.md) for information on configuring and interpreting CLIP scores.

#### Stable evaluation loss

If you wish to use stable MSE loss to score the model's performance, see [this document](/documentation/evaluation/EVAL_LOSS.md) for information on configuring and interpreting evaluation loss.

#### Flow schedule shifting

Qwen Image, as a flow-matching model, supports timestep schedule shifting to control which parts of the generation process are trained.

The `flow_schedule_shift` parameter controls this:
- Lower values (0.1-1.0): Focus on fine details
- Medium values (1.0-3.0): Balanced training (recommended)
- Higher values (3.0-6.0): Focus on large compositional features

##### Auto-shift
You can enable resolution-dependent timestep shift with `--flow_schedule_auto_shift`, which uses higher shift values for larger images and lower shift values for smaller images. This can provide stable but potentially mediocre training results.

##### Manual specification
A `--flow_schedule_shift` value of 1.73 is recommended as a starting point for Qwen Image, though you may need to experiment based on your dataset and goals.

#### Dataset considerations

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively.

> ℹ️ With few enough images, you might see a message **no images detected in dataset** - increasing the `repeats` value will overcome this limitation.

> ⚠️ **Important**: Due to current limitations, keep `train_batch_size` at 1 and use `gradient_accumulation_steps` instead to simulate larger batch sizes.

Create a `--data_backend_config` (`config/multidatabackend.json`) document containing this:

```json
[
  {
    "id": "pseudo-camera-10k-qwen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/qwen_image",
    "disabled": false,
    "write_batch_size": 16
  }
]
```

> ℹ️ Use `caption_strategy=textfile` if you have `.txt` files containing captions.
> ℹ️ Note the reduced `write_batch_size` for text embeds to avoid OOM issues.

Then, create a `datasets` directory:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

This will download about 10k photograph samples to your `datasets/pseudo-camera-10k` directory, which will be automatically created for you.

Your Dreambooth images should go into the `datasets/dreambooth-subject` directory.

#### Login to WandB and Huggingface Hub

You'll want to login to WandB and HF Hub before beginning training, especially if you're using `--push_to_hub` and `--report_to=wandb`.

If you're going to be pushing items to a Git LFS repository manually, you should also run `git config --global credential.helper store`

Run the following commands:

```bash
wandb login
```

and

```bash
huggingface-cli login
```

Follow the instructions to log in to both services.

### Executing the training run

From the SimpleTuner directory, one simply has to run:

```bash
./train.sh
```

This will begin the text embed and VAE output caching to disk.

For more information, see the [dataloader](/documentation/DATALOADER.md) and [tutorial](/documentation/TUTORIAL.md) documents.

### Memory optimization tips

#### Lowest VRAM config (24GB minimum)

The lowest VRAM Qwen Image configuration requires approximately 24GB:

- OS: Ubuntu Linux 24
- GPU: A single NVIDIA CUDA device (24GB minimum)
- System memory: 64GB+ recommended
- Base model precision:
  - For NVIDIA systems: `int2-quanto` or `nf4-bnb` (required for 24GB cards)
  - `int4-quanto` can work but may have lower quality
- Optimizer: `optimi-lion` or `bnb-lion8bit-paged` for memory efficiency
- Resolution: Start with 512px or 768px, work up to 1024px if memory allows
- Batch size: 1 (mandatory due to current limitations)
- Gradient accumulation steps: 2-8 to simulate larger batches
- Enable `--gradient_checkpointing` (required)
- Use `--quantize_via=cpu` to avoid OOM during startup
- Use a small LoRA rank (1-8)
- Setting the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps minimize VRAM usage

**NOTE**: Pre-caching of VAE embeds and text encoder outputs will use significant memory. Enable `offload_during_startup=true` if you encounter OOM issues.

### Running inference on the LoRA afterward

Since Qwen Image is a newer model, here's a functioning example for inference:

```python
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

model_id = 'Qwen/Qwen-Image'
adapter_id = 'your-username/your-lora-name'

# Load the pipeline
pipeline = QwenImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
pipeline.load_lora_weights(adapter_id)

# Optional: quantize the model to save VRAM
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

# Move to device
pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

# Generate an image
prompt = "Your test prompt here"
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=4.0,
    generator=torch.Generator(device='cuda').manual_seed(42),
    width=1024,
    height=1024,
).images[0]

image.save("output.png", format="PNG")
```

### Notes & troubleshooting tips

#### Batch size limitations

Currently, Qwen Image has issues with batch sizes > 1 due to sequence length handling in the text encoder. Always use:
- `train_batch_size: 1`
- `gradient_accumulation_steps: 2-8` to simulate larger batches

#### Quantization

- `int2-quanto` provides the most aggressive memory savings but may impact quality
- `nf4-bnb` offers a good balance between memory and quality
- `int4-quanto` is a middle ground option
- Avoid `int8` unless you have 40GB+ VRAM

#### Learning rates

For LoRA training:
- Small LoRAs (rank 1-8): Use learning rates around 1e-4
- Larger LoRAs (rank 16-32): Use learning rates around 5e-5
- With Prodigy optimizer: Start with 1.0 and let it adapt

#### Image artifacts

If you encounter artifacts:
- Lower your learning rate
- Increase gradient accumulation steps
- Ensure your images are high quality and properly preprocessed
- Consider using lower resolutions initially

#### Multiple-resolution training

Start training at lower resolutions (512px or 768px) to speed up initial learning, then fine-tune at 1024px. Enable `--flow_schedule_auto_shift` when training at different resolutions.

### Platform limitations

**Not supported on:**
- AMD ROCm (lacks efficient flash attention implementation)
- Apple Silicon/MacOS (memory and attention limitations)
- Consumer GPUs with less than 24GB VRAM

### Current known issues

1. Batch size > 1 doesn't work correctly (use gradient accumulation)
2. TREAD is not yet supported
3. High memory usage from text encoder (~16GB before quantization)
4. Sequence length handling issues ([upstream issue](https://github.com/huggingface/diffusers/issues/12075))

For additional help and troubleshooting, consult the [SimpleTuner documentation](/documentation) or join the community Discord.