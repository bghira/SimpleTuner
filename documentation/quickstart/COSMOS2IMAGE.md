## Cosmos2 Predict (Image) Quickstart

In this example, we'll be training a Lycoris LoKr for Cosmos2 Predict (Image), a flow-matching model from NVIDIA.

### Hardware requirements

Cosmos2 Predict (Image) is a vision transformer-based model that uses flow matching.

**Note**: Due to its architecture, it really should not be quantized during training, which means you'll need sufficient VRAM to accommodate the full bf16 precision.

A 24GB GPU is recommended as the minimum for comfortable training without extensive optimizations.

### Memory offloading (optional)

To squeeze Cosmos2 into smaller GPUs, enable grouped offloading:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams are only honoured on CUDA; other devices fall back automatically.
- Do not combine this with `--enable_model_cpu_offload`.
- Disk staging is optional and helps when system RAM is the bottleneck.

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

For Vast, RunPod, and TensorDock (among others), the following will work on a CUDA 12.2-12.8 image to enable compiling of CUDA extensions:

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
- `lora_type` - Set this to `lycoris`.
- `model_family` - Set this to `cosmos2image`.
- `base_model_precision` - **Important**: Set this to `no_change` - Cosmos2 should not be quantized.
- `output_dir` - Set this to the directory where you want to store your checkpoints and validation images. It's recommended to use a full path here.
- `train_batch_size` - Start with 1 and increase if you have sufficient VRAM.
- `validation_resolution` - The default is `1024x1024` for Cosmos2.
  - Other resolutions may be specified using commas to separate them: `1024x1024,768x768`
- `validation_guidance` - Use a value around 4.0 for Cosmos2.
- `validation_num_inference_steps` - Use around 20 steps.
- `use_ema` - Setting this to `true` will greatly help obtain a more smoothed result alongside your main trained checkpoint.
- `optimizer` - The example uses `adamw_bf16`.
- `mixed_precision` - It's recommended to set this to `bf16` for the most efficient training configuration.
- `gradient_checkpointing` - Enable this to reduce VRAM usage at the cost of training speed.

Your config.json will look something like this:

```json
{
    "base_model_precision": "no_change",
    "checkpoint_step_interval": 500,
    "data_backend_config": "config/cosmos2image/multidatabackend.json",
    "disable_bucket_pruning": true,
    "flow_schedule_shift": 0.0,
    "flow_schedule_auto_shift": true,
    "gradient_checkpointing": true,
    "hub_model_id": "cosmos2image-lora",
    "learning_rate": 6e-5,
    "lora_type": "lycoris",
    "lycoris_config": "config/cosmos2image/lycoris_config.json",
    "lr_scheduler": "constant",
    "lr_warmup_steps": 100,
    "max_train_steps": 10000,
    "model_family": "cosmos2image",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/cosmos2image",
    "push_checkpoints_to_hub": false,
    "push_to_hub": false,
    "quantize_via": "cpu",
    "report_to": "tensorboard",
    "seed": 42,
    "tracker_project_name": "cosmos2image-training",
    "tracker_run_name": "cosmos2image-lora",
    "train_batch_size": 1,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_disable_unconditional": true,
    "validation_guidance": 4.0,
    "validation_guidance_rescale": 0.0,
    "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
    "validation_num_inference_steps": 20,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_prompt_library": false,
    "validation_resolution": "512x512",
    "validation_seed": 42,
    "validation_step_interval": 500
}
```

> ℹ️ Multi-GPU users can reference [this document](/documentation/OPTIONS.md#environment-configuration-variables) for information on configuring the number of GPUs to use.

And a `config/cosmos2image/lycoris_config.json` file:

```json
{
    "bypass_mode": true,
    "algo": "lokr",
    "multiplier": 1.0,
    "full_matrix": true,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 4,
    "apply_preset": {
        "target_module": [
            "Attention"
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 4
            }
        }
    }
}
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

To point the trainer to this prompt library, add it to your config by setting:
```json
"validation_prompt_library": "config/user_prompt_library.json"
```

A set of diverse prompts will help determine whether the model is collapsing as it trains. In this example, the word `<token>` should be replaced with your subject name (instance_prompt).

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```

#### CLIP score tracking

If you wish to enable evaluations to score the model's performance, see [this document](/documentation/evaluation/CLIP_SCORES.md) for information on configuring and interpreting CLIP scores.

#### Validation previews

SimpleTuner supports streaming intermediate validation previews during generation using Tiny AutoEncoder models. This allows you to see validation images being generated step-by-step in real-time via webhook callbacks.

To enable:
```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

**Requirements:**
- Webhook configuration
- Validation enabled

Set `validation_preview_steps` to a higher value (e.g., 3 or 5) to reduce Tiny AutoEncoder overhead. With `validation_num_inference_steps=20` and `validation_preview_steps=5`, you'll receive preview images at steps 5, 10, 15, and 20.

#### Flow schedule shifting

As a flow-matching model, Cosmos2 has a property called "shift" that allows us to shift the trained portion of the timestep schedule using a simple decimal value.

The configuration includes `flow_schedule_auto_shift` enabled by default, which uses resolution-dependent timestep shift - higher shift values for larger images, and lower shift values for smaller images.

#### Dataset considerations

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively. Note that the bare minimum dataset size is `train_batch_size * gradient_accumulation_steps` as well as more than `vae_batch_size`. The dataset will not be useable if it is too small.

> ℹ️ With few enough images, you might see a message **no images detected in dataset** - increasing the `repeats` value will overcome this limitation.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) as the dataset.

Create a `--data_backend_config` (`config/cosmos2image/multidatabackend.json`) document containing this:

```json
[
  {
    "id": "pseudo-camera-10k-cosmos2",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/cosmos2/pseudo-camera-10k",
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
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/cosmos2/dreambooth-subject",
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
    "cache_dir": "cache/text/cosmos2",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> ℹ️ Use `caption_strategy=textfile` if you have `.txt` files containing captions.

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

For more information, see the [dataloader](/documentation/DATALOADER.md) and [tutorial](/documentation/TUTORIAL.md) documents.

### Running inference on the LoKr afterward

Since Cosmos2 is a newer model with limited documentation, inference examples may need adjustment. A basic example structure would be:

```py
import torch
from lycoris import create_lycoris_from_weights

# Model and adapter paths
model_id = 'nvidia/Cosmos-1.0-Predict-Image-Text2World-12B'
adapter_repo_id = 'your-username/your-cosmos2-lora'
adapter_filename = 'pytorch_lora_weights.safetensors'

def download_adapter(repo_id: str):
    import os
    from huggingface_hub import hf_hub_download
    adapter_filename = "pytorch_lora_weights.safetensors"
    cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
    cleaned_adapter_path = repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
    path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
    os.makedirs(path_to_adapter, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
    )

    return path_to_adapter_file

# Load the model and adapter

import torch
from diffusers import Cosmos2TextToImagePipeline

# Available checkpoints: nvidia/Cosmos-Predict2-2B-Text2Image, nvidia/Cosmos-Predict2-14B-Text2Image
model_id = "nvidia/Cosmos-Predict2-2B-Text2Image"
adapter_repo_id = "youruser/your-repo-name"

adapter_file_path = download_adapter(repo_id=adapter_repo_id)
pipe = Cosmos2TextToImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

pipe.to("cuda")

prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

output = pipe(
    prompt=prompt, negative_prompt=negative_prompt, generator=torch.Generator().manual_seed(1)
).images[0]
output.save("output.png")

```

## Notes & troubleshooting tips

### Memory considerations

Since Cosmos2 cannot be quantized during training, memory usage will be higher than quantized models. Key settings for lower VRAM usage:

- Enable `gradient_checkpointing`
- Use batch size of 1
- Consider using `adamw_8bit` optimizer if memory is tight
- Setting the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps minimize VRAM usage when training multiple aspect ratios

### Training considerations

As Cosmos2 is a newer model, optimal training parameters are still being explored:

- The example uses a learning rate of `6e-5` with AdamW
- Flow schedule auto-shift is enabled to handle different resolutions
- CLIP evaluation is used to monitor training progress

### Aspect bucketing

The configuration has `disable_bucket_pruning` set to true, which may be adjusted based on your dataset characteristics.

### Multiple-resolution training

The model can be trained at 512px initially, with potential for training at higher resolutions later. The `flow_schedule_auto_shift` setting helps with multi-resolution training.

### Masked loss

If you are training a subject or style and would like to mask one or the other, see the [masked loss training](/documentation/DREAMBOOTH.md#masked-loss) section of the Dreambooth guide.

### Known limitations

- System prompt handling is not yet implemented
- The model's trainability characteristics are still being explored
- Quantization is not supported and should be avoided
