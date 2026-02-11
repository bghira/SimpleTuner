## Auraflow Quickstart

In this example, we'll be training a Lycoris LoKr for Auraflow.

Full fine-tuning for this model will require a lot of VRAM due to the 6B parameters, and you'd need to use [DeepSpeed](../DEEPSPEED.md) for that to work.

### Hardware requirements

Auraflow v0.3 was released as a 6B parameter MMDiT that uses Pile T5 for its encoded text representation and the 4ch SDXL VAE for its latent image representation.

This model is somewhat slow for inference, but trains at a decent speed.

### Memory offloading (optional)

Auraflow benefits greatly from the new grouped offloading path. Add the following to your training flags if you are limited to a single 24G (or smaller) GPU:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams are automatically disabled on non-CUDA backends, so the command is safe to reuse on ROCm and MPS.
- Do not combine this with `--enable_model_cpu_offload`.
- Disk offloading trades throughput for lower host RAM pressure; keep it on a local SSD for best results.

### Prerequisites

Make sure that you have python installed; SimpleTuner does well with 3.10 through 3.13.

You can check this by running:

```bash
python --version
```

If you don't have python 3.13 installed on Ubuntu, you can try the following:

```bash
apt -y install python3.13 python3.13-venv
```

#### Container image dependencies

For Vast, RunPod, and TensorDock (among others), the following will work on a CUDA 12.2-12.8 image to enable compiling of CUDA extensions:

```bash
apt -y install nvidia-cuda-toolkit
```

### Installation

Install SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

For manual installation or development setup, see the [installation documentation](../INSTALL.md).

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
- `model_family` - Set this to `auraflow`.
- `model_flavour` - Set this to `pony`, or leave it unset to use the default model.
- `output_dir` - Set this to the directory where you want to store your checkpoints and validation images. It's recommended to use a full path here.
- `train_batch_size` - 1 to 4 should work for 24G card.
- `validation_resolution` - You should set this to `1024x1024` or one of Auraflow's other supported resolutions.
  - Other resolutions may be specified using commas to separate them: `1024x1024,1280x768,1536x1536`
  - Note that Auraflow's positional embeds are a bit strange and training with multi-scale images (multiple base resolutions) has an uncertain outcome.
- `validation_guidance` - Use whatever you are used to selecting at inference time for Auraflow; a lower value around 3.5-4.0 makes more realistic results
- `validation_num_inference_steps` - Use somewhere around 30-50
- `use_ema` - setting this to `true` will greatly help obtain a more smoothed result alongside your main trained checkpoint.

- `optimizer` - You can use any optimiser you are comfortable and familiar with, but we will use `optimi-lion` for this example.
  - The author of Pony Flow recommends using `adamw_bf16` for the fewest issues and most stably reliable training results
  - We're using Lion for this demonstration to help you see the model train more rapidly, but for long term runs, `adamw_bf16` will be a safe bet.
- `learning_rate` - For Lion optimiser with Lycoris LoKr, a value of `4e-5` is a good starting point.
  - If you went with `adamw_bf16`, you'll want the LR to be about 10x larger than this, or `2.5e-4`
  - Smaller Lycoris/LoRA ranks will require **higher learning rates** and larger Lycoris/LoRA require **smaller learning rates**
- `mixed_precision` - It's recommended to set this to `bf16` for the most efficient training configuration, or `no` for better results (but will consume more memory and be slower).
- `gradient_checkpointing` - Disabling this will go the fastest, but limits your batch sizes. It is required to enable this to get the lowest VRAM usage.


The impact of these options are currently unknown.

Your config.json will look something like mine by the end:

<details>
<summary>View example config</summary>

```json
{
    "validation_torch_compile": "false",
    "validation_step_interval": 200,
    "validation_seed": 42,
    "validation_resolution": "1024x1024",
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_num_inference_steps": "20",
    "validation_guidance": 2.0,
    "validation_guidance_rescale": "0.0",
    "vae_cache_ondemand": true,
    "vae_batch_size": 1,
    "train_batch_size": 1,
    "tracker_run_name": "eval_loss_test1",
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "output_dir": "output/models-auraflow",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "auraflow",
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 10000,
    "max_grad_norm": 0.01,
    "lycoris_config": "config/lycoris_config.json",
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant",
    "lora_type": "lycoris",
    "learning_rate": "4e-5",
    "gradient_checkpointing": "true",
    "grad_clip_method": "value",
    "eval_steps_interval": 100,
    "disable_benchmark": false,
    "data_backend_config": "config/auraflow/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "int8-quanto",
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ Multi-GPU users can reference [this document](../OPTIONS.md#environment-configuration-variables) for information on configuring the number of GPUs to use.

And a simple `config/lycoris_config.json` file:

<details>
<summary>View example config</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 16,
    "apply_preset": {
        "target_module": [
            "Attention",
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 8
            },
        }
    }
}
```
</details>

### Advanced Experimental Features

<details>
<summary>Show advanced experimental details</summary>


SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> ⚠️ These features increase the computational overhead of training.

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

To point the trainer to this prompt library, add it to TRAINER_EXTRA_ARGS by adding a new line at the end of `config.json`:
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

A set of diverse prompt will help determine whether the model is collapsing as it trains. In this example, the word `<token>` should be replaced with your subject name (instance_prompt).

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing her essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing her unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing her dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```

> ℹ️ Auraflow defaults to 128 tokens and then truncates.

#### CLIP score tracking

If you wish to enable evaluations to score the model's performance, see [this document](../evaluation/CLIP_SCORES.md) for information on configuring and interpreting CLIP scores.

</details>

# Stable evaluation loss

If you wish to use stable MSE loss to score the model's performance, see [this document](../evaluation/EVAL_LOSS.md) for information on configuring and interpreting evaluation loss.

#### Validation previews

SimpleTuner supports streaming intermediate validation previews during generation using Tiny AutoEncoder models. This allows you to see validation images being generated step-by-step in real-time via webhook callbacks.

To enable:
<details>
<summary>View example config</summary>

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```
</details>

**Requirements:**
- Webhook configuration
- Validation enabled

Set `validation_preview_steps` to a higher value (e.g., 3 or 5) to reduce Tiny AutoEncoder overhead. With `validation_num_inference_steps=20` and `validation_preview_steps=5`, you'll receive preview images at steps 5, 10, 15, and 20.

#### Flow schedule shifting

Flow-matching models such as OmniGen, Sana, Flux, and SD3 have a property called "shift" that allows us to shift the trained portion of the timestep schedule using a simple decimal value.

##### Auto-shift
A commonly-recommended approach is to follow several recent works and enable resolution-dependent timestep shift, `--flow_schedule_auto_shift` which uses higher shift values for larger images, and lower shift values for smaller images. This results in stable but potentially mediocre training results.

##### Manual specification
_Thanks to General Awareness from Discord for the following examples_

When using a `--flow_schedule_shift` value of 0.1 (a very low value), only the finer details of the image are affected:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

When using a `--flow_schedule_shift` value of 4.0 (a very high value), the large compositional features and potentially colour space of the model becomes impacted:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### Dataset considerations

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively. Note that the bare minimum dataset size is `train_batch_size * gradient_accumulation_steps` as well as more than `vae_batch_size`. The dataset will not be useable if it is too small.

> ℹ️ With few enough images, you might see a message **no images detected in dataset** - increasing the `repeats` value will overcome this limitation.

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) as the dataset.

Create a `--data_backend_config` (`config/multidatabackend.json`) document containing this:

<details>
<summary>View example config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-auraflow",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/auraflow/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/auraflow/dreambooth-subject",
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
    "cache_dir": "cache/text/auraflow",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> ℹ️ Use `caption_strategy=textfile` if you have `.txt` files containing captions.
> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

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
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130

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

For more information, see the [dataloader](../DATALOADER.md) and [tutorial](../TUTORIAL.md) documents.

### Running inference on the LoKr afterward

Since it's a new model, the example will need some adjustment to work. Here's a functioning example:

<details>
<summary>Show Python inference example</summary>

```py
import torch
from helpers.models.auraflow.pipeline import AuraFlowPipeline
from helpers.models.auraflow.transformer import AuraFlowTransformer2DModel
from lycoris import create_lycoris_from_weights
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

model_id = 'terminusresearch/auraflow-v0.3'
adapter_repo_id = 'bghira/auraflow-photo-1mp-Prodigy'
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

adapter_file_path = download_adapter(repo_id=adapter_repo_id)
transformer = AuraFlowTransformer2DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, subfolder="transformer")
pipeline = AuraFlowPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    transformer=transformer,
)
lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

prompt = "Place your test prompt here."
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

## Optional: quantise the model to save on vram.
## Note: The model was quantised during training, and so it is recommended to do the same during inference time.
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # the pipeline is already in its target precision level
t5_embeds, negative_t5_embeds, attention_mask, negative_attention_mask = pipeline.encode_prompt(
    prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt, num_images_per_prompt=1
)
# We'll nuke the text encoders to save memory.
pipeline.text_encoder.to("meta")
pipeline.text_encoder_2.to("meta")
pipeline.text_encoder_3.to("meta")
model_output = pipeline(
    prompt_embeds=t5_embeds,
    prompt_attention_mask=attention_mask,
    negative_prompt_embeds=negative_t5_embeds,
    negative_prompt_attention_mask=negative_attention_mask,
    num_inference_steps=30,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=3.2,
).images[0]

model_output.save("output.png", format="PNG")

```
</details>

## Notes & troubleshooting tips

### Lowest VRAM config

The lowest VRAM Auraflow configuration is about 20-22G:

- OS: Ubuntu Linux 24
- GPU: A single NVIDIA CUDA device (10G, 12G)
- System memory: 50G of system memory approximately (could be more, could be less)
- Base model precision:
  - For Apple and AMD systems, `int8-quanto` (or `fp8-torchao`, `int8-torchao` all follow similar memory use profiles)
    - `int4-quanto` works as well, but you might have lower accuracy / worse results
  - For NVIDIA systems, `nf4-bnb` is reported to work well, but will be slower than `int8-quanto`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 1024px
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: disabled / unconfigured
- PyTorch: 2.7+
- Using `--quantize_via=cpu` to avoid outOfMemory error during startup on <=16G cards.
- Enable `--gradient_checkpointing`
- Use a tiny LoRA or Lycoris configuration (eg. LoRA rank 1 or Lokr factor 25)

**NOTE**: Pre-caching of VAE embeds and text encoder outputs may use more memory and still OOM. VAE tiling and slicing are enabled by default. If you see OOM, you might need to enable `offload_during_startup=true`; otherwise, you may just be out of luck.

Speed was approximately 3 iterations per second on an NVIDIA 4090 using Pytorch 2.7 and CUDA 12.8

### Masked loss

If you are training a subject or style and would like to mask one or the other, see the [masked loss training](../DREAMBOOTH.md#masked-loss) section of the Dreambooth guide.

### Quantisation

Auraflow tends to respond well down to `int4` precision level, though `int8` will be a sweet spot for quality and stability if you can't afford `bf16`.

### Learning rates

#### LoRA (--lora_type=standard)

*Not supported.*

#### LoKr (--lora_type=lycoris)
- Mild learning rates are better for LoKr (`1e-4` with AdamW, `2e-5` with Lion)
- Other algo need more exploration.
- Setting `is_regularisation_data` has unknown impact/effect with Auraflow (not tested, but, should be fine?)

### Image artifacts

Auraflow has an unknown response to image artifacts, though it uses the Flux VAE, and has similar fine-details limitations.

If any image quality issues arise, please open an issue on Github.

### Aspect bucketing

Some limitations with the model's patch embed implementation mean that there are certain resolutions that will cause an error.

Experimentation will be helpful, as well as thorough bug reports.

### Full-rank tuning

DeepSpeed will use a LOT of system memory with Auraflow, and full tuning might not perform the way you hope in terms of learning concepts or avoiding model collapse.

Lycoris LoKr is recommended in lieu of full-rank tuning, as it is more stable and has a lower memory footprint.
