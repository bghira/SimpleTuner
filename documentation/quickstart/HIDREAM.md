## HiDream Quickstart

In this example, we'll be training a Lycoris LoKr for HiDream, hoping to god we've got enough memory to pull it off.

A 24G GPU is likely the minimum you'll get away with without extensive block offloading and fused backward pass. A Lycoris LoKr will function just as well!

### Hardware requirements

HiDream is a 17B total parameters with ~8B active at any given time using a learnt MoE gate to distribute its work. It uses **four** text encoders, and the Flux VAE.

Overall the model suffers from architectural complexity, and seems to be a derivative of Flux Dev, either by direct distillation or by continued fine-tuning, evident from some validation samples that look like they share the same weights.

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

pip install -U poetry pip

# Necessary on some systems to prevent it from deciding it knows better than us.
poetry config virtualenvs.create false
```

**Note:** We're currently installing the `release` branch here; the `main` branch may contain experimental features that might have better results or lower memory use.

Depending on your system, you will run one of 3 commands:

```bash
# Linux
poetry install
```

### Setting up the environment

To run SimpleTuner, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

#### Configuration file

An experimental script, `configure.py`, may allow you to entirely skip this section through an interactive step-by-step configuration. It contains some safety features that help avoid common pitfalls.

**Note:** This doesn't configure your dataloader. You will still have to do that manually, later.

To run it:

```bash
python configure.py
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
- `model_family` - Set this to `hidream`.
- `model_flavour` - Set this to `full`, because `dev` is distilled in a way that it is not easily directly trained unless you want to go the distance and break its distillation.
  - In fact, the `full` model is also difficult to train, but is the only one that has not been distilled.
- `output_dir` - Set this to the directory where you want to store your checkpoints and validation images. It's recommended to use a full path here.
- `train_batch_size` - 1, maybe?.
- `validation_resolution` - You should set this to `1024x1024` or one of HiDream's other supported resolutions.
  - Other resolutions may be specified using commas to separate them: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - Use whatever you are used to selecting at inference time for HiDream; a lower value around 2.5-3.0 makes more realistic results
- `validation_num_inference_steps` - Use somewhere around 30
- `use_ema` - setting this to `true` will greatly help obtain a more smoothed result alongside your main trained checkpoint.

- `optimizer` - You can use any optimiser you are comfortable and familiar with, but we will use `optimi-lion` for this example.
- `mixed_precision` - It's recommended to set this to `bf16` for the most efficient training configuration, or `no` (but will consume more memory and be slower).
- `gradient_checkpointing` - Disabling this will go the fastest, but limits your batch sizes. It is required to enable this to get the lowest VRAM usage.

Some advanced HiDream options can be set to include MoE auxiliary loss during training. By adding the MoE loss, the value will naturally be much higher than usual.

- `hidream_use_load_balancing_loss` - Set this to `true` to enable load balancing loss.
- `hidream_load_balancing_loss_weight` - This is the magnitude of the auxiliary loss. A value of `0.01` is the default, but you can set it to `0.1` or `0.2` for a more aggressive training run.

The impact of these options are currently unknown.

Your config.json will look something like mine by the end:

```json
{
    "validation_torch_compile": "false",
    "validation_steps": 200,
    "validation_seed": 42,
    "validation_resolution": "1024x1024",
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_num_inference_steps": "20",
    "validation_guidance": 3.0,
    "validation_guidance_rescale": "0.0",
    "vae_batch_size": 1,
    "train_batch_size": 1,
    "tracker_run_name": "eval_loss_test1",
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "output_dir": "output/models-hidream",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "hidream",
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 10000,
    "max_grad_norm": 0.01,
    "lycoris_config": "config/lycoris_config.json",
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "lora_type": "lycoris",
    "learning_rate": "4e-5",
    "gradient_checkpointing": "true",
    "grad_clip_method": "value",
    "eval_steps_interval": 100,
    "disable_benchmark": false,
    "data_backend_config": "config/hidream/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpointing_steps": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "int8-quanto",
    "text_encoder_3_precision": "int8-quanto",
    "text_encoder_4_precision": "int8-quanto",
    "aspect_bucket_rounding": 2
}
```

> ℹ️ Multi-GPU users can reference [this document](/OPTIONS.md#environment-configuration-variables) for information on configuring the number of GPUs to use.

> ℹ️ This configuration sets the T5 (#3) and Llama (#4) text encoder precision levels to int8 to save memory for 24G cards. You can remove these options or set them to `no_change` if you have more memory available.

And a simple `config/lycoris_config.json` file - note that the `FeedForward` may be removed for additional training stability.

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 16384,
    "linear_alpha": 1,
    "full_matrix": true,
    "use_scalar": true,
    "factor": 16,
    "apply_preset": {
        "target_module": [
            "Attention",
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 16
            },
        }
    }
}
```

Setting either `"use_scalar": true` in `config/lycoris_config.json` or setting `"init_lokr_norm": 1e-4` in `config/config.json` will speed up training considerably. Enabling both seems to slow down training slightly. Note that setting `init_lokr_norm` will slightly change the validation images at step 0.

Adding the `FeedForward` module to `config/lycoris_config.json` will train a much larger number of parameters, including all the experts. Training the experts seems to be rather difficult though.

An easier option is to only train the feed forward parameters outside the experts using the following `config/lycoris_config.json` file.

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 16384,
    "linear_alpha": 1,
    "full_matrix": true,
    "use_scalar": true,
    "factor": 16,
    "apply_preset": {
        "name_algo_map": {
            "double_stream_blocks.*.block.attn*": {
                "factor": 16
            },
            "double_stream_blocks.*.block.ff_t*": {
                "factor": 16
            },
            "double_stream_blocks.*.block.ff_i.shared_experts*": {
                "factor": 16
            },
            "single_stream_blocks.*.block.attn*": {
                "factor": 16
            },
            "single_stream_blocks.*.block.ff_i.shared_experts*": {
                "factor": 16
            }
        },
        "use_fnmatch": true
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

> ℹ️ HiDream defaults to 128 tokens and then truncates.

#### CLIP score tracking

If you wish to enable evaluations to score the model's performance, see [this document](/documentation/evaluation/CLIP_SCORES.md) for information on configuring and interpreting CLIP scores.

# Stable evaluation loss

If you wish to use stable MSE loss to score the model's performance, see [this document](/documentation/evaluation/EVAL_LOSS.md) for information on configuring and interpreting evaluation loss.

#### Flow schedule shifting

Flow-matching models such as OmniGen, Sana, Flux, and SD3 have a property called "shift" that allows us to shift the trained portion of the timestep schedule using a simple decimal value.

The `full` model is trained with a value of `3.0` and `dev` used `6.0`.

In practice, using such a high shift value tends to destroy either model. A value of `1.0` is a good starting point, but may move the model too little, and 3.0 may be too high.

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

Depending on the dataset you have, you will need to set up your dataset directory and dataloader configuration file differently. In this example, we will be using [pseudo-camera-10k](https://huggingface.co/datasets/ptx0/pseudo-camera-10k) as the dataset.

Create a `--data_backend_config` (`config/multidatabackend.json`) document containing this:

```json
[
  {
    "id": "pseudo-camera-10k-hidream",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/hidream/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/hidream/dreambooth-subject",
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
    "cache_dir": "cache/text/hidream",
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

From the SimpleTuner directory, one simply has to run:

```bash
./train.sh
```

This will begin the text embed and VAE output caching to disk.

For more information, see the [dataloader](/documentation/DATALOADER.md) and [tutorial](/TUTORIAL.md) documents.

### Running inference on the LoKr afterward

Since it's a new model, the example will need some adjustment to work. Here's a functioning example:

```py
import torch
from helpers.models.hidream.pipeline import HiDreamImagePipeline
from helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
from lycoris import create_lycoris_from_weights
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

llama_repo = "unsloth/Meta-Llama-3.1-8B-Instruct"
model_id = 'HiDream-ai/HiDream-I1-Dev'
adapter_repo_id = 'bghira/hidream5m-photo-1mp-Prodigy'
adapter_filename = 'pytorch_lora_weights.safetensors'

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
    llama_repo,
)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    llama_repo,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)

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
transformer = HiDreamImageTransformer2DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, subfolder="transformer")
pipeline = HiDreamImagePipeline.from_pretrained(
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
t5_embeds, llama_embeds, negative_t5_embeds, negative_llama_embeds, pooled_embeds, negative_pooled_embeds = pipeline.encode_prompt(
    prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt, num_images_per_prompt=1
)
# We'll nuke the text encoders to save memory.
pipeline.text_encoder.to("meta")
pipeline.text_encoder_2.to("meta")
pipeline.text_encoder_3.to("meta")
pipeline.text_encoder_4.to("meta")
model_output = pipeline(
    t5_prompt_embeds=t5_embeds,
    llama_prompt_embeds=llama_embeds,
    pooled_prompt_embeds=pooled_embeds,
    negative_t5_prompt_embeds=negative_t5_embeds,
    negative_llama_prompt_embeds=negative_llama_embeds,
    negative_pooled_prompt_embeds=negative_pooled_embeds,
    num_inference_steps=30,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=3.2,
).images[0]

model_output.save("output.png", format="PNG")

```

## Notes & troubleshooting tips

### Lowest VRAM config

The lowest VRAM HiDream configuration is about 20-22G:

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
- Setting the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps minimize VRAM usage when training multiple aspect ratios.

**NOTE**: Pre-caching of VAE embeds and text encoder outputs may use more memory and still OOM. VAE tiling and slicing are enabled by default. If you see OOM, you might just be out of luck.

Speed was approximately 3 iterations per second on an NVIDIA 4090 using Pytorch 2.7 and CUDA 12.8

### Masked loss

Masked loss training is currently not supported with HiDream.

If you are training a subject or style and would like to mask one or the other, see the [masked loss training](/documentation/DREAMBOOTH.md#masked-loss) section of the Dreambooth guide.

### Quantisation

HiDream tends to respond well down to `int4` precision level, though `int8` will be a sweet spot for quality and stability if you can't afford `bf16`.

### Learning rates

#### LoRA (--lora_type=standard)

*Not supported.*

#### LoKr (--lora_type=lycoris)
- Mild learning rates are better for LoKr (`1e-4` with AdamW, `2e-5` with Lion)
- Other algo need more exploration.
- Setting `is_regularisation_data` has unknown impact/effect with HiDream (not tested, but, should be fine?)

### Image artifacts

HiDream has an unknown response to image artifacts, though it uses the Flux VAE, and has similar fine-details limitations.

If any image quality issues arise, please open an issue on Github.

### Aspect bucketing

Some limitations with the model's patch embed implementation mean that there are certain resolutions that will cause an error.

Experimentation will be helpful, as well as thorough bug reports.

### Multiple-resolution training

The model can be initially trained at a lower resolution such as 512px to speed up training. It is a good idea to enable `--flow_schedule_auto_shift` when training resolutions different from 1024px. Lower resolutions use less VRAM, allowing higher batch sizes to be used.

### Full-rank tuning

DeepSpeed will use a LOT of system memory with HiDream, and full tuning might not perform the way you hope in terms of learning concepts or avoiding model collapse.

Lycoris LoKr is recommended in lieu of full-rank tuning, as it is more stable and has a lower memory footprint.
