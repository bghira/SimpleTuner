# Mixture-of-Experts

SimpleTuner allows splitting the task of training in two, such that the self-attention and cross-attention stages of inference can effectively be split between two entirely different sets of weights.

In this example, we will use SegMind's collaborative effort with Hugging Face, [SSD-1B](https://huggingface.co/segmind/ssd-1b) to create two new models that train more reliably and have better resulting fine details than a single model.

Thanks to the small size of the SSD-1B model, training on even lighter-weight hardware is possible. Since we're starting our model from their pretrained weights, we have to abide by their Apache 2.0 license - but this is relatively straightforward. You can even use the resulting weights in a commercial setting!

When SDXL 0.9 and 1.0 were introduced, they both contained a full base model with a split-schedule refiner.

- The base model was trained on steps 999 to 0
  - The base model is more than 3B parameters, and functions entirely standalone.
- The refiner model was trained on steps 199 to 0
  - The refiner model is also more than 3B parameters, a seemingly unnecessary waste of resources. It does not function on its own without substantial deformations and a bias toward cartoonish outputs.

Let's see how we can improve this situation.


## The Base model, "Stage One"

The first portion of a mixture-of-experts is known as the base model. As mentioned in SDXL's case, it is trained on all 1000 timesteps - but it doesn't need to be. The following configuration will train the base model on just 650 steps out of the total 1000, saving time and training more reliably.

### Environment configuration

Setting the following values in your `config/config.env` will get us started:

```bash
# Ensure these aren't incorrectly set.
export USE_BITFIT=false
export USE_DORA=false
# lora could be used here instead, but the concept hasn't been explored.
export MODEL_TYPE="full"
export MODEL_FAMILY="sdxl"
export MODEL_NAME="segmind/SSD-1B"
# The original Segmind model used a learning rate of 1e-5, which is
# probably too high for whatever batch size most users can pull off.
export LEARNING_RATE=4e-7

# We really want this as high as you can tolerate.
# - If training is very slow, ensure your CHECKPOINT_STEPS and VALIDATION_STEPS
#   are set low enough that you'll get a checkpoint every couple hours.
# - The original Segmind models used a batch size of 32 with 4 accumulations.
export TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1

# If you are running on a beefy machine that doesn't fully utilise its VRAM during training, set this to "false" and your training will go faster.
export USE_GRADIENT_CHECKPOINTING=true

# Enable first stage model training
export TRAINER_EXTRA_ARGS="--refiner_training --refiner_training_strength=0.35 --refiner_training_invert_schedule"

# Optionally reparameterise it to v-prediction/zero-terminal SNR. 'sample' prediction_type can be used instead for x-prediction.
# This will start out looking pretty terrible until about 1500-2500 steps have passed, but it could be very worthwhile.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --prediction_type=v_prediction --rescale_betas_zero_snr --training_scheduler_timestep_spacing=trailing"
```

### Dataloader configuration

No special considerations for dataloader configuration are necessary. See the [dataloader config guide](/documentation/DATALOADER.md) for more information on this step.

### Validation

Currently, SimpleTuner does not engage the second stage model during stage one evaluations.

Future work will support this as an option, in case a stage two model already exists, or is being trained concurrently.

---

## The Refiner model, "Stage Two"

### Comparison to training the SDXL refiner

- Unlike the SDXL refiner, when using Segmind SSD-1B for both stages the text embeds **can** be shared between the two training jobs
  - The SDXL refiner uses a different text embed layout versus the SDXL base model.
- The VAE embeds **can** be shared between the training jobs, just like the SDXL refiner. Both models use the same input layout.
- No aesthetic score is used for the Segmind models, instead they use the same microconditioning inputs as SDXL, eg. crop coordinates
- Training goes much faster, as the model is much smaller, and text embeds can be reused from stage one training

### Environment Configuration

Update the following values in your `config/config.env` to swap training over to your stage two model. It might be convenient to keep a copy of the base model configuration.

```bash
# Update your OUTPUT_DIR value, so that we don't overwrite the stage one model checkpoints.
export OUTPUT_DIR="/some/new/path"

# We'll swap --refiner_training_invert_schedule for --validation_using_datasets
# - Train the end of the model instead of the beginning
# - Validate using images as input for partial denoising to evaluate fine detail improvements
export TRAINER_EXTRA_ARGS="--refiner_training --refiner_training_strength=0.35 --validation_using_datasets"

# Don't update these values if you've set them on the stage one. Be sure to use the same parameterisation for both models!
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --prediction_type=v_prediction --rescale_betas_zero_snr --training_scheduler_timestep_spacing=trailing"
```

### Dataset format

The images should be purely high-quality - remove any datasets you find questionable in terms of compression artifacts or other errors.

Other than that, the same exact dataloader configuration can be used between the two training jobs.

If you'd like a demonstration dataset, [pseudo-camera-10k](https://huggingface.co/datasets/ptx0/pseudo-camera-10k) is a solid choice with permissive licensing.

### Validation

Stage two refiner training will automatically select images from each of your training sets, and use those as inputs for partial denoising at validation time.

## CLIP score tracking

If you wish to enable evaluations to score the model's performance, see [this document](/documentation/evaluation/CLIP_SCORES.md) for information on configuring and interpreting CLIP scores.

# Stable evaluation loss

If you wish to use stable MSE loss to score the model's performance, see [this document](/documentation/evaluation/EVAL_LOSS.md) for information on configuring and interpreting evaluation loss.

## Putting it all together at inference time

If you'd like to plug both of the models together to experiment with in a simple script, this will get you started:

```py
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, UniPCMultistepScheduler
from torch import float16, cuda
from torch.backends import mps

# For a training_refiner_strength of .35, you'll set the base model strength to 0.65.
# Formula: 1 - training_refiner_strength
training_refiner_strength = 0.35
base_model_power = 1 - training_refiner_strength
# Reduce this for lower quality but speed-up.
num_inference_steps = 40
# Update these to your local or hugging face hub paths.
stage_1_model_id = 'ptx0/terminus-xl-velocity-v2'
stage_2_model_id = 'ptx0/terminus-xl-refiner'
torch_device = 'cuda' if cuda.is_available() else 'mps' if mps.is_available() else 'cpu'

pipe = StableDiffusionXLPipeline.from_pretrained(stage_1_model_id, add_watermarker=False, torch_dtype=float16).to(torch_device)
pipe.scheduler = UniPCMultistepScheduler.from_pretrained(stage_1_model_id, subfolder="scheduler")
img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(stage_2_model_id).to(device=torch_device, dtype=float16)
img2img_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(stage_1_model_id, subfolder="scheduler")

prompt = "An astronaut riding a green horse"

# Important: update this to True if you reparameterised the models.
use_zsnr = True

image = pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_end=base_model_power,
    guidance_scale=9.2,
    guidance_rescale=0.7 if use_zsnr else 0.0,
    output_type="latent",
).images
image = img2img_pipe(
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    denoising_start=base_model_power,
    guidance_scale=9.2,
    guidance_rescale=0.7 if use_zsnr else 0.0,
    image=image,
).images[0]
image.save('demo.png', format="PNG")
```

Some experimentations you can run:
- Play with some values here such as `base_model_power` or `num_inference_steps`, which must be the same for both pipelines.
- You can also play with `guidance_scale`, `guidance_rescale` which can be set differently for each stage. These impact contrast and realism.
- Using separate prompts between the base and refiner models to shift the guidance focus for fine details.
