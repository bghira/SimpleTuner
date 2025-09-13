from accelerate import Accelerator
from diffusers import (
    DiffusionPipeline,
    UNet2DConditionModel,
    DDPMScheduler,
    DDIMScheduler,
)
from transformers import CLIPTextModel
from simpletuner.helpers.prompts import prompts
from compel import Compel

import torch, os, logging

logger = logging.getLogger("SimpleTuner-inference")
logger.setLevel(logging.INFO)
# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "stabilityai/stable-diffusion-2"
model_id = "ptx0/terminus-xl-gamma-v2-1"
base_dir = "/Volumes/models/training"
model_path = os.path.join(base_dir, "models")
# output_test_dir = os.path.join(base_dir, 'test_results')
output_test_dir = os.path.join(base_dir, "encoder_test")
save_pretrained = False
torch_seed = 4202420420

# Find the latest checkpoint
import os

checkpoints = [
    int(x.split("-")[1]) for x in os.listdir(model_path) if x.startswith("checkpoint-")
]
checkpoints.sort()
range_begin = 0
range_step = 100
base_checkpoint_for_unet = (
    0  # Use the unet from this model for comparison against text encoder progress.
)
try:
    range_end = checkpoints[-1]
except Exception as e:
    range_end = range_begin
logging.info(f"Highest checkpoint found so far: {range_end}")

# Convert numeric range to an array of string numerics:
# checkpoints = [ str(x) for x in range(range_begin, range_end + range_step, range_step) ]
checkpoints.reverse()
torch.set_float32_matmul_precision("high")
negative = "deep fried watermark cropped out-of-frame low quality low res oorly drawn bad anatomy wrong anatomy extra limb missing limb floating limbs (mutated hands and fingers)1.4 disconnected limbs mutation mutated ugly disgusting blurry amputation synthetic rendering"
for checkpoint in checkpoints:
    for enable_textencoder in [False]:
        suffix = (
            "t" if enable_textencoder else "b" if enable_textencoder is None else "u"
        )
        if len(checkpoints) > 1 and os.path.isfile(
            f"{output_test_dir}/target-{checkpoint}_{base_checkpoint_for_unet}{suffix}.png"
        ):
            continue
        try:
            logging.info(f"Loading checkpoint: {model_path}/checkpoint-{checkpoint}")
            # Does the checkpoint path exist?
            if checkpoint != "0" and not os.path.exists(
                f"{model_path}/checkpoint-{checkpoint}"
            ):
                logging.info(f"Checkpoint {checkpoint} does not exist.")
                continue

            if checkpoint != "0":
                logging.info(f"Loading non-base ckpt.")
                if enable_textencoder is None:
                    logging.info(f"Loading full unet and te")
                    # Enable fully-trained text_encoder and unet
                    text_encoder = CLIPTextModel.from_pretrained(
                        f"{model_path}/checkpoint-{checkpoint}/text_encoder"
                    )
                    unet = UNet2DConditionModel.from_pretrained(
                        f"{model_path}/checkpoint-{checkpoint}/unet"
                    )
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_id, unet=unet, text_encoder=text_encoder
                    )
                elif enable_textencoder:
                    # Enable the fully-trained text encoder with the 4200 ckpt unet
                    logging.info(f"Loading full te and base unet")
                    text_encoder = CLIPTextModel.from_pretrained(
                        f"{model_path}/checkpoint-{checkpoint}/text_encoder"
                    )
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_id, text_encoder=text_encoder
                    )
                else:
                    # Enable the fully-trained unet with the 4200 ckpt text encoder
                    logging.info(f"Loading full unet and base te")
                    unet = UNet2DConditionModel.from_pretrained(
                        f"{model_path}/checkpoint-{checkpoint}/unet"
                    )
                    pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet)
            else:
                # Do the base model.
                logging.info(f"Loading base ckpt.")
                pipeline = DiffusionPipeline.from_pretrained(model_id)
            pipeline.unet = torch.compile(pipeline.unet)
            compel = Compel(
                tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder
            )
            negative_embed = compel.build_conditioning_tensor(negative)

            pipeline.scheduler = DDIMScheduler.from_pretrained(
                model_id,
                subfolder="scheduler",
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
            )
            pipeline.to(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        except Exception as e:
            logging.info(
                f"Could not generate pipeline for checkpoint {checkpoint}: {e}"
            )
            continue
        # Does the file exist already?
        import os

        for shortname, prompt in prompts.items():
            if not os.path.isfile(
                f"{output_test_dir}/{shortname}-{checkpoint}_{base_checkpoint_for_unet}{suffix}.png"
            ):
                logging.info(
                    f"Generating {shortname} at {checkpoint}_{base_checkpoint_for_unet}{suffix}"
                )
                logging.info(f"Shortname: {shortname}, Prompt: {prompt}")
                logging.info(f"Negative: {negative}")
                conditioning = compel.build_conditioning_tensor(prompt)
                generator = torch.Generator(device="cuda").manual_seed(torch_seed)
                pipeline.do_guidance_rescale_before = 20
                output = pipeline(
                    generator=generator,
                    negative_prompt_embeds=negative_embed,
                    prompt_embeds=conditioning,
                    guidance_scale=7.5,
                    guidance_rescale=0.0,
                    width=1152,
                    height=768,
                    num_inference_steps=25,
                ).images[0]
                output.save(
                    f"{output_test_dir}/{shortname}-{checkpoint}_{base_checkpoint_for_unet}{suffix}.png"
                )
                del output

        if save_pretrained and not os.path.exists(f"{model_path}/pipeline"):
            logging.info(f"Saving pretrained pipeline.")
            pipeline.save_pretrained(
                f"{model_path}/pseudo-real", safe_serialization=True
            )
        elif save_pretrained:
            raise Exception("Can not save pretrained model, path already exists.")
logging.info(f"Exit.")
