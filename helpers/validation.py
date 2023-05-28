import logging, torch, numpy as np

from diffusers.utils import is_wandb_available
if is_wandb_available():
    import wandb

from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline

def log_validation(
    text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch
):
    logging.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline.unet = torch.compile(pipeline.unet)
    # pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        None
        if args.seed is None
        else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    )
    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            negative = "negative low quality, low res, messy, grainy, smooth, sand, big eyes, anime, fractured, cracked, wrinkles, makeup (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, synthetic, rendering"
            image = pipeline(
                args.validation_prompt,
                negative_prompt=negative,
                num_inference_steps=35,
                generator=generator,
            ).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", np_images, epoch, dataformats="NHWC"
            )
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images
