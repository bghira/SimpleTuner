import logging
import os

import numpy as np
import torch
import wandb
from PIL import Image

from simpletuner.helpers.image_manipulation.brightness import calculate_luminance
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)


def save_images(save_dir, validation_images, validation_shortname, validation_resolutions, config):
    """
    Save validation images to disk.
    """
    validation_img_idx = 0
    resolutions = [_res for res in validation_resolutions for _res in [res] * config.num_eval_images]

    for validation_image in validation_images.get(validation_shortname, []):
        if validation_img_idx >= len(resolutions):
            # Fallback if we have more images than resolutions
            if hasattr(validation_image, "size"):
                res_label = f"{validation_image.size[0]}x{validation_image.size[1]}"
            else:
                res_label = "unknown"
        else:
            res = resolutions[validation_img_idx]
            if isinstance(res, str) and "x" in res:
                res_label = str(res)
            elif isinstance(res, tuple):
                res_label = f"{res[0]}x{res[1]}"
            else:
                res_label = f"{res}x{res}"

        filename = f"step_{StateTracker.get_global_step()}_{validation_shortname}_{validation_img_idx}_{res_label}.png"
        save_path = os.path.join(save_dir, filename)

        try:
            validation_image.save(save_path)
        except Exception as e:
            logger.error(f"Failed to save validation image to {save_path}: {e}")

        validation_img_idx += 1


def log_images_to_trackers(accelerator, validation_images, validation_resolutions, config):
    """
    Log validation images to available trackers.
    """
    for tracker in accelerator.trackers:
        if tracker.name == "comet_ml":
            experiment = accelerator.get_tracker("comet_ml").tracker
            for shortname, image_list in validation_images.items():
                for idx, image in enumerate(image_list):
                    res_label = str(validation_resolutions[idx]) if idx < len(validation_resolutions) else "unknown"
                    experiment.log_image(
                        image,
                        name=f"{shortname} - {res_label}",
                    )
        elif tracker.name == "tensorboard":
            tracker = accelerator.get_tracker("tensorboard")
            for shortname, image_list in validation_images.items():
                tracker.log_images(
                    {
                        f"{shortname} - {validation_resolutions[idx] if idx < len(validation_resolutions) else 'unknown'}": np.moveaxis(
                            np.array(image), -1, 0
                        )[
                            np.newaxis, ...
                        ]
                        for idx, image in enumerate(image_list)
                    },
                    step=StateTracker.get_global_step(),
                )
        elif tracker.name == "wandb":
            resolution_list = []
            for res in validation_resolutions:
                if isinstance(res, tuple):
                    resolution_list.append(f"{res[0]}x{res[1]}")
                else:
                    resolution_list.append(str(res))

            if config.tracker_image_layout == "table":
                columns = [
                    "Prompt",
                    *resolution_list,
                    "Mean Luminance",
                ]
                table = wandb.Table(columns=columns)

                # Process each prompt and its associated images
                for prompt_shortname, image_list in validation_images.items():
                    wandb_images = []
                    luminance_values = []
                    logger.debug(f"Prompt {prompt_shortname} has {len(image_list)} images")
                    for image in image_list:
                        wandb_image = wandb.Image(image)
                        wandb_images.append(wandb_image)
                        luminance = calculate_luminance(image)
                        luminance_values.append(luminance)

                    if luminance_values:
                        mean_luminance = torch.tensor(luminance_values).mean().item()
                    else:
                        mean_luminance = 0.0

                    while len(wandb_images) < len(resolution_list):
                        # any missing images will crash it. use None so they are indexed.
                        wandb_images.append(None)

                    # Ensure we don't have more images than columns
                    if len(wandb_images) > len(resolution_list):
                        wandb_images = wandb_images[: len(resolution_list)]

                    table.add_data(prompt_shortname, *wandb_images, mean_luminance)

                # Log the table to Weights & Biases
                tracker.log(
                    {"Validation Gallery": table},
                    step=StateTracker.get_global_step(),
                )

            elif config.tracker_image_layout == "gallery":
                gallery_images = {}
                for prompt_shortname, image_list in validation_images.items():
                    for idx, image in enumerate(image_list):
                        # if it's a list of images, make a grid
                        if isinstance(image, list) and isinstance(image[0], Image.Image):
                            image = image[0]

                        res_label = resolution_list[idx] if idx < len(resolution_list) else "unknown"
                        wandb_image = wandb.Image(
                            image,
                            caption=f"{prompt_shortname} - {res_label}",
                        )
                        gallery_images[f"{prompt_shortname} - {res_label}"] = wandb_image

                # Log all images in one call to prevent the global step from ticking
                tracker.log(gallery_images, step=StateTracker.get_global_step())


def log_images_to_webhook(validation_images, validation_shortname, validation_prompt, eval_scores):
    """
    Log validation images to webhook.
    """
    webhook_handler = StateTracker.get_webhook_handler()
    if webhook_handler is None:
        return

    message = (
        f"Validation image for `{validation_shortname if validation_shortname != '' else '(blank shortname)'}`"
        f"\\nValidation prompt: `{validation_prompt if validation_prompt != '' else '(blank prompt)'}`"
        f"\\nEvaluation score: {eval_scores.get(validation_shortname, 'N/A')}"
    )

    images_payload = validation_images.get(validation_shortname, [])

    webhook_handler.send(
        message,
        images=images_payload,
    )

    webhook_handler.send_raw(
        structured_data={"message": f"Validation: {validation_shortname}"},
        message_type="training.validation",
        message_level="info",
        job_id=StateTracker.get_job_id(),
        images=images_payload,
    )
