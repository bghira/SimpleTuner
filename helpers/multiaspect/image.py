from torchvision import transforms
from PIL import Image
import logging
import os
import numpy as np
from math import sqrt
from helpers.training.state_tracker import StateTracker
from helpers.models.common import VideoModelFoundation, ImageModelFoundation

logger = logging.getLogger("MultiaspectImage")
logger.setLevel(os.environ.get("SIMPLETUNER_IMAGE_PREP_LOG_LEVEL", "INFO"))

import torch
from torchvision import transforms
from PIL import Image
import numpy as np


class MultiaspectImage:
    @staticmethod
    def limit_canvas_size(width: int, height: int, max_size: int) -> dict:
        """
        Limit the canvas size to a maximum value.
        If the canvas size exceeds the maximum, it will be reduced proportionally.
        It's important that the canvas size returned by this function is used for cropping,
          since the aspect ratio of the canvas size is not guaranteed to match the
          aspect ratio of the original image.

        Args:
            width (int): The width of the image.
            height (int): The height of the image.
            max_size (int): The maximum allowed canvas size.

        Returns:
            dict: A dictionary containing the adjusted width, height, and canvas size.
        """
        if width * height <= max_size:
            # If the canvas size is already within limits, return the original dimensions.
            return {"width": width, "height": height, "canvas_size": width * height}

        align = StateTracker.get_args().aspect_bucket_alignment
        dims = [("width", width), ("height", height)]

        # Sort by size descending
        dims.sort(key=lambda x: x[1], reverse=True)
        # Subtract alignment from the larger dimension
        if dims[0][0] == "width":
            width -= align
        else:
            height -= align

        # If still too large, subtract from the other dimension.
        # Note: we're referencing the original width and height here, not the adjusted ones.
        if width * height > max_size:
            if dims[1][0] == "width":
                width -= align
            else:
                height -= align

        return {"width": width, "height": height, "canvas_size": width * height}

    @staticmethod
    def _round_to_nearest_multiple(value, override_value: int = None):
        """Round a value to the nearest multiple."""
        multiple = override_value or StateTracker.get_args().aspect_bucket_alignment
        rounded = round(value / multiple) * multiple
        return max(rounded, multiple)  # Ensure it's at least the value of 'multiple'

    @staticmethod
    def is_image_too_large(image_size: tuple, resolution: float, resolution_type: str):
        """
        Determine if an image is too large to be processed.

        Args:
            image (PIL.Image): The image to check.
            resolution (float): The maximum resolution to allow.
            resolution_type (str): What form of resolution to check, choices: "pixel", "area".

        Returns:
            bool: True if the image is too large, False otherwise.
        """
        if resolution_type == "pixel":
            return image_size[0] > resolution or image_size[1] > resolution
        elif resolution_type == "area":
            image_area = image_size[0] * image_size[1]
            target_area = resolution * 1e6  # Convert megapixels to pixels
            logger.debug(
                f"Image is too large? {image_area > target_area} (image area: {image_area}, target area: {target_area})"
            )
            return image_area > target_area
        else:
            raise ValueError(f"Unknown resolution type: {resolution_type}")

    @staticmethod
    def calculate_new_size_by_pixel_edge(
        aspect_ratio: float, resolution: int, original_size: tuple
    ):
        if type(aspect_ratio) != float:
            raise ValueError(f"Aspect ratio must be a float, not {type(aspect_ratio)}")
        if type(resolution) != int and (
            type(resolution) != float or int(resolution) != resolution
        ):
            raise ValueError(f"Resolution must be an int, not {type(resolution)}")

        W_original, H_original = original_size

        # Start by determining the potential initial sizes
        if W_original < H_original:  # Portrait or square orientation
            W_initial = resolution
            H_initial = int(W_initial / aspect_ratio)
        else:  # Landscape orientation
            H_initial = resolution
            W_initial = int(H_initial * aspect_ratio)

        # Round down to ensure we do not exceed original dimensions
        W_adjusted = MultiaspectImage._round_to_nearest_multiple(W_initial)
        H_adjusted = MultiaspectImage._round_to_nearest_multiple(H_initial)

        # Intermediary size might be less than the reformed size.
        # This situation is difficult.
        # If the original image is roughly the size of the reformed image, and the intermediary is too small,
        #  we can't really just boost the size of the reformed image willy-nilly. The intermediary size needs to be larger.
        # We can't increase the intermediary size larger than the original size.
        if W_initial < W_adjusted or H_initial < H_adjusted:
            logger.debug(
                f"Intermediary size {W_initial}x{H_initial} would be smaller than {W_adjusted}x{H_adjusted} (original size: {original_size}, aspect ratio: {aspect_ratio})."
            )
            # How much leeway to we have between the intermediary size and the reformed size?
            reformed_W_diff = W_adjusted - W_initial
            reformed_H_diff = H_adjusted - H_initial
            bigger_difference = max(reformed_W_diff, reformed_H_diff)
            logger.debug(
                f"We have {reformed_W_diff}x{reformed_H_diff} leeway to the reformed image {W_adjusted}x{H_adjusted} from {W_initial}x{H_initial}, adjusting by {bigger_difference}px to both sides: {W_initial + bigger_difference}x{H_initial + bigger_difference}."
            )
            W_initial += bigger_difference
            H_initial += bigger_difference

        adjusted_aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(
            (W_adjusted, H_adjusted)
        )

        return (W_adjusted, H_adjusted), (W_initial, H_initial), adjusted_aspect_ratio

    @staticmethod
    def calculate_new_size_by_pixel_area(
        aspect_ratio: float, megapixels: float, original_size: tuple
    ):
        if type(aspect_ratio) not in [float, np.float64]:
            raise ValueError(f"Aspect ratio must be a float, not {type(aspect_ratio)}")
        target_pixel_area = (
            megapixels * 1e6
        )  # Convert megapixels to pixel area, eg. 1.0 mp = 1000000 pixels
        target_pixel_edge = MultiaspectImage._round_to_nearest_multiple(
            int(sqrt(target_pixel_area))
        )
        logger.debug(
            f"Converted {megapixels} megapixels to {target_pixel_area} pixels with a square edge of {target_pixel_edge}."
        )

        W_initial, H_initial = original_size
        if aspect_ratio == 1.0:
            # If the aspect ratio is 1.0, we can just use the square edge as the target size.
            logger.debug(
                f"Returning the square edge {target_pixel_edge}x{target_pixel_edge} as the target size and original size as intermediary."
            )
            if W_initial == H_initial:
                # if we have squares, resizing straight to the target is alright.
                return (
                    (target_pixel_edge, target_pixel_edge),
                    (target_pixel_edge, target_pixel_edge),
                    aspect_ratio,
                )
            return (
                (target_pixel_edge, target_pixel_edge),
                (W_initial, H_initial),
                aspect_ratio,
            )

        # Calculate the target size. This is what will be cropped-to.
        W_target = MultiaspectImage._round_to_nearest_multiple(
            target_pixel_edge * sqrt(aspect_ratio)
        )
        H_target = MultiaspectImage._round_to_nearest_multiple(
            target_pixel_edge / sqrt(aspect_ratio)
        )
        calculated_resulting_megapixels = (W_target * H_target) / 1e6
        target_aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(
            (W_target, H_target)
        )

        if not np.isclose(calculated_resulting_megapixels, megapixels, rtol=1e-1):
            logger.debug(
                f"-!- This image will not have the correct target megapixel size: {calculated_resulting_megapixels}"
            )

        # Calculate the intermediary size. This will maintain aspect ratio and be resized-to.
        if W_target < H_target:  # Portrait or square orientation
            W_intermediary = W_target
            H_intermediary = int(W_intermediary / aspect_ratio)
        else:  # Landscape orientation
            H_intermediary = H_target
            W_intermediary = int(H_intermediary * aspect_ratio)

        # retrieve the static mapping.
        adjusted_aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(
            (W_target, H_target)
        )
        previously_stored_resolution = StateTracker.get_resolution_by_aspect(
            dataloader_resolution=megapixels, aspect=adjusted_aspect_ratio
        )

        if previously_stored_resolution:
            logger.debug(
                f"Using cached aspect-resolution map value for {adjusted_aspect_ratio}: {previously_stored_resolution}"
            )
            W_target, H_target = previously_stored_resolution
        target_resolution = (W_target, H_target)

        # The intermediary size might be smaller than the target. This is bad.
        # If it happens, the cropped image will be cropped past the boundaries of the intermediary size.
        if W_target > W_intermediary or H_target > H_intermediary:
            _W_intermediary, _H_intermediary = W_intermediary, H_intermediary
            if W_target > W_intermediary:
                W_diff = W_target - W_intermediary
                H_diff = int(W_diff / aspect_ratio)
            else:
                H_diff = H_target - H_intermediary
                W_diff = int(H_diff * aspect_ratio)
            H_intermediary += H_diff
            W_intermediary += W_diff
            logger.debug(
                f"Intermediary size {_W_intermediary}x{_H_intermediary} would be smaller than {W_target}x{H_target} with a difference in size of {W_diff}x{H_diff}."
                f" The size will be adjusted to maintain the aspect ratio: {W_intermediary}x{H_intermediary}."
            )
            calculated_resulting_megapixels = (W_intermediary * H_intermediary) / 1e6

        intermediary_resolution = (W_intermediary, H_intermediary)

        logger.debug(
            f"Using target size of {megapixels} megapixels:"
            f"\n-> initial size is {W_initial}x{H_initial}, original aspect ratio {aspect_ratio}."
            f"\n-> intermediary size is {W_intermediary}x{H_intermediary}, with aspect ratio {adjusted_aspect_ratio}."
            f"\n-> cropped size is {W_target}x{H_target}, with aspect ratio {target_aspect_ratio}."
            f"\n-> cropped sample will be {calculated_resulting_megapixels} megapixels"
        )
        # Attempt to retrieve previously stored resolution by adjusted aspect ratio
        if not previously_stored_resolution:
            logger.debug(
                f"No cached resolution found for aspect ratio {adjusted_aspect_ratio}. Storing {target_resolution}."
            )
            StateTracker.set_resolution_by_aspect(
                dataloader_resolution=megapixels,
                aspect=adjusted_aspect_ratio,
                resolution=target_resolution,
            )

        return (target_resolution, intermediary_resolution, adjusted_aspect_ratio)

    @staticmethod
    def adjust_resolution_to_bucket_interval(
        initial_resolution: tuple, target_resolution: tuple
    ):
        W_initial, H_initial = initial_resolution
        W_adjusted, H_adjusted = target_resolution
        # If W_initial or H_initial are < W_adjusted or H_adjusted, add the greater of the two differences to both values.
        W_diff = W_adjusted - W_initial
        H_diff = H_adjusted - H_initial
        if W_diff > 0 and (W_diff > H_diff or W_diff == H_diff):
            logger.debug(
                f"Intermediary size {W_initial}x{H_initial} would be smaller than {W_adjusted}x{H_adjusted} with a difference in size of {W_diff}x{H_diff}. Adjusting both sides by {max(W_diff, H_diff)} pixels."
            )
            H_initial += W_diff
            W_initial += W_diff
        elif H_diff > 0 and H_diff > W_diff:
            logger.debug(
                f"Intermediary size {W_initial}x{H_initial} would be smaller than {W_adjusted}x{H_adjusted} with a difference in size of {W_diff}x{H_diff}. Adjusting both sides by {max(W_diff, H_diff)} pixels."
            )
            W_initial += H_diff
            H_initial += H_diff

        return W_initial, H_initial

    @staticmethod
    def calculate_image_aspect_ratio(image, rounding: int = 2):
        """
        Calculate the aspect ratio of an image and round it to a specified precision.

        Args:
            image (PIL.Image): The image to calculate the aspect ratio for.

        Returns:
            float: The rounded aspect ratio of the image.
        """
        to_round = StateTracker.get_args().aspect_bucket_rounding
        if to_round is None:
            to_round = rounding
        if isinstance(image, Image.Image):
            # An actual image was passed in.
            width, height = image.size
        elif isinstance(image, tuple) or isinstance(image, list):
            # An image.size or a similar (W, H) tuple was provided.
            width, height = image
        elif isinstance(image, float):
            # An externally-calculated aspect ratio was given to round.
            return round(image, to_round)
        elif isinstance(image, np.ndarray):
            # A video was passed in as a numpy array.
            width, height = image.shape[2], image.shape[1]
        else:
            raise ValueError(f"Unexpected type {image}")
            width, height = image.size
        aspect_ratio = round(width / height, to_round)
        return aspect_ratio

    @staticmethod
    def numpy_list_to_pil(numpy_list):
        if isinstance(numpy_list, list) and isinstance(numpy_list[0], np.ndarray):
            numpy_list = [
                Image.fromarray(np.uint8(image)).convert("RGB") for image in numpy_list
            ]
        return numpy_list


resize_helpers = {
    "pixel": MultiaspectImage.calculate_new_size_by_pixel_edge,
    "area": MultiaspectImage.calculate_new_size_by_pixel_area,
}
