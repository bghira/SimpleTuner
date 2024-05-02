from torchvision import transforms
from helpers.image_manipulation.brightness import calculate_luminance
from io import BytesIO
from PIL import Image
from PIL.ImageOps import exif_transpose
import logging, os, random
from math import sqrt
from helpers.training.state_tracker import StateTracker
from helpers.image_manipulation.cropping import crop_handlers

logger = logging.getLogger("MultiaspectImage")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class MultiaspectImage:
    @staticmethod
    def get_image_transforms():
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    @staticmethod
    def _round_to_nearest_multiple(value):
        """Round a value to the nearest multiple."""
        multiple = StateTracker.get_args().aspect_bucket_alignment
        rounded = round(value / multiple) * multiple
        return max(rounded, multiple)  # Ensure it's at least the value of 'multiple'

    @staticmethod
    def _resize_image(
        input_image: Image.Image,
        target_width: int,
        target_height: int,
        image_metadata: dict = None,
    ) -> Image:
        """Resize the input image to the target width and height in stages, ensuring a higher quality end result."""
        if input_image:
            if not hasattr(input_image, "convert"):
                raise Exception(
                    f"Unknown data received instead of PIL.Image object: {type(input_image)}"
                )
            logger.debug(f"Received image for processing: {input_image}")
            input_image = input_image.convert("RGB")
            logger.debug(f"Converted image to RGB for processing: {input_image}")
            current_width, current_height = input_image.size
        elif image_metadata:
            current_width, current_height = image_metadata["original_size"]

        if (target_width, target_height) == (current_width, current_height):
            if input_image:
                return input_image
            elif image_metadata:
                return image_metadata["original_size"]

        msg = f"Resizing image of size {input_image.size} to its new size: {target_width}x{target_height}."
        logger.debug(msg)

        # Resize in stages
        while (
            current_width > target_width * 1.5 or current_height > target_height * 1.5
        ):
            # Calculate intermediate size
            intermediate_width = int(current_width * 0.75)
            intermediate_height = int(current_height * 0.75)

            # Ensure intermediate size is not smaller than the target size
            intermediate_width = max(intermediate_width, target_width)
            intermediate_height = max(intermediate_height, target_height)

            new_image_size = (intermediate_width, intermediate_height)
            if input_image:
                input_image = input_image.resize(
                    new_image_size, resample=Image.Resampling.LANCZOS
                )
            current_width, current_height = new_image_size
            logger.debug(
                f"Resized image to intermediate size: {current_width}x{current_height}."
            )

        # Final resize to target dimensions
        if input_image:
            input_image = input_image.resize(
                (target_width, target_height), resample=Image.Resampling.LANCZOS
            )
            logger.debug(f"Final image size: {input_image.size}.")
            return input_image
        elif image_metadata:
            image_metadata["original_size"] = (target_width, target_height)
            logger.debug(f"Final image size: {image_metadata['original_size']}.")
            return image_metadata

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
    def calculate_new_size_by_pixel_edge(aspect_ratio: float, resolution: int):
        """
        Calculate the width, height, and new AR of a pixel-aligned size, where resolution is the smaller edge length.

        Args:
            aspect_ratio (float): The aspect ratio of the image.
            resolution (int): The resolution of the smaller edge of the image.

        return int(W), int(H), new_aspect_ratio
        """
        if type(aspect_ratio) != float:
            raise ValueError(f"Aspect ratio must be a float, not {type(aspect_ratio)}")
        if type(resolution) != int and type(resolution) != float:
            raise ValueError(f"Resolution must be an int, not {type(resolution)}")
        if aspect_ratio > 1:
            W_initial = resolution * aspect_ratio
            H_initial = resolution
        elif aspect_ratio < 1:
            W_initial = resolution
            H_initial = resolution / aspect_ratio
        else:
            W_initial = resolution
            H_initial = resolution

        W_adjusted = MultiaspectImage._round_to_nearest_multiple(W_initial)
        H_adjusted = MultiaspectImage._round_to_nearest_multiple(H_initial)

        # Ensure the adjusted dimensions meet the resolution requirement
        while min(W_adjusted, H_adjusted) < resolution:
            W_adjusted += StateTracker.get_args().aspect_bucket_alignment
            H_adjusted = MultiaspectImage._round_to_nearest_multiple(
                int(round(W_adjusted * aspect_ratio))
            )

        return (
            W_adjusted,
            H_adjusted,
            MultiaspectImage.calculate_image_aspect_ratio((W_adjusted, H_adjusted)),
        )

    @staticmethod
    def calculate_new_size_by_pixel_area(aspect_ratio: float, megapixels: float):
        if type(aspect_ratio) != float:
            raise ValueError(f"Aspect ratio must be a float, not {type(aspect_ratio)}")
        # Special case for 1024px (1.0) megapixel images
        if aspect_ratio == 1.0 and megapixels == 1.0:
            return 1024, 1024, 1.0
        # Special case for 768px (0.75mp) images
        if aspect_ratio == 1.0 and megapixels == 0.75:
            return 768, 768, 1.0
        # Special case for 512px (0.5mp) images
        if aspect_ratio == 1.0 and megapixels == 0.25:
            return 512, 512, 1.0
        total_pixels = max(megapixels * 1e3, 1e6)
        W_initial = int(round((total_pixels * aspect_ratio) ** 0.5))
        H_initial = int(round((total_pixels / aspect_ratio) ** 0.5))

        W_adjusted = MultiaspectImage._round_to_nearest_multiple(W_initial)
        H_adjusted = MultiaspectImage._round_to_nearest_multiple(H_initial)

        # Ensure the adjusted dimensions meet the megapixel requirement
        while W_adjusted * H_adjusted < total_pixels:
            W_adjusted += StateTracker.get_args().aspect_bucket_alignment
            H_adjusted = MultiaspectImage._round_to_nearest_multiple(
                int(round(W_adjusted / aspect_ratio))
            )

        return (
            W_adjusted,
            H_adjusted,
            MultiaspectImage.calculate_image_aspect_ratio((W_adjusted, H_adjusted)),
        )

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
        elif isinstance(image, tuple):
            # An image.size or a similar (W, H) tuple was provided.
            width, height = image
        elif isinstance(image, float):
            # An externally-calculated aspect ratio was given to round.
            return round(image, to_round)
        else:
            width, height = image.size
        aspect_ratio = round(width / height, to_round)
        return aspect_ratio


resize_helpers = {
    "pixel": MultiaspectImage.calculate_new_size_by_pixel_edge,
    "area": MultiaspectImage.calculate_new_size_by_pixel_area,
}
