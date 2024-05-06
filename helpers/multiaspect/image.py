from torchvision import transforms
from helpers.image_manipulation.brightness import calculate_luminance
from io import BytesIO
from PIL import Image
from PIL.ImageOps import exif_transpose
import logging, os, random
from math import sqrt, floor
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
    def calculate_new_size_by_pixel_edge(
        aspect_ratio: float, resolution: int, original_size: tuple
    ):
        if type(aspect_ratio) != float:
            raise ValueError(f"Aspect ratio must be a float, not {type(aspect_ratio)}")
        if type(resolution) != int:
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
                f"Intermediary size {W_initial}x{H_initial} would be smaller than {W_adjusted}x{H_adjusted} (original size: {original_size})."
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
        if type(aspect_ratio) != float:
            raise ValueError(f"Aspect ratio must be a float, not {type(aspect_ratio)}")
        pixels = megapixels * 1e6  # Convert megapixels to pixels
        logger.debug(f"Converted {megapixels} megapixels to {pixels} pixels.")
        W_initial = int(round((pixels * aspect_ratio) ** 0.5))
        H_initial = int(round((pixels / aspect_ratio) ** 0.5))
        # Special case for 1024px (1.0) megapixel images
        if aspect_ratio == 1.0 and megapixels == 1.0:
            return ((1024, 1024), (W_initial, H_initial), 1.0)
        # Special case for 768px (0.75mp) images
        if aspect_ratio == 1.0 and megapixels == 0.75:
            return ((768, 768), (W_initial, H_initial), 1.0)
        # Special case for 512px (0.5mp) images
        if aspect_ratio == 1.0 and megapixels == 0.25:
            return ((512, 512), (W_initial, H_initial), 1.0)

        W_adjusted = MultiaspectImage._round_to_nearest_multiple(W_initial)
        H_adjusted = MultiaspectImage._round_to_nearest_multiple(H_initial)

        # Ensure the adjusted dimensions meet the megapixel requirement
        while W_adjusted * H_adjusted < pixels:
            W_adjusted += StateTracker.get_args().aspect_bucket_alignment
            H_adjusted = MultiaspectImage._round_to_nearest_multiple(
                int(round(W_adjusted / aspect_ratio))
            )

        # If W_initial or H_initial are < W_adjusted or H_adjusted, add the greater of the two differences to both values.
        W_diff = W_adjusted - W_initial
        H_diff = H_adjusted - H_initial
        logger.debug(f"Differences: {W_diff}, {H_diff}")
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

        return (
            (W_adjusted, H_adjusted),
            (W_initial, H_initial),
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
