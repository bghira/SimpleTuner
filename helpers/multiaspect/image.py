from torchvision import transforms
from io import BytesIO
from PIL import Image
from PIL.ImageOps import exif_transpose
import logging, os
from math import sqrt

logger = logging.getLogger("MultiaspectImage")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING"))


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
    def process_for_bucket(
        data_backend,
        image_path_str,
        resolution: float,
        resolution_type: str,
        aspect_ratio_bucket_indices,
        aspect_ratio_rounding: int = 2,
    ):
        try:
            image_data = data_backend.read(image_path_str)
            with Image.open(BytesIO(image_data)) as image:
                # Apply EXIF transforms
                image = MultiaspectImage.prepare_image(
                    image, resolution, resolution_type
                )
                # Round to avoid excessive unique buckets
                aspect_ratio = round(image.width / image.height, aspect_ratio_rounding)
                logger.debug(
                    f"Image {image_path_str} has aspect ratio {aspect_ratio} and size {image.size}."
                )
            # Create a new bucket if it doesn't exist
            if str(aspect_ratio) not in aspect_ratio_bucket_indices:
                aspect_ratio_bucket_indices[str(aspect_ratio)] = []
            aspect_ratio_bucket_indices[str(aspect_ratio)].append(image_path_str)
        except Exception as e:
            logger.error(f"Error processing image {image_path_str}.")
            logger.error(e)
        return aspect_ratio_bucket_indices

    @staticmethod
    def prepare_image(image: Image, resolution: float, resolution_type: str = "pixel"):
        """Prepare an image for training.

        Args:
            image (Image): A Pillow image.
            resolution (int): An integer for the image size.
            resolution_type (str, optional): Whether to use the size as pixel edge or area. If area, the image will be resized overall area. Defaults to "pixel".

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        if not hasattr(image, "convert"):
            raise Exception(
                f"Unknown data received instead of PIL.Image object: {type(image)}"
            )
        # Strip transparency
        image = image.convert("RGB")
        # Rotate, maybe.
        logger.debug(f"Processing image filename: {image}")
        logger.debug(f"Image size before EXIF transform: {image.size}")
        image = exif_transpose(image)
        logger.debug(f"Image size after EXIF transform: {image.size}")
        if resolution_type == "pixel":
            image = MultiaspectImage.resize_by_pixel_edge(image, resolution)
        elif resolution_type == "area":
            image = MultiaspectImage.resize_by_pixel_area(image, resolution)
        else:
            raise ValueError(f"Unknown resolution type: {resolution_type}")
        return image

    @staticmethod
    def _round_to_nearest_multiple(value, multiple):
        """Round a value to the nearest multiple."""
        rounded = round(value / multiple) * multiple
        return max(rounded, multiple)  # Ensure it's at least the value of 'multiple'

    @staticmethod
    def _resize_image(
        input_image: Image, target_width: int, target_height: int
    ) -> Image:
        """Resize the input image to the target width and height."""
        if not hasattr(input_image, "convert"):
            raise Exception(
                f"Unknown data received instead of PIL.Image object: {type(input_image)}"
            )
        logger.debug(f"Received image for processing: {input_image}")
        input_image = input_image.convert("RGB")
        logger.debug(f"Converted image to RGB for processing: {input_image}")
        if (target_width, target_height) == input_image.size:
            return input_image
        msg = f"Resizing image of size {input_image.size} to its new size: {target_width}x{target_height}."
        logger.debug(msg)
        return input_image.resize((target_width, target_height), resample=Image.BICUBIC)

    @staticmethod
    def resize_by_pixel_edge(input_image: Image, resolution: float) -> Image:
        W, H = input_image.size
        aspect_ratio = W / H
        if W < H:
            W = resolution
            H = MultiaspectImage._round_to_nearest_multiple(
                resolution / aspect_ratio, 64
            )
        elif H < W:
            H = resolution
            W = MultiaspectImage._round_to_nearest_multiple(
                resolution * aspect_ratio, 64
            )
        else:
            W = H = MultiaspectImage._round_to_nearest_multiple(resolution, 64)
        return MultiaspectImage._resize_image(input_image, W, H)

    @staticmethod
    def resize_by_pixel_area(input_image: Image, megapixels: float) -> Image:
        W, H = input_image.size
        aspect_ratio = W / H
        total_pixels = megapixels * 1e6  # Convert megapixels to pixels

        W_new = int(round(sqrt(total_pixels * aspect_ratio)))
        H_new = int(round(sqrt(total_pixels / aspect_ratio)))

        # Ensure they are divisible by 8
        W_new = MultiaspectImage._round_to_nearest_multiple(W_new, 64)
        H_new = MultiaspectImage._round_to_nearest_multiple(H_new, 64)

        return MultiaspectImage._resize_image(input_image, W_new, H_new)
