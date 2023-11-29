from torchvision import transforms
from helpers.image_manipulation.brightness import calculate_luminance
from io import BytesIO
from PIL import Image
from PIL.ImageOps import exif_transpose
import logging, os
from math import sqrt
from helpers.training.state_tracker import StateTracker

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
    def _crop_center(img: Image, target_width: int, target_height: int) -> Image:
        """
        Crop img to the target width and height, retaining the center of the img.
        """
        width, height = img.size
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2

        return img.crop((left, top, right, bottom)), (left, top)

    @staticmethod
    def process_for_bucket(
        data_backend,
        bucket_manager,
        image_path_str,
        aspect_ratio_bucket_indices,
        aspect_ratio_rounding: int = 2,
        metadata_updates=None,
        delete_problematic_images: bool = False,
        minimum_image_size: int = None,
        resolution_type: str = "pixel"
    ):
        try:
            image_metadata = {}
            image_data = data_backend.read(image_path_str)
            with Image.open(BytesIO(image_data)) as image:
                # Apply EXIF transforms
                image_metadata["original_size"] = image.size
                image, crop_coordinates = MultiaspectImage.prepare_image(
                    image, bucket_manager.resolution, bucket_manager.resolution_type
                )
                image_metadata["crop_coordinates"] = crop_coordinates
                image_metadata["target_size"] = image.size
                # Round to avoid excessive unique buckets
                aspect_ratio = round(image.width / image.height, aspect_ratio_rounding)
                image_metadata["aspect_ratio"] = aspect_ratio
                image_metadata["luminance"] = calculate_luminance(image)
                logger.debug(
                    f"Image {image_path_str} has aspect ratio {aspect_ratio} and size {image.size}."
                )
                if not BucketManager.meets_resolution_requirements(
                    image=image,
                    minimum_image_size=minimum_image_size,
                    resolution_type=resolution_type,
                ):
                    logger.debug(
                        f"Image {image_path_str} does not meet minimum image size requirements. Skipping image."
                    )
                    return aspect_ratio_bucket_indices

            # Create a new bucket if it doesn't exist
            if str(aspect_ratio) not in aspect_ratio_bucket_indices:
                aspect_ratio_bucket_indices[str(aspect_ratio)] = []
            aspect_ratio_bucket_indices[str(aspect_ratio)].append(image_path_str)
            # Instead of directly updating, just fill the provided dictionary
            if metadata_updates is not None:
                metadata_updates[image_path_str] = image_metadata
        except Exception as e:
            import traceback

            logger.error(f"Error processing image: {e}")
            logging.debug(f"Error traceback: {traceback.format_exc()}")
            logger.error(e)
            if delete_problematic_images:
                logger.error(f"Deleting image.")
                data_backend.delete(image_path_str)
        return aspect_ratio_bucket_indices

    @staticmethod
    def prepare_image(image: Image, resolution: float, resolution_type: str = "pixel"):
        """Prepare an image for training.

        Args:
            image (Image): A Pillow image.
            resolution (float): A float for the image size.
            resolution_type (str, optional): Whether to use the size as pixel edge or area. If area, the image will be resized overall area. Defaults to "pixel".

        Raises:
            Exception: _description_

        Returns:
            _type_: (Image, (cLeft, cTop))
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
        original_width, original_height = image.size
        if resolution_type == "pixel":
            (
                target_width,
                target_height,
            ) = MultiaspectImage.calculate_new_size_by_pixel_edge(
                original_width, original_height, resolution
            )
        elif resolution_type == "area":
            (
                target_width,
                target_height,
            ) = MultiaspectImage.calculate_new_size_by_pixel_area(
                original_width, original_height, resolution
            )
        else:
            raise ValueError(f"Unknown resolution type: {resolution_type}")

        if StateTracker.get_args().crop:
            if original_width < target_width or original_height < target_height:
                # Upscale if the original image is smaller than the target size
                image = MultiaspectImage._resize_image(image, target_width, target_height)
            image, crop_coordinates = MultiaspectImage._crop_center(
                image, resolution, resolution
            )
        else:
            # Resize unconditionally if center cropping is not enabled
            image = MultiaspectImage._resize_image(image, target_width, target_height)
            crop_coordinates = (0, 0)

        return image, crop_coordinates

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
    def calculate_new_size_by_pixel_edge(W: int, H: int, resolution: float):
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
        return int(W), int(H)

    @staticmethod
    def calculate_new_size_by_pixel_area(W: int, H: int, megapixels: float):
        aspect_ratio = W / H
        total_pixels = megapixels * 1e6  # Convert megapixels to pixels

        W_new = int(round(sqrt(total_pixels * aspect_ratio)))
        H_new = int(round(sqrt(total_pixels / aspect_ratio)))

        # Ensure they are divisible by 8
        W_new = MultiaspectImage._round_to_nearest_multiple(W_new, 64)
        H_new = MultiaspectImage._round_to_nearest_multiple(H_new, 64)

        return W_new, H_new
