from torchvision import transforms
from helpers.image_manipulation.brightness import calculate_luminance
from io import BytesIO
from PIL import Image
from PIL.ImageOps import exif_transpose
import logging, os, random
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
    def process_for_bucket(
        data_backend,
        bucket_manager,
        image_path_str,
        aspect_ratio_bucket_indices,
        aspect_ratio_rounding: int = 2,
        metadata_updates=None,
        delete_problematic_images: bool = False,
        minimum_image_size: int = None,
        resolution_type: str = "pixel",
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
                if not StateTracker.get_bucket_manager().meets_resolution_requirements(
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

        # Calculate new size
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
            # Convert 'resolution' to pixels from megapixels
            resolution = resolution / 1e6
            logging.debug(
                f"After area resize, our image will be {target_width}x{target_height} with an overridden resolution of {resolution} pixels."
            )
        else:
            raise ValueError(f"Unknown resolution type: {resolution_type}")

        crop_style = StateTracker.get_args().crop_style
        crop_aspect = StateTracker.get_args().crop_aspect

        if StateTracker.get_args().crop:
            crop_width, crop_height = (
                (resolution, resolution)
                if crop_aspect == "square"
                else (target_width, target_height)
            )

            if crop_style == "corner":
                image, crop_coordinates = MultiaspectImage._crop_corner(
                    image, crop_width, crop_height
                )
            elif crop_style in ["centre", "center"]:
                image, crop_coordinates = MultiaspectImage._crop_center(
                    image, crop_width, crop_height
                )
            elif crop_style == "random":
                image, crop_coordinates = MultiaspectImage._crop_random(
                    image, crop_width, crop_height
                )
            else:
                raise ValueError(f"Unknown crop style: {crop_style}")
        else:
            # Resize unconditionally if cropping is not enabled
            image = MultiaspectImage._resize_image(image, target_width, target_height)
            crop_coordinates = (0, 0)

        return image, crop_coordinates

    @staticmethod
    def _crop_corner(image, target_width, target_height):
        """Crop the image from the bottom-right corner."""
        original_width, original_height = image.size
        left = max(0, original_width - target_width)
        top = max(0, original_height - target_height)
        right = original_width
        bottom = original_height
        return image.crop((left, top, right, bottom)), (left, top)

    @staticmethod
    def _crop_center(image, target_width, target_height):
        """Crop the image from the center."""
        original_width, original_height = image.size
        left = (original_width - target_width) / 2
        top = (original_height - target_height) / 2
        right = (original_width + target_width) / 2
        bottom = (original_height + target_height) / 2
        return image.crop((left, top, right, bottom)), (left, top)

    @staticmethod
    def _crop_random(image, target_width, target_height):
        """Crop the image from a random position."""
        original_width, original_height = image.size
        left = random.randint(0, max(0, original_width - target_width))
        top = random.randint(0, max(0, original_height - target_height))
        right = left + target_width
        bottom = top + target_height
        return image.crop((left, top, right, bottom)), (left, top)

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
        return input_image.resize((target_width, target_height), resample=Image.LANCZOS)

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
