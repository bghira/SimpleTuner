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
    def prepare_image(
        resolution: float,
        image: Image = None,
        image_metadata: dict = None,
        resolution_type: str = "pixel",
        id: str = "foo",
    ):
        if image:
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
            image_size = image.size
        elif image_metadata:
            image_size = (
                image_metadata["original_size"][0],
                image_metadata["original_size"][1],
            )
        original_width, original_height = image_size
        original_resolution = resolution
        # Convert 'resolution' from eg. "1 megapixel" to "1024 pixels"
        original_resolution = original_resolution * 1e3
        # Make resolution a multiple of 64
        original_resolution = MultiaspectImage._round_to_nearest_multiple(
            original_resolution, 64
        )

        # Downsample before we handle, if necessary.
        downsample_before_crop = False
        crop = StateTracker.get_data_backend_config(data_backend_id=id).get(
            "crop", StateTracker.get_args().crop
        )
        maximum_image_size = StateTracker.get_data_backend_config(
            data_backend_id=id
        ).get("maximum_image_size", None)
        target_downsample_size = StateTracker.get_data_backend_config(
            data_backend_id=id
        ).get("target_downsample_size", None)
        logger.debug(
            f"Dataset: {id}, maximum_image_size: {maximum_image_size}, target_downsample_size: {target_downsample_size}"
        )
        if crop and maximum_image_size and target_downsample_size:
            if MultiaspectImage.is_image_too_large(
                image_size, maximum_image_size, resolution_type=resolution_type
            ):
                # Override the target resolution with the target downsample size
                logger.debug(
                    f"Overriding resolution {resolution} with target downsample size: {target_downsample_size}"
                )
                resolution = target_downsample_size
                downsample_before_crop = True

        # Calculate new size
        if resolution_type == "pixel":
            (target_width, target_height, new_aspect_ratio) = (
                MultiaspectImage.calculate_new_size_by_pixel_edge(
                    original_width, original_height, resolution
                )
            )
        elif resolution_type == "area":
            (target_width, target_height, new_aspect_ratio) = (
                MultiaspectImage.calculate_new_size_by_pixel_area(
                    original_width, original_height, resolution
                )
            )
            # Convert 'resolution' from eg. "1 megapixel" to "1024 pixels"
            resolution = resolution * 1e3
            # Make resolution a multiple of 64
            resolution = MultiaspectImage._round_to_nearest_multiple(resolution, 64)
            logger.debug(
                f"After area resize, our image will be {target_width}x{target_height} with an overridden resolution of {resolution} pixels."
            )
        else:
            raise ValueError(f"Unknown resolution type: {resolution_type}")

        crop_style = StateTracker.get_data_backend_config(data_backend_id=id).get(
            "crop_style", StateTracker.get_args().crop_style
        )
        crop_aspect = StateTracker.get_data_backend_config(data_backend_id=id).get(
            "crop_aspect", StateTracker.get_args().crop_aspect
        )

        if crop:
            if downsample_before_crop:
                logger.debug(
                    f"Resizing image before crop, as its size is too large. Data backend: {id}, image size: {image.size}, target size: {target_width}x{target_height}"
                )
                if image:
                    image = MultiaspectImage._resize_image(
                        image, target_width, target_height
                    )
                elif image_metadata:
                    image_metadata = MultiaspectImage._resize_image(
                        None, target_width, target_height, image_metadata
                    )
                if resolution_type == "area":
                    # Convert original_resolution back from eg. 1024 pixels to 1.0 mp
                    original_megapixel_resolution = original_resolution / 1e3
                    (target_width, target_height, new_aspect_ratio) = (
                        MultiaspectImage.calculate_new_size_by_pixel_area(
                            original_width,
                            original_height,
                            original_megapixel_resolution,
                        )
                    )
                elif resolution_type == "pixel":
                    (target_width, target_height, new_aspect_ratio) = (
                        MultiaspectImage.calculate_new_size_by_pixel_edge(
                            original_width, original_height, original_resolution
                        )
                    )
                logger.debug(
                    f"Recalculated target_width and target_height {target_width}x{target_height} based on original_resolution: {original_resolution}"
                )

            logger.debug(f"We are cropping the image. Data backend: {id}")
            crop_width, crop_height = (
                (original_resolution, original_resolution)
                if crop_aspect == "square"
                else (target_width, target_height)
            )

            if image:
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
            elif image_metadata:
                if crop_style == "corner":
                    _, crop_coordinates = MultiaspectImage._crop_corner(
                        image_metadata=image_metadata,
                        crop_width=crop_width,
                        crop_height=crop_height,
                    )
                elif crop_style in ["centre", "center"]:
                    _, crop_coordinates = MultiaspectImage._crop_center(
                        image_metadata=image_metadata,
                        crop_width=crop_width,
                        crop_height=crop_height,
                    )
                elif crop_style == "random":
                    _, crop_coordinates = MultiaspectImage._crop_random(
                        image_metadata=image_metadata,
                        crop_width=crop_width,
                        crop_height=crop_height,
                    )
                else:
                    raise ValueError(f"Unknown crop style: {crop_style}")

            logger.debug(f"After cropping, our image size: {image.size}")
        else:
            # Resize unconditionally if cropping is not enabled
            if image:
                image = MultiaspectImage._resize_image(
                    image, target_width, target_height
                )
            crop_coordinates = (0, 0)

        if image:
            return image, crop_coordinates, new_aspect_ratio
        elif image_metadata:
            return (target_width, target_height), crop_coordinates, new_aspect_ratio

    @staticmethod
    def _crop_corner(
        image: Image = None,
        target_width=None,
        target_height=None,
        image_metadata: dict = None,
    ):
        """Crop the image from the bottom-right corner."""
        if image:
            original_width, original_height = image.size
        elif image_metadata:
            original_width, original_height = image_metadata["original_size"]
        left = max(0, original_width - target_width)
        top = max(0, original_height - target_height)
        right = original_width
        bottom = original_height
        if image:
            return image.crop((left, top, right, bottom)), (left, top)
        elif image_metadata:
            return image_metadata, (left, top)

    @staticmethod
    def _crop_center(
        image: Image = None,
        target_width=None,
        target_height=None,
        image_metadata: dict = None,
    ):
        """Crop the image from the center."""
        original_width, original_height = image.size
        left = (original_width - target_width) / 2
        top = (original_height - target_height) / 2
        right = (original_width + target_width) / 2
        bottom = (original_height + target_height) / 2
        if image:
            return image.crop((left, top, right, bottom)), (left, top)
        elif image_metadata:
            return image_metadata, (left, top)

    @staticmethod
    def _crop_random(
        image: Image = None,
        target_width=None,
        target_height=None,
        image_metadata: dict = None,
    ):
        """Crop the image from a random position."""
        original_width, original_height = image.size
        left = random.randint(0, max(0, original_width - target_width))
        top = random.randint(0, max(0, original_height - target_height))
        right = left + target_width
        bottom = top + target_height
        if image:
            return image.crop((left, top, right, bottom)), (left, top)
        elif image_metadata:
            return image_metadata, (left, top)

    @staticmethod
    def _round_to_nearest_multiple(value, multiple):
        """Round a value to the nearest multiple."""
        rounded = round(value / multiple) * multiple
        return max(rounded, multiple)  # Ensure it's at least the value of 'multiple'

    @staticmethod
    def _resize_image(
        input_image: Image,
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
                input_image = input_image.resize(new_image_size, resample=Image.LANCZOS)
            current_width, current_height = new_image_size
            logger.debug(
                f"Resized image to intermediate size: {current_width}x{current_height}."
            )

        # Final resize to target dimensions
        if input_image:
            input_image = input_image.resize(
                (target_width, target_height), resample=Image.LANCZOS
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
    def calculate_new_size_by_pixel_edge(W: int, H: int, resolution: float):
        aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio((W, H))
        if W < H:
            W = resolution
            H = MultiaspectImage._round_to_nearest_multiple(
                resolution / aspect_ratio, 8
            )
        elif H < W:
            H = resolution
            W = MultiaspectImage._round_to_nearest_multiple(
                resolution * aspect_ratio, 8
            )
        else:
            W = H = MultiaspectImage._round_to_nearest_multiple(resolution, 8)

        new_aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio((W, H))
        return int(W), int(H), new_aspect_ratio

    @staticmethod
    def calculate_new_size_by_pixel_area(W: int, H: int, megapixels: float):
        # Calculate initial dimensions based on aspect ratio and target megapixels
        aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio((W, H))
        total_pixels = max(megapixels * 1e6, 1e6)
        W_initial = int(round((total_pixels * aspect_ratio) ** 0.5))
        H_initial = int(round((total_pixels / aspect_ratio) ** 0.5))

        # Ensure divisibility by 64 for both dimensions with minimal adjustment
        def adjust_for_divisibility(n):
            return (n + 63) // 64 * 64

        W_adjusted = adjust_for_divisibility(W_initial)
        H_adjusted = adjust_for_divisibility(H_initial)

        # Ensure the adjusted dimensions meet the megapixel requirement
        while W_adjusted * H_adjusted < total_pixels:
            W_adjusted += 64
            H_adjusted = adjust_for_divisibility(int(round(W_adjusted / aspect_ratio)))

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
            rounding (int): The number of decimal places to round the aspect ratio to.

        Returns:
            float: The rounded aspect ratio of the image.
        """
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, tuple):
            width, height = image
        else:
            width, height = image.size
        aspect_ratio = round(width / height, rounding)
        return aspect_ratio

    @staticmethod
    def determine_bucket_for_aspect_ratio(aspect_ratio, rounding: int = 3):
        """
        Determine the correct bucket for a given aspect ratio.

        Args:
            aspect_ratio (float): The aspect ratio of an image.

        Returns:
            str: The bucket corresponding to the aspect ratio.
        """
        # The logic for determining the bucket can be based on the aspect ratio directly
        return str(round(aspect_ratio, rounding))
