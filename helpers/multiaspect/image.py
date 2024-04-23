from torchvision import transforms
from helpers.image_manipulation.brightness import calculate_luminance
from io import BytesIO
from PIL import Image
from PIL.ImageOps import exif_transpose
import logging, os, random
from math import sqrt
from helpers.training.state_tracker import StateTracker

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
        if resolution_type == "area":
            original_resolution = original_resolution * 1e3
        # Make resolution a multiple of StateTracker.get_args().aspect_bucket_alignment
        original_resolution = MultiaspectImage._round_to_nearest_multiple(
            original_resolution
        )

        # Downsample before we handle, if necessary.
        downsample_before_crop = False
        crop = StateTracker.get_data_backend_config(data_backend_id=id).get(
            "crop", False
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
        original_aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(
            (original_width, original_height)
        )
        if resolution_type == "pixel":
            (target_width, target_height, new_aspect_ratio) = (
                MultiaspectImage.calculate_new_size_by_pixel_edge(
                    original_aspect_ratio, resolution
                )
            )
        elif resolution_type == "area":
            (target_width, target_height, new_aspect_ratio) = (
                MultiaspectImage.calculate_new_size_by_pixel_area(
                    original_aspect_ratio, resolution
                )
            )
            # Convert 'resolution' from eg. "1 megapixel" to "1024 pixels"
            resolution = resolution * 1e3
            # Make resolution a multiple of StateTracker.get_args().aspect_bucket_alignment
            resolution = MultiaspectImage._round_to_nearest_multiple(resolution)
            logger.debug(
                f"After area resize, our image will be {target_width}x{target_height} with an overridden resolution of {resolution} pixels."
            )
        else:
            raise ValueError(f"Unknown resolution type: {resolution_type}")

        crop_style = StateTracker.get_data_backend_config(data_backend_id=id).get(
            "crop_style", "random"
        )
        crop_aspect = StateTracker.get_data_backend_config(data_backend_id=id).get(
            "crop_aspect", "square"
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
                            original_aspect_ratio,
                            original_megapixel_resolution,
                        )
                    )
                elif resolution_type == "pixel":
                    (target_width, target_height, new_aspect_ratio) = (
                        MultiaspectImage.calculate_new_size_by_pixel_edge(
                            original_aspect_ratio, original_resolution
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
                elif crop_style == "face":
                    image, crop_coordinates = MultiaspectImage._crop_face(
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
                elif crop_style == "random" or crop_style == "face":
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
    def _crop_face(
        image: Image,
        target_width: int,
        target_height: int,
    ):
        """Crop the image to include a face, or the most 'interesting' part of the image, without a face."""
        # Import modules
        import cv2
        import numpy as np

        # Detect a face in the image
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        image = image.convert("RGB")
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            # Get the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            left = max(0, x - 0.5 * w)
            top = max(0, y - 0.5 * h)
            right = min(image.shape[1], x + 1.5 * w)
            bottom = min(image.shape[0], y + 1.5 * h)
            image = Image.fromarray(image)
            return image.crop((left, top, right, bottom)), (left, top)
        else:
            # Crop the image from a random position
            return MultiaspectImage._crop_random(image, target_width, target_height)

    @staticmethod
    def _round_to_nearest_multiple(value):
        """Round a value to the nearest multiple."""
        multiple = StateTracker.get_args().aspect_bucket_alignment
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
        total_pixels = max(megapixels * 1e6, 1e6)
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
        if not isinstance(image, int):
            to_round = rounding
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, tuple):
            width, height = image
        elif isinstance(image, float):
            return round(image, to_round)
        else:
            width, height = image.size
        aspect_ratio = round(width / height, to_round)
        return aspect_ratio

    @staticmethod
    def determine_bucket_for_aspect_ratio(aspect_ratio):
        """
        Determine the correct bucket for a given aspect ratio.

        Args:
            aspect_ratio (float): The aspect ratio of an image.

        Returns:
            str: The bucket corresponding to the aspect ratio.
        """
        # The logic for determining the bucket can be based on the aspect ratio directly
        return str(aspect_ratio)
