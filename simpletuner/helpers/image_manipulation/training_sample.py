try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
import logging
import os
import random
import time
from math import sqrt

import numpy as np
from diffusers.utils.export_utils import export_to_gif
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm import tqdm

from simpletuner.helpers.image_manipulation.batched_training_samples import BatchedTrainingSamples
from simpletuner.helpers.image_manipulation.cropping import crop_handlers
from simpletuner.helpers.multiaspect.image import MultiaspectImage, resize_helpers
from simpletuner.helpers.multiaspect.video import resize_video_frames
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class TrainingSample:
    def __init__(
        self,
        image: Image.Image,
        data_backend_id: str,
        image_metadata: dict = None,
        image_path: str = None,
        conditioning_type: str = None,
        model=None,
    ):
        # transforms normalize pixels for VAE
        self.model = model
        self.transforms = None
        self.caption = None
        self.batch_processor = BatchedTrainingSamples()
        if model is None:
            self.model = StateTracker.get_model()
        if self.model is not None:
            self.transforms = self.model.get_transforms()
        self.image = image
        self.target_size = None
        self.intermediary_size = None
        self.original_size = None
        self.conditioning_type = conditioning_type
        self.data_backend_id = data_backend_id
        self.image_metadata = (
            image_metadata if image_metadata else StateTracker.get_metadata_by_filepath(image_path, data_backend_id)
        )
        if isinstance(image, np.ndarray):
            if len(image.shape) == 4:
                logger.debug(f"Received 4D Shape: {image.shape}")
                self.original_size = (
                    image.shape[2],
                    image.shape[1],
                )  # mapping image.shape (F, H, W, C) to (W, H)
            elif len(image.shape) == 5:
                raise ValueError(f"Received invalid shape: {image.shape}, expected 4D item instead")

            self.original_aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(self.original_size)
        elif hasattr(image, "size"):
            self.original_size = self.image.size
            self.original_aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(self.original_size)
        elif image_metadata is not None:
            self.original_size = image_metadata.get("original_size")
            self.original_aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(self.original_size)
        self.current_size = self.original_size

        if not self.original_size:
            raise Exception(f"Original size not found in metadata: {image_metadata}")

        # backend config
        self.data_backend_config = StateTracker.get_data_backend_config(data_backend_id)
        self.crop_enabled = self.data_backend_config.get("crop", False)
        self.crop_style = self.data_backend_config.get("crop_style", "random")
        self.crop_aspect = self.data_backend_config.get("crop_aspect", "square")
        self.crop_aspect_buckets = self.data_backend_config.get("crop_aspect_buckets", [])
        self.crop_coordinates = (0, 0)
        crop_handler_cls = crop_handlers.get(self.crop_style)
        if not crop_handler_cls:
            raise ValueError(f"Unknown crop style: {self.crop_style}")
        self.cropper = crop_handler_cls(image=self.image, image_metadata=image_metadata)
        self.resolution = self.data_backend_config.get("resolution")
        self.resolution_type = self.data_backend_config.get("resolution_type")
        self.target_size_calculator = resize_helpers.get(self.resolution_type)
        if self.target_size_calculator is None and conditioning_type not in [
            "mask",
            "controlnet",
        ]:
            raise ValueError(f"Unknown resolution type: {self.resolution_type}")
        self._set_resolution()
        self.target_downsample_size = self.data_backend_config.get("target_downsample_size", None)
        self.maximum_image_size = self.data_backend_config.get("maximum_image_size", None)
        self._image_path = image_path
        # RGB/EXIF conversion
        self.correct_image()
        self._validate_image_metadata()

    def save_debug_image(self, path: str):
        if self.image is not None:
            if os.environ.get("SIMPLETUNER_DEBUG_IMAGE_PREP", "") == "true":
                if hasattr(self.image, "save"):
                    self.image.save(path)
                else:
                    # switch .png to .mp4
                    if path.endswith(".png"):
                        path = path.replace(".png", ".mp4")
                    logger.debug(f"Not saving debug video output: {path}")
                    # write to path
                    from io import BytesIO

                    import imageio

                    video_byte_array = BytesIO()
                    imageio.v3.imwrite(
                        video_byte_array,
                        self.image,  # a list of NumPy arrays
                        plugin="pyav",  # or "ffmpeg"
                        fps=StateTracker.get_args().framerate,
                        extension=".mp4",
                        codec="libx264",
                    )
                    video_byte_array.seek(0)
                    with open(path, "wb") as f:
                        f.write(video_byte_array.read())

        return self

    @staticmethod
    def from_image_path(image_path: str, data_backend_id: str):
        data_backend = StateTracker.get_data_backend(data_backend_id)
        image = data_backend["data_backend"].read_image(image_path)
        return TrainingSample(image, data_backend_id, image_path=image_path)

    def training_sample_path(self, training_dataset_id: str) -> str:
        """get primary training sample path for conditioning sample"""
        training_backend = StateTracker.get_data_backend(training_dataset_id)
        cond_backend = StateTracker.get_data_backend(self.data_backend_id)
        if training_backend is None:
            raise ValueError(f"No training dataset registered for backend “{training_dataset_id}”.")
        training_data_dir = training_backend["config"]["instance_data_dir"]
        cond_data_dir = cond_backend["config"]["instance_data_dir"]
        cond_relpath = self._image_path.replace(cond_data_dir, training_data_dir, 1)
        if not cond_relpath:
            raise ValueError("Cannot determine training sample path: no image path provided.")
        training_sample_path = training_backend["data_backend"].get_abs_path(cond_relpath)

        return training_sample_path

    def _validate_image_metadata(self) -> bool:
        """check if metadata has required keys to skip calculations
        preserves randomized aspect buckets across runs to avoid tensor shape mismatches
        """
        required_keys = [
            "original_size",
            "target_size",
            "intermediary_size",
            "crop_coordinates",
            "aspect_ratio",
        ]
        if type(self.image_metadata) is not dict:
            self.valid_metadata = False
        else:
            self.valid_metadata = all(key in self.image_metadata for key in required_keys)
        if self.valid_metadata:
            self.original_size = self.image_metadata["original_size"]
            self.target_size = self.image_metadata["target_size"]
            self.intermediary_size = self.image_metadata["intermediary_size"]
            self.crop_coordinates = self.image_metadata["crop_coordinates"]
            self.aspect_ratio = self.image_metadata["aspect_ratio"]

        self.original_aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(self.original_size)

        if not self.valid_metadata and hasattr(self.image, "size") and isinstance(self.image, Image.Image):
            self.original_size = self.image.size

        return self.valid_metadata

    def set_caption(self, caption: str) -> None:
        self.caption = caption

    def _set_resolution(self):
        if self.resolution_type == "pixel":
            self.target_area = self.resolution
            # Store the pixel value, eg. 1024
            self.pixel_resolution = int(self.resolution)
            # Store the megapixel value, eg. 1.0
            self.megapixel_resolution = self.resolution / 1e3
        elif self.resolution_type == "area":
            # Convert pixel area to megapixels, remapping commonly used round values
            # to their pixel_area equivalents for compatibility purposes.
            resolution_map = {
                0.25: 512**2,
                0.5: 768**2,
                1.0: 1024**2,
                2.0: 1536**2,
                4.0: 2048**2,
            }
            # Find the closest match within a small tolerance
            target_area = None
            for key, value in resolution_map.items():
                if abs(self.resolution - key) < 0.05:  # Allow 0.05 tolerance
                    target_area = value
                    break

            if target_area is None:
                target_area = self.resolution * 1e6

            self.target_area = target_area

            # Store the pixel value, eg. 1024
            self.pixel_resolution = int(MultiaspectImage._round_to_nearest_multiple(sqrt(self.target_area)))
            # Store the megapixel value, eg. 1.0
            self.megapixel_resolution = self.resolution
        else:
            raise Exception(f"Unknown resolution type: {self.resolution_type}")

    def _trim_aspect_bucket_list(self):
        """filter buckets to avoid upscaling >20%"""
        available_buckets = []
        for bucket in self.crop_aspect_buckets:
            # skip buckets that would upscale >20%
            if type(bucket) is dict:
                aspect = bucket["aspect_ratio"]
            elif type(bucket) is float or type(bucket) is int:
                aspect = bucket
            else:
                raise ValueError("Aspect buckets must be a list of floats or dictionaries.")
            # Calculate new size
            target_size, _, _ = self.target_size_calculator(aspect, self.resolution, self.original_size)
            # check 20% upscale threshold
            if target_size[0] * 1.2 < self.original_size[0] and target_size[1] * 1.2 < self.original_size[1]:
                available_buckets.append(aspect)
        return available_buckets

    def _select_random_aspect(self):
        """select aspect bucket based on crop_aspect config (closest/random)"""
        if not self.crop_aspect_buckets:
            raise ValueError("Aspect buckets are not defined in the data backend config.")

        if self.valid_metadata:
            self.aspect_ratio = self.image_metadata["aspect_ratio"]
            return self.aspect_ratio

        # Handle 'preserve' crop_aspect mode by picking the closest aspect ratio
        if self.crop_aspect == "closest":
            closest_aspect = min(
                self.crop_aspect_buckets,
                key=lambda bucket: abs((bucket["aspect"] if isinstance(bucket, dict) else bucket) - self.aspect_ratio),
            )
            closest_aspect_value = closest_aspect["aspect"] if isinstance(closest_aspect, dict) else closest_aspect
            # logger.debug(f"Selected closest aspect: {closest_aspect_value} for aspect ratio: {self.aspect_ratio}")
            return closest_aspect_value

        # Handle 'random' crop_aspect mode by picking a random aspect ratio based on weights
        if self.crop_aspect == "random":
            if len(self.crop_aspect_buckets) > 0 and type(self.crop_aspect_buckets[0]) is dict:
                has_portrait_buckets = any(bucket["aspect"] < 1.0 for bucket in self.crop_aspect_buckets)
                has_landscape_buckets = any(bucket["aspect"] > 1.0 for bucket in self.crop_aspect_buckets)
                logger.error(f"has_portrait_buckets: {has_portrait_buckets}, has_landscape_buckets: {has_landscape_buckets}")

                # Instead of defaulting to 1.0, use whatever buckets are available
                aspects = [bucket["aspect"] for bucket in self.crop_aspect_buckets]
                weights = [bucket["weight"] for bucket in self.crop_aspect_buckets]

                # Ensure that the weights add up to 1.0
                total_weight = sum(weights)
                if total_weight != 1.0:
                    raise ValueError("The weights of aspect buckets must add up to 1.")

                selected_aspect = random.choices(aspects, weights)[0]
                return selected_aspect

            elif len(self.crop_aspect_buckets) > 0 and type(self.crop_aspect_buckets[0]) is float:
                available_aspects = self._trim_aspect_bucket_list()
                if len(available_aspects) == 0:
                    selected_aspect = 1.0
                    if should_log():
                        tqdm.write(
                            "[WARNING] Image dimensions do not fit into the configured aspect buckets. Using square crop."
                        )
                else:
                    selected_aspect = random.choice(available_aspects)
                return selected_aspect

            else:
                raise ValueError(
                    "Aspect buckets must be a list of floats or dictionaries."
                    " If using a dictionary, it is expected to be in the format {'aspect': 1.0, 'weight': 0.5}."
                    " To provide multiple aspect ratios, use a list of dictionaries: [{'aspect': 1.0, 'weight': 0.5}, {'aspect': 1.5, 'weight': 0.5}]."
                )

        # Default to 1.0 if none of the conditions above match
        return 1.0

    def prepare_like(self, other_sample, return_tensor=False):
        """
        Prepare the current TrainingSample in the same way as other_sample.

        Args:
            other_sample (TrainingSample): The sample to mimic.
            return_tensors (bool): Whether to return tensors.

        Returns:
            PreparedSample: The prepared sample.
        """
        if other_sample.image_metadata:
            self.image_metadata = other_sample.image_metadata.copy()
        # copy derived geometry so prepare() skips recalculation
        self.original_size = other_sample.original_size
        self.intermediary_size = other_sample.intermediary_size
        self.target_size = other_sample.target_size
        self.crop_coordinates = other_sample.crop_coordinates
        self.aspect_ratio = other_sample.aspect_ratio
        self._validate_image_metadata()

        return self.prepare(return_tensor=return_tensor)

    def prepare(self, return_tensor: bool = False):
        """
        Perform initial image preparations such as converting to RGB and applying EXIF transformations.

        Args:
            return_tensor (bool): Whether to return tensors.

        Returns: tuple
            - image data (PIL.Image)
            - crop_coordinates (tuple)
            - aspect_ratio (float)
        """
        self.save_debug_image(f"images/{time.time()}-0-original.png")
        self.crop()
        self.save_debug_image(f"images/{time.time()}-1-cropped.png")
        if not self.crop_enabled:
            self.save_debug_image(f"images/{time.time()}-1b-nocrop-resize.png")
            self.resize()
            self.save_debug_image(f"images/{time.time()}-2-final-output.png")

        image = self.image
        if return_tensor and self.transforms is not None:
            # Return normalised tensor.
            image = self.transforms(image)
        webhook_handler = StateTracker.get_webhook_handler()

        # For square crops, ensure aspect ratio is exactly 1.0
        final_aspect_ratio = self.aspect_ratio
        if self.crop_enabled and self.crop_aspect == "square":
            final_aspect_ratio = 1.0

        prepared_sample = PreparedSample(
            image=image,
            original_size=self.original_size,
            crop_coordinates=self.crop_coordinates,
            aspect_ratio=final_aspect_ratio,  # Use the corrected aspect ratio
            image_metadata=self.image_metadata,
            target_size=self.target_size,
            intermediary_size=self.intermediary_size,
        )
        if webhook_handler:
            # Send the debug message with the image using send() (following validation pattern)
            webhook_handler.send(
                message=f"Debug info for prepared sample, {str(prepared_sample)}",
                images=[self.image],
                message_level="debug",
            )

            # Send structured data using send_raw() with proper event structure
            from simpletuner.helpers.webhooks.events import notification_event

            event = notification_event(
                message=f"Prepared sample debug info",
                severity="debug",
                job_id=StateTracker.get_job_id(),
                data={
                    "prepared_sample": {
                        "original_size": prepared_sample.original_size,
                        "target_size": prepared_sample.target_size,
                        "aspect_ratio": prepared_sample.aspect_ratio,
                        "crop_coordinates": prepared_sample.crop_coordinates,
                    }
                },
            )
            webhook_handler.send_raw(event, message_level="debug", job_id=StateTracker.get_job_id())
        return prepared_sample

    def area(self) -> int:
        """
        Calculate the area of the image.

        Returns:
            int: The area of the image.
        """
        if self.image is not None:
            if isinstance(self.image, np.ndarray):
                # it's a numpy array of frames, probably?
                if len(self.image.shape) == 4:
                    # frames, height, width, channels (195, 360, 640, 3) as an example
                    return self.image.shape[2] * self.image.shape[1]
                else:
                    raise NotImplementedError(f"NumPy array shape not supported: {self.image.shape}")
            elif hasattr(self.image, "size") and isinstance(self.image.size, tuple):
                return self.image.size[0] * self.image.size[1]
        if self.original_size:
            return self.original_size[0] * self.original_size[1]

    def _should_resize_before_crop(self) -> bool:
        """
        If the options to do so are enabled, or, the image require it; we will resize before cropping.

        Returns:
            bool: True if the image should be resized before cropping, False otherwise.
        """
        if not self.crop_enabled or not self.maximum_image_size or not self.target_downsample_size:
            return False
        if self.data_backend_config.get("resolution_type") == "pixel":
            return (self.current_size[0] > self.pixel_resolution or self.current_size[1] > self.pixel_resolution) or (
                self.current_size[0] < self.pixel_resolution or self.current_size[1] < self.pixel_resolution
            )
        elif self.data_backend_config.get("resolution_type") == "area":
            should_resize = (
                self.area() > self.target_area
                or self.area() < self.target_area
                or self.current_size[0] < self.target_size[0]
                or self.current_size[1] < self.target_size[1]
            )
            logger.debug(f"Should resize? {should_resize}")
            return should_resize
        else:
            raise ValueError(f"Unknown resolution type: {self.data_backend_config.get('resolution_type')}")

    def _limit_maximum_size(self, size_to_check: tuple) -> tuple:
        """
        If self.model.MAXIMUM_CANVAS_SIZE is not None, we have to limit the area of the image to this value.

        The image aspect ratio must be preserved, as well as the bucket alignment interval (eg. 64px)

        Args:
            size_to_check (tuple): The current size of the image as (width, height).
        Returns:
            tuple: The limited size as (width, height).
        """
        if self.model is None or self.model.MAXIMUM_CANVAS_SIZE is None:
            logger.debug("No canvas size constraint required, value is None.")
            return size_to_check
        max_size = self.model.MAXIMUM_CANVAS_SIZE
        width, height = size_to_check
        canvas_size = width * height
        if canvas_size <= max_size:
            # no adjustment needed, we're good to go.
            logger.debug(f"No canvas size constraint required for {size_to_check}.")
            return size_to_check

        # Calculate the scale factor to fit within the maximum canvas size
        scale_factor = sqrt(max_size / canvas_size)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        # Ensure the new dimensions are divisible by 8 or 64, depending on the model's requirements
        new_width = MultiaspectImage._round_to_nearest_multiple(new_width)
        new_height = MultiaspectImage._round_to_nearest_multiple(new_height)
        new_canvas_size = new_width * new_height
        if new_canvas_size > max_size:
            new_canvas_details = MultiaspectImage.limit_canvas_size(width=new_width, height=new_height, max_size=max_size)
            new_width, new_height, new_canvas_size = (
                new_canvas_details["width"],
                new_canvas_details["height"],
                new_canvas_details["canvas_size"],
            )
        logger.debug(
            f"Canvas size constraint applied: {size_to_check} -> ({new_width}, {new_height}). "
            f"Original canvas: {canvas_size}, New canvas: {new_canvas_size}, "
            f"Limit: {max_size}"
        )

        return new_width, new_height

    def _calculate_target_downsample_size(self):
        """
        When cropping images, it is optional to disturb them with a resize before the crop.
        This is desirable when a large image is being cropped to a small size, as it will preserve scene details and maintain aspect ratio.

        Returns:
            tuple: The target downsample size as (width, height).
        """
        # We'll run the target size calculator logic without updating any of the object attributes.
        # This will prevent contamination of the final values that the image will represent.
        _, calculated_intermediary_size, _ = self.target_size_calculator(
            self.original_aspect_ratio, self.target_downsample_size, self.original_size
        )
        # The calculated_intermediary_size's purpose is to resize to this value before cropping to target_size.
        # If the intermediary size is smaller than target_size on either edge, the cropping will result in black bars.
        # We have to calculate the scale factor and adjust the image edges proportionally to avoid squishing it.
        if calculated_intermediary_size[0] < self.target_size[0]:
            scale_factor = self.target_size[0] / calculated_intermediary_size[0]
            calculated_intermediary_size = (
                self.target_size[0],
                int(calculated_intermediary_size[1] * scale_factor),
            )
        elif calculated_intermediary_size[1] < self.target_size[1]:
            scale_factor = self.target_size[1] / calculated_intermediary_size[1]
            calculated_intermediary_size = (
                int(calculated_intermediary_size[0] * scale_factor),
                self.target_size[1],
            )

        return calculated_intermediary_size

    def _downsample_before_crop(self):
        """
        Downsample the image before cropping, to preserve scene details and maintain aspect ratio.

        Returns:
            TrainingSample: The current TrainingSample instance.
        """
        if self._should_resize_before_crop():
            target_downsample_size = self._calculate_target_downsample_size()
            logger.debug(f"Calculated target_downsample_size, resizing to {target_downsample_size}")
            self.resize(target_downsample_size)
        return self

    def correct_intermediary_square_size(self):
        """
        When an intermediary size is calculated, we don't adjust it to be divisible by 8 or 64.
        However, the aspect ratio 1.0 needs special consideration for our base resolutions 512, 768, and 1024, because they typically result in 500x500, 750x750, and 1000x1000 images.

        Returns:
            TrainingSample: The current TrainingSample instance.
        """
        if self.aspect_ratio == 1.0 and self.intermediary_size[0] < self.pixel_resolution:
            self.intermediary_size = (
                self.pixel_resolution,
                self.pixel_resolution,
            )
            self.crop_coordinates = (0, 0)
        return self

    def calculate_target_size(self):
        """
        This method will populate the values for self.{target_size,intermediary_size,aspect_ratio} based on the image's original size and the data backend configuration.

        Returns:
            tuple:
                - The target size as (width, height).
                - The intermediary size as (width, height).
                - The aspect ratio of the target size. This will likely be different from the original aspect ratio.
        """
        self.aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(self.original_size)
        is_square_crop = False  # Track if we want square output

        if self.crop_enabled:
            if self.crop_aspect == "square":
                is_square_crop = True
                self.target_size = (self.pixel_resolution, self.pixel_resolution)
                # ensure the area isn't past the allowed range.
                self.target_size = self._limit_maximum_size(self.target_size)

                # Force square dimensions after limiting
                if self.target_size[0] != self.target_size[1]:
                    min_dim = min(self.target_size[0], self.target_size[1])
                    self.target_size = (min_dim, min_dim)

                _, self.intermediary_size, _ = self.target_size_calculator(
                    self.aspect_ratio, self.resolution, self.original_size
                )
                self.aspect_ratio = 1.0
                self.correct_intermediary_square_size()
                square_crop_metadata = (
                    self.target_size,
                    self.intermediary_size,
                    self.aspect_ratio,
                )
                return square_crop_metadata

        if self.crop_enabled and (self.crop_aspect == "random" or self.crop_aspect == "closest"):
            # Grab a random aspect ratio from a list.
            self.aspect_ratio = self._select_random_aspect()

        self.target_size, calculated_intermediary_size, self.aspect_ratio = self.target_size_calculator(
            self.aspect_ratio, self.resolution, self.original_size
        )
        self.target_size = self._limit_maximum_size(self.target_size)

        if (self.crop_enabled and self.crop_aspect != "random") or not self.valid_metadata:
            self.intermediary_size = calculated_intermediary_size

        # Only recalculate aspect ratio if it's not a square crop
        if not is_square_crop:
            self.aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(self.target_size)
        else:
            self.aspect_ratio = 1.0

        self.correct_intermediary_square_size()

        if self.aspect_ratio == 1.0:
            self.target_size = (self.pixel_resolution, self.pixel_resolution)

        return (
            self.target_size,
            (int(self.intermediary_size[0]), int(self.intermediary_size[1])),
            self.aspect_ratio,
        )

    def correct_image(self):
        """
        Apply a series of transformations to the image to "correct" it, such as EXIF rotation and conversion to RGB.

        Returns:
            TrainingSample: The current TrainingSample instance.
        """
        if self.image is not None and hasattr(self.image, "convert"):
            # Convert image to RGB to remove any alpha channel and apply EXIF data transformations
            self.image = self.image.convert("RGB")
            self.image = exif_transpose(self.image)
        return self

    def crop(self):
        """
        Crop the image using trainingsample batch operations when possible, or crop handler as fallback.
        If cropping is not enabled, we do nothing.

        Returns:
            TrainingSample: The current TrainingSample instance.
        """
        if not self.crop_enabled:
            return self
        self.calculate_target_size()
        self._downsample_before_crop()
        self.save_debug_image(f"images/{time.time()}-0.5-downsampled.png")

        # Try to use trainingsample for efficient cropping when possible
        if self.image is not None and isinstance(self.image, Image.Image) and self.crop_style in ["center", "random"]:
            try:
                img_array = np.array(self.image)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    if self.crop_style == "center":
                        # Use BatchedTrainingSamples center crop
                        cropped_arrays = self.batch_processor.batch_center_crop_images([img_array], [self.target_size])
                        if cropped_arrays and len(cropped_arrays) > 0:
                            self.image = Image.fromarray(cropped_arrays[0])
                            # Calculate crop coordinates for center crop
                            original_size = img_array.shape[:2]  # (H, W)
                            crop_x = (original_size[1] - self.target_size[0]) // 2
                            crop_y = (original_size[0] - self.target_size[1]) // 2
                            self.crop_coordinates = (crop_x, crop_y)
                            self.current_size = self.target_size
                            logger.debug(f"Used trainingsample center crop: {self.target_size}")
                            return self
                    elif self.crop_style == "random":
                        # Use BatchedTrainingSamples random crop
                        cropped_arrays = self.batch_processor.batch_random_crop_images([img_array], [self.target_size])
                        if cropped_arrays and len(cropped_arrays) > 0:
                            self.image = Image.fromarray(cropped_arrays[0])
                            # Note: We can't get exact coordinates from random crop, so estimate
                            original_size = img_array.shape[:2]  # (H, W)
                            max_x = original_size[1] - self.target_size[0]
                            max_y = original_size[0] - self.target_size[1]
                            # Use a placeholder since we don't know the exact random coordinates
                            self.crop_coordinates = (max_x // 2, max_y // 2)
                            self.current_size = self.target_size
                            logger.debug(f"Used trainingsample random crop: {self.target_size}")
                            return self
            except Exception as e:
                logger.debug(f"Trainingsample crop failed, falling back to crop handler: {e}")

        # Fallback to original crop handler
        if self.image is not None:
            self.cropper.set_image(self.image)
        self.cropper.set_intermediary_size(self.current_size[0], self.current_size[1])
        self.image, self.crop_coordinates = self.cropper.crop(self.target_size[0], self.target_size[1])
        self.current_size = self.target_size
        return self

    def resize(self, size: tuple = None):
        _ = self.image.size if self.image is not None else self.original_size
        if size is None:
            if not self.valid_metadata:
                self.target_size, self.intermediary_size, self.target_aspect_ratio = self.calculate_target_size()
            size = self.target_size
            if self.target_size != self.intermediary_size:
                logger.debug(
                    f"we have to crop because target size {self.target_size} != intermediary size {self.intermediary_size}"
                )
                # Now we can resize the image to the intermediary size.
                self.current_size = self.intermediary_size
                if self.image is not None:
                    if isinstance(self.image, Image.Image):
                        # Use trainingsample for efficient resizing
                        try:
                            img_array = np.array(self.image)
                            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                                resized_arrays = self.batch_processor.batch_resize_images(
                                    [img_array], [self.intermediary_size]
                                )
                                if resized_arrays and len(resized_arrays) > 0:
                                    self.image = Image.fromarray(resized_arrays[0])
                                else:
                                    self.image = self.image.resize(self.intermediary_size, Image.Resampling.LANCZOS)
                            else:
                                self.image = self.image.resize(self.intermediary_size, Image.Resampling.LANCZOS)
                        except Exception as e:
                            logger.debug(f"Trainingsample resize failed, falling back to PIL: {e}")
                            self.image = self.image.resize(self.intermediary_size, Image.Resampling.LANCZOS)
                        self.current_size = self.image.size
                    elif isinstance(self.image, np.ndarray):
                        # we have a video to resize - use trainingsample
                        logger.debug(f"Resizing {self.image.shape} to {self.intermediary_size}, ")
                        try:
                            if len(self.image.shape) == 4:  # (T, H, W, C)
                                resized_videos = self.batch_processor.batch_resize_videos(
                                    [self.image], [self.intermediary_size]
                                )
                                if resized_videos and len(resized_videos) > 0:
                                    self.image = resized_videos[0]
                                else:
                                    self.image = resize_video_frames(
                                        self.image,
                                        (
                                            self.intermediary_size[0],
                                            self.intermediary_size[1],
                                        ),
                                    )
                            else:
                                self.image = resize_video_frames(
                                    self.image,
                                    (
                                        self.intermediary_size[0],
                                        self.intermediary_size[1],
                                    ),
                                )
                        except Exception as e:
                            logger.debug(f"Trainingsample video resize failed, falling back: {e}")
                            self.image = resize_video_frames(
                                self.image,
                                (self.intermediary_size[0], self.intermediary_size[1]),
                            )
                        width, height = (
                            self.image.shape[2],
                            self.image.shape[1],
                        )  # shape (F, H, W, C)
                        self.current_size = (width, height)
                        logger.debug(f"Post resize: {self.current_size} / {self.image.shape}")
                if self.image is not None and self.cropper:
                    self.cropper.set_image(self.image)
                self.cropper.set_intermediary_size(self.intermediary_size[0], self.intermediary_size[1])
                self.image, self.crop_coordinates = self.cropper.crop(self.target_size[0], self.target_size[1])
                logger.debug(
                    f"Cropped to {self.target_size} via crop coordinates {self.crop_coordinates} (resulting in current_size of {self.current_size})"
                )
                self.current_size = self.target_size
                logger.debug(f"crop coordinates: {self.crop_coordinates}")
                return self

        if self.image is not None and hasattr(self.image, "resize"):
            if isinstance(self.image, Image.Image):
                # Use trainingsample for efficient resizing
                try:
                    img_array = np.array(self.image)
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        # Use BatchedTrainingSamples for better performance
                        resized_arrays = self.batch_processor.batch_resize_images([img_array], [size])
                        if resized_arrays and len(resized_arrays) > 0:
                            self.image = Image.fromarray(resized_arrays[0])
                        else:
                            # Fallback to PIL resize
                            self.image = self.image.resize(size, Image.Resampling.LANCZOS)
                    else:
                        # Fallback to PIL resize for non-standard formats
                        self.image = self.image.resize(size, Image.Resampling.LANCZOS)
                except Exception as e:
                    logger.debug(f"Trainingsample resize failed, falling back to PIL: {e}")
                    self.image = self.image.resize(size, Image.Resampling.LANCZOS)

                self.aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(self.image.size)
            elif isinstance(self.image, np.ndarray):
                # we have a video to resize - use trainingsample for videos
                logger.debug(f"Resizing {self.image.shape} to {size}, ")
                try:
                    # For video arrays, use trainingsample batch resize
                    if len(self.image.shape) == 4:  # (T, H, W, C)
                        resized_videos = self.batch_processor.batch_resize_videos([self.image], [size])
                        if resized_videos and len(resized_videos) > 0:
                            self.image = resized_videos[0]
                        else:
                            # Fallback to original method
                            self.image = resize_video_frames(self.image, (size[0], size[1]))
                    else:
                        # Fallback for unexpected shapes
                        self.image = resize_video_frames(self.image, (size[0], size[1]))
                except Exception as e:
                    logger.debug(f"Trainingsample video resize failed, falling back: {e}")
                    self.image = resize_video_frames(self.image, (size[0], size[1]))

                width, height = self.image.shape[2], self.image.shape[1]
                self.current_size = (width, height)
                self.aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(size)
                logger.debug(f"Now {self.image.shape} @ {self.aspect_ratio}")
        self.current_size = size
        return self

    def get_image(self):
        """get current image (may be None during parquet bucketing)"""
        return self.image

    def is_conditioning_sample(self):
        return self.conditioning_type is not None

    def get_conditioning_type(self):
        return self.conditioning_type

    def cache_path(self):
        """get VAE cache path for image"""
        vae_cache = StateTracker.get_data_backend(self.data_backend_id)["vaecache"]

        return vae_cache.image_path_to_vae_path.get(self._image_path, None)

    def image_path(self, basename_only=False):
        """
        Returns the absolute or basename path for the current training sample.

        Args:
            basename_only (bool): Whether to return the basename only.
        Returns:
            str: The image path
        """
        if basename_only:
            return os.path.basename(self._image_path)
        return self._image_path


class PreparedSample:
    def __init__(
        self,
        image: Image.Image,
        image_metadata: dict,
        original_size: tuple,
        intermediary_size: tuple,
        target_size: tuple,
        aspect_ratio: float,
        crop_coordinates: tuple,
    ):
        self.image = image
        self.image_metadata = image_metadata if image_metadata else {}
        self.original_size = original_size
        self.intermediary_size = intermediary_size
        self.target_size = target_size
        self.aspect_ratio = aspect_ratio
        self.crop_coordinates = crop_coordinates

    def __str__(self):
        return f"PreparedSample(image={self.image}, original_size={self.original_size}, intermediary_size={self.intermediary_size}, target_size={self.target_size}, aspect_ratio={self.aspect_ratio}, crop_coordinates={self.crop_coordinates})"

    def to_dict(self):
        return {
            "image": self.image,
            "original_size": self.original_size,
            "intermediary_size": self.intermediary_size,
            "target_size": self.target_size,
            "aspect_ratio": self.aspect_ratio,
            "crop_coordinates": self.crop_coordinates,
        }
