from PIL import Image
from PIL.ImageOps import exif_transpose
from helpers.multiaspect.image import MultiaspectImage, resize_helpers
from helpers.multiaspect.image import crop_handlers
from helpers.training.state_tracker import StateTracker
import logging

logger = logging.getLogger(__name__)


class TrainingSample:
    def __init__(
        self, image: Image.Image, data_backend_id: str, image_metadata: dict = None
    ):
        """
        Initializes a new TrainingSample instance with a provided PIL.Image object and a data backend identifier.

        Args:
        image (Image.Image): A PIL Image object.
        data_backend_id (str): Identifier for the data backend used for additional operations.
        metadata (dict): Optional metadata associated with the image.
        """
        self.image = image
        self.data_backend_id = data_backend_id
        self.image_metadata = image_metadata if image_metadata else {}
        if hasattr(image, "size"):
            self.original_size = self.image.size
        elif image_metadata is not None:
            self.original_size = image_metadata.get("original_size")
            logger.debug(
                f"Metadata for training sample given instead of image? {image_metadata}"
            )

        if not self.original_size:
            raise Exception("Original size not found in metadata.")

        # Torchvision transforms turn the pixels into a Tensor and normalize them for the VAE.
        self.transforms = MultiaspectImage.get_image_transforms()
        # EXIT, RGB conversions.
        self.correct_image()

        # Backend config details
        self.data_backend_config = StateTracker.get_data_backend_config(data_backend_id)
        self.crop_enabled = self.data_backend_config.get("crop", False)
        self.crop_style = self.data_backend_config.get("crop_style", "random")
        self.crop_aspect = self.data_backend_config.get("crop_aspect", "square")
        self.crop_coordinates = (0, 0)
        crop_handler_cls = crop_handlers.get(self.crop_style)
        if not crop_handler_cls:
            raise ValueError(f"Unknown crop style: {self.crop_style}")
        self.cropper = crop_handler_cls(image=self.image, image_metadata=image_metadata)
        self.resolution = self.data_backend_config.get("resolution")
        self.resolution_type = self.data_backend_config.get("resolution_type")
        self.target_size_calculator = resize_helpers.get(self.resolution_type)
        if self.target_size_calculator is None:
            raise ValueError(f"Unknown resolution type: {self.resolution_type}")
        if self.resolution_type == "pixel":
            self.target_area = self.resolution
            # Store the pixel value, eg. 1024
            self.pixel_resolution = self.resolution
            # Store the megapixel value, eg. 1.0
            self.megapixel_resolution = self.resolution / 1e6
        elif self.resolution_type == "area":
            self.target_area = self.resolution * 1e6  # Convert megapixels to pixels
            # Store the pixel value, eg. 1024
            self.pixel_resolution = self.resolution * 1e6
            # Store the megapixel value, eg. 1.0
            self.megapixel_resolution = self.resolution
        else:
            raise Exception(f"Unknown resolution type: {self.resolution_type}")
        self.target_downsample_size = self.data_backend_config.get(
            "target_downsample_size", None
        )
        self.maximum_image_size = self.data_backend_config.get(
            "maximum_image_size", None
        )

    def prepare(self, return_tensor: bool = False):
        """
        Perform initial image preparations such as converting to RGB and applying EXIF transformations.

        Args:
        image (Image.Image): The image to prepare.

        Returns:
        (image, crop_coordinates, aspect_ratio)
        """
        self.crop()
        if not self.crop_enabled:
            self.resize()

        image = self.image
        if return_tensor:
            # Return normalised tensor.
            image = self.transforms(image)
        return PreparedSample(
            image=image,
            original_size=self.original_size,
            crop_coordinates=self.crop_coordinates,
            aspect_ratio=self.aspect_ratio,
            image_metadata=self.image_metadata,
            target_size=self.target_size,
        )

    def area(self) -> int:
        """
        Calculate the area of the image.

        Returns:
        int: The area of the image.
        """
        if self.image is not None:
            return self.image.size[0] * self.image.size[1]
        if self.original_size:
            return self.original_size[0] * self.original_size[1]

    def should_downsample_before_crop(self) -> bool:
        """
        Returns:
        bool: True if the image should be downsampled before cropping, False otherwise.
        """
        if (
            not self.crop_enabled
            or not self.maximum_image_size
            or not self.target_downsample_size
        ):
            return False
        if self.data_backend_config.get("resolution_type") == "pixel":
            return (
                self.image.size[0] > self.pixel_resolution
                or self.image.size[1] > self.pixel_resolution
            )
        elif self.data_backend_config.get("resolution_type") == "area":
            logger.debug(
                f"Image is too large? {self.area() > self.target_area} (image area: {self.area()}, target area: {self.target_area})"
            )
            return self.area() > self.target_area
        else:
            raise ValueError(
                f"Unknown resolution type: {self.data_backend_config.get('resolution_type')}"
            )

    def downsample_before_crop(self):
        """
        Downsample the image before cropping, to preserve scene details.
        """
        if self.image and self.should_downsample_before_crop():
            width, height, _ = self.calculate_target_size(downsample_before_crop=True)
            logger.debug(
                f"Downsampling image from {self.image.size} to {width}x{height} before cropping."
            )
            self.resize((width, height))
        return self

    def calculate_target_size(self, downsample_before_crop: bool = False):
        # Square crops are always {self.pixel_resolution}x{self.pixel_resolution}
        if self.crop_aspect == "square" and not downsample_before_crop:
            self.aspect_ratio = 1.0
            self.target_size = (self.pixel_resolution, self.pixel_resolution)
            return self.target_size[0], self.target_size[1], self.aspect_ratio
        self.aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(
            self.original_size
        )
        if downsample_before_crop and self.target_downsample_size is not None:
            target_width, target_height, self.aspect_ratio = (
                self.target_size_calculator(
                    self.aspect_ratio, self.target_downsample_size
                )
            )
        else:
            target_width, target_height, self.aspect_ratio = (
                self.target_size_calculator(self.aspect_ratio, self.resolution)
            )
        self.target_size = (target_width, target_height)
        self.aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(
            self.target_size
        )

        return self.target_size[0], self.target_size[1], self.aspect_ratio

    def correct_image(self):
        """
        Apply a series of transformations to the image to "correct" it.
        """
        if self.image:
            # Convert image to RGB to remove any alpha channel and apply EXIF data transformations
            self.image = self.image.convert("RGB")
            self.image = exif_transpose(self.image)
            self.original_size = self.image.size
        return self

    def crop(self):
        """
        Crop the image using the detected crop handler class.
        """
        if not self.crop_enabled:
            return self
        logger.debug(
            f"Cropping image with {self.crop_style} style and {self.crop_aspect}."
        )

        # Too-big of an image, resize before we crop.
        self.downsample_before_crop()
        width, height, aspect_ratio = self.calculate_target_size(
            downsample_before_crop=False
        )
        logger.debug(
            f"Pre-crop size: {self.image.size if hasattr(self.image, 'size') else 'Unknown'}."
        )
        self.image, self.crop_coordinates = self.cropper.crop(width, height)
        logger.debug(
            f"Post-crop size: {self.image.size if hasattr(self.image, 'size') else 'Unknown'}."
        )
        return self

    def resize(self, target_size: tuple = None):
        """
        Resize the image to a new size.

        Args:
        target_size (tuple): The target size as (width, height).
        """
        if target_size is None:
            target_width, target_height, aspect_ratio = self.calculate_target_size()
            target_size = (target_width, target_height)
        if self.image:
            self.image = self.image.resize(target_size, Image.Resampling.LANCZOS)
            self.aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(
                self.image.size
            )
        return self

    def get_image(self):
        """
        Returns the current state of the image.

        Returns:
        Image.Image: The current image.
        """
        return self.image


class PreparedSample:
    def __init__(
        self,
        image: Image.Image,
        image_metadata: dict,
        original_size: tuple,
        target_size: tuple,
        aspect_ratio: float,
        crop_coordinates: tuple,
    ):
        """
        Initializes a new PreparedSample instance with a provided PIL.Image object and optional metadata.

        Args:
        image (Image.Image): A PIL Image object.
        metadata (dict): Optional metadata associated with the image.
        """
        self.image = image
        self.image_metadata = image_metadata if image_metadata else {}
        self.original_size = original_size
        self.target_size = target_size
        self.aspect_ratio = aspect_ratio
        self.crop_coordinates = crop_coordinates
