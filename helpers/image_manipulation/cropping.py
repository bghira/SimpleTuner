from PIL import Image
import logging, os

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class BaseCropping:
    def __init__(self, image: Image = None, image_metadata: dict = None):
        self.image = image
        self.image_metadata = image_metadata
        if self.image:
            self.original_width, self.original_height = self.image.size
        elif self.image_metadata:
            self.original_width, self.original_height = self.image_metadata[
                "original_size"
            ]
        logger.debug(
            "Cropper intialized with image size: %s x %s",
            self.original_width,
            self.original_height,
        )

    def crop(self, target_width, target_height):
        raise NotImplementedError("Subclasses must implement this method")

    def set_image(self, image: Image):
        logger.debug(
            f"Cropper image being refreshed. Before size: {self.original_width} x {self.original_height}"
        )
        self.image = image
        self.original_width, self.original_height = self.image.size

    def set_image_metadata(self, image_metadata: dict):
        logger.debug(
            f"Cropper image metadata being refreshed. Before size: {self.original_width} x {self.original_height}"
        )
        self.image_metadata = image_metadata
        self.original_width, self.original_height = self.image_metadata["original_size"]
        if "current_size" in self.image_metadata:
            self.original_width, self.original_height = self.image_metadata[
                "current_size"
            ]


class CornerCropping(BaseCropping):
    def crop(self, target_width, target_height):
        left = max(0, self.original_width - target_width)
        top = max(0, self.original_height - target_height)
        right = self.original_width
        bottom = self.original_height
        if self.image:
            return self.image.crop((left, top, right, bottom)), (top, left)
        elif self.image_metadata:
            return self.image_metadata, (top, left)


class CenterCropping(BaseCropping):
    def crop(self, target_width, target_height):
        left = (self.original_width - target_width) / 2
        top = (self.original_height - target_height) / 2
        right = (self.original_width + target_width) / 2
        bottom = (self.original_height + target_height) / 2
        if self.image:
            return self.image.crop((left, top, right, bottom)), (top, left)
        elif self.image_metadata:
            return self.image_metadata, (top, left)


class RandomCropping(BaseCropping):
    def crop(self, target_width, target_height):
        import random

        left = random.randint(0, max(0, self.original_width - target_width))
        top = random.randint(0, max(0, self.original_height - target_height))
        logger.debug(
            f"Random cropping from {left}, {top} - {self.original_width}x{self.original_height} to {target_width}x{target_height}"
        )
        right = left + target_width
        bottom = top + target_height
        if self.image:
            return self.image.crop((left, top, right, bottom)), (top, left)
        elif self.image_metadata:
            return self.image_metadata, (top, left)


class FaceCropping(RandomCropping):
    def crop(
        self,
        image: Image.Image,
        target_width: int,
        target_height: int,
    ):
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
            return image.crop((left, top, right, bottom)), (top, left)
        else:
            # Crop the image from a random position
            return super.crop(image, target_width, target_height)


crop_handlers = {
    "corner": CornerCropping,
    "centre": CenterCropping,
    "center": CenterCropping,
    "random": RandomCropping,
}
