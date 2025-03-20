from PIL import Image
import logging
import os
from typing import Union
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class BaseCropping:
    def __init__(
        self, image: Union[Image.Image, np.ndarray] = None, image_metadata: dict = None
    ):
        self.original_height = None
        self.original_width = None
        self.intermediary_height = None
        self.intermediary_width = None
        self.image = image
        self.image_metadata = image_metadata
        # When we've only got metadata, we can't crop the image.
        self.meta_crop = False
        if self.image is not None:
            if isinstance(self.image, Image.Image):
                self.original_width, self.original_height = self.image.size
            elif isinstance(self.image, np.ndarray):
                # Support both single image (3D) and video (4D)
                if self.image.ndim == 4:  # video: (num_frames, height, width, channels)
                    _, h, w, _ = self.image.shape
                    self.original_width, self.original_height = w, h
                elif self.image.ndim == 3:  # single image: (height, width, channels)
                    h, w = self.image.shape[:2]
                    self.original_width, self.original_height = w, h
                else:
                    raise ValueError(
                        f"Unexpected shape for training sample: {self.image.shape}"
                    )
        elif self.image_metadata:
            self.original_width, self.original_height = self.image_metadata[
                "original_size"
            ]

    def crop(self, target_width, target_height):
        raise NotImplementedError("Subclasses must implement this method")

    def set_image(self, image: Union[Image.Image, np.ndarray]):
        if type(image) not in [Image.Image, np.ndarray]:
            raise TypeError("Image must be a PIL Image or a NumPy ndarray")
        self.image = image
        return self

    def set_intermediary_size(self, width, height):
        self.intermediary_width = width
        self.intermediary_height = height
        return self


class CornerCropping(BaseCropping):
    def crop(self, target_width, target_height):
        left = max(0, self.intermediary_width - target_width)
        top = max(0, self.intermediary_height - target_height)
        right = self.intermediary_width
        bottom = self.intermediary_height
        if self.image is not None:
            if isinstance(self.image, Image.Image):
                return self.image.crop((left, top, right, bottom)), (top, left)
            elif isinstance(self.image, np.ndarray):
                # Handle both video (4D) and single image (3D)
                if self.image.ndim == 4:
                    cropped = self.image[:, top:bottom, left:right, :]
                else:
                    cropped = self.image[top:bottom, left:right, :]
                return cropped, (top, left)
        elif self.image_metadata:
            return None, (top, left)


class CenterCropping(BaseCropping):
    def crop(self, target_width, target_height):
        left = int((self.intermediary_width - target_width) / 2)
        top = int((self.intermediary_height - target_height) / 2)
        right = left + target_width
        bottom = top + target_height
        if self.image is not None:
            if isinstance(self.image, Image.Image):
                return self.image.crop((left, top, right, bottom)), (top, left)
            elif isinstance(self.image, np.ndarray):
                if self.image.ndim == 4:
                    cropped = self.image[:, top:bottom, left:right, :]
                else:
                    cropped = self.image[top:bottom, left:right, :]
                return cropped, (top, left)
        elif self.image_metadata:
            return None, (top, left)


class RandomCropping(BaseCropping):
    def crop(self, target_width, target_height):
        import random

        left = random.randint(0, max(0, self.intermediary_width - target_width))
        top = random.randint(0, max(0, self.intermediary_height - target_height))
        right = left + target_width
        bottom = top + target_height
        if self.image is not None:
            if isinstance(self.image, Image.Image):
                return self.image.crop((left, top, right, bottom)), (top, left)
            elif isinstance(self.image, np.ndarray):
                if self.image.ndim == 4:
                    cropped = self.image[:, top:bottom, left:right, :]
                else:
                    cropped = self.image[top:bottom, left:right, :]
                return cropped, (top, left)
        elif self.image_metadata:
            return None, (top, left)


class FaceCropping(RandomCropping):
    def crop(
        self,
        image: Union[Image.Image, np.ndarray],
        target_width: int,
        target_height: int,
    ):
        import cv2
        import numpy as np

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        if isinstance(image, np.ndarray):
            # Assume it's a video (4D) and use the first frame for face detection.
            sample_frame = image[0]
            gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                # Get the largest face.
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face
                left = int(max(0, x - 0.5 * w))
                top = int(max(0, y - 0.5 * h))
                right = int(min(sample_frame.shape[1], x + 1.5 * w))
                bottom = int(min(sample_frame.shape[0], y + 1.5 * h))
            else:
                # Fallback to random cropping on the sample frame.
                import random

                left = random.randint(0, max(0, sample_frame.shape[1] - target_width))
                top = random.randint(0, max(0, sample_frame.shape[0] - target_height))
                right = left + target_width
                bottom = top + target_height
            # Crop all frames in the video.
            cropped = image[:, top:bottom, left:right, :]
            return cropped, (top, left)
        elif isinstance(image, Image.Image):
            image_rgb = image.convert("RGB")
            image_np = np.array(image_rgb)
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face
                left = max(0, x - 0.5 * w)
                top = max(0, y - 0.5 * h)
                right = min(image_np.shape[1], x + 1.5 * w)
                bottom = min(image_np.shape[0], y + 1.5 * h)
                return image.crop((left, top, right, bottom)), (top, left)
            else:
                return super().crop(target_width, target_height)


# Dictionary mapping crop types to classes.
crop_handlers = {
    "corner": CornerCropping,
    "centre": CenterCropping,
    "center": CenterCropping,
    "random": RandomCropping,
    "face": FaceCropping,
}
