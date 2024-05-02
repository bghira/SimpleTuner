from PIL import Image


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

    def crop(self, target_width, target_height):
        raise NotImplementedError("Subclasses must implement this method")


class CornerCropping(BaseCropping):
    def crop(self, target_width, target_height):
        left = max(0, self.original_width - target_width)
        top = max(0, self.original_height - target_height)
        right = self.original_width
        bottom = self.original_height
        if self.image:
            return self.image.crop((left, top, right, bottom)), (left, top)
        elif self.image_metadata:
            return self.image_metadata, (left, top)


class CenterCropping(BaseCropping):
    def crop(self, target_width, target_height):
        left = (self.original_width - target_width) / 2
        top = (self.original_height - target_height) / 2
        right = (self.original_width + target_width) / 2
        bottom = (self.original_height + target_height) / 2
        if self.image:
            return self.image.crop((left, top, right, bottom)), (left, top)
        elif self.image_metadata:
            return self.image_metadata, (left, top)


class RandomCropping(BaseCropping):
    def crop(self, target_width, target_height):
        import random

        left = random.randint(0, max(0, self.original_width - target_width))
        top = random.randint(0, max(0, self.original_height - target_height))
        right = left + target_width
        bottom = top + target_height
        if self.image:
            return self.image.crop((left, top, right, bottom)), (left, top)
        elif self.image_metadata:
            return self.image_metadata, (left, top)


class FaceCropping(RandomCropping):
    def crop(
        self,
        image: Image,
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
            return image.crop((left, top, right, bottom)), (left, top)
        else:
            # Crop the image from a random position
            return super.crop(image, target_width, target_height)
