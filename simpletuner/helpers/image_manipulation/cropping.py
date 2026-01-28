import logging
import os
from typing import Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class BaseCropping:
    def __init__(self, image: Union[Image.Image, np.ndarray] = None, image_metadata: dict = None):
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
                    raise ValueError(f"Unexpected shape for training sample: {self.image.shape}")
        elif self.image_metadata:
            self.original_width, self.original_height = self.image_metadata["original_size"]

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
        # Corner crop from top-left (0, 0)
        left = 0
        top = 0
        right = min(target_width, self.intermediary_width)
        bottom = min(target_height, self.intermediary_height)

        left = int(round(left))
        top = int(round(top))
        right = int(round(right))
        bottom = int(round(bottom))

        logger.debug(
            f"CornerCropping: intermediary_size=({self.intermediary_width}, {self.intermediary_height}), target_size=({target_width}, {target_height})"
        )
        logger.debug(f"CornerCropping: crop_box=(left={left}, top={top}, right={right}, bottom={bottom})")

        if self.image is not None:
            if isinstance(self.image, Image.Image):
                cropped_image = self.image.crop((left, top, right, bottom))
                logger.debug(f"CornerCropping PIL result: {cropped_image.size}")
                return cropped_image, (top, left)
            elif isinstance(self.image, np.ndarray):
                if self.image.ndim == 4:  # video: (num_frames, height, width, channels)
                    logger.debug(f"CornerCropping video input shape: {self.image.shape}")
                    # Standard numpy slicing: [frames, height_slice, width_slice, channels]
                    cropped = self.image[:, top:bottom, left:right, :]
                    logger.debug(f"CornerCropping video result shape: {cropped.shape}")
                    return cropped, (top, left)
                else:  # single image: (height, width, channels)
                    logger.debug(f"CornerCropping image input shape: {self.image.shape}")
                    cropped = self.image[top:bottom, left:right, :]
                    logger.debug(f"CornerCropping image result shape: {cropped.shape}")
                    return cropped, (top, left)
        elif self.image_metadata:
            return None, (top, left)


class CenterCropping(BaseCropping):
    def crop(self, target_width, target_height):
        # Calculate center crop coordinates
        left = max(0, int((self.intermediary_width - target_width) / 2))
        top = max(0, int((self.intermediary_height - target_height) / 2))
        right = left + min(target_width, self.intermediary_width - left)
        bottom = top + min(target_height, self.intermediary_height - top)

        left = int(round(left))
        top = int(round(top))
        right = int(round(right))
        bottom = int(round(bottom))

        logger.debug(
            f"CenterCropping: intermediary_size=({self.intermediary_width}, {self.intermediary_height}), target_size=({target_width}, {target_height})"
        )
        logger.debug(f"CenterCropping: crop_box=(left={left}, top={top}, right={right}, bottom={bottom})")

        if self.image is not None:
            if isinstance(self.image, Image.Image):
                cropped_image = self.image.crop((left, top, right, bottom))
                logger.debug(f"CenterCropping PIL result: {cropped_image.size}")
                return cropped_image, (top, left)
            elif isinstance(self.image, np.ndarray):
                if self.image.ndim == 4:  # video: (num_frames, height, width, channels)
                    logger.debug(f"CenterCropping video input shape: {self.image.shape}")
                    # Standard numpy slicing: [frames, height_slice, width_slice, channels]
                    cropped = self.image[:, top:bottom, left:right, :]
                    logger.debug(f"CenterCropping video result shape: {cropped.shape}")
                    return cropped, (top, left)
                else:  # single image: (height, width, channels)
                    logger.debug(f"CenterCropping image input shape: {self.image.shape}")
                    cropped = self.image[top:bottom, left:right, :]
                    logger.debug(f"CenterCropping image result shape: {cropped.shape}")
                    return cropped, (top, left)
        elif self.image_metadata:
            return None, (top, left)


class RandomCropping(BaseCropping):
    def crop(self, target_width, target_height):
        import random

        # Calculate maximum possible crop coordinates
        max_left = max(0, self.intermediary_width - target_width)
        max_top = max(0, self.intermediary_height - target_height)

        left = random.randint(0, max_left) if max_left > 0 else 0
        top = random.randint(0, max_top) if max_top > 0 else 0
        right = left + min(target_width, self.intermediary_width - left)
        bottom = top + min(target_height, self.intermediary_height - top)

        left = int(round(left))
        top = int(round(top))
        right = int(round(right))
        bottom = int(round(bottom))

        logger.debug(
            f"RandomCropping: intermediary_size=({self.intermediary_width}, {self.intermediary_height}), target_size=({target_width}, {target_height})"
        )
        logger.debug(f"RandomCropping: max_offsets=(max_left={max_left}, max_top={max_top})")
        logger.debug(f"RandomCropping: crop_box=(left={left}, top={top}, right={right}, bottom={bottom})")

        if self.image is not None:
            if isinstance(self.image, Image.Image):
                cropped_image = self.image.crop((left, top, right, bottom))
                logger.debug(f"RandomCropping PIL result: {cropped_image.size}")
                return cropped_image, (top, left)
            elif isinstance(self.image, np.ndarray):
                if self.image.ndim == 4:  # video: (num_frames, height, width, channels)
                    logger.debug(f"RandomCropping video input shape: {self.image.shape}")
                    # Standard numpy slicing: [frames, height_slice, width_slice, channels]
                    cropped = self.image[:, top:bottom, left:right, :]
                    logger.debug(f"RandomCropping video result shape: {cropped.shape}")
                    return cropped, (top, left)
                else:  # single image: (height, width, channels)
                    logger.debug(f"RandomCropping image input shape: {self.image.shape}")
                    cropped = self.image[top:bottom, left:right, :]
                    logger.debug(f"RandomCropping image result shape: {cropped.shape}")
                    return cropped, (top, left)
        elif self.image_metadata:
            return None, (top, left)


class FaceCropping(RandomCropping):
    def crop(self, target_width, target_height):
        import numpy as np
        import trainingsample as tsr

        # Get OpenCV data path and create classifier
        opencv_data_path = tsr.get_opencv_data_path_py()
        face_cascade = tsr.PyCascadeClassifier(opencv_data_path + "haarcascades/haarcascade_frontalface_default.xml")

        if isinstance(self.image, np.ndarray):
            # Use the first frame for face detection in videos
            sample_frame = self.image[0] if self.image.ndim == 4 else self.image

            # Convert to grayscale for face detection
            if sample_frame.shape[-1] == 3:  # RGB
                gray = tsr.cvt_color_py(sample_frame, 7)  # 7 = COLOR_RGB2GRAY
            else:
                gray = sample_frame

            faces = face_cascade.detect_multi_scale(gray, 1.1, 4)

            if len(faces) > 0:
                # Get the largest face
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face

                # Calculate crop area centered on face
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                left = max(0, face_center_x - target_width // 2)
                top = max(0, face_center_y - target_height // 2)

                # Ensure we don't go beyond image boundaries
                if self.image.ndim == 4:  # video: (frames, height, width, channels)
                    img_height, img_width = self.image.shape[1:3]
                else:
                    img_height, img_width = self.image.shape[:2]

                right = min(img_width, left + target_width)
                bottom = min(img_height, top + target_height)

                # Adjust left/top if we hit the right/bottom boundary
                if right - left < target_width:
                    left = max(0, right - target_width)
                if bottom - top < target_height:
                    top = max(0, bottom - target_height)

                logger.debug(f"FaceCropping: detected face at ({x}, {y}, {w}, {h})")
                logger.debug(f"FaceCropping: crop_box=(left={left}, top={top}, right={right}, bottom={bottom})")

                if self.image.ndim == 4:  # video: (frames, height, width, channels)
                    # Standard numpy slicing: [frames, height_slice, width_slice, channels]
                    cropped = self.image[:, top:bottom, left:right, :]
                    logger.debug(f"FaceCropping video result shape: {cropped.shape}")
                else:  # single image
                    cropped = self.image[top:bottom, left:right, :]
                    logger.debug(f"FaceCropping image result shape: {cropped.shape}")

                return cropped, (top, left)
            else:
                logger.debug("FaceCropping: No faces detected, falling back to random cropping")
                return super().crop(target_width, target_height)

        elif isinstance(self.image, Image.Image):
            image_rgb = self.image.convert("RGB")
            image_np = np.array(image_rgb)
            gray = tsr.cvt_color_py(image_np, 7)  # 7 = COLOR_RGB2GRAY
            faces = face_cascade.detect_multi_scale(gray, 1.1, 4)

            if len(faces) > 0:
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face

                face_center_x = x + w // 2
                face_center_y = y + h // 2

                left = max(0, face_center_x - target_width // 2)
                top = max(0, face_center_y - target_height // 2)
                right = min(image_np.shape[1], left + target_width)
                bottom = min(image_np.shape[0], top + target_height)

                if right - left < target_width:
                    left = max(0, right - target_width)
                if bottom - top < target_height:
                    top = max(0, bottom - target_height)

                logger.debug(f"FaceCropping PIL: detected face at ({x}, {y}, {w}, {h})")
                logger.debug(f"FaceCropping PIL: crop_box=(left={left}, top={top}, right={right}, bottom={bottom})")

                cropped_image = self.image.crop((left, top, right, bottom))
                logger.debug(f"FaceCropping PIL result: {cropped_image.size}")
                return cropped_image, (top, left)
            else:
                logger.debug("FaceCropping PIL: No faces detected, falling back to random cropping")
                return super().crop(target_width, target_height)


# Dictionary mapping crop types to classes.
crop_handlers = {
    "corner": CornerCropping,
    "centre": CenterCropping,
    "center": CenterCropping,
    "random": RandomCropping,
    "face": FaceCropping,
}
