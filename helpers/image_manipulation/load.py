import logging

from io import BytesIO
from typing import Union, IO, Any

import numpy as np

from PIL import Image, PngImagePlugin


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

try:
    import cv2
except Exception as e:
    if "libGL" in str(e):
        print(
            "An error occurred while importing OpenCV2 due to a missing LibGL dependency on your system or container."
            " Unfortunately, this is not a dependency that SimpleTuner can include during install time."
            "\nFor Ubuntu systems, you can typically resolve this by running the following command:\n"
            "sudo apt-get install libgl1-mesa-glx"
            "\nor, if that does not work:\n"
            "sudo apt-get install libgl1-mesa-dri"
            "\nIf all else fails, you may need to contact the support department for your chosen platform."
            " You can find the full error message at the end of debug.log inside the SimpleTuner directory."
        )
        from sys import exit

        exit(1)
    else:
        raise e


LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def decode_image_with_opencv(nparr: np.ndarray) -> Union[Image.Image, None]:
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_cv is not None:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # Ensuring we only convert to RGB if needed.
        if len(img_cv.shape) == 2 or (img_cv.shape[2] != 3 and img_cv.shape[2] == 1):
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
    return img_cv if img_cv is None else Image.fromarray(img_cv)


def decode_image_with_pil(img_data: bytes) -> Image.Image:
    try:
        if isinstance(img_data, bytes):
            img_pil = Image.open(BytesIO(img_data))
        else:
            img_pil = Image.open(img_data)

        if img_pil.mode not in ["RGB", "RGBA"] and "transparency" in img_pil.info:
            img_pil = img_pil.convert("RGBA")

        # For transparent images, add a white background as this is correct
        # most of the time.
        if img_pil.mode == "RGBA":
            canvas = Image.new("RGBA", img_pil.size, (255, 255, 255))
            canvas.alpha_composite(img_pil)
            img_pil = canvas.convert("RGB")
        else:
            img_pil = img_pil.convert("RGB")
    except (OSError, Image.DecompressionBombError, ValueError) as e:
        logger.warning(f"Error decoding image: {e}")
        raise
    return img_pil


def load_image(img_data: Union[bytes, IO[Any], str]) -> Image.Image:
    """
    Load an image using CV2. If that fails, fall back to PIL.

    The image is returned as a PIL object.
    """
    if isinstance(img_data, str):
        with open(img_data, "rb") as file:
            img_data = file.read()
    elif hasattr(img_data, "read"):
        # Check if it's file-like object.
        img_data = img_data.read()

    # Preload the image bytes with channels unchanged and ensure determine
    # if the image has an alpha channel. If it does we should add a white
    # background to it using PIL.
    nparr = np.frombuffer(img_data, np.uint8)
    image_preload = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    has_alpha = False
    if (
        image_preload is not None
        and len(image_preload.shape) >= 3
        and image_preload.shape[2] == 4
    ):
        has_alpha = True
    del image_preload

    img = None
    if not has_alpha:
        img = decode_image_with_opencv(nparr)
    if img is None:
        img = decode_image_with_pil(img_data)
    return img
