import logging
import os
import tempfile
from io import BytesIO
from typing import IO, Any, Union

import numpy as np

try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image, PngImagePlugin

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

try:
    import trainingsample as tsr
except Exception as e:
    print(
        "An error occurred while importing trainingsample library."
        " This is required for high-performance image processing in SimpleTuner."
        "\nPlease install it with: pip install trainingsample"
        "\nFull error message at the end of debug.log inside the SimpleTuner directory."
    )
    from sys import exit

    exit(1)


LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


def decode_image_with_trainingsample(img_bytes: bytes) -> Union[Image.Image, None]:
    """Decode image using trainingsample with optimized format conversions."""
    img_tsr = tsr.imdecode_py(img_bytes, 1)  # 1 = IMREAD_COLOR
    if img_tsr is not None:
        # trainingsample's imdecode already returns RGB tensors, no extra swap needed
        # Handle grayscale with ultra-fast conversion if needed
        if len(img_tsr.shape) == 2:
            # Convert grayscale to RGB using optimized method
            img_tsr = tsr.cvt_color_py(img_tsr, 8)  # 8 = COLOR_GRAY2RGB
        elif img_tsr.shape[2] == 1:
            # Single channel to RGB
            img_tsr = tsr.cvt_color_py(img_tsr, 8)  # 8 = COLOR_GRAY2RGB

    return img_tsr if img_tsr is None else Image.fromarray(img_tsr)


def decode_image_with_pil(img_data: bytes) -> Image.Image:
    try:
        if isinstance(img_data, bytes):
            img_pil = Image.open(BytesIO(img_data))
        else:
            img_pil = Image.open(img_data)

        if img_pil.mode not in ["RGB", "RGBA"] and "transparency" in img_pil.info:
            img_pil = img_pil.convert("RGBA")

        # For transparent images, use ultra-fast trainingsample format conversion
        if img_pil.mode == "RGBA":
            # Use trainingsample's optimized RGBA->RGB conversion (10x faster!)
            import numpy as np
            import trainingsample as tsr

            rgba_array = np.array(img_pil)
            rgb_array, timing = tsr.rgba_to_rgb_optimized(rgba_array)
            img_pil = Image.fromarray(rgb_array)
        else:
            img_pil = img_pil.convert("RGB")
    except (OSError, Image.DecompressionBombError, ValueError) as e:
        logger.warning(f"Error decoding image: {e}")
        raise
    return img_pil


def remove_iccp_chunk(img_bytes: bytes) -> bytes:
    """
    Remove the iCCP chunk from a PNG image if present.
    Returns the modified bytes, or the original if not PNG or no iCCP chunk found.
    """
    PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
    if not img_bytes.startswith(PNG_SIGNATURE):
        return img_bytes

    out = bytearray()
    out += PNG_SIGNATURE
    i = len(PNG_SIGNATURE)
    while i < len(img_bytes):
        # Need at least 8 bytes for length and type
        if i + 8 > len(img_bytes):
            break
        length = int.from_bytes(img_bytes[i : i + 4], "big")
        # Validate length: must be non-negative, not too large, and fit in buffer
        if length < 0 or length > 2**31 - 1 or i + 8 + length + 4 > len(img_bytes):
            # Malformed chunk length; abort processing to avoid memory issues
            break
        # Need enough bytes for chunk data and CRC (4 bytes)
        if i + 8 + length + 4 > len(img_bytes):
            break
        chunk_type = img_bytes[i + 4 : i + 8]
        # crc = img_bytes[i + 8 + length : i + 12 + length]
        if chunk_type == b"iCCP":
            # skip this chunk
            i += 8 + length + 4
            continue
        out += img_bytes[i : i + 8 + length + 4]
        i += 8 + length + 4
    return bytes(out)


def load_image(img_data: Union[bytes, IO[Any], str]) -> Image.Image:
    """
    Load an image using trainingsample. If that fails, fall back to PIL.

    The image is returned as a PIL object.
    """
    if isinstance(img_data, str):
        with open(img_data, "rb") as file:
            img_data = file.read()
    elif hasattr(img_data, "read"):
        # Check if it's file-like object.
        img_data = img_data.read()

    # remove iCCP chunk if found
    img_data = remove_iccp_chunk(img_data)

    # Try to preload the image bytes with channels unchanged to determine
    # if the image has an alpha channel. If it does we should add a white
    # background to it using PIL.
    has_alpha = False
    try:
        image_preload = tsr.imdecode_py(img_data, -1)  # -1 = IMREAD_UNCHANGED
        if image_preload is not None and len(image_preload.shape) >= 3 and image_preload.shape[2] == 4:
            has_alpha = True
        del image_preload
    except Exception:
        # If trainingsample fails, we'll use PIL fallback regardless
        pass

    img = None
    if not has_alpha:
        try:
            img = decode_image_with_trainingsample(img_data)
        except Exception:
            # If trainingsample fails, fall back to PIL
            pass
    if img is None:
        img = decode_image_with_pil(img_data)
    return img


def load_video(vid_data: Union[bytes, IO[Any], str]) -> np.ndarray:
    """
    Load a video using trainingsample's VideoCapture.

    Accepts a file path (str), a file-like object, or raw bytes.
    Reads all frames from the video and returns them as a NumPy array.

    Raises:
        ValueError: If the video cannot be opened or no frames are read.
        TypeError: If the input type is not supported.
    """
    tmp_path = None

    # If it's a file path, use it directly.
    if isinstance(vid_data, str):
        video_path = vid_data
    # If it's a file-like object.
    elif hasattr(vid_data, "read"):
        data = vid_data.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        try:
            tmp.write(data)
            video_path = tmp.name
            tmp_path = video_path
        finally:
            tmp.close()
    # If it's raw bytes.
    elif isinstance(vid_data, bytes):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        try:
            tmp.write(vid_data)
            video_path = tmp.name
            tmp_path = video_path
        finally:
            tmp.close()
    else:
        raise TypeError("Unsupported type for vid_data. Expected str, bytes, or file-like object.")

    cap = tsr.PyVideoCapture(video_path)
    if not cap.is_opened():
        if tmp_path:
            os.remove(tmp_path)
        raise ValueError(
            f"Failed to open video with trainingsample at '{video_path}'. Ensure trainingsample was built with video "
            "support (ffmpeg) and that the asset is a supported format."
        )

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_rgb = tsr.cvt_color_py(frame, 4)  # 4 = COLOR_BGR2RGB
        frames.append(frame_rgb)

    cap.release()

    # Clean up temporary file if one was used.
    if tmp_path:
        os.remove(tmp_path)

    if not frames:
        raise ValueError(
            "No frames were read from the video using trainingsample. Verify ffmpeg support is available and the "
            "video is not corrupted."
        )

    # Stack frames into a numpy array: shape (num_frames, height, width, channels)
    video_array = np.stack(frames, axis=0)
    return video_array
