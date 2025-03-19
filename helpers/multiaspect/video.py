import cv2
import numpy as np


def resize_video_frames(
    video_frames: np.ndarray, dsize=None, fx=None, fy=None
) -> np.ndarray:
    """
    Resize each frame in a video (NumPy array with shape (num_frames, height, width, channels)).
    You can either provide a fixed destination size (dsize) or scaling factors (fx and fy).
    """
    resized_frames = []
    for frame in video_frames:
        # Optionally, add a check to make sure frame is valid.
        if frame is None or frame.size == 0:
            continue
        resized_frame = cv2.resize(frame, dsize=dsize, fx=fx, fy=fy)
        resized_frames.append(resized_frame)

    if not resized_frames:
        raise ValueError(
            "No frames were resized. Check your video data and resize parameters."
        )

    return np.stack(resized_frames, axis=0)
