import multiprocessing

import numpy as np
from PIL import Image


def calculate_luminance(img: Image.Image):
    if isinstance(img, np.ndarray):
        np_img = img
    elif isinstance(img, Image.Image):
        np_img = np.asarray(img.convert("RGB"))
    else:
        raise ValueError(f"Unexpected image type for luminance calculation: {type(img)}")
    r, g, b = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    avg_luminance = np.mean(luminance)
    return avg_luminance


def worker_batch_luminance(imgs: list):
    return [calculate_luminance(img) for img in imgs]


def calculate_batch_luminance(imgs: list):
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_processes) as pool:
        # Splitting images into batches for each process
        img_batches = [imgs[i::num_processes] for i in range(num_processes)]
        results = pool.map(worker_batch_luminance, img_batches)

    # Flatten the results and calculate average luminance
    all_luminance_values = [lum for sublist in results for lum in sublist]
    return sum(all_luminance_values) / len(all_luminance_values)
