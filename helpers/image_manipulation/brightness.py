from PIL import Image
import multiprocessing


def calculate_luminance(img: Image):
    pixels = list(img.getdata())

    luminance_values = []
    for pixel in pixels:
        r, g, b = pixel[:3]  # Assuming the image is RGB or RGBA
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        luminance_values.append(luminance)

    # Return average luminance for the entire image
    avg_luminance = sum(luminance_values) / len(luminance_values)
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
