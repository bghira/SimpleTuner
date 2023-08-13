from PIL import Image

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

def calculate_batch_luminance(imgs: list):
    luminance_values = []
    for img in imgs:
        luminance_values.append(calculate_luminance(img))
    return sum(luminance_values) / len(luminance_values)