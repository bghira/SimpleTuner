from PIL import Image
import os


def handle_broken_images(dir_path, delete=False):
    """Handle broken images in a given directory.

    Args:
        dir_path (str): The directory path to scan for images.
        delete (bool, optional): If True, delete broken images.
            Otherwise, just print their names. Defaults to False.
    """
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        ):
            try:
                img_path = os.path.join(dir_path, filename)
                with Image.open(img_path) as img:
                    img.verify()  # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                logging.info(f"Bad file: {img_path} - {e}")
                if delete:
                    os.remove(img_path)
                    logging.info(f"Removed: {img_path}")
