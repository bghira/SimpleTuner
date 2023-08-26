import os, torch, logging
from tqdm import tqdm
from PIL import Image
from helpers.multiaspect.image import MultiaspectImage
from helpers.data_backend.base import BaseDataBackend

logger = logging.getLogger("VAECache")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")


class VAECache:
    def __init__(
        self,
        vae,
        accelerator,
        data_backend: BaseDataBackend,
        cache_dir="vae_cache",
        resolution: int = 1024,
        delete_problematic_images: bool = False
    ):
        self.data_backend = data_backend
        self.vae = vae
        self.vae.enable_slicing()
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        self.resolution = resolution
        self.data_backend.create_directory(self.cache_dir)
        self.delete_problematic_images = delete_problematic_images

    def _generate_filename(self, filepath: str):
        """Get the cache filename for a given image filepath."""
        # Extract the base name from the filepath and replace the image extension with .pt
        return os.path.join(
            self.cache_dir, os.path.splitext(os.path.basename(filepath))[0] + ".pt"
        )

    def save_to_cache(self, filename, embeddings):
        self.data_backend.torch_save(embeddings, filename)

    def load_from_cache(self, filename):
        return self.data_backend.torch_load(filename)

    def discover_unprocessed_files(self, directory):
        """Identify files that haven't been processed yet."""
        all_files = {
            os.path.join(subdir, file)
            for subdir, _, files in self.data_backend.list_files(directory)
            for file in files
            if file.endswith((".png", ".jpg", ".jpeg"))
        }
        processed_files = {self._generate_filename(file) for file in all_files}
        unprocessed_files = {
            file
            for file in all_files
            if self._generate_filename(file) not in processed_files
        }
        return list(unprocessed_files)

    def encode_image(self, pixel_values, filepath: str):
        filename = self._generate_filename(filepath)
        logger.debug(
            f"Created filename {filename} from filepath {filepath} for resulting .pt filename."
        )
        if self.data_backend.exists(filename):
            latents = self.load_from_cache(filename)
            logger.debug(
                f"Loading latents of shape {latents.shape} from existing cache file: {filename}"
            )
        else:
            with torch.no_grad():
                latents = self.vae.encode(
                    pixel_values.unsqueeze(0).to(
                        self.accelerator.device, dtype=torch.bfloat16
                    )
                ).latent_dist.sample()
                logger.debug(
                    f"Using shape {latents.shape}, creating new latent cache: {filename}"
                )
            latents = latents * self.vae.config.scaling_factor
            logger.debug(f"Latent shape after re-scale: {latents.shape}")
            self.save_to_cache(filename, latents.squeeze())

        output_latents = latents.squeeze().to(
            self.accelerator.device, dtype=self.vae.dtype
        )
        logger.debug(f"Output latents shape: {output_latents.shape}")
        return output_latents

    def process_directory(self, directory):
        # Define a transform to convert the image to tensor
        transform = MultiaspectImage.get_image_transforms()

        # Get a list of all the files to process (customize as needed)
        files_to_process = []
        logger.debug(f"Beginning processing of VAECache directory {directory}")
        for subdir, _, files in self.data_backend.list_files(
            instance_data_root=directory, str_pattern="*.[jJpP][pPnN][gG]"
        ):
            for file in files:
                logger.debug(f"Discovered image: {os.path.join(subdir, file)}")
                files_to_process.append(os.path.join(subdir, file))

        # Iterate through the files, displaying a progress bar
        for filepath in tqdm(files_to_process, desc="Processing images"):
            # Create a hash based on the filename
            filename = self._generate_filename(filepath)

            # If processed file already exists, skip processing for this image
            if self.data_backend.exists(filename):
                logger.debug(
                    f"Skipping processing for {filepath} as cached file {filename} already exists."
                )
                continue

            # Open the image using PIL
            try:
                logger.debug(f"Loading image: {filepath}")
                image = self.data_backend.read_image(filepath)
                image = image.convert("RGB")
                image = self._resize_for_condition_image(image, self.resolution)
            except Exception as e:
                logger.error(f"Encountered error opening image: {e}")
                try:
                    if self.delete_problematic_images:
                        self.data_backend.delete(filepath)
                except Exception as e:
                    logger.error(f'Could not delete file: {filepath} via {type(self.data_backend)}. Error: {e}')
                continue

            # Convert the image to a tensor
            try:
                pixel_values = transform(image).to(
                    self.accelerator.device, dtype=self.vae.dtype
                )
            except OSError as e:
                logger.error(f"Encountered error converting image to tensor: {e}")
                continue

            # Process the image with the VAE
            self.encode_image(pixel_values, filepath)

            logger.debug(f"Processed image {filepath}")

    def _resize_for_condition_image(self, input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        aspect_ratio = round(W / H, 2)
        msg = f"Inspecting image of aspect {aspect_ratio} and size {W}x{H} to "
        if W < H:
            W = resolution
            H = int(resolution / aspect_ratio)  # Calculate the new height
        elif H < W:
            H = resolution
            W = int(resolution * aspect_ratio)  # Calculate the new width
        if W == H:
            W = resolution
            H = resolution
        msg = f"{msg} {W}x{H}."
        logger.debug(msg)
        img = input_image.resize((W, H), resample=Image.BICUBIC)
        return img
