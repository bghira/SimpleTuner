import os, torch, logging
from tqdm import tqdm
from PIL import Image
from helpers.multiaspect.image import MultiaspectImage
from helpers.multiaspect.sampler import MultiAspectSampler
from helpers.multiaspect.bucket import BucketManager
from helpers.data_backend.base import BaseDataBackend
from helpers.data_backend.aws import S3DataBackend

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
        delete_problematic_images: bool = False,
        write_batch_size: int = 25,
        vae_batch_size: int = 4,
    ):
        self.data_backend = data_backend
        self.vae = vae
        self.vae.enable_slicing()
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        self.resolution = resolution
        self.data_backend.create_directory(self.cache_dir)
        self.delete_problematic_images = delete_problematic_images
        self.write_batch_size = write_batch_size
        self.vae_batch_size = vae_batch_size

    def _generate_filename(self, filepath: str) -> tuple:
        """Get the cache filename for a given image filepath and its base name."""
        # Extract the base name from the filepath and replace the image extension with .pt
        base_filename = os.path.splitext(os.path.basename(filepath))[0] + ".pt"
        full_filename = os.path.join(self.cache_dir, base_filename)
        return full_filename, base_filename

    def save_to_cache(self, filename, embeddings):
        self.data_backend.torch_save(embeddings, filename)

    def load_from_cache(self, filename):
        return self.data_backend.torch_load(filename)

    def discover_unprocessed_files(self, directory):
        """Identify files that haven't been processed yet."""
        all_files = {
            os.path.join(subdir, file)
            for subdir, _, files in self.data_backend.list_files(
                "*.[jJpP][pPnN][gG]", directory
            )
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
        full_filename, base_filename = self._generate_filename(filepath)
        logger.debug(
            f"Created filename {full_filename} from filepath {filepath} for resulting .pt filename."
        )
        if self.data_backend.exists(full_filename):
            latents = self.load_from_cache(full_filename)
            logger.debug(
                f"Loading latents of shape {latents.shape} from existing cache file: {full_filename}"
            )
        else:
            # Print the shape of the pixel values:
            logger.debug(f"Pixel values shape: {pixel_values.shape}")
            with torch.no_grad():
                latents = self.vae.encode(
                    pixel_values.unsqueeze(0).to(
                        self.accelerator.device, dtype=torch.bfloat16
                    )
                ).latent_dist.sample()
                logger.debug(
                    f"Using shape {latents.shape}, creating new latent cache: {full_filename}"
                )
            latents = latents * self.vae.config.scaling_factor
            logger.debug(f"Latent shape after re-scale: {latents.shape}")

        output_latents = latents.squeeze().to(
            self.accelerator.device, dtype=self.vae.dtype
        )
        logger.debug(f"Output latents shape: {output_latents.shape}")
        return output_latents

    def split_cache_between_processes(self):
        all_unprocessed_files = self.discover_unprocessed_files(self.cache_dir)
        # Use the accelerator to split the data
        with self.accelerator.split_between_processes(
            all_unprocessed_files
        ) as split_files:
            self.local_unprocessed_files = split_files

    def _process_image(self, filepath):
        full_filename, base_filename = self._generate_filename(filepath)
        if self.data_backend.exists(full_filename):
            return None

        try:
            image = self.data_backend.read_image(filepath)
            image = MultiaspectImage.prepare_image(image, self.resolution)
            pixel_values = self.transform(image).to(
                self.accelerator.device, dtype=self.vae.dtype
            )
            latents = self.encode_image(pixel_values, filepath)
            return latents
        except Exception as e:
            logger.error(f"Error processing image {filepath}: {e}")
            if self.delete_problematic_images:
                self.data_backend.delete(filepath)
            else:
                raise e

    def _accumulate_batch(self, latents, filepath, batch_data, batch_filepaths):
        full_filename, _ = self._generate_filename(filepath)
        batch_data.append(latents.squeeze())
        batch_filepaths.append(full_filename)

        if len(batch_filepaths) >= self.write_batch_size:
            self.data_backend.write_batch(batch_filepaths, batch_data)
            batch_filepaths.clear()
            batch_data.clear()

        return batch_data, batch_filepaths

    def process_directory(self, directory, bucket_manager):
        files_to_process = self.discover_unprocessed_files(directory)
        batch_data, batch_filepaths = [], []
        aspect_bucket_cache = bucket_manager.read_cache().deepcopy()

        for bucket in aspect_bucket_cache:
            for raw_filepath in tqdm(
                aspect_bucket_cache[bucket], desc="Processing images"
            ):
                if type(raw_filepath) == str or len(raw_filepath) == 1:
                    filepath = raw_filepath
                elif len(raw_filepath) == 2:
                    idx, filepath = raw_filepath
                else:
                    raise ValueError(f"Received unknown filepath value: {raw_filepath}")

                latents = self._process_image(filepath)
                if latents is not None:
                    batch_data, batch_filepaths = self._accumulate_batch(
                        latents, filepath, batch_data, batch_filepaths
                    )

            if batch_data:  # Write remaining batches
                self.data_backend.write_batch(batch_filepaths, batch_data)
