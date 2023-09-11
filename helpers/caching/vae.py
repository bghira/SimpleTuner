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
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        self.resolution = resolution
        self.data_backend.create_directory(self.cache_dir)
        self.delete_problematic_images = delete_problematic_images
        self.write_batch_size = write_batch_size
        self.vae_batch_size = vae_batch_size
        self.transform = MultiaspectImage.get_image_transforms()

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

    def _list_cached_images(self):
        """
        Return a set of filenames (without the .pt extension) that have been processed.
        """
        # Extract array of tuple into just, an array of files:
        pt_files = [
            f
            for _, _, files in self.data_backend.list_files("*.pt", self.cache_dir)
            for f in files
        ]
        logging.debug(
            f"Found {len(pt_files)} cached files in {self.cache_dir}: {pt_files}"
        )
        # Extract just the base filename without the extension
        return {os.path.splitext(os.path.basename(f))[0] for f in pt_files}

    def encode_image(self, image, filepath):
        """
        Use the encode_images method to emulate a single image encoding.
        """
        return self.encode_images([image], [filepath])[0]

    def encode_images(self, images, filepaths):
        """
        Encode a batch of input images. Images must be the same dimension.
        """
        batch_size = len(images)
        if batch_size != len(filepaths):
            raise ValueError("Mismatch between number of images and filepaths.")

        full_filenames = [
            self._generate_filename(filepath)[0] for filepath in filepaths
        ]

        # Check cache for each image and filter out already cached ones
        uncached_image_indices = [
            i
            for i, filename in enumerate(full_filenames)
            if not self.data_backend.exists(filename)
        ]
        uncached_images = [images[i] for i in uncached_image_indices]

        if not uncached_images:
            # If all images are cached, simply load them
            latents = [self.load_from_cache(filename) for filename in full_filenames]
        else:
            # Only process images not found in cache
            with torch.no_grad():
                processed_images = torch.stack(uncached_images).to(
                    self.accelerator.device, dtype=torch.bfloat16
                )
                latents_uncached = self.vae.encode(
                    processed_images
                ).latent_dist.sample()
                latents_uncached = latents_uncached * self.vae.config.scaling_factor

            # Prepare final latents list by combining cached and newly computed latents
            latents = []
            cached_idx, uncached_idx = 0, 0
            for i in range(batch_size):
                if i in uncached_image_indices:
                    latents.append(latents_uncached[uncached_idx])
                    uncached_idx += 1
                else:
                    latents.append(self.load_from_cache(full_filenames[i]))
                    cached_idx += 1

        return latents

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

    def process_buckets(self, bucket_manager):
        processed_images = self._list_cached_images()
        batch_data, batch_filepaths, vae_input_images, vae_input_filepaths = (
            [],
            [],
            [],
            [],
        )

        aspect_bucket_cache = bucket_manager.read_cache().copy()

        for bucket in aspect_bucket_cache:
            relevant_files = [
                f
                for f in aspect_bucket_cache[bucket]
                if os.path.splitext(os.path.basename(f))[0] not in processed_images
            ]
            if len(relevant_files) == 0:
                continue

            for raw_filepath in tqdm(
                relevant_files, desc=f"Processing bucket {bucket}"
            ):
                if type(raw_filepath) == str or len(raw_filepath) == 1:
                    filepath = raw_filepath
                elif len(raw_filepath) == 2:
                    idx, filepath = raw_filepath
                else:
                    raise ValueError(f"Received unknown filepath value: {raw_filepath}")

                try:
                    image = self.data_backend.read_image(filepath)
                    image = MultiaspectImage.prepare_image(image, self.resolution)
                    pixel_values = self.transform(image).to(
                        self.accelerator.device, dtype=self.vae.dtype
                    )
                    vae_input_images.append(pixel_values)
                    vae_input_filepaths.append(filepath)
                except Exception as e:
                    logger.error(f"Error processing image {filepath}: {e}")
                    if self.delete_problematic_images:
                        self.data_backend.delete(filepath)
                    else:
                        raise e

                # If VAE input batch is ready
                if len(vae_input_images) >= self.vae_batch_size:
                    latents_batch = self.encode_images(
                        vae_input_images, vae_input_filepaths
                    )
                    batch_data.extend(latents_batch)
                    batch_filepaths.extend(
                        [self._generate_filename(f)[0] for f in vae_input_filepaths]
                    )
                    vae_input_images, vae_input_filepaths = [], []

                # If write batch is ready
                if len(batch_filepaths) >= self.write_batch_size:
                    self.data_backend.write_batch(batch_filepaths, batch_data)
                    batch_filepaths.clear()
                    batch_data.clear()

            # Handle remainders after processing the bucket
            if vae_input_images:  # If there are images left to be encoded
                latents_batch = self.encode_images(
                    vae_input_images, vae_input_filepaths
                )
                batch_data.extend(latents_batch)
                batch_filepaths.extend(
                    [self._generate_filename(f)[0] for f in vae_input_filepaths]
                )
                vae_input_images, vae_input_filepaths = [], []

            # Write the remaining batches
            if batch_data:
                self.data_backend.write_batch(batch_filepaths, batch_data)
                batch_filepaths.clear()
                batch_data.clear()
