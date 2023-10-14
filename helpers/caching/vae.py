import os, torch, logging
from random import shuffle
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from numpy import str_ as numpy_str
from helpers.multiaspect.image import MultiaspectImage
from helpers.data_backend.base import BaseDataBackend
from helpers.training.state_tracker import StateTracker
from helpers.training.multi_process import _get_rank as get_rank
from helpers.training.multi_process import rank_info

logger = logging.getLogger("VAECache")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")


class VAECache:
    def __init__(
        self,
        vae,
        accelerator,
        instance_data_root: str,
        data_backend: BaseDataBackend,
        cache_dir="vae_cache",
        resolution: float = 1024,
        delete_problematic_images: bool = False,
        write_batch_size: int = 25,
        vae_batch_size: int = 4,
        resolution_type: str = "pixel",
    ):
        self.data_backend = data_backend
        self.vae = vae
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        self.resolution = resolution
        self.resolution_type = resolution_type
        self.data_backend.create_directory(self.cache_dir)
        self.delete_problematic_images = delete_problematic_images
        self.write_batch_size = write_batch_size
        self.vae_batch_size = vae_batch_size
        self.instance_data_root = instance_data_root
        self.transform = MultiaspectImage.get_image_transforms()
        self.rank_info = rank_info()

    def debug_log(self, msg: str):
        logger.debug(f"{self.rank_info}{msg}")

    def generate_vae_cache_filename(self, filepath: str) -> tuple:
        """Get the cache filename for a given image filepath and its base name."""
        # Extract the base name from the filepath and replace the image extension with .pt
        base_filename = os.path.splitext(os.path.basename(filepath))[0] + ".pt"
        full_filename = os.path.join(self.cache_dir, base_filename)
        return full_filename, base_filename

    def save_to_cache(self, filename, embeddings):
        self.data_backend.torch_save(embeddings, filename)

    def load_from_cache(self, filename):
        return self.data_backend.torch_load(filename)

    def discover_all_files(self, directory: str = None):
        """Identify all files in a directory."""
        all_image_files = (
            StateTracker.get_image_files()
            or StateTracker.set_image_files(
                self.data_backend.list_files(
                    instance_data_root=self.instance_data_root,
                    str_pattern="*.[jJpP][pPnN][gG]",
                )
            )
        )
        # This isn't returned, because we merely check if it's stored, or, store it.
        (
            StateTracker.get_vae_cache_files()
            or StateTracker.set_vae_cache_files(
                self.data_backend.list_files(
                    instance_data_root=self.cache_dir,
                    str_pattern="*.pt",
                )
            )
        )
        self.debug_log(
            f"VAECache discover_all_files found {len(all_image_files)} images"
        )
        return all_image_files

    def discover_unprocessed_files(self, directory: str = None):
        """Identify files that haven't been processed yet."""
        all_image_files = StateTracker.get_image_files()
        existing_cache_files = StateTracker.get_vae_cache_files()
        self.debug_log(
            f"discover_unprocessed_files found {len(all_image_files)} images from StateTracker (truncated): {list(all_image_files)[:5]}"
        )
        self.debug_log(
            f"discover_unprocessed_files found {len(existing_cache_files)} already-processed cache files (truncated): {list(existing_cache_files)[:5]}"
        )
        cache_filenames = {
            self.generate_vae_cache_filename(file)[1] for file in all_image_files
        }
        self.debug_log(
            f"discover_unprocessed_files found {len(cache_filenames)} cache filenames (truncated): {list(cache_filenames)[:5]}"
        )
        unprocessed_files = {
            f"{os.path.splitext(file)[0]}.png"
            for file in cache_filenames
            if file not in existing_cache_files
        }

        return list(unprocessed_files)

    def _list_cached_images(self):
        """
        Return a set of filenames (without the .pt extension) that have been processed.
        """
        # Extract array of tuple into just, an array of files:
        pt_files = StateTracker.get_vae_cache_files()
        # Extract just the base filename without the extension
        results = {os.path.splitext(f)[0] for f in pt_files}
        logging.debug(
            f"Found {len(pt_files)} cached files in {self.cache_dir} (truncated): {list(results)[:5]}"
        )
        return results

    def encode_image(self, image, filepath):
        """
        Use the encode_images method to emulate a single image encoding.
        """
        return self.encode_images([image], [filepath])[0]

    def encode_images(self, images, filepaths, load_from_cache=True):
        """
        Encode a batch of input images. Images must be the same dimension.

        If load_from_cache=True, we read from the VAE cache rather than encode.
        If load_from_cache=True, we will throw an exception if the entry is not found.
        """
        batch_size = len(images)
        if batch_size != len(filepaths):
            raise ValueError("Mismatch between number of images and filepaths.")

        full_filenames = [
            self.generate_vae_cache_filename(filepath)[0] for filepath in filepaths
        ]

        # Check cache for each image and filter out already cached ones
        uncached_images = []
        uncached_image_indices = [
            i
            for i, filename in enumerate(full_filenames)
            if not self.data_backend.exists(filename)
        ]
        uncached_images = [images[i] for i in uncached_image_indices]

        if len(uncached_image_indices) > 0 and load_from_cache:
            # We wanted only uncached images. Something went wrong.
            raise Exception(
                f"Some images were not correctly cached during the VAE Cache operations. Ensure --skip_file_discovery=vae is not set.\nProblematic images: {uncached_images}"
            )

        if load_from_cache:
            # If all images are cached, simply load them
            latents = [self.load_from_cache(filename) for filename in full_filenames]
        elif len(uncached_images) > 0:
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
        else:
            return None
        return latents

    def split_cache_between_processes(self):
        all_unprocessed_files = self.discover_unprocessed_files(self.cache_dir)
        self.debug_log(
            f"All unprocessed files: {all_unprocessed_files[:5]} (truncated)"
        )
        # Use the accelerator to split the data
        with self.accelerator.split_between_processes(
            all_unprocessed_files
        ) as split_files:
            self.local_unprocessed_files = split_files
        # Print the first 5 as a debug log:
        self.debug_log(
            f"Local unprocessed files: {self.local_unprocessed_files[:5]} (truncated)"
        )

    def _process_image(self, filepath):
        full_filename, base_filename = self.generate_vae_cache_filename(filepath)
        if self.data_backend.exists(full_filename):
            return None

        try:
            image = self.data_backend.read_image(filepath)
            image, crop_coordinates = MultiaspectImage.prepare_image(
                image, self.resolution, self.resolution_type
            )
            pixel_values = self.transform(image).to(
                self.accelerator.device, dtype=self.vae.dtype
            )
            latents = self.encode_image(pixel_values, filepath)
            return latents
        except Exception as e:
            import traceback

            logger.error(f"Error processing image {filepath}: {e}")
            logging.debug(f"Error traceback: {traceback.format_exc()}")
            if self.delete_problematic_images:
                self.data_backend.delete(filepath)
            else:
                raise e

    def _accumulate_batch(self, latents, filepath, batch_data, batch_filepaths):
        full_filename, _ = self.generate_vae_cache_filename(filepath)
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

        # Extract and shuffle the keys of the dictionary
        shuffled_keys = list(aspect_bucket_cache.keys())
        shuffle(shuffled_keys)

        for bucket in shuffled_keys:
            relevant_files = [
                f
                for f in aspect_bucket_cache[bucket]
                if os.path.splitext(os.path.basename(f))[0] not in processed_images
                and f in self.local_unprocessed_files
            ]
            for sample in aspect_bucket_cache[bucket]:
                quick_piece = os.path.splitext(os.path.basename(sample))[0]
                if quick_piece in processed_images:
                    self.debug_log(
                        f"Skipping {quick_piece} because it is in processed images"
                    )
                    continue
                if sample not in self.local_unprocessed_files:
                    self.debug_log(
                        f"Skipping {sample} because it is not in local unprocessed files"
                    )
                    continue
                self.debug_log(
                    f"Processing bucket {bucket} sample {sample}  (quick_piece {quick_piece}) because it is in local unprocessed files"
                )
            self.debug_log(
                f"Reduced bucket {bucket} down from {len(aspect_bucket_cache[bucket])} to {len(relevant_files)} relevant files"
            )
            if len(relevant_files) == 0:
                continue

            for raw_filepath in tqdm(
                relevant_files,
                desc=f"Processing bucket {bucket}",
                position=get_rank(),
                ncols=100,
                leave=False,
            ):
                if type(raw_filepath) == str or len(raw_filepath) == 1:
                    filepath = raw_filepath
                elif len(raw_filepath) == 2:
                    basename, filepath = raw_filepath
                elif type(raw_filepath) == Path or type(raw_filepath) == numpy_str:
                    filepath = str(raw_filepath)
                else:
                    raise ValueError(
                        f"Received unknown filepath type ({type(raw_filepath)}) value: {raw_filepath}"
                    )
                test_filepath = f"{os.path.splitext(self.generate_vae_cache_filename(filepath)[1])[0]}.png"
                if test_filepath not in self.local_unprocessed_files:
                    self.debug_log(
                        f"Skipping {test_filepath} because it is not in local unprocessed files"
                    )
                    continue
                try:
                    # Does it exist on the backend?
                    if self.data_backend.exists(
                        self.generate_vae_cache_filename(filepath)[0]
                    ):
                        self.debug_log(
                            f"Skipping {filepath} because it is already in the cache"
                        )
                        continue
                    self.debug_log(
                        f"Processing {filepath} because it is in local unprocessed files"
                    )
                    image = self.data_backend.read_image(filepath)
                    image, crop_coordinates = MultiaspectImage.prepare_image(
                        image, self.resolution, self.resolution_type
                    )
                    aspect_ratio = float(round(image.width / image.height, 2))
                    pixel_values = self.transform(image).to(
                        self.accelerator.device, dtype=self.vae.dtype
                    )
                    vae_input_images.append(pixel_values)
                    vae_input_filepaths.append(filepath)
                except ValueError as e:
                    logger.error(f"Received fatal error: {e}")
                    raise e
                except Exception as e:
                    import traceback

                    logger.error(f"Error processing image {filepath}: {e}")
                    logging.debug(f"Error traceback: {traceback.format_exc()}")
                    if self.delete_problematic_images:
                        self.data_backend.delete(filepath)
                    else:
                        raise e

                # If VAE input batch is ready
                if len(vae_input_images) >= self.vae_batch_size:
                    self.debug_log(
                        f"Reached a VAE batch size of {self.vae_batch_size} pixel groups, so we will now encode them into latents."
                    )
                    latents_batch = self.encode_images(
                        vae_input_images, vae_input_filepaths, load_from_cache=False
                    )
                    if latents_batch is not None:
                        batch_data.extend(latents_batch)
                        batch_filepaths.extend(
                            [
                                self.generate_vae_cache_filename(f)[0]
                                for f in vae_input_filepaths
                            ]
                        )
                    vae_input_images, vae_input_filepaths = [], []

                # If write batch is ready
                if len(batch_filepaths) >= self.write_batch_size:
                    logging.debug(
                        f"We have accumulated {self.write_batch_size} latents, so we will now write them to disk."
                    )
                    self.data_backend.write_batch(batch_filepaths, batch_data)
                    batch_filepaths.clear()
                    batch_data.clear()

            # Handle remainders after processing the bucket
            if vae_input_images:  # If there are images left to be encoded
                latents_batch = self.encode_images(
                    vae_input_images, vae_input_filepaths, load_from_cache=False
                )
                if latents_batch is not None:
                    batch_data.extend(latents_batch)
                    batch_filepaths.extend(
                        [
                            self.generate_vae_cache_filename(f)[0]
                            for f in vae_input_filepaths
                        ]
                    )
                vae_input_images, vae_input_filepaths = [], []

            # Write the remaining batches
            if batch_data:
                self.data_backend.write_batch(batch_filepaths, batch_data)
                batch_filepaths.clear()
                batch_data.clear()
