import os, torch, logging, traceback
from concurrent.futures import ThreadPoolExecutor
from random import shuffle
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from numpy import str_ as numpy_str
from helpers.multiaspect.image import MultiaspectImage
from helpers.data_backend.base import BaseDataBackend
from helpers.multiaspect.bucket import BucketManager
from helpers.training.state_tracker import StateTracker
from helpers.training.multi_process import _get_rank as get_rank
from helpers.training.multi_process import rank_info
from queue import Queue
from concurrent.futures import as_completed

logger = logging.getLogger("VAECache")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL") or "INFO")


class VAECache:
    read_queue = Queue()
    process_queue = Queue()
    write_queue = Queue()
    vae_input_queue = Queue()

    def __init__(
        self,
        id: str,
        vae,
        accelerator,
        bucket_manager: BucketManager,
        instance_data_root: str,
        data_backend: BaseDataBackend,
        cache_dir="vae_cache",
        resolution: float = 1024,
        delete_problematic_images: bool = False,
        write_batch_size: int = 25,
        read_batch_size: int = 25,
        process_queue_size: int = 16,
        vae_batch_size: int = 4,
        resolution_type: str = "pixel",
        minimum_image_size: int = None,
    ):
        self.id = id
        if data_backend.id != id:
            raise ValueError(
                f"VAECache received incorrect data_backend: {data_backend}"
            )
        self.data_backend = data_backend
        self.vae = vae
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        self.resolution = resolution
        self.resolution_type = resolution_type
        self.minimum_image_size = minimum_image_size
        self.data_backend.create_directory(self.cache_dir)
        self.delete_problematic_images = delete_problematic_images
        self.write_batch_size = write_batch_size
        self.read_batch_size = read_batch_size
        self.process_queue_size = process_queue_size
        self.vae_batch_size = vae_batch_size
        self.instance_data_root = instance_data_root
        self.transform = MultiaspectImage.get_image_transforms()
        self.rank_info = rank_info()
        self.bucket_manager = bucket_manager

    def debug_log(self, msg: str):
        logger.debug(f"{self.rank_info}{msg}")

    def generate_vae_cache_filename(self, filepath: str) -> tuple:
        """Get the cache filename for a given image filepath and its base name."""
        # Extract the base name from the filepath and replace the image extension with .pt
        base_filename = os.path.splitext(os.path.basename(filepath))[0] + ".pt"
        full_filename = os.path.join(self.cache_dir, base_filename)
        return full_filename, base_filename

    def _image_filename_from_vaecache_filename(self, filepath: str) -> str:
        test_filepath_png = (
            f"{os.path.splitext(self.generate_vae_cache_filename(filepath)[0])[0]}.png"
        )
        if str(self.cache_dir) in test_filepath_png:
            # replace cache_dir with instance_data_root:
            test_filepath_png = test_filepath_png.replace(
                self.cache_dir, self.instance_data_root
            )
        elif str(self.instance_data_root) not in test_filepath_png:
            test_filepath_png = os.path.join(self.instance_data_root, test_filepath_png)

        test_filepath_jpg = (
            f"{os.path.splitext(self.generate_vae_cache_filename(filepath)[0])[0]}.jpg"
        )

        return test_filepath_png, test_filepath_jpg

    def already_cached(self, filepath: str) -> bool:
        if self.data_backend.exists(self.generate_vae_cache_filename(filepath)[0]):
            self.debug_log(f"Skipping {filepath} because it is already in the cache")
            return True
        return False

    def _read_from_storage(self, filename: str) -> torch.Tensor:
        """Read a cache object from the storage backend.

        Args:
            filename (str): The path to the cache item, eg. `vae_cache/foo.pt`

        Returns:
            torch.Tensor: The cached Tensor object.
        """
        return self.data_backend.torch_load(filename)

    def retrieve_from_cache(self, filepath: str):
        """
        Use the encode_images method to emulate a single image encoding.
        """
        return self.encode_images([None], [filepath])[0]

    def discover_all_files(self, directory: str = None):
        """Identify all files in a directory."""
        all_image_files = StateTracker.get_image_files(
            data_backend_id=self.id
        ) or StateTracker.set_image_files(
            self.data_backend.list_files(
                instance_data_root=self.instance_data_root,
                str_pattern="*.[jJpP][pPnN][gG]",
            ),
            data_backend_id=self.id,
        )
        # This isn't returned, because we merely check if it's stored, or, store it.
        (
            StateTracker.get_vae_cache_files(data_backend_id=self.id)
            or StateTracker.set_vae_cache_files(
                self.data_backend.list_files(
                    instance_data_root=self.cache_dir,
                    str_pattern="*.pt",
                ),
                data_backend_id=self.id,
            )
        )
        self.debug_log(
            f"VAECache discover_all_files found {len(all_image_files)} images"
        )
        return all_image_files

    def _list_cached_images(self):
        """
        Return a set of filenames (without the .pt extension) that have been processed.
        """
        # Extract array of tuple into just, an array of files:
        pt_files = StateTracker.get_vae_cache_files(data_backend_id=self.id)
        # Extract just the base filename without the extension
        results = {os.path.splitext(f)[0] for f in pt_files}
        logging.debug(
            f"Found {len(pt_files)} cached files in {self.cache_dir} (truncated): {list(results)[:5]}"
        )
        return results

    def discover_unprocessed_files(self, directory: str = None):
        """Identify files that haven't been processed yet."""
        all_image_files = StateTracker.get_image_files(data_backend_id=self.id)
        existing_cache_files = StateTracker.get_vae_cache_files(data_backend_id=self.id)
        self.debug_log(
            f"discover_unprocessed_files found {len(all_image_files)} images from StateTracker (truncated): {list(all_image_files)[:5]}"
        )
        self.debug_log(
            f"discover_unprocessed_files found {len(existing_cache_files)} already-processed cache files (truncated): {list(existing_cache_files)[:5]}"
        )

        # Convert cache filenames to their corresponding image filenames
        existing_image_filenames = {
            os.path.splitext(
                self._image_filename_from_vaecache_filename(cache_file)[0]
            )[0]
            for cache_file in existing_cache_files
        }

        # Identify unprocessed files
        unprocessed_files = [
            file
            for file in all_image_files
            if os.path.splitext(file)[0] not in existing_image_filenames
        ]

        self.debug_log(
            f"discover_unprocessed_files found {len(unprocessed_files)} unprocessed files (truncated): {list(unprocessed_files)[:5]}"
        )
        return unprocessed_files

    def _reduce_bucket(
        self,
        bucket: str,
        aspect_bucket_cache: dict,
        processed_images: dict,
        do_shuffle: bool = True,
    ):
        """
        Given a bucket, return the relevant files for that bucket.
        """
        relevant_files = []
        for f in aspect_bucket_cache[bucket]:
            if os.path.splitext(f)[0] in processed_images:
                self.debug_log(
                    f"Skipping {f} because it is already in the processed images list"
                )
                continue
            if f not in self.local_unprocessed_files:
                self.debug_log(
                    f"Skipping {f} because it is not in local unprocessed files (truncated): {self.local_unprocessed_files[:5]}"
                )
                continue
            relevant_files.append(f)
        if do_shuffle:
            shuffle(relevant_files)
        self.debug_log(
            f"Reduced bucket {bucket} down from {len(aspect_bucket_cache[bucket])} to {len(relevant_files)} relevant files."
            f" Our system has {len(self.local_unprocessed_files)} images in its assigned slice for processing."
        )
        return relevant_files

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
        self.debug_log(
            f"Before splitting, we had {len(all_unprocessed_files)} unprocessed files. After splitting, we have {len(self.local_unprocessed_files)} unprocessed files."
        )
        # Print the first 5 as a debug log:
        self.debug_log(
            f"Local unprocessed files: {self.local_unprocessed_files[:5]} (truncated)"
        )

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
        logger.debug(
            f"Found {len(uncached_image_indices)} uncached images (truncated): {uncached_image_indices[:5]}"
        )
        logger.debug(
            f"Received full filenames {len(full_filenames)} (truncated): {full_filenames[:5]}"
        )
        uncached_images = [images[i] for i in uncached_image_indices]

        if len(uncached_image_indices) > 0 and load_from_cache:
            # We wanted only uncached images. Something went wrong.
            raise Exception(
                f"Some images were not correctly cached during the VAE Cache operations. Ensure --skip_file_discovery=vae is not set.\nProblematic images: {uncached_images}"
            )

        if load_from_cache:
            # If all images are cached, simply load them
            latents = [self._read_from_storage(filename) for filename in full_filenames]
        elif len(uncached_images) > 0:
            # Only process images not found in cache
            with torch.no_grad():
                # Debug log the size of each image:
                for image in uncached_images:
                    self.debug_log(f"Image size: {image.size()}")
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
                    latents.append(self._read_from_storage(full_filenames[i]))
                    cached_idx += 1
        else:
            return None
        return latents

    def _write_latents_in_batch(self):
        # Pull the 'filepaths' and 'latents' from self.write_queue
        filepaths, latents = [], []
        qlen = self.write_queue.qsize()
        logger.debug(f"We have {qlen} latents to write to disk.")
        for idx in range(0, qlen):
            output_file, filepath, latent_vector = self.write_queue.get()
            file_extension = os.path.splitext(output_file)[1]
            if (
                file_extension == ".png"
                or file_extension == ".jpg"
                or file_extension == ".jpeg"
            ):
                raise ValueError(
                    f"Cannot write a latent embedding to an image path, {output_file}"
                )
            filepaths.append(output_file)
            latents.append(latent_vector)
        self.bucket_manager.save_image_metadata()
        self.data_backend.write_batch(filepaths, latents)

    def _process_images_in_batch(self) -> None:
        """Process a queue of images. This method assumes our batch size has been reached.

        Returns:
            None
        """
        try:
            filepaths = []
            qlen = self.process_queue.qsize()
            for idx in range(0, qlen):
                filepath, image = self.process_queue.get()
                filepaths.append(filepath)
                self.debug_log(f"Processing {filepath}")
                if self.minimum_image_size is not None:
                    if not self.bucket_manager.meets_resolution_requirements(
                        image_path=filepath,
                        minimum_image_size=self.minimum_image_size,
                        resolution_type=self.resolution_type,
                    ):
                        self.debug_log(
                            f"Skipping {filepath} because it does not meet the minimum image size requirement of {self.minimum_image_size}"
                        )
                        continue
                image, crop_coordinates = MultiaspectImage.prepare_image(
                    image, self.resolution, self.resolution_type, self.id
                )
                pixel_values = self.transform(image).to(
                    self.accelerator.device, dtype=self.vae.dtype
                )
                self.vae_input_queue.put((pixel_values, filepath))
                # Update the crop_coordinates in the metadata document
                self.bucket_manager.set_metadata_attribute_by_filepath(
                    filepath=filepath,
                    attribute="crop_coordinates",
                    value=crop_coordinates,
                    update_json=False,
                )
                self.debug_log(f"Completed processing {filepath}")
        except Exception as e:
            logger.error(f"Error processing images {filepaths}: {e}")
            logging.debug(f"Error traceback: {traceback.format_exc()}")
            raise e

    def _encode_images_in_batch(self) -> None:
        """Encode the batched Image objects using the VAE model.

        Raises:
            ValueError: If we receive any invalid results.
        """
        try:
            qlen = self.vae_input_queue.qsize()
            if qlen == 0:
                return
            while qlen > 0:
                vae_input_images, vae_input_filepaths, vae_output_filepaths = [], [], []
                for idx in range(0, min(qlen, self.vae_batch_size)):
                    pixel_values, filepath = self.vae_input_queue.get()
                    vae_input_images.append(pixel_values)
                    vae_input_filepaths.append(filepath)
                    vae_output_filepaths.append(
                        self.generate_vae_cache_filename(filepath)[0]
                    )
                latents = self.encode_images(
                    vae_input_images, vae_input_filepaths, load_from_cache=False
                )
                if latents is None:
                    raise ValueError("Received None from encode_images")
                for output_file, latent_vector, filepath in zip(
                    vae_output_filepaths, latents, vae_input_filepaths
                ):
                    if latent_vector is None:
                        raise ValueError(
                            f"Latent vector is None for filepath {filepath}"
                        )
                    self.write_queue.put((output_file, filepath, latent_vector))
                qlen = self.vae_input_queue.qsize()
        except Exception as e:
            logger.error(f"Error encoding images {vae_input_filepaths}: {e}")
            logging.debug(f"Error traceback: {traceback.format_exc()}")
            raise e

    def read_images_in_batch(self) -> None:
        """Immediately read a batch of images.

        The images are added to a Queue, for later processing.

        Args:
            filepaths (list): A list of image file paths.

        Returns:
            None
        """
        filepaths = []
        qlen = self.read_queue.qsize()
        for idx in range(0, qlen):
            path = self.read_queue.get()
            logger.debug(f"Read path '{path}' from read_queue.")
            filepaths.append(path)
        logger.debug(f"Beginning to read images in batch: {filepaths}")
        available_filepaths, batch_output = self.data_backend.read_image_batch(
            filepaths, delete_problematic_images=self.delete_problematic_images
        )
        if len(available_filepaths) != len(filepaths):
            logging.warning(
                f"Received {len(batch_output)} items from the batch read, when we requested {len(filepaths)}: {batch_output}"
            )
        for filepath, element in zip(available_filepaths, batch_output):
            if type(filepath) != str:
                raise ValueError(
                    f"Received unknown filepath type ({type(filepath)}) value: {filepath}"
                )
            # Add the element to the queue for later processing.
            # This allows us to have separate read and processing queue size limits.
            self.process_queue.put((filepath, element))

    def _process_raw_filepath(self, raw_filepath: str):
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
        return filepath

    def _accumulate_read_queue(self, filepath):
        self.debug_log(
            f"Adding {filepath} to read queue because it is in local unprocessed files"
        )
        self.read_queue.put(filepath)

    def _process_futures(self, futures: list, executor: ThreadPoolExecutor):
        completed_futures = []
        for future in as_completed(futures):
            try:
                future.result()
                completed_futures.append(future)
            except Exception as e:
                logging.error(
                    f"An error occurred in a future: {e}, file {e.__traceback__.tb_frame}, {e.__traceback__.tb_lineno}"
                )
                completed_futures.append(future)
        return [f for f in futures if f not in completed_futures]

    def process_buckets(self):
        futures = []
        processed_images = self._list_cached_images()
        aspect_bucket_cache = self.bucket_manager.read_cache().copy()

        # Extract and shuffle the keys of the dictionary
        do_shuffle = (
            os.environ.get("SIMPLETUNER_SHUFFLE_ASPECTS", "true").lower() == "true"
        )
        if do_shuffle:
            shuffled_keys = list(aspect_bucket_cache.keys())
            shuffle(shuffled_keys)

        with ThreadPoolExecutor() as executor:
            for bucket in shuffled_keys:
                relevant_files = self._reduce_bucket(
                    bucket, aspect_bucket_cache, processed_images, do_shuffle
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
                    filepath = self._process_raw_filepath(raw_filepath)
                    (
                        test_filepath_png,
                        test_filepath_jpg,
                    ) = self._image_filename_from_vaecache_filename(filepath)
                    if (
                        test_filepath_png not in self.local_unprocessed_files
                        and test_filepath_jpg not in self.local_unprocessed_files
                    ):
                        self.debug_log(
                            f"Skipping {raw_filepath} because it is not in local unprocessed files"
                        )
                        continue
                    try:
                        # Convert whatever we have, into the VAE cache basename.
                        filepath = self._process_raw_filepath(raw_filepath)
                        # Does it exist on the backend?
                        if self.already_cached(filepath):
                            continue
                        # It does not exist. We can add it to the read queue.
                        logger.debug(f"Adding {filepath} to the read queue.")
                        self._accumulate_read_queue(filepath)
                        # We will check to see whether the queue is ready.
                        if self.read_queue.qsize() >= self.read_batch_size:
                            # We have an adequate number of samples to read. Let's now do that in a batch, to reduce I/O wait.
                            future_to_read = executor.submit(self.read_images_in_batch)
                            futures.append(future_to_read)

                        # Now we try and process the images, if we have a process batch size large enough.
                        if self.process_queue.qsize() >= self.process_queue_size:
                            future_to_process = executor.submit(
                                self._process_images_in_batch
                            )
                            futures.append(future_to_process)

                        # Now we encode the images.
                        if self.vae_input_queue.qsize() >= self.vae_batch_size:
                            future_to_process = executor.submit(
                                self._encode_images_in_batch
                            )
                            futures.append(future_to_process)

                        # If we have accumulated enough write objects, we can write them to disk at once.
                        if self.write_queue.qsize() >= self.write_batch_size:
                            future_to_write = executor.submit(
                                self._write_latents_in_batch
                            )
                            futures.append(future_to_write)
                    except ValueError as e:
                        logger.error(f"Received fatal error: {e}")
                        raise e
                    except Exception as e:
                        logger.error(f"Error processing image {filepath}: {e}")
                        logging.debug(f"Error traceback: {traceback.format_exc()}")
                        raise e

                    # Now, see if we have any futures to complete, and execute them.
                    # Cleanly removes futures from the list, once they are completed.
                    futures = self._process_futures(futures, executor)

                try:
                    # Handle remainders after processing the bucket
                    if self.read_queue.qsize() > 0:
                        # We have an adequate number of samples to read. Let's now do that in a batch, to reduce I/O wait.
                        future_to_read = executor.submit(self.read_images_in_batch)
                        futures.append(future_to_read)

                    futures = self._process_futures(futures, executor)

                    # Now we try and process the images, if we have a process batch size large enough.
                    if self.process_queue.qsize() > 0:
                        future_to_process = executor.submit(
                            self._process_images_in_batch
                        )
                        futures.append(future_to_process)

                    futures = self._process_futures(futures, executor)

                    if self.vae_input_queue.qsize() > 0:
                        future_to_process = executor.submit(
                            self._encode_images_in_batch
                        )
                        futures.append(future_to_process)

                    futures = self._process_futures(futures, executor)

                    # Write the remaining batches. This is not strictly necessary, since they do not need to be written with matching dimensions.
                    # However, it's simply easiest to do this now, even if we have less-than a single batch size.
                    if self.write_queue.qsize() > 0:
                        future_to_write = executor.submit(self._write_latents_in_batch)
                        futures.append(future_to_write)

                    futures = self._process_futures(futures, executor)
                    logger.debug(
                        "Completed process_buckets, all futures have been returned."
                    )
                except Exception as e:
                    logger.error(f"Fatal error when processing bucket {bucket}: {e}")
                    continue
