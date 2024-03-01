import os, torch, logging, traceback
from concurrent.futures import ThreadPoolExecutor
from random import shuffle
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from numpy import str_ as numpy_str
from helpers.multiaspect.image import MultiaspectImage
from helpers.data_backend.base import BaseDataBackend
from helpers.metadata.backends.base import MetadataBackend
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
        metadata_backend: MetadataBackend,
        instance_data_root: str,
        data_backend: BaseDataBackend,
        cache_dir="vae_cache",
        resolution: float = 1024,
        maximum_image_size: float = None,
        target_downsample_size: float = None,
        delete_problematic_images: bool = False,
        write_batch_size: int = 25,
        read_batch_size: int = 25,
        process_queue_size: int = 16,
        vae_batch_size: int = 4,
        resolution_type: str = "pixel",
        minimum_image_size: int = None,
        max_workers: int = 32,
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
        if len(self.cache_dir) > 0 and self.cache_dir[-1] == "/":
            # Remove trailing slash
            self.cache_dir = self.cache_dir[:-1]
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
        self.metadata_backend = metadata_backend
        self.max_workers = max_workers
        if (maximum_image_size and not target_downsample_size) or (
            target_downsample_size and not maximum_image_size
        ):
            raise ValueError(
                "Both maximum_image_size and target_downsample_size must be specified."
                f"Only {'maximum_image_size' if maximum_image_size else 'target_downsample_size'} was specified."
            )
        self.maximum_image_size = maximum_image_size
        self.target_downsample_size = target_downsample_size

    def debug_log(self, msg: str):
        logger.debug(f"{self.rank_info}{msg}")

    def generate_vae_cache_filename(self, filepath: str) -> tuple:
        """Get the cache filename for a given image filepath and its base name."""
        # Extract the base name from the filepath and replace the image extension with .pt
        base_filename = os.path.splitext(os.path.basename(filepath))[0] + ".pt"
        # Find the subfolders the sample was in, and replace the instance_data_root with the cache_dir
        subfolders = os.path.dirname(filepath).replace(self.instance_data_root, "")
        if len(subfolders) > 0 and subfolders[0] == "/":
            subfolders = subfolders[1:]
            full_filename = os.path.join(self.cache_dir, subfolders, base_filename)
        else:
            full_filename = os.path.join(self.cache_dir, base_filename)
        return full_filename, base_filename

    def _image_filename_from_vaecache_filename(self, filepath: str) -> str:
        generated_names = self.generate_vae_cache_filename(filepath)
        logger.debug(f"VAE cache generated names: {generated_names}")
        test_filepath_png = f"{os.path.splitext(generated_names[0])[0]}.png"
        logger.debug(f"Translated into test_filepath_png: {test_filepath_png}")
        if str(self.cache_dir) in test_filepath_png:
            # replace cache_dir with instance_data_root:
            test_filepath_png = test_filepath_png.replace(
                self.cache_dir, self.instance_data_root
            )
            logger.debug(
                f"Replacing cache_dir value {self.cache_dir} with {self.instance_data_root} in {test_filepath_png}"
            )
        elif str(self.instance_data_root) not in test_filepath_png:
            test_filepath_png = os.path.join(self.instance_data_root, test_filepath_png)
            logger.debug(
                f"Joining instance_data_root value {self.instance_data_root} with {test_filepath_png}"
            )

        test_filepath_jpg = os.path.splitext(test_filepath_png)[0] + ".jpg"
        return test_filepath_png, test_filepath_jpg

    def already_cached(self, filepath: str) -> bool:
        test_path = self.generate_vae_cache_filename(filepath)[0]
        if self.data_backend.exists(test_path):
            # self.debug_log(f"Skipping {test_path} because it is already in the cache")
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

    def discover_all_files(self):
        """Identify all files in the data backend."""
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

    def init_vae(self):
        from diffusers import AutoencoderKL

        args = StateTracker.get_args()
        vae_path = (
            args.pretrained_model_name_or_path
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        precached_vae = StateTracker.get_vae()
        self.debug_log(f"Was the VAE loaded? {precached_vae}")
        self.vae = precached_vae or AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            force_upcast=False,
        ).to(self.accelerator.device)
        StateTracker.set_vae(self.vae)

    def rebuild_cache(self):
        """
        First, we'll clear the cache before rebuilding it.
        """
        self.debug_log("Rebuilding cache.")
        self.debug_log("-> Clearing cache objects")
        self.clear_cache()
        self.debug_log("-> Split tasks between GPU(s)")
        self.split_cache_between_processes()
        self.debug_log("-> Load VAE")
        self.init_vae()
        self.debug_log("-> Process VAE cache")
        self.process_buckets()
        self.debug_log("-> Completed cache rebuild")
        if self.accelerator.is_local_main_process:
            self.debug_log("Updating StateTracker with new VAE cache entry list.")
            StateTracker.set_vae_cache_files(
                self.data_backend.list_files(
                    instance_data_root=self.cache_dir,
                    str_pattern="*.pt",
                ),
                data_backend_id=self.id,
            )
        self.accelerator.wait_for_everyone()

    def clear_cache(self):
        """
        Clear all .pt files in our data backend's cache prefix, as obtained from self.discover_all_files().

        We can't simply clear the directory, because it might be mixed with the image samples (in the case of S3)

        We want to thread this, using the data_backend.delete function as the worker function.
        """
        futures = []
        all_cache_files = StateTracker.get_vae_cache_files(data_backend_id=self.id)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for filename in all_cache_files:
                full_path = os.path.join(self.cache_dir, filename)
                self.debug_log(f"Would delete: {full_path}")
                futures.append(executor.submit(self.data_backend.delete, full_path))
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Deleting files for backend {self.id}",
                position=get_rank(),
                ncols=125,
                leave=False,
            ):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error deleting file {filename}: {e}")
                    self.debug_log(f"Error traceback: {traceback.format_exc()}")
                    raise e
        # Clear the StateTracker list of VAE objects:
        StateTracker.set_vae_cache_files([], data_backend_id=self.id)

    def _list_cached_images(self):
        """
        Return a set of filenames (without the .pt extension) that have been processed.
        """
        # Extract array of tuple into just, an array of files:
        pt_files = StateTracker.get_vae_cache_files(data_backend_id=self.id)
        # Extract just the base filename without the extension
        results = {os.path.splitext(f)[0] for f in pt_files}
        # self.debug_log(
        #     f"Found {len(pt_files)} cached files in {self.cache_dir} (truncated): {list(results)[:5]}"
        # )
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
                # self.debug_log(
                #     f"Skipping {f} because it is already in the processed images list"
                # )
                continue
            if f not in self.local_unprocessed_files:
                # self.debug_log(
                #     f"Skipping {f} because it is not in local unprocessed files (truncated): {self.local_unprocessed_files[:5]}"
                # )
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
        # # Print the first 5 as a debug log:
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
        uncached_image_paths = [
            filepaths[i]
            for i, filename in enumerate(full_filenames)
            if i in uncached_image_indices
        ]
        # if len(uncached_image_indices) > 0:
        #     self.debug_log(
        #         f"Found {len(uncached_image_indices)} uncached images (truncated): {uncached_image_indices[:5]}"
        #     )
        #     self.debug_log(
        #         f"Received full filenames {len(full_filenames)} (truncated): {full_filenames[:5]}"
        #     )
        uncached_images = [images[i] for i in uncached_image_indices]

        if len(uncached_image_indices) > 0 and load_from_cache:
            # We wanted only uncached images. Something went wrong.
            raise Exception(
                f"Some images were not correctly cached during the VAE Cache operations. Ensure --skip_file_discovery=vae is not set.\nProblematic images: {uncached_image_paths}"
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
        self.debug_log(f"We have {qlen} latents to write to disk.")
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
        self.metadata_backend.save_image_metadata()
        self.data_backend.write_batch(filepaths, latents)

    def _process_images_in_batch(self) -> None:
        """Process a queue of images. This method assumes our batch size has been reached.

        Returns:
            None
        """
        try:
            filepaths = []
            qlen = self.process_queue.qsize()
            is_final_sample = False
            for idx in range(0, qlen):
                if idx == qlen - 1:
                    is_final_sample = True
                filepath, image, aspect_bucket = self.process_queue.get()
                filepaths.append(filepath)
                self.debug_log(f"Processing {filepath}")
                if self.minimum_image_size is not None:
                    if not self.metadata_backend.meets_resolution_requirements(
                        image_path=filepath,
                    ):
                        self.debug_log(
                            f"Skipping {filepath} because it does not meet the minimum image size requirement of {self.minimum_image_size}"
                        )
                        continue
                image, crop_coordinates, new_aspect_ratio = (
                    MultiaspectImage.prepare_image(
                        image=image,
                        resolution=self.resolution,
                        resolution_type=self.resolution_type,
                        id=self.id,
                    )
                )
                pixel_values = self.transform(image).to(
                    self.accelerator.device, dtype=self.vae.dtype
                )
                self.vae_input_queue.put(
                    (pixel_values, filepath, aspect_bucket, is_final_sample)
                )
                # Update the crop_coordinates in the metadata document
                self.metadata_backend.set_metadata_attribute_by_filepath(
                    filepath=filepath,
                    attribute="crop_coordinates",
                    value=crop_coordinates,
                    update_json=False,
                )
                self.debug_log(f"Completed processing {filepath}")
        except Exception as e:
            logger.error(f"Error processing images {filepaths}: {e}")
            self.debug_log(f"Error traceback: {traceback.format_exc()}")
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
                batch_aspect_bucket = None
                for idx in range(0, min(qlen, self.vae_batch_size)):
                    pixel_values, filepath, aspect_bucket, is_final_sample = (
                        self.vae_input_queue.get()
                    )
                    self.debug_log(
                        f"Queue values: {pixel_values.shape}, {filepath}, {aspect_bucket}, {is_final_sample}"
                    )
                    if batch_aspect_bucket is None:
                        batch_aspect_bucket = aspect_bucket
                    vae_input_images.append(pixel_values)
                    vae_input_filepaths.append(filepath)
                    vae_output_filepaths.append(
                        self.generate_vae_cache_filename(filepath)[0]
                    )
                    if is_final_sample:
                        # When we have fewer samples in a bucket than our VAE batch size might indicate,
                        # we need to respect is_final_sample value and not retrieve the *next* element yet.
                        break

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
            # Remove all of the errored images from the bucket. They will be captured on restart.
            for filepath in vae_input_filepaths:
                self.metadata_backend.remove_image(filepath)
            self.debug_log(f"Error traceback: {traceback.format_exc()}")
            raise Exception(
                f"Error encoding images {vae_input_filepaths}: {e}, traceback: {traceback.format_exc()}"
            )

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
            read_queue_item = self.read_queue.get()
            self.debug_log(f"Read path '{read_queue_item}' from read_queue.")
            path, aspect_bucket = read_queue_item
            self.debug_log(f"Read path '{path}' from read_queue.")
            filepaths.append(path)
        self.debug_log(f"Beginning to read images in batch: {filepaths}")
        available_filepaths, batch_output = self.data_backend.read_image_batch(
            filepaths, delete_problematic_images=self.delete_problematic_images
        )
        missing_image_count = len(filepaths) - len(available_filepaths)
        if len(available_filepaths) != len(filepaths):
            logging.warning(
                f"Failed to request {missing_image_count} sample{'s' if missing_image_count > 1 else ''} during batched read, out of {len(filepaths)} total samples requested."
                " These samples likely do not exist in the storage pool any longer."
            )
        for filepath, element in zip(available_filepaths, batch_output):
            if type(filepath) != str:
                raise ValueError(
                    f"Received unknown filepath type ({type(filepath)}) value: {filepath}"
                )
            # Add the element to the queue for later processing.
            # This allows us to have separate read and processing queue size limits.
            self.process_queue.put((filepath, element, aspect_bucket))

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

    def _accumulate_read_queue(self, filepath, aspect_bucket):
        self.debug_log(
            f"Adding {filepath} to read queue because it is in local unprocessed files"
        )
        self.read_queue.put((filepath, aspect_bucket))

    def _process_futures(self, futures: list, executor: ThreadPoolExecutor):
        completed_futures = []
        for future in as_completed(futures):
            try:
                future.result()
                completed_futures.append(future)
            except Exception as e:
                logging.error(
                    f"An error occurred in a future: {e}, file {e.__traceback__.tb_frame}, {e.__traceback__.tb_lineno}, future traceback {traceback.format_exc()}"
                )
                completed_futures.append(future)
        return [f for f in futures if f not in completed_futures]

    def process_buckets(self):
        futures = []
        processed_images = self._list_cached_images()
        aspect_bucket_cache = self.metadata_backend.read_cache().copy()

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
                statistics = {
                    "not_local": 0,
                    "already_cached": 0,
                    "cached": 0,
                    "total": 0,
                }
                for raw_filepath in tqdm(
                    relevant_files,
                    desc=f"Processing bucket {bucket}",
                    position=get_rank(),
                    ncols=125,
                    leave=False,
                ):
                    statistics["total"] += 1
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
                            f"Skipping {raw_filepath} because it is not in local unprocessed files:"
                            f"\n -> {test_filepath_jpg}"
                            f"\n -> {test_filepath_png}"
                            f"\n -> {self.local_unprocessed_files[:5]}"
                        )
                        statistics["not_local"] += 1
                        continue
                    try:
                        # Convert whatever we have, into the VAE cache basename.
                        filepath = self._process_raw_filepath(raw_filepath)
                        # Does it exist on the backend?
                        if self.already_cached(filepath):
                            statistics["already_cached"] += 1
                            continue
                        # It does not exist. We can add it to the read queue.
                        self.debug_log(f"Adding {filepath} to the read queue.")
                        self._accumulate_read_queue(filepath, aspect_bucket=bucket)
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
                            statistics["cached"] += 1
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
                        self.debug_log(f"Error traceback: {traceback.format_exc()}")
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
                    import json

                    logger.info(f"Bucket {bucket} caching results: {statistics}")
                    self.debug_log(
                        "Completed process_buckets, all futures have been returned."
                    )
                except Exception as e:
                    logger.error(f"Fatal error when processing bucket {bucket}: {e}")
                    continue

    def scan_cache_contents(self):
        """
        A generator method that iterates over the VAE cache, yielding each cache file's path and its contents.

        This is likely a very expensive operation for extra-large cloud datasets, but it could save time and
        computational resources if finding a problem with surgical precision can prevent the need for removing
        all cache entries in a dataset for a complete rebuild.

        Yields:
            Tuple[str, Any]: A tuple containing the file path and its contents.
        """
        try:
            all_cache_files = StateTracker.get_vae_cache_files(data_backend_id=self.id)
            for cache_file in all_cache_files:
                full_path = os.path.join(self.cache_dir, cache_file)
                cache_content = self._read_from_storage(full_path)
                yield (full_path, cache_content)
        except Exception as e:
            logger.error(f"Error in scan_cache_contents: {e}")
            self.debug_log(f"Error traceback: {traceback.format_exc()}")
