import os
import torch
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from random import shuffle
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from numpy import str_ as numpy_str
from helpers.multiaspect.image import MultiaspectImage
from helpers.image_manipulation.training_sample import TrainingSample, PreparedSample
from helpers.data_backend.base import BaseDataBackend
from helpers.metadata.backends.base import MetadataBackend
from helpers.training.state_tracker import StateTracker
from helpers.training.multi_process import _get_rank as get_rank
from helpers.training.multi_process import rank_info
from queue import Queue
from concurrent.futures import as_completed
from hashlib import sha256
from helpers.training import image_file_extensions
from helpers.webhooks.mixin import WebhookMixin
from helpers.models.ltxvideo import normalize_ltx_latents
from helpers.models.wan import compute_wan_posterior

logger = logging.getLogger("VAECache")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


def prepare_sample(
    image: Image.Image = None,
    data_backend_id: str = None,
    filepath: str = None,
    model=None,
):
    metadata = StateTracker.get_metadata_by_filepath(
        filepath, data_backend_id=data_backend_id
    )
    data_backend = StateTracker.get_data_backend(data_backend_id)
    data_sampler = data_backend.get("sampler")
    image_data = image
    if image_data is None:
        image_data = data_sampler.yield_single_image(filepath)
    training_sample = TrainingSample(
        image=image_data,
        data_backend_id=data_backend_id,
        image_metadata=metadata,
        image_path=filepath,
        model=model,
    )
    is_cond = data_backend_id in StateTracker.get_conditioning_mappings().values()

    if is_cond:
        # On-demand VAE caching for conditioning inputs can get complicated.
        if data_backend_id in StateTracker.get_conditioning_mappings().values():
            conditioning_sample_path = training_sample.image_path()
            # locate the partner backend id
            for train_id, cond_id in StateTracker.get_conditioning_mappings().items():
                if cond_id == data_backend_id:
                    # We found a conditioning dataset.
                    train_data_backend = StateTracker.get_data_backend(train_id)
                    train_sample_path = training_sample.training_sample_path(training_dataset_id=train_id)
                    cond_meta = StateTracker.get_metadata_by_filepath(
                        conditioning_sample_path, data_backend_id=cond_id
                    )
                    if not cond_meta:
                        train_meta = train_data_backend["metadata_backend"].get_metadata_by_filepath(
                            train_sample_path
                        )
                        prepared_sample = training_sample.prepare_like(
                            TrainingSample(
                                image=None,
                                data_backend_id=train_id,
                                image_metadata=train_meta,
                                image_path=train_sample_path,
                                model=model,
                            )
                        )
                    else:
                        # prepare the sample independently of the training sample,
                        # since the metadata scan built an element for this.
                        # a metadata object will exist for conditioning samples that
                        # have their dataset configured to operate somewhat independently.
                        prepared_sample = training_sample.prepare()
    else:
        # If this VAECache is attached to a *training* dataset, we prepare the
        # sample for training, which includes cropping and resizing.
        prepared_sample = training_sample.prepare()

    return (
        prepared_sample.image,
        prepared_sample.crop_coordinates,
        prepared_sample.aspect_ratio,
    )


class VAECache(WebhookMixin):
    def __init__(
        self,
        id: str,
        model,
        vae,
        accelerator,
        metadata_backend: MetadataBackend,
        instance_data_dir: str,
        image_data_backend: BaseDataBackend,
        webhook_progress_interval: int = 100,
        cache_data_backend: BaseDataBackend = None,
        cache_dir="vae_cache",
        resolution: float = 1024,
        maximum_image_size: float = None,
        target_downsample_size: float = None,
        num_video_frames: int = 125,
        delete_problematic_images: bool = False,
        write_batch_size: int = 25,
        read_batch_size: int = 25,
        process_queue_size: int = 16,
        vae_batch_size: int = 4,
        resolution_type: str = "pixel",
        minimum_image_size: int = None,
        max_workers: int = 32,
        vae_cache_ondemand: bool = False,
        hash_filenames: bool = False,
        dataset_type: str = None,
    ):
        self.id = id
        self.dataset_type = dataset_type
        if image_data_backend and image_data_backend.id != id:
            raise ValueError(
                f"VAECache received incorrect image_data_backend: {image_data_backend}"
            )
        self.image_data_backend = image_data_backend
        self.cache_data_backend = (
            cache_data_backend if cache_data_backend is not None else image_data_backend
        )
        self.hash_filenames = hash_filenames
        self.vae = vae
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        if len(self.cache_dir) > 0 and self.cache_dir[-1] == "/":
            # Remove trailing slash
            self.cache_dir = self.cache_dir[:-1]
        if self.cache_data_backend and self.cache_data_backend.type == "local":
            self.cache_dir = os.path.abspath(self.cache_dir)
            self.cache_data_backend.create_directory(self.cache_dir)
        self.resolution = resolution
        self.resolution_type = resolution_type
        self.minimum_image_size = minimum_image_size
        self.webhook_progress_interval = webhook_progress_interval
        self.delete_problematic_images = delete_problematic_images
        self.write_batch_size = write_batch_size
        self.read_batch_size = read_batch_size
        self.process_queue_size = process_queue_size
        self.vae_batch_size = vae_batch_size
        self.instance_data_dir = instance_data_dir
        self.model = model
        self.transform_sample = model.get_transforms(dataset_type=dataset_type)
        self.num_video_frames = None
        if self.dataset_type == "video":
            self.num_video_frames = num_video_frames
        self.rank_info = rank_info()
        self.metadata_backend = metadata_backend
        if self.metadata_backend and not self.metadata_backend.image_metadata_loaded:
            self.metadata_backend.load_image_metadata()

        self.vae_cache_ondemand = vae_cache_ondemand

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
        self.read_queue = Queue()
        self.process_queue = Queue()
        self.write_queue = Queue()
        self.vae_input_queue = Queue()

    def debug_log(self, msg: str):
        logger.debug(f"{self.rank_info}{msg}")

    def generate_vae_cache_filename(self, filepath: str) -> tuple:
        """Get the cache filename for a given image filepath and its base name."""
        if filepath.endswith(".pt"):
            return filepath, os.path.basename(filepath)
        # Extract the base name from the filepath and replace the image extension with .pt
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        if self.hash_filenames:
            base_filename = str(sha256(str(base_filename).encode()).hexdigest())
        base_filename = str(base_filename) + ".pt"
        # Find the subfolders the sample was in, and replace the instance_data_dir with the cache_dir
        subfolders = ""
        if self.instance_data_dir is not None:
            subfolders = os.path.dirname(filepath).replace(self.instance_data_dir, "")
            subfolders = subfolders.lstrip(os.sep)

        if len(subfolders) > 0:
            full_filename = os.path.join(self.cache_dir, subfolders, base_filename)
            # logger.debug(
            #     f"full_filename: {full_filename} = os.path.join({self.cache_dir}, {subfolders}, {base_filename})"
            # )
        else:
            full_filename = os.path.join(self.cache_dir, base_filename)
            # logger.debug(
            #     f"full_filename: {full_filename} = os.path.join({self.cache_dir}, {base_filename})"
            # )
        return full_filename, base_filename

    def _image_filename_from_vaecache_filename(self, filepath: str) -> tuple[str, str]:
        test_filepath, _ = self.generate_vae_cache_filename(filepath)
        result = self.vae_path_to_image_path.get(test_filepath, None)

        return result

    def build_vae_cache_filename_map(self, all_image_files: list):
        """Build a map of image filepaths to their corresponding cache filenames."""
        self.image_path_to_vae_path = {}
        self.vae_path_to_image_path = {}
        for image_file in all_image_files:
            cache_filename, _ = self.generate_vae_cache_filename(image_file)
            if self.cache_data_backend.type == "local":
                cache_filename = os.path.abspath(cache_filename)
            self.image_path_to_vae_path[image_file] = cache_filename
            self.vae_path_to_image_path[cache_filename] = image_file

    def already_cached(self, filepath: str) -> bool:
        test_path = self.image_path_to_vae_path.get(filepath, None)
        if self.cache_data_backend.exists(test_path):
            return True
        return False

    def _read_from_storage(
        self, filename: str, hide_errors: bool = False
    ) -> torch.Tensor:
        """Read an image or cache object from the storage backend.

        Args:
            filename (str): The path to the cache item, eg. `vae_cache/foo.pt` or `instance_data_dir/foo.png`

        Returns:
            Image or cache object
        """
        if os.path.splitext(filename)[1] != ".pt":
            try:
                return self.image_data_backend.read_image(filename)
            except Exception as e:
                if self.delete_problematic_images:
                    self.metadata_backend.remove_image(filename)
                    self.image_data_backend.delete(filename)
                    self.debug_log(
                        f"Deleted {filename} because it was problematic: {e}"
                    )
                raise e
        try:
            torch_data = self.cache_data_backend.torch_load(filename)
            if isinstance(torch_data, torch.Tensor):
                torch_data = torch_data.to("cpu")
            elif isinstance(torch_data, dict):
                torch_data["latents"] = torch_data["latents"].to("cpu")

            return torch_data
        except Exception as e:
            if hide_errors:
                self.debug_log(
                    f"Filename: {filename}, returning None even though read_from_storage found no object, since hide_errors is True: {e}"
                )
                return None
            raise e

    def retrieve_from_cache(self, filepath: str):
        """
        Use the encode_images method to emulate a single image encoding.
        """
        return self.encode_images([None], [filepath])[0]

    def retreve_batch_from_cache(self, filepaths: list):
        """
        Use the encode_images method to emulate a batch of image encodings.
        """
        return self.encode_images([None] * len(filepaths), filepaths)

    def discover_all_files(self):
        """Identify all files in the data backend."""
        all_image_files = StateTracker.get_image_files(
            data_backend_id=self.id
        ) or StateTracker.set_image_files(
            self.image_data_backend.list_files(
                instance_data_dir=self.instance_data_dir,
                file_extensions=image_file_extensions,
            ),
            data_backend_id=self.id,
        )
        # This isn't returned, because we merely check if it's stored, or, store it.
        (
            StateTracker.get_vae_cache_files(data_backend_id=self.id)
            or StateTracker.set_vae_cache_files(
                self.cache_data_backend.list_files(
                    instance_data_dir=self.cache_dir,
                    file_extensions=["pt"],
                ),
                data_backend_id=self.id,
            )
        )
        self.debug_log(
            f"VAECache discover_all_files found {len(all_image_files)} images"
        )
        return all_image_files

    def init_vae(self):
        if StateTracker.get_args().model_family == "sana":
            from diffusers import AutoencoderDC as AutoencoderClass
        elif StateTracker.get_args().model_family == "ltxvideo":
            from diffusers import AutoencoderKLLTXVideo as AutoencoderClass
        elif StateTracker.get_args().model_family == "wan":
            from diffusers import AutoencoderKLWan as AutoencoderClass
        else:
            from diffusers import AutoencoderKL as AutoencoderClass

        args = StateTracker.get_args()
        vae_path = (
            args.pretrained_model_name_or_path
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        self.vae = self.model.get_vae()

    def rebuild_cache(self):
        """
        First, we'll clear the cache before rebuilding it.
        """
        self.debug_log("Rebuilding cache.")
        if self.accelerator.is_local_main_process:
            self.debug_log("Updating StateTracker with new VAE cache entry list.")
            StateTracker.set_vae_cache_files(
                self.cache_data_backend.list_files(
                    instance_data_dir=self.cache_dir,
                    file_extensions=["pt"],
                ),
                data_backend_id=self.id,
            )
        self.accelerator.wait_for_everyone()
        self.debug_log("-> Clearing cache objects")
        self.clear_cache()
        self.debug_log("-> Split tasks between GPU(s)")
        self.discover_unprocessed_files()
        self.debug_log("-> Load VAE")
        self.init_vae()
        if not StateTracker.get_args().vae_cache_ondemand:
            self.debug_log("-> Process VAE cache")
            self.process_buckets()
            if self.accelerator.is_local_main_process:
                self.debug_log("Updating StateTracker with new VAE cache entry list.")
                StateTracker.set_vae_cache_files(
                    self.cache_data_backend.list_files(
                        instance_data_dir=self.cache_dir,
                        file_extensions=["pt"],
                    ),
                    data_backend_id=self.id,
                )
            self.accelerator.wait_for_everyone()
        self.debug_log("-> Completed cache rebuild")

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
                futures.append(
                    executor.submit(self.cache_data_backend.delete, full_path)
                )
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
                    logger.debug(f"Error deleting file {filename}", e)

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
        all_image_files = set(StateTracker.get_image_files(data_backend_id=self.id))
        existing_cache_files = set(
            StateTracker.get_vae_cache_files(data_backend_id=self.id)
        )
        # Convert cache filenames to their corresponding image filenames
        already_cached_images = []
        for cache_file in existing_cache_files:
            try:
                n = self._image_filename_from_vaecache_filename(cache_file)
                if n is None:
                    continue
                already_cached_images.append(n)
            except Exception as e:
                logger.error(
                    f"Could not find image path for cache file {cache_file}: {e}"
                )
                continue

        # Identify unprocessed files
        self.local_unprocessed_files = list(
            set(all_image_files) - set(already_cached_images)
        )

        return self.local_unprocessed_files

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
        total_files = 0
        skipped_files = 0
        for full_image_path in aspect_bucket_cache[bucket]:
            total_files += 1
            comparison_path = self.generate_vae_cache_filename(full_image_path)[0]
            if os.path.splitext(comparison_path)[0] in processed_images:
                # processed_images contains basename *cache* paths:
                skipped_files += 1
                # self.debug_log(
                #     f"Reduce bucket {bucket}, skipping ({skipped_files}/{total_files}) {full_image_path} because it is in processed_images"
                # )
                continue
            if full_image_path not in self.local_unprocessed_files:
                # full_image_path is the full *image* path:
                skipped_files += 1
                # self.debug_log(
                #     f"Reduce bucket {bucket}, skipping ({skipped_files}/{total_files}) {full_image_path} because it is not in local_unprocessed_files"
                # )
                continue
            # self.debug_log(
            #     f"Reduce bucket {bucket}, adding ({len(relevant_files)}/{total_files}) {full_image_path}"
            # )
            relevant_files.append(full_image_path)
        if do_shuffle:
            shuffle(relevant_files)
        # self.debug_log(
        #     f"Reduced bucket {bucket} down from {len(aspect_bucket_cache[bucket])} to {len(relevant_files)} relevant files."
        #     f" Our system has {len(self.local_unprocessed_files)} total images in its assigned slice for processing across all buckets."
        # )
        return relevant_files

    def prepare_video_latents(self, samples):
        if StateTracker.get_model_family() in ["ltxvideo", "wan"]:
            if samples.ndim == 4:
                original_shape = samples.shape
                samples = samples.unsqueeze(2)
                logger.debug(
                    "PROCESSING IMAGE to VIDEO LATENTS CONVERSION ({original_shape} to {samples.shape})"
                )
            assert samples.ndim == 5, f"Expected 5D tensor, got {samples.ndim}D tensor"
            logger.debug(
                f"PROCESSING VIDEO to VIDEO LATENTS CONVERSION ({samples.shape})"
            )
            # images are torch.Size([1, 3, 1, 640, 448]) (B, C, F, H, W) but videos are torch.Size([1, 600, 3, 384, 395]) (B, F, C, H, W)
            # we have to permute the video latent samples to match the image latent samples
            num_frames = samples.shape[1]
            if samples.shape[2] == 3:
                original_shape = samples.shape
                samples = samples.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)
                num_frames = samples.shape[2]
                logger.debug(
                    f"Found video latent of shape: {original_shape} (B, F, C, H, W) to (B, C, F, H, W) {samples.shape}"
                )

            num_frames = samples.shape[1]
            if (
                self.num_video_frames is not None
                and self.num_video_frames != num_frames
            ):
                # we'll discard along dim2 after num_video_frames
                samples = samples[:, :, : self.num_video_frames, :, :]
        elif StateTracker.get_model_family() in ["hunyuan-video", "mochi"]:
            raise Exception(
                f"{StateTracker.get_model_family()} not supported for VAE Caching yet."
            )
        logger.debug(f"Final samples shape: {samples.shape}")
        return samples

    def process_video_latents(self, latents_uncached):
        output_cache_entry = latents_uncached
        if StateTracker.get_model_family() in ["ltxvideo"]:
            # hardcode patch size to 1 for LTX Video.
            # patch_size, patch_size_t = self.transformer.config.patch_size, self.transformer.config.patch_size_t
            patch_size, patch_size_t = 1, 1
            _, _, num_frames, height, width = latents_uncached.shape
            logger.debug(f"Latents shape: {latents_uncached.shape}")
            latents_uncached = normalize_ltx_latents(
                latents_uncached, self.vae.latents_mean, self.vae.latents_std
            )

            output_cache_entry = {
                "latents": latents_uncached.shape,  # we'll log the shape first
                "num_frames": self.num_video_frames,
                "height": height,
                "width": width,
            }
            logger.debug(f"Video latent processing results: {output_cache_entry}")
            # we'll now overwrite the latents after logging.
            output_cache_entry["latents"] = latents_uncached
        elif StateTracker.get_model_family() in ["wan"]:
            logger.debug(
                f"Shape for Wan VAE encode: {latents_uncached.shape} with latents_mean: {self.vae.latents_mean} and latents_std: {self.vae.latents_std}"
            )
            latents_uncached = compute_wan_posterior(
                latents_uncached, self.vae.latents_mean, self.vae.latents_std
            )
        elif StateTracker.get_model_family() in ["hunyuan-video", "mochi"]:
            raise Exception(
                f"{StateTracker.get_model_family()} not supported for VAE Caching yet."
            )

        return output_cache_entry

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
            if not self.cache_data_backend.exists(filename)
        ]
        uncached_image_paths = [
            filepaths[i]
            for i, filename in enumerate(full_filenames)
            if i in uncached_image_indices
        ]

        # We need to populate any uncached images with the actual image data if they are None.
        missing_images = [
            i
            for i, image in enumerate(images)
            if i in uncached_image_indices and image is None
        ]
        missing_image_pixel_values = []
        written_latents = []
        if len(missing_images) > 0 and self.vae_cache_ondemand:
            missing_image_paths = [filepaths[i] for i in missing_images]
            missing_image_data_generator = self._read_from_storage_concurrently(
                missing_image_paths, hide_errors=True
            )
            # extract images from generator:
            missing_image_data = [
                retrieved_image_data[1]
                for retrieved_image_data in missing_image_data_generator
            ]
            missing_image_pixel_values = self._process_images_in_batch(
                missing_image_paths, missing_image_data, disable_queue=True
            )
            missing_image_vae_outputs = self._encode_images_in_batch(
                image_pixel_values=missing_image_pixel_values, disable_queue=True
            )
            written_latents = self._write_latents_in_batch(missing_image_vae_outputs)
            if len(written_latents) == len(images):
                return written_latents

        if len(uncached_image_indices) > 0:
            uncached_images = [images[i] for i in uncached_image_indices]
        elif len(missing_images) > 0 and len(missing_image_pixel_values) > 0:
            uncached_images = []
            for i in uncached_image_indices:
                if images[i] is not None:
                    uncached_images.append(images[i])
                elif i in missing_image_pixel_values:
                    uncached_images.append(missing_image_pixel_values[i])

        if (
            len(uncached_image_indices) > 0
            and load_from_cache
            and not self.vae_cache_ondemand
        ):
            # We wanted only uncached images. Something went wrong.
            raise Exception(
                f"(id={self.id}) Some images were not correctly cached during the VAE Cache operations. Ensure --skip_file_discovery=vae is not set.\nProblematic images: {uncached_image_paths}"
            )

        latents = []
        if load_from_cache:
            # If all images are cached, simply load them
            latents = [
                self._read_from_storage(filename, hide_errors=self.vae_cache_ondemand)
                for filename in full_filenames
                if filename not in uncached_images
            ]

        if len(uncached_images) > 0 and (
            len(images) != len(latents) or len(filepaths) != len(latents)
        ):
            # Process images not found in cache
            with torch.no_grad():
                processed_images = torch.stack(uncached_images).to(
                    self.accelerator.device, dtype=StateTracker.get_vae_dtype()
                )
                processed_images = self.prepare_video_latents(processed_images)
                latents_uncached = self.vae.encode(processed_images)

                if hasattr(latents_uncached, "latent_dist"):
                    latents_uncached = latents_uncached.latent_dist.sample()
                elif hasattr(latents_uncached, "sample"):
                    latents_uncached = latents_uncached.sample()
                latents_uncached = self.process_video_latents(latents_uncached)
                if (
                    hasattr(self.vae, "config")
                    and hasattr(self.vae.config, "shift_factor")
                    and self.vae.config.shift_factor is not None
                ):
                    latents_uncached = (
                        latents_uncached - self.vae.config.shift_factor
                    ) * getattr(
                        self.model,
                        "AUTOENCODER_SCALING_FACTOR",
                        self.vae.config.scaling_factor,
                    )
                elif isinstance(latents_uncached, torch.Tensor) and hasattr(
                    self.vae.config, "scaling_factor"
                ):
                    latents_uncached = getattr(
                        latents_uncached, "latent", latents_uncached
                    ) * getattr(
                        self.model,
                        "AUTOENCODER_SCALING_FACTOR",
                        self.vae.config.scaling_factor,
                    )
                    logger.debug(f"Latents shape: {latents_uncached.shape}")

            # Prepare final latents list by combining cached and newly computed latents
            if isinstance(latents_uncached, dict) and "latents" in latents_uncached:
                # video models tend to return a dict with latents.
                raw_latents = latents_uncached["latents"]
                num_samples = raw_latents.shape[0]
                for i in range(num_samples):
                    # Each sub-dict is shape [1, 128, F, H, W]
                    single_latent = raw_latents[i : i + 1].squeeze(0)
                    chunk = {
                        "latents": single_latent,
                        "num_frames": latents_uncached["num_frames"],
                        "height": latents_uncached["height"],
                        "width": latents_uncached["width"],
                    }
                    latents.append(chunk)
            elif hasattr(latents_uncached, "latent"):
                # this one happens with sana really, so far.
                raw_latents = latents_uncached["latent"]
                num_samples = raw_latents.shape[0]
                for i in range(num_samples):
                    # Each sub-dict is shape [b, c, H, W], we want just 1 b at a time
                    single_latent = raw_latents[i : i + 1].squeeze(0)
                    logger.debug(f"Adding shape: {single_latent.shape}")
                    latents.append(single_latent)
            elif isinstance(latents_uncached, torch.Tensor):
                # it seems like sdxl and some others end up here
                cached_idx, uncached_idx = 0, 0
                for i in range(batch_size):
                    if i in uncached_image_indices:
                        # logger.info(
                        #     f"Adding latent {uncached_idx} of ({len(latents_uncached)}: {latents_uncached})"
                        # )
                        latents.append(latents_uncached[uncached_idx])
                        uncached_idx += 1
                    else:
                        latents.append(self._read_from_storage(full_filenames[i]))
                        cached_idx += 1
            else:
                raise ValueError(
                    f"Unknown handler for latent encoding type: {type(latents_uncached)}"
                )
        return latents

    def _write_latents_in_batch(self, input_latents: list = None):
        # Pull the 'filepaths' and 'latents' from self.write_queue
        filepaths, latents = [], []
        if input_latents is not None:
            qlen = len(input_latents)
        else:
            qlen = self.write_queue.qsize()

        for idx in range(0, qlen):
            if input_latents:
                output_file, filepath, latent_vector = input_latents.pop()
            else:
                output_file, filepath, latent_vector = self.write_queue.get()
            file_extension = os.path.splitext(output_file)[1]
            if file_extension != ".pt":
                raise ValueError(
                    f"Cannot write a latent embedding to an image path, {output_file}"
                )
            filepaths.append(output_file)
            # pytorch will hold onto all of the tensors in the list if we do not use clone()
            if isinstance(latent_vector, dict):
                latent_vector["latents"] = latent_vector["latents"].clone()
                latents.append(latent_vector)
            else:
                latents.append(latent_vector.clone())

        self.cache_data_backend.write_batch(filepaths, latents)

        return latents

    def _process_images_in_batch(
        self,
        image_paths: list = None,
        image_data: list = None,
        disable_queue: bool = False,
    ) -> None:
        """Process a queue of images. This method assumes our batch size has been reached.

        Args:
            image_paths: list If given, image_data must also be supplied. This will avoid the use of the Queues.
            image_data: list Provided Image objects for corresponding image_paths.

        Returns:
            None
        """
        try:
            # self.debug_log(
            #     f"Processing batch of images into VAE embeds. image_paths: {type(image_paths)}, image_data: {type(image_data)}"
            # )
            initial_data = []
            filepaths = []
            if image_paths is not None and image_data is not None:
                qlen = len(image_paths)
            else:
                qlen = self.process_queue.qsize()

            # First Loop: Preparation and Filtering
            for _ in range(qlen):
                if image_paths:
                    # retrieve image data from Generator, image_data:
                    filepath = image_paths.pop()
                    image = image_data.pop()
                    aspect_bucket = (
                        self.metadata_backend.get_metadata_attribute_by_filepath(
                            filepath=filepath, attribute="aspect_bucket"
                        )
                    )
                else:
                    filepath, image, aspect_bucket = self.process_queue.get()
                if self.minimum_image_size is not None:
                    if not self.metadata_backend.meets_resolution_requirements(
                        image_path=filepath
                    ):
                        self.debug_log(
                            f"Skipping {filepath} because it does not meet the minimum image size requirement of {self.minimum_image_size}"
                        )
                        continue
                # image.save(f"test_{os.path.basename(filepath)}.png")
                initial_data.append((filepath, image, aspect_bucket))

            # Process Pool Execution
            processed_images = []
            with ThreadPoolExecutor(self.max_workers) as executor:
                futures = [
                    executor.submit(
                        prepare_sample,
                        data_backend_id=self.id,
                        filepath=data[0],
                        model=self.model,
                    )
                    for data in initial_data
                ]
                first_aspect_ratio = None
                for future in futures:
                    try:
                        result = (
                            future.result()
                        )  # Returns PreparedSample or tuple(image, crop_coordinates, aspect_ratio)
                        if result:  # Ensure result is not None or invalid
                            processed_images.append(result)
                            if first_aspect_ratio is None:
                                first_aspect_ratio = result[2]
                            elif (
                                type(result) is PreparedSample
                                and result.aspect_ratio is not None
                                and first_aspect_ratio is not None
                                and result.aspect_ratio != first_aspect_ratio
                            ):
                                raise ValueError(
                                    f"({type(result)}) Image {filepath} has a different aspect ratio ({result.aspect_ratio}) than the first image in the batch ({first_aspect_ratio})."
                                )
                            elif (
                                type(result) is tuple
                                and result[2]
                                and first_aspect_ratio is not None
                                and result[2] != first_aspect_ratio
                            ):
                                raise ValueError(
                                    f"({type(result)}) Image {filepath} has a different aspect ratio ({result[2]}) than the first image in the batch ({first_aspect_ratio})."
                                )

                    except Exception as e:
                        logger.error(
                            f"Error processing image in pool: {e}, traceback: {traceback.format_exc()}"
                        )

            # Second Loop: Final Processing
            is_final_sample = False
            output_values = []
            first_aspect_ratio = None
            for idx, (image, crop_coordinates, new_aspect_ratio) in enumerate(
                processed_images
            ):
                if idx == len(processed_images) - 1:
                    is_final_sample = True
                if first_aspect_ratio is None:
                    first_aspect_ratio = new_aspect_ratio
                elif new_aspect_ratio != first_aspect_ratio:
                    is_final_sample = True
                    first_aspect_ratio = new_aspect_ratio
                filepath, _, aspect_bucket = initial_data[idx]
                filepaths.append(filepath)

                pixel_values = self.transform_sample(image).to(
                    self.accelerator.device, dtype=self.vae.dtype
                )
                output_value = (pixel_values, filepath, aspect_bucket, is_final_sample)
                output_values.append(output_value)
                if not disable_queue:
                    self.vae_input_queue.put(
                        (pixel_values, filepath, aspect_bucket, is_final_sample)
                    )
                # Update the crop_coordinates in the metadata document
                # NOTE: This is currently a no-op because the metadata is now considered 'trustworthy'.
                #       The VAE encode uses the preexisting metadata, and the TrainingSample class will not update.
                #       However, we'll check that the values didn't change anyway, just in case.
                if crop_coordinates:
                    current_crop_coordinates = (
                        self.metadata_backend.get_metadata_attribute_by_filepath(
                            filepath=filepath,
                            attribute="crop_coordinates",
                        )
                    )
                    if current_crop_coordinates is not None and tuple(current_crop_coordinates) != tuple(crop_coordinates):
                        logger.debug(
                            f"Should be updating crop_coordinates for {filepath} from {current_crop_coordinates} to {crop_coordinates}. But we won't.."
                        )

            self.debug_log(
                f"Completed processing gathered {len(output_values)} output values."
            )
        except Exception as e:
            logger.error(
                f"Error processing images {filepaths if len(filepaths) > 0 else image_paths}: {e}"
            )
            self.debug_log(f"Error traceback: {traceback.format_exc()}")
            raise e
        return output_values

    def _encode_images_in_batch(
        self, image_pixel_values: list = None, disable_queue: bool = False
    ) -> None:
        """Encode the batched Image objects using the VAE model.

        Raises:
            ValueError: If we receive any invalid results.
        """
        try:
            if image_pixel_values is not None:
                qlen = len(image_pixel_values)
                if self.vae_batch_size != len(image_pixel_values):
                    self.vae_batch_size = len(image_pixel_values)
            else:
                qlen = self.vae_input_queue.qsize()

            if qlen == 0:
                return
            output_values = []
            while qlen > 0:
                vae_input_images, vae_input_filepaths, vae_output_filepaths = [], [], []
                batch_aspect_bucket = None
                count_to_process = min(qlen, self.vae_batch_size)
                for idx in range(0, count_to_process):
                    if image_pixel_values:
                        (
                            pixel_values,
                            filepath,
                            aspect_bucket,
                            is_final_sample,
                        ) = image_pixel_values.pop()
                    else:
                        (
                            pixel_values,
                            filepath,
                            aspect_bucket,
                            is_final_sample,
                        ) = self.vae_input_queue.get()

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
                    [
                        sample.to(dtype=StateTracker.get_vae_dtype())
                        for sample in vae_input_images
                    ],
                    vae_input_filepaths,
                    load_from_cache=False,
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
                    output_value = (output_file, filepath, latent_vector)
                    output_values.append(output_value)
                    if not disable_queue:
                        logger.debug("Adding outputs to write queue")
                        self.write_queue.put(output_value)
                if image_pixel_values is not None:
                    qlen = len(image_pixel_values)
                else:
                    qlen = self.vae_input_queue.qsize()
        except Exception as e:
            logger.error(f"Error encoding images {vae_input_filepaths}: {e}")
            if "out of memory" in str(e).lower():
                import sys

                sys.exit(1)
            # Remove all of the errored images from the bucket. They will be captured on restart.
            for filepath in vae_input_filepaths:
                self.metadata_backend.remove_image(filepath)
            self.debug_log(f"Error traceback: {traceback.format_exc()}")
            raise Exception(
                f"Error encoding images {vae_input_filepaths}: {e}, traceback: {traceback.format_exc()}"
            )
        return output_values

    def _read_from_storage_concurrently(self, paths, hide_errors: bool = False):
        """
        A helper method to read files from storage concurrently, without Queues.

        Args:
            paths (List[str]): A list of file paths to read.

        Returns:
            Generator[Tuple[str, Any], None, None]: Yields file path and contents.
        """

        def read_file(path):
            try:
                return path, self._read_from_storage(path, hide_errors=hide_errors)
            except Exception as e:
                import traceback

                logger.error(
                    f"Error reading {path}: {e}, traceback: {traceback.format_exc()}"
                )
                # If --delete_problematic_images is supplied, we remove the image now:
                if self.delete_problematic_images:
                    self.metadata_backend.remove_image(path)
                    self.image_data_backend.delete(path)
                return path, None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map read_file operation over all paths
            future_to_path = {executor.submit(read_file, path): path for path in paths}
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    yield future.result()
                except Exception as exc:
                    logger.error(f"{path} generated an exception: {exc}")

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
            path, aspect_bucket = read_queue_item
            filepaths.append(path)
        available_filepaths, batch_output = self.image_data_backend.read_image_batch(
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

        if self.webhook_handler is not None:
            total_count = len(
                [item for sublist in aspect_bucket_cache.values() for item in sublist]
            )
            self.send_progress_update(
                type="init_cache_vae_processing_started",
                progress=int(len(processed_images) / total_count * 100),
                total=total_count,
                current=len(processed_images),
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
                last_reported_index = 0

                for raw_filepath in tqdm(
                    relevant_files,
                    desc=f"Processing bucket {bucket}",
                    position=get_rank(),
                    ncols=125,
                    leave=False,
                ):
                    statistics["total"] += 1
                    filepath = self._process_raw_filepath(raw_filepath)
                    test_filepath = self._image_filename_from_vaecache_filename(
                        filepath
                    )
                    if test_filepath is None:
                        continue
                    if test_filepath not in self.local_unprocessed_files:
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
                            if (
                                self.webhook_handler is not None
                                and int(
                                    statistics["total"]
                                    // self.webhook_progress_interval
                                )
                                > last_reported_index
                            ):
                                last_reported_index = (
                                    statistics["total"]
                                    // self.webhook_progress_interval
                                )
                                self.send_progress_update(
                                    type="vaecache",
                                    progress=int(
                                        statistics["total"] / len(relevant_files) * 100
                                    ),
                                    total=len(relevant_files),
                                    current=statistics["total"],
                                )

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
                    log_msg = (
                        f"(id={self.id}) Bucket {bucket} caching results: {statistics}"
                    )
                    if get_rank() == 0:
                        logger.debug(log_msg)
                        tqdm.write(log_msg)
                    if self.webhook_handler is not None:
                        self.send_progress_update(
                            type="init_cache_vae_processing_complete",
                            progress=100,
                            total=statistics["total"],
                            current=statistics["total"],
                        )
                    self.debug_log(
                        "Completed process_buckets, all futures have been returned."
                    )
                except Exception as e:
                    logger.error(f"Fatal error when processing bucket {bucket}: {e}")
                    continue

    def scan_cache_contents(self):
        """
        A generator method that iterates over the VAE cache, yielding each cache file's path and its contents
        using multi-threading for improved performance.

        Yields:
            Tuple[str, Any]: A tuple containing the file path and its contents.
        """
        try:
            all_cache_files = StateTracker.get_vae_cache_files(data_backend_id=self.id)
            try:
                yield from self._read_from_storage_concurrently(
                    all_cache_files, hide_errors=True
                )
            except FileNotFoundError:
                yield (None, None)
        except Exception as e:
            if "is not iterable" not in str(e):
                logger.error(f"Error in scan_cache_contents: {e}")
                self.debug_log(f"Error traceback: {traceback.format_exc()}")
