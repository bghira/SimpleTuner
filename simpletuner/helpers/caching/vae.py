import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from queue import Queue
from random import shuffle

import numpy as np
import torch
from numpy import str_ as numpy_str
from PIL import Image
from tqdm import tqdm

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.image_manipulation.batched_training_samples import BatchedTrainingSamples
from simpletuner.helpers.image_manipulation.training_sample import PreparedSample, TrainingSample
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.models.ltxvideo import normalize_ltx_latents
from simpletuner.helpers.models.wan import compute_wan_posterior
from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.training import image_file_extensions
from simpletuner.helpers.training.multi_process import _get_rank as get_rank
from simpletuner.helpers.training.multi_process import rank_info, should_log
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.webhooks.events import lifecycle_stage_event
from simpletuner.helpers.webhooks.mixin import WebhookMixin

logger = logging.getLogger("VAECache")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


def prepare_sample(
    image: Image.Image = None,
    data_backend_id: str = None,
    filepath: str = None,
    model=None,
):
    metadata = StateTracker.get_metadata_by_filepath(filepath, data_backend_id=data_backend_id)
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
    # python will raise an error here if any cond_datasets are set to back multiple train_datasets
    # this would be a problem since we wouldn't know how to prepare our sample
    cond_mapping = {y: x for (x, y) in StateTracker.get_conditioning_mappings()}

    if data_backend_id in cond_mapping:
        conditioning_sample_path = training_sample.image_path()
        # locate the partner backend id
        train_id = cond_mapping[data_backend_id]
        train_data_backend = StateTracker.get_data_backend(train_id)
        train_sample_path = training_sample.training_sample_path(training_dataset_id=train_id)
        cond_meta = StateTracker.get_metadata_by_filepath(conditioning_sample_path, data_backend_id=data_backend_id)
        if not cond_meta:
            train_meta = train_data_backend["metadata_backend"].get_metadata_by_filepath(train_sample_path)
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
        num_video_frames: int = 125,
        delete_problematic_images: bool = False,
        write_batch_size: int = 25,
        read_batch_size: int = 25,
        process_queue_size: int = 16,
        vae_batch_size: int = 4,
        max_workers: int = 32,
        vae_cache_ondemand: bool = False,
        hash_filenames: bool = False,
        dataset_type: str = None,
    ):
        self.id = id
        self.dataset_type = dataset_type
        if image_data_backend and image_data_backend.id != id:
            raise ValueError(f"VAECache received incorrect image_data_backend: {image_data_backend}")
        self.image_data_backend = image_data_backend
        self.cache_data_backend = cache_data_backend if cache_data_backend is not None else image_data_backend
        self.hash_filenames = hash_filenames
        self.vae = vae
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        if len(self.cache_dir) > 0 and self.cache_dir[-1] == "/":
            # Remove trailing slash
            self.cache_dir = self.cache_dir[:-1]
        if self.cache_data_backend and self.cache_data_backend.type in [
            "local",
            "huggingface",
        ]:
            self.cache_dir = os.path.abspath(self.cache_dir)
            self.cache_data_backend.create_directory(self.cache_dir)
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
        self.read_queue = Queue()
        self.process_queue = Queue()
        self.write_queue = Queue()
        self.vae_input_queue = Queue()

        # Initialize batch processing helper
        self.batch_processor = BatchedTrainingSamples()

    def debug_log(self, msg: str):
        logger.debug(f"{self.rank_info}{msg}")

    def generate_vae_cache_filename(self, filepath: str) -> tuple:
        if filepath.endswith(".pt"):
            return filepath, os.path.basename(filepath)
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        if self.hash_filenames:
            base_filename = str(sha256(str(base_filename).encode()).hexdigest())
        base_filename = str(base_filename) + ".pt"
        subfolders = ""
        if self.instance_data_dir is not None:
            subfolders = os.path.dirname(filepath).replace(self.instance_data_dir, "")
            subfolders = subfolders.lstrip(os.sep)

        if len(subfolders) > 0:
            full_filename = os.path.join(self.cache_dir, subfolders, base_filename)
        else:
            full_filename = os.path.join(self.cache_dir, base_filename)
        return full_filename, base_filename

    def _image_filename_from_vaecache_filename(self, filepath: str) -> tuple[str, str]:
        test_filepath, _ = self.generate_vae_cache_filename(filepath)
        result = self.vae_path_to_image_path.get(test_filepath, None)

        return result

    def build_vae_cache_filename_map(self, all_image_files: list):
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

    def _read_from_storage(self, filename: str, hide_errors: bool = False) -> torch.Tensor:
        if os.path.splitext(filename)[1] != ".pt":
            try:
                sample = self.image_data_backend.read_image(filename)
                return self._normalise_loaded_sample(sample)
            except Exception as e:
                if self.delete_problematic_images:
                    self.metadata_backend.remove_image(filename)
                    self.image_data_backend.delete(filename)
                    self.debug_log(f"Deleted {filename} because it was problematic: {e}")
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

    def _normalise_loaded_sample(self, sample):
        if isinstance(sample, Image.Image):
            return sample
        if torch.is_tensor(sample):
            return self._first_frame_to_pil(sample.detach().cpu().numpy())
        if isinstance(sample, np.ndarray):
            return self._first_frame_to_pil(sample)
        return sample

    def _first_frame_to_pil(self, array: np.ndarray) -> Image.Image:
        if array.ndim == 5:
            array = array[0]
        if array.ndim == 4:
            # Prefer channel-last; if not, attempt conversion
            if array.shape[-1] in (1, 3, 4):
                array = array[0]
            elif array.shape[1] in (1, 3, 4):
                array = array[0].transpose(1, 2, 0)
            else:
                array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = array.transpose(1, 2, 0)
        if array.ndim == 2:
            pass
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = array.squeeze(-1)
        elif array.ndim != 3:
            raise ValueError(f"Unsupported sample shape for conditioning embed: {array.shape}")

        if array.dtype.kind in {"f", "d"}:
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    def retrieve_from_cache(self, filepath: str):
        return self.encode_images([None], [filepath])[0]

    def retreve_batch_from_cache(self, filepaths: list):
        return self.encode_images([None] * len(filepaths), filepaths)

    def discover_all_files(self):
        all_image_files = StateTracker.get_image_files(data_backend_id=self.id) or StateTracker.set_image_files(
            self.image_data_backend.list_files(
                instance_data_dir=self.instance_data_dir,
                file_extensions=image_file_extensions,
            ),
            data_backend_id=self.id,
        )
        logger.debug(f"Checking {self.cache_dir=}")
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
        self.debug_log(f"VAECache discover_all_files found {len(all_image_files)} images")
        return all_image_files

    def init_vae(self):
        if StateTracker.get_args().model_family == "sana":
            from diffusers import AutoencoderDC as AutoencoderClass
        elif StateTracker.get_args().model_family == "ltxvideo":
            from simpletuner.helpers.models.ltxvideo.autoencoder import AutoencoderKLLTXVideo as AutoencoderClass
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
        # can't simply clear directory since it might be mixed with image samples (s3 case)
        futures = []
        all_cache_files = StateTracker.get_vae_cache_files(data_backend_id=self.id)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for filename in all_cache_files:
                full_path = os.path.join(self.cache_dir, filename)
                self.debug_log(f"Would delete: {full_path}")
                futures.append(executor.submit(self.cache_data_backend.delete, full_path))
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

        StateTracker.set_vae_cache_files([], data_backend_id=self.id)

    def _list_cached_images(self):
        pt_files = StateTracker.get_vae_cache_files(data_backend_id=self.id)
        results = {os.path.splitext(f)[0] for f in pt_files}
        return results

    def discover_unprocessed_files(self):
        all_image_files = set(StateTracker.get_image_files(data_backend_id=self.id))
        existing_cache_files = set(StateTracker.get_vae_cache_files(data_backend_id=self.id))
        already_cached_images = []
        for cache_file in existing_cache_files:
            try:
                n = self._image_filename_from_vaecache_filename(cache_file)
                if n is None:
                    continue
                already_cached_images.append(n)
            except Exception as e:
                logger.error(f"Could not find image path for cache file {cache_file}: {e}")
                continue

        self.local_unprocessed_files = list(set(all_image_files) - set(already_cached_images))
        if os.environ.get("SIMPLETUNER_LOG_LEVEL", None) == "DEBUG":
            self.debug_log(f"All ({len(all_image_files)}) image files: (truncated) {list(all_image_files)[:5]}")
            self.debug_log(f"Existing cache files: (truncated) {list(existing_cache_files)[:5]}")
            self.debug_log(f"Already cached images: (truncated) {already_cached_images[:5]}")
            self.debug_log(f"VAECache Mapping: (truncated) {list(self.image_path_to_vae_path.items())[:5]}")

        return self.local_unprocessed_files

    def _reduce_bucket(
        self,
        bucket: str,
        aspect_bucket_cache: dict,
        processed_images: dict,
    ):
        relevant_files = []
        total_files = 0
        skipped_files = 0
        for full_image_path in aspect_bucket_cache[bucket]:
            total_files += 1
            comparison_path = self.generate_vae_cache_filename(full_image_path)[0]
            if os.path.splitext(comparison_path)[0] in processed_images:
                skipped_files += 1
                continue
            if full_image_path not in self.local_unprocessed_files:
                skipped_files += 1
                continue
            relevant_files.append(full_image_path)
        self.debug_log(
            f"Reduced bucket {bucket} down from {len(aspect_bucket_cache[bucket])} to {len(relevant_files)} relevant files."
            f" Our system has {len(self.local_unprocessed_files)} total images in its assigned slice for processing across all buckets."
        )
        return relevant_files

    def prepare_video_latents(self, samples):
        if StateTracker.get_model_family() in ["ltxvideo", "wan"]:
            if samples.ndim == 4:
                original_shape = samples.shape
                samples = samples.unsqueeze(2)
                logger.debug(f"PROCESSING IMAGE to VIDEO LATENTS CONVERSION ({original_shape} to {samples.shape})")
            assert samples.ndim == 5, f"Expected 5D tensor, got {samples.ndim}D tensor"
            logger.debug(f"PROCESSING VIDEO to VIDEO LATENTS CONVERSION ({samples.shape})")
            # permute video latents (B, F, C, H, W) to match image latents (B, C, F, H, W)
            num_frames = samples.shape[1]
            if samples.shape[2] == 3:
                original_shape = samples.shape
                samples = samples.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)
                num_frames = samples.shape[2]
                logger.debug(
                    f"Found video latent of shape: {original_shape} (B, F, C, H, W) to (B, C, F, H, W) {samples.shape}"
                )

            num_frames = samples.shape[2]
            if self.num_video_frames is not None and self.num_video_frames != num_frames:
                samples = samples[:, :, : self.num_video_frames, :, :]

            spatial_ratio = getattr(self.vae, "spatial_compression_ratio", None)
            if spatial_ratio and spatial_ratio > 1:
                # The encoder expects latent spatial dims to be divisible by its stride (typically 2).
                # Ensure that (height / spatial_ratio) and (width / spatial_ratio) remain divisible by 2
                # by trimming down in spatial_ratio-sized steps when necessary.
                height = samples.shape[-2]
                width = samples.shape[-1]

                def _align_dimension(dim: int) -> int:
                    aligned = (dim // spatial_ratio) * spatial_ratio
                    min_dim = spatial_ratio * 2  # keep at least two stride steps
                    if aligned < min_dim:
                        # Not enough room to align; fall back to original (will likely error later)
                        return dim
                    while aligned >= min_dim and ((aligned // spatial_ratio) % 2 != 0):
                        aligned -= spatial_ratio
                    return max(min_dim, aligned)

                target_height = _align_dimension(height)
                target_width = _align_dimension(width)

                if target_height != height or target_width != width:
                    logger.warning(
                        "Adjusted video frames from (%s, %s) to (%s, %s) to satisfy VAE stride requirements",
                        height,
                        width,
                        target_height,
                        target_width,
                    )
                    samples = samples[..., :target_height, :target_width]
        elif StateTracker.get_model_family() in ["hunyuan-video", "mochi"]:
            raise Exception(f"{StateTracker.get_model_family()} not supported for VAE Caching yet.")
        logger.debug(f"Final samples shape: {samples.shape}")
        return samples

    def process_video_latents(self, latents_uncached):
        output_cache_entry = latents_uncached
        if StateTracker.get_model_family() in ["ltxvideo"]:
            _, _, _, height, width = latents_uncached.shape
            logger.debug(f"Latents shape: {latents_uncached.shape}")
            latents_uncached = normalize_ltx_latents(latents_uncached, self.vae.latents_mean, self.vae.latents_std)

            output_cache_entry = {
                "latents": latents_uncached.shape,
                "num_frames": self.num_video_frames,
                "height": height,
                "width": width,
            }
            logger.debug(f"Video latent processing results: {output_cache_entry}")
            output_cache_entry["latents"] = latents_uncached
        elif StateTracker.get_model_family() in ["wan"]:
            logger.debug(
                f"Shape for Wan VAE encode: {latents_uncached.shape} with latents_mean: {self.vae.latents_mean} and latents_std: {self.vae.latents_std}"
            )
            posterior = compute_wan_posterior(latents_uncached, self.vae.latents_mean, self.vae.latents_std)
            # use deterministic posterior sampling (mode) for reproducibility
            latents_uncached = posterior.mode()
            output_cache_entry = latents_uncached
        elif StateTracker.get_model_family() in ["hunyuan-video", "mochi"]:
            raise Exception(f"{StateTracker.get_model_family()} not supported for VAE Caching yet.")

        return output_cache_entry

    def encode_images(self, images, filepaths, load_from_cache=True):
        # images must be same dimension
        # if load_from_cache=True and entry not found, throws exception
        batch_size = len(images)
        if batch_size != len(filepaths):
            raise ValueError("Mismatch between number of images and filepaths.")

        full_filenames = [self.generate_vae_cache_filename(filepath)[0] for filepath in filepaths]

        uncached_images = []
        uncached_image_indices = [
            i for i, filename in enumerate(full_filenames) if not self.cache_data_backend.exists(filename)
        ]
        uncached_image_paths = [filepaths[i] for i, filename in enumerate(full_filenames) if i in uncached_image_indices]

        missing_images = [i for i, image in enumerate(images) if i in uncached_image_indices and image is None]
        missing_image_pixel_values = []
        written_latents = []
        if len(missing_images) > 0 and self.vae_cache_ondemand:
            missing_image_paths = [filepaths[i] for i in missing_images]
            missing_image_data_generator = self._read_from_storage_concurrently(missing_image_paths, hide_errors=True)
            missing_image_data = [retrieved_image_data[1] for retrieved_image_data in missing_image_data_generator]
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

        if len(uncached_image_indices) > 0 and load_from_cache and not self.vae_cache_ondemand:
            raise Exception(
                f"(id={self.id}) Some images were not correctly cached during the VAE Cache operations. Ensure --skip_file_discovery=vae is not set.\nProblematic images: {uncached_image_paths}"
            )

        latents = []
        if load_from_cache:
            latents = [
                self._read_from_storage(filename, hide_errors=self.vae_cache_ondemand)
                for filename in full_filenames
                if filename not in uncached_images
            ]

        if len(uncached_images) > 0 and (len(images) != len(latents) or len(filepaths) != len(latents)):
            with torch.no_grad():
                processed_images = torch.stack(uncached_images).to(
                    self.accelerator.device, dtype=StateTracker.get_vae_dtype()
                )
                processed_images = self.prepare_video_latents(processed_images)
                processed_images = self.model.pre_vae_encode_transform_sample(processed_images)
                latents_uncached = self.model.encode_with_vae(self.vae, processed_images)
                latents_uncached = self.model.post_vae_encode_transform_sample(latents_uncached)

                if StateTracker.get_model_family() in ["wan"]:
                    if hasattr(latents_uncached, "latent_dist"):
                        latents_uncached = latents_uncached.latent_dist.parameters
                    latents_uncached = self.process_video_latents(latents_uncached)
                else:
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
                    latents_uncached = (latents_uncached - self.vae.config.shift_factor) * getattr(
                        self.model,
                        "AUTOENCODER_SCALING_FACTOR",
                        self.vae.config.scaling_factor,
                    )
                elif isinstance(latents_uncached, torch.Tensor) and hasattr(self.vae.config, "scaling_factor"):
                    latents_uncached = latents_uncached * getattr(
                        self.model,
                        "AUTOENCODER_SCALING_FACTOR",
                        self.vae.config.scaling_factor,
                    )
                    logger.debug(f"Latents shape after scaling: {latents_uncached.shape}")
            if isinstance(latents_uncached, dict) and "latents" in latents_uncached:
                raw_latents = latents_uncached["latents"]
                num_samples = raw_latents.shape[0]
                for i in range(num_samples):
                    single_latent = raw_latents[i : i + 1].squeeze(0)
                    chunk = {
                        "latents": single_latent,
                        "num_frames": latents_uncached["num_frames"],
                        "height": latents_uncached["height"],
                        "width": latents_uncached["width"],
                    }
                    latents.append(chunk)
            elif hasattr(latents_uncached, "latent"):
                raw_latents = latents_uncached["latent"]
                num_samples = raw_latents.shape[0]
                for i in range(num_samples):
                    single_latent = raw_latents[i : i + 1].squeeze(0)
                    logger.debug(f"Adding shape: {single_latent.shape}")
                    latents.append(single_latent)
            elif isinstance(latents_uncached, torch.Tensor):
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
                raise ValueError(f"Unknown handler for latent encoding type: {type(latents_uncached)}")
        return latents

    def _write_latents_in_batch(self, input_latents: list = None):
        # Pull the 'filepaths' and 'latents' from self.write_queue
        filepaths, latents = [], []
        if input_latents is not None:
            qlen = len(input_latents)
        else:
            qlen = self.write_queue.qsize()

        for _ in range(0, qlen):
            if input_latents:
                output_file, filepath, latent_vector = input_latents.pop()
            else:
                output_file, filepath, latent_vector = self.write_queue.get()
            file_extension = os.path.splitext(output_file)[1]
            if file_extension != ".pt":
                raise ValueError(f"Cannot write a latent embedding to an image path, {output_file}")
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
        """Process a queue of images using trainingsample for better performance.
        Replaced complex threading with batch operations from trainingsample.

        Args:
            image_paths: list If given, image_data must also be supplied. This will avoid the use of the Queues.
            image_data: list Provided Image objects for corresponding image_paths.

        Returns:
            None
        """
        try:
            initial_data = []
            filepaths = []
            if image_paths is not None and image_data is not None:
                qlen = len(image_paths)
            else:
                qlen = self.process_queue.qsize()

            # First Loop: Preparation and Filtering
            for _ in range(qlen):
                if image_paths:
                    filepath = image_paths.pop()
                    image = image_data.pop()
                    aspect_bucket = self.metadata_backend.get_metadata_attribute_by_filepath(
                        filepath=filepath, attribute="aspect_bucket"
                    )
                else:
                    filepath, image, aspect_bucket = self.process_queue.get()
                initial_data.append((filepath, image, aspect_bucket))

            # Use BatchedTrainingSamples for efficient batch processing
            processed_images = []

            # Group images by aspect ratio for batch processing
            aspect_groups = {}
            for filepath, image, aspect_bucket in initial_data:
                if aspect_bucket not in aspect_groups:
                    aspect_groups[aspect_bucket] = []
                aspect_groups[aspect_bucket].append((filepath, image, aspect_bucket))

            # Process using the batch processor
            try:
                batch_results = self.batch_processor.process_aspect_grouped_images(
                    aspect_groups,
                    metadata_backend=self.metadata_backend,
                )

                # Convert batch results to processed samples
                for filepath, processed_image_array, metadata in batch_results:
                    try:
                        # Convert back to PIL for TrainingSample compatibility
                        prepared_input = processed_image_array
                        if isinstance(processed_image_array, np.ndarray):
                            if processed_image_array.ndim == 3:
                                prepared_input = Image.fromarray(processed_image_array)
                            elif processed_image_array.ndim == 4:
                                # Leave video tensors as-is; TrainingSample handles multi-frame arrays.
                                prepared_input = processed_image_array
                            else:
                                logger.warning(f"Skipping {filepath}: unexpected array shape {processed_image_array.shape}")
                                continue
                        else:
                            prepared_input = processed_image_array

                        result = prepare_sample(
                            image=prepared_input,
                            data_backend_id=self.id,
                            filepath=filepath,
                            model=self.model,
                        )
                        if result:
                            processed_images.append(result)
                    except Exception as e:
                        logger.error(f"Error processing batch result {filepath}: {e}, traceback: {traceback.format_exc()}")

            except Exception as e:
                logger.error(f"Batch processing failed, falling back to individual processing: {e}")
                # Fallback to individual processing
                for filepath, image, _ in initial_data:
                    try:
                        result = prepare_sample(
                            image=image,
                            data_backend_id=self.id,
                            filepath=filepath,
                            model=self.model,
                        )
                        if result:
                            processed_images.append(result)
                    except Exception as e:
                        logger.error(f"Error processing individual image {filepath}: {e}")

            # Final Processing - simplified without complex threading
            output_values = []
            first_aspect_ratio = None
            for idx, (image, crop_coordinates, new_aspect_ratio) in enumerate(processed_images):
                is_final_sample = idx == len(processed_images) - 1
                if first_aspect_ratio is None:
                    first_aspect_ratio = new_aspect_ratio
                elif new_aspect_ratio != first_aspect_ratio:
                    is_final_sample = True
                    first_aspect_ratio = new_aspect_ratio

                filepath, _, aspect_bucket = initial_data[idx]
                filepaths.append(filepath)

                if (
                    self.dataset_type == "conditioning"
                    and hasattr(self.model, "_is_i2v_like_flavour")
                    and callable(self.model._is_i2v_like_flavour)
                    and self.model._is_i2v_like_flavour()
                ):
                    if isinstance(image, np.ndarray) and image.ndim >= 4:
                        image = image[0]
                    elif torch.is_tensor(image) and image.ndim >= 4:
                        image = image[0]
                    if torch.is_tensor(image) and image.ndim == 3:
                        image = image.cpu().numpy()
                    elif isinstance(image, list) and len(image) > 0:
                        image = image[0]
                    if isinstance(image, np.ndarray) and image.ndim == 3:
                        image = Image.fromarray(image.astype(np.uint8))

                pixel_values = self.transform_sample(image).to(self.accelerator.device, dtype=self.vae.dtype)
                output_value = (pixel_values, filepath, aspect_bucket, is_final_sample)
                output_values.append(output_value)
                if not disable_queue:
                    self.vae_input_queue.put((pixel_values, filepath, aspect_bucket, is_final_sample))

                # Update crop coordinates metadata if needed
                if crop_coordinates:
                    current_crop_coordinates = self.metadata_backend.get_metadata_attribute_by_filepath(
                        filepath=filepath,
                        attribute="crop_coordinates",
                    )
                    if current_crop_coordinates is not None and tuple(current_crop_coordinates) != tuple(crop_coordinates):
                        logger.debug(
                            f"Should be updating crop_coordinates for {filepath} from {current_crop_coordinates} to {crop_coordinates}. But we won't.."
                        )

            self.debug_log(f"Completed processing gathered {len(output_values)} output values.")
        except Exception as e:
            logger.error(f"Error processing images {filepaths if len(filepaths) > 0 else image_paths}: {e}")
            self.debug_log(f"Error traceback: {traceback.format_exc()}")
            raise e
        return output_values

    def _encode_images_in_batch(self, image_pixel_values: list = None, disable_queue: bool = False) -> None:
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
                for _ in range(0, count_to_process):
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
                    vae_output_filepaths.append(self.generate_vae_cache_filename(filepath)[0])
                    if is_final_sample:
                        # When we have fewer samples in a bucket than our VAE batch size might indicate,
                        # we need to respect is_final_sample value and not retrieve the *next* element yet.
                        break

                latents = self.encode_images(
                    [sample.to(dtype=StateTracker.get_vae_dtype()) for sample in vae_input_images],
                    vae_input_filepaths,
                    load_from_cache=False,
                )
                if latents is None:
                    raise ValueError("Received None from encode_images")
                for output_file, latent_vector, filepath in zip(vae_output_filepaths, latents, vae_input_filepaths):
                    if latent_vector is None:
                        raise ValueError(f"Latent vector is None for filepath {filepath}")
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
            raise Exception(f"Error encoding images {vae_input_filepaths}: {e}, traceback: {traceback.format_exc()}")
        return output_values

    def _read_from_storage_concurrently(self, paths, hide_errors: bool = False):
        """
        A helper method to read files from storage concurrently, using simplified approach.
        Replaced complex threading with direct batch operations.

        Args:
            paths (List[str]): A list of file paths to read.

        Returns:
            Generator[Tuple[str, Any], None, None]: Yields file path and contents.
        """
        image_paths = [p for p in paths if not p.endswith(".pt")]
        cache_paths = [p for p in paths if p.endswith(".pt")]

        # Read images in batch if available
        if image_paths:
            try:
                available_paths, batch_images = self.image_data_backend.read_image_batch(
                    image_paths,
                    delete_problematic_images=self.delete_problematic_images,
                )
                for path, image in zip(available_paths, batch_images):
                    yield path, image
            except Exception as e:
                # Fallback to individual reads
                for path in image_paths:
                    try:
                        yield path, self._read_from_storage(path, hide_errors=hide_errors)
                    except Exception as read_e:
                        logger.error(f"Error reading {path}: {read_e}")
                        if self.delete_problematic_images:
                            self.metadata_backend.remove_image(path)
                            self.image_data_backend.delete(path)
                        yield path, None

        # Read cache files individually (they're typically small)
        for path in cache_paths:
            try:
                yield path, self._read_from_storage(path, hide_errors=hide_errors)
            except Exception as e:
                logger.error(f"Error reading cache {path}: {e}")
                yield path, None

    def read_images_in_batch(self) -> None:
        """Immediately read a batch of images using simplified approach.
        Replaced complex queue management with direct batch operations.

        Returns:
            None
        """
        filepaths = []
        aspect_buckets = []
        qlen = self.read_queue.qsize()
        for _ in range(0, qlen):
            read_queue_item = self.read_queue.get()
            path, aspect_bucket = read_queue_item
            filepaths.append(path)
            aspect_buckets.append(aspect_bucket)

        if not filepaths:
            return

        # Use backend batch reading capabilities
        try:
            available_filepaths, batch_output = self.image_data_backend.read_image_batch(
                filepaths, delete_problematic_images=self.delete_problematic_images
            )
            missing_image_count = len(filepaths) - len(available_filepaths)
            if len(available_filepaths) != len(filepaths):
                logging.warning(
                    f"Failed to request {missing_image_count} sample{'s' if missing_image_count > 1 else ''} during batched read, out of {len(filepaths)} total samples requested."
                    " These samples likely do not exist in the storage pool any longer."
                )

            # Add to process queue with corresponding aspect buckets
            for i, (filepath, element) in enumerate(zip(available_filepaths, batch_output)):
                if type(filepath) != str:
                    raise ValueError(f"Received unknown filepath type ({type(filepath)}) value: {filepath}")
                # Find the corresponding aspect bucket
                original_index = filepaths.index(filepath) if filepath in filepaths else i
                bucket = aspect_buckets[original_index] if original_index < len(aspect_buckets) else aspect_buckets[0]
                self.process_queue.put((filepath, element, bucket))
        except Exception as e:
            logger.error(f"Error in batch image reading: {e}")
            # Fallback: process individually
            for filepath, aspect_bucket in zip(filepaths, aspect_buckets):
                try:
                    image = self._read_from_storage(filepath)
                    if image is not None:
                        self.process_queue.put((filepath, image, aspect_bucket))
                except Exception as read_e:
                    logger.error(f"Error reading individual image {filepath}: {read_e}")

    def _process_raw_filepath(self, raw_filepath: str):
        if type(raw_filepath) == str or len(raw_filepath) == 1:
            filepath = raw_filepath
        elif len(raw_filepath) == 2:
            _, filepath = raw_filepath
        elif type(raw_filepath) == Path or type(raw_filepath) == numpy_str:
            filepath = str(raw_filepath)
        else:
            raise ValueError(f"Received unknown filepath type ({type(raw_filepath)}) value: {raw_filepath}")
        return filepath

    def _accumulate_read_queue(self, filepath, aspect_bucket):
        self.read_queue.put((filepath, aspect_bucket))

    def _process_futures(self, futures: list, executor):
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
        self.debug_log("Listing cached images")
        processed_images = self._list_cached_images()
        self.debug_log("Reading the cache and copying")
        aspect_bucket_cache = self.metadata_backend.read_cache().copy()

        # Extract and shuffle the keys of the dictionary
        do_shuffle = os.environ.get("SIMPLETUNER_SHUFFLE_ASPECTS", "true").lower() == "true"
        if do_shuffle:
            shuffled_keys = list(aspect_bucket_cache.keys())
            shuffle(shuffled_keys)

        if self.webhook_handler is not None:
            total_count = len([item for sublist in aspect_bucket_cache.values() for item in sublist])
            self.send_progress_update(
                type="init_vae_cache",
                readable_type="VAE Cache initialising",
                progress=int(len(processed_images) / max(1, total_count) * 100),
                total=total_count,
                current=0,
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for bucket in shuffled_keys:
                relevant_files = self._reduce_bucket(bucket, aspect_bucket_cache, processed_images)
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
                    test_filepath = self._image_filename_from_vaecache_filename(filepath)
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
                        self._accumulate_read_queue(filepath, aspect_bucket=bucket)
                        if self.read_queue.qsize() >= self.read_batch_size:
                            future_to_read = executor.submit(self.read_images_in_batch)
                            futures.append(future_to_read)

                        if self.process_queue.qsize() >= self.process_queue_size:
                            future_to_process = executor.submit(self._process_images_in_batch)
                            futures.append(future_to_process)

                        if self.vae_input_queue.qsize() >= self.vae_batch_size:
                            statistics["cached"] += 1
                            future_to_process = executor.submit(self._encode_images_in_batch)
                            futures.append(future_to_process)
                            if self.webhook_handler is not None:
                                last_reported_index = statistics["total"] // self.webhook_progress_interval
                                self.send_progress_update(
                                    type="vaecache",
                                    readable_type=f"VAE Caching (bucket {bucket})",
                                    progress=int(statistics["total"] / len(relevant_files) * 100),
                                    total=len(relevant_files),
                                    current=statistics["total"],
                                )

                        if self.write_queue.qsize() >= self.write_batch_size:
                            future_to_write = executor.submit(self._write_latents_in_batch)
                            futures.append(future_to_write)
                    except ValueError as e:
                        logger.error(f"Received fatal error: {e}")
                        raise e
                    except Exception as e:
                        logger.error(f"Error processing image {filepath}: {e}")
                        self.debug_log(f"Error traceback: {traceback.format_exc()}")
                        raise e

                    try:
                        futures = self._process_futures(futures, executor)
                    except Exception as e:
                        logger.error(
                            f"Error processing futures for bucket {bucket}: {e}, traceback: {traceback.format_exc()}"
                        )
                        continue
                logger.debug(f"bucket {bucket} statistics: {statistics}")
                try:
                    # Handle remainders after processing the bucket
                    if self.read_queue.qsize() > 0:
                        # We have an adequate number of samples to read. Let's now do that in a batch, to reduce I/O wait.
                        future_to_read = executor.submit(self.read_images_in_batch)
                        futures.append(future_to_read)

                    futures = self._process_futures(futures, executor)

                    # Now we try and process the images, if we have a process batch size large enough.
                    if self.process_queue.qsize() > 0:
                        future_to_process = executor.submit(self._process_images_in_batch)
                        futures.append(future_to_process)

                    futures = self._process_futures(futures, executor)

                    if self.vae_input_queue.qsize() > 0:
                        future_to_process = executor.submit(self._encode_images_in_batch)
                        futures.append(future_to_process)

                    futures = self._process_futures(futures, executor)

                    # Write the remaining batches. This is not strictly necessary, since they do not need to be written with matching dimensions.
                    # However, it's simply easiest to do this now, even if we have less-than a single batch size.
                    if self.write_queue.qsize() > 0:
                        future_to_write = executor.submit(self._write_latents_in_batch)
                        futures.append(future_to_write)

                    futures = self._process_futures(futures, executor)
                    log_msg = f"(id={self.id}) Bucket {bucket} caching results: {statistics}"
                    if get_rank() == 0:
                        logger.debug(log_msg)
                        tqdm.write(log_msg)
                    if self.webhook_handler is not None:
                        self.send_progress_update(
                            type="init_cache_vae_processing_complete",
                            progress=100,
                            total=statistics["total"],
                            current=statistics["total"],
                            readable_type=f"VAE Caching (bucket {bucket})",
                        )
                    self.debug_log("Completed process_buckets, all futures have been returned.")
                except Exception as e:
                    logger.error(f"Fatal error when processing bucket {bucket}: {e}")
                    continue

        # Send completion event for VAE cache initialization
        if self.webhook_handler is not None:
            event = lifecycle_stage_event(
                key="init_vae_cache",
                label="VAE Cache initialising",
                status="completed",
                message="VAE cache initialization complete",
                percent=100,
                current=1,
                total=1,
                job_id=StateTracker.get_job_id(),
            )
            self.webhook_handler.send_raw(event, message_level="info", job_id=StateTracker.get_job_id())

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
                yield from self._read_from_storage_concurrently(all_cache_files, hide_errors=True)
            except FileNotFoundError:
                yield (None, None)
        except Exception as e:
            if "is not iterable" not in str(e):
                logger.error(f"Error in scan_cache_contents: {e}")
                self.debug_log(f"Error traceback: {traceback.format_exc()}")
