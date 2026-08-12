import logging
import os

from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class DatasetDuplicator:
    @staticmethod
    def _translate_conditioning_path(path: str, source_dir: str, target_dir: str, conditioning_data_type: str | None) -> str:
        source_dir_abs = os.path.abspath(source_dir)
        target_dir_abs = os.path.abspath(target_dir)
        if os.path.isabs(path):
            path_abs = os.path.abspath(path)
            try:
                path_is_under_source = os.path.commonpath([path_abs, source_dir_abs]) == source_dir_abs
            except ValueError:
                path_is_under_source = False
            if path_is_under_source:
                rel_path = os.path.relpath(path_abs, source_dir_abs)
                new_path = os.path.join(target_dir_abs, rel_path)
            else:
                new_path = os.path.join(target_dir_abs, os.path.basename(path_abs))
        else:
            new_path = os.path.join(target_dir_abs, os.path.basename(path))
        if conditioning_data_type == "i2v_first_frame":
            new_path = os.path.splitext(new_path)[0] + ".png"
        return new_path

    @staticmethod
    def copy_metadata(source_backend, target_backend):
        """Copy metadata from source backend to target backend with path updates."""
        source_meta = source_backend.get("metadata_backend", None)
        target_meta = target_backend.get("metadata_backend", None)

        if source_meta is None or target_meta is None:
            raise ValueError(f"Both backends must have metadata_backend defined. Received {source_meta} \n\n {target_meta}")

        logger.debug("Reloading target metadata cache...")
        target_meta.reload_cache(set_config=False)

        # Get the instance directories for path translation
        source_dir = source_backend.get("instance_data_dir", "")
        target_dir = target_backend.get("instance_data_dir", "")
        target_config = target_backend.get("config", {}) or {}
        conditioning_config = target_config.get("conditioning_config") or target_backend.get("conditioning_config") or {}
        conditioning_data_type = conditioning_config.get("type")

        # Check if we need to update paths (for conditioning datasets)
        needs_path_update = source_dir != target_dir and target_backend.get("dataset_type") == "conditioning"

        if needs_path_update:
            logger.info(f"Copying metadata with path translation: '{source_dir}' -> '{target_dir}'")

            # Copy and update bucket indices
            target_meta.aspect_ratio_bucket_indices = {}
            for bucket, paths in source_meta.aspect_ratio_bucket_indices.items():
                updated_paths = []
                for path in paths:
                    new_path = DatasetDuplicator._translate_conditioning_path(
                        path,
                        source_dir,
                        target_dir,
                        conditioning_data_type,
                    )
                    updated_paths.append(new_path)
                target_meta.aspect_ratio_bucket_indices[bucket] = updated_paths

            # Copy other metadata
            if hasattr(source_meta, "image_metadata") and source_meta.image_metadata:
                target_meta.image_metadata = {}
                for path, metadata in source_meta.image_metadata.items():
                    new_path = DatasetDuplicator._translate_conditioning_path(
                        path,
                        source_dir,
                        target_dir,
                        conditioning_data_type,
                    )
                    copied_metadata = dict(metadata)
                    if conditioning_data_type == "i2v_first_frame":
                        copied_metadata["training_sample_path"] = path
                        copied_metadata["image_path"] = new_path
                    target_meta.image_metadata[new_path] = copied_metadata
                logger.debug(f"Copied {len(target_meta.image_metadata)} image_metadata entries")
            else:
                logger.debug("No image_metadata to copy from source")

            # Copy any other attributes that need to be preserved
            for attr in ["metadata_update_interval", "cache_file_suffix"]:
                if hasattr(source_meta, attr):
                    setattr(target_meta, attr, getattr(source_meta, attr))

        else:
            # Regular copy without path translation
            logger.info("Copying metadata without path translation")
            target_meta.set_metadata(metadata_backend=source_meta, update_json=False)

        source_config = source_backend.get("config", {}) or {}
        conditioning_type = target_config.get("conditioning_type") or target_backend.get("conditioning_type")

        propagated_fields = [
            "resolution",
            "resolution_type",
            "minimum_image_size",
            "maximum_image_size",
            "target_downsample_size",
            "repeats",
        ]
        for field in propagated_fields:
            if field in source_config:
                target_config[field] = source_config[field]

        if conditioning_type in ("reference_strict", "grounding"):
            alignment_fields = ["crop", "crop_aspect", "crop_style", "crop_aspect_buckets"]
            for field in alignment_fields:
                if field in source_config:
                    target_config[field] = source_config[field]

        target_backend["config"] = target_config
        if target_backend.get("id"):
            StateTracker.set_data_backend_config(target_backend["id"], target_config)

        if "repeats" in target_config:
            target_meta.repeats = int(target_config.get("repeats") or 0)
        if "resolution" in target_config and target_config["resolution"] is not None:
            target_meta.resolution = float(target_config["resolution"])
        if "resolution_type" in target_config and target_config["resolution_type"] is not None:
            target_meta.resolution_type = target_config["resolution_type"]
        if "minimum_image_size" in target_config:
            target_meta.minimum_image_size = target_config["minimum_image_size"]
        if "maximum_image_size" in target_config:
            target_meta.maximum_image_size = target_config["maximum_image_size"]
        if "target_downsample_size" in target_config:
            target_meta.target_downsample_size = target_config["target_downsample_size"]

        target_meta.config = target_config

        if conditioning_data_type == "i2v_first_frame" and hasattr(target_meta, "save_cache"):
            target_meta.save_cache()

        # Bucket indices may be rank-local here; do not overwrite the canonical target cache.
        target_meta.set_readonly()
        if hasattr(target_meta, "save_image_metadata"):
            target_meta.save_image_metadata()
            logger.debug("Saved image_metadata to disk")
        else:
            logger.warning("target_meta does not have save_image_metadata method")

        logger.info("Metadata copied successfully.")
        source_meta.print_debug_info()
        target_meta.print_debug_info()

    @staticmethod
    def generate_conditioning_datasets(global_config, source_backend_config):
        """Generate conditioning dataset configs from source dataset."""
        source_dataset_id = source_backend_config.get("id")
        source_conditioning_config = source_backend_config.get("conditioning", None)

        if source_conditioning_config is None:
            return []

        # Normalize to list
        if type(source_conditioning_config) is dict:
            source_conditioning_config = [source_conditioning_config]
        elif type(source_conditioning_config) is not list:
            raise ValueError("Conditioning config must be a dict or a list of dicts.")

        target_backend_configs = []
        target_backend_ids = []

        for conditioning_config in source_conditioning_config:
            target_cfg = DatasetDuplicator._create_single_conditioning_config(
                source_backend_config,
                conditioning_config,
                source_dataset_id,
                global_config,
            )
            # if the target cfg has captions defined and we're in conditioning_multidataset_sampling=combined mode, we error out.
            if (
                global_config.conditioning_multidataset_sampling == "combined"
                and target_cfg.get("caption_strategy", None) is not None
            ):
                raise ValueError(
                    f"Conditioning config {target_cfg['id']} has captions defined, but 'conditioning_multidataset_sampling' is set to 'combined'. "
                    "Please remove captions from the conditioning config or change the sampling mode."
                )

            target_backend_configs.append(target_cfg)
            target_backend_ids.append(target_cfg["id"])

        # Remove the conditioning config from the source backend config
        source_backend_config.pop("conditioning", None)
        # Link all conditioning datasets to source
        source_backend_config["conditioning_data"] = target_backend_ids

        return source_backend_config, target_backend_configs

    @staticmethod
    def _create_single_conditioning_config(source_cfg, cond_cfg, source_id, global_cfg):
        """Create a single conditioning config - extracted for clarity."""
        target_cfg = source_cfg.copy()

        # Get conditioning type
        conditioning_data_type = cond_cfg.get("type", None)
        if conditioning_data_type is None:
            raise ValueError(
                "Conditioning config must have a 'type' field containing a value like 'canny', 'depth_midas', etc."
            )

        # Set ID
        target_cfg["id"] = f"{source_id}_conditioning_{conditioning_data_type}"

        # Set instance data directory
        target_dataset_path = cond_cfg.get("instance_data_dir", None)
        if target_dataset_path is None:
            target_dataset_path = os.path.join(
                global_cfg.cache_dir,
                "conditioning_data",
                target_cfg["id"],
            )
        target_cfg["instance_data_dir"] = target_dataset_path

        # Remove conditioning from target config
        del target_cfg["conditioning"]

        # Conditioning datasets don't use audio - remove audio settings
        # (e.g., IC-LoRA reference videos are visual-only conditioning)
        target_cfg.pop("audio", None)
        target_cfg.pop("s2v_datasets", None)
        target_cfg.pop("_s2v_audio_autoinjected", None)
        if conditioning_data_type == "i2v_first_frame":
            target_cfg.pop("video", None)

        # Set core fields
        target_cfg["auto_generated"] = True
        target_cfg["source_dataset_id"] = source_id
        target_cfg["dataset_type"] = "conditioning"
        target_cfg["conditioning_config"] = cond_cfg
        target_cfg["conditioning_type"] = cond_cfg.get("conditioning_type", "reference_strict")

        # Auto-generated conditioning datasets must use local storage
        # even if the source dataset is HuggingFace, because conditioning
        # images are derived/generated data that need to be cached locally
        target_cfg["type"] = "local"

        # Auto-generated conditioning datasets must use discovery metadata backend
        # to work with local storage, even if source used huggingface metadata backend
        target_cfg["metadata_backend"] = "discovery"

        # Override for controlnet
        if global_cfg.controlnet:
            target_cfg["conditioning_type"] = "controlnet"

        # Set VAE cache directory
        source_vae_path = source_cfg.get("cache_dir_vae", None)
        if source_vae_path is not None:
            target_vae_path = os.path.join(source_vae_path, target_cfg["id"])
        else:
            target_vae_path = os.path.join(global_cfg.cache_dir, "vae", target_cfg["id"])
        target_cfg["cache_dir_vae"] = target_vae_path

        # Create directories and set absolute paths for local backends
        if target_cfg.get("type", "local") == "local":
            os.makedirs(target_cfg["instance_data_dir"], exist_ok=True)
            os.makedirs(target_cfg["cache_dir_vae"], exist_ok=True)
            target_cfg["cache_dir_vae"] = os.path.abspath(target_cfg["cache_dir_vae"])
            target_cfg["instance_data_dir"] = os.path.abspath(target_cfg["instance_data_dir"])

        # Handle caption strategy
        target_cfg["caption_strategy"] = cond_cfg.get("caption_strategy", None)
        target_cfg["instance_prompt"] = None

        if target_cfg["caption_strategy"] not in [None, "instanceprompt"]:
            logger.warning(
                f"Caption strategy {target_cfg['caption_strategy']} in base model will be overridden by instanceprompt strategy in the conditioning config."
            )
            target_cfg["caption_strategy"] = "instanceprompt"

        # Check for captions with exact original logic
        if cond_cfg.get("captions", False) not in [False, None]:
            target_cfg["caption_strategy"] = "instanceprompt"
            target_cfg["instance_prompt"] = cond_cfg.get("captions", None) or cond_cfg.get("instance_prompt", None)

        return target_cfg
