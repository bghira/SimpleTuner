import logging
import os
from simpletuner.helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if _get_rank() == 0 else logging.WARNING)


class DatasetDuplicator:
    @staticmethod
    def copy_metadata(source_backend, target_backend):
        """Copy metadata from source backend to target backend with path updates."""
        source_meta = source_backend.get("metadata_backend", None)
        target_meta = target_backend.get("metadata_backend", None)

        if source_meta is None or target_meta is None:
            raise ValueError(
                f"Both backends must have metadata_backend defined. Received {source_meta} \n\n {target_meta}"
            )

        logger.info("Reloading metadata caches...")
        source_meta.reload_cache(set_config=False)
        target_meta.reload_cache(set_config=False)

        # Get the instance directories for path translation
        source_dir = source_backend.get("instance_data_dir", "")
        target_dir = target_backend.get("instance_data_dir", "")

        # Check if we need to update paths (for conditioning datasets)
        needs_path_update = (
            source_dir != target_dir
            and target_backend.get("dataset_type") == "conditioning"
        )

        if needs_path_update:
            logger.info(
                f"Copying metadata with path translation: '{source_dir}' -> '{target_dir}'"
            )

            # Copy and update bucket indices
            target_meta.aspect_ratio_bucket_indices = {}
            for bucket, paths in source_meta.aspect_ratio_bucket_indices.items():
                updated_paths = []
                for path in paths:
                    # Update the path to point to the target directory
                    # Handle both absolute and relative paths
                    if os.path.isabs(path):
                        # For absolute paths, replace the directory
                        rel_path = os.path.relpath(path, source_dir)
                        new_path = os.path.join(target_dir, rel_path)
                    else:
                        # For relative paths, just prepend the new directory
                        new_path = os.path.join(target_dir, os.path.basename(path))
                    updated_paths.append(new_path)
                target_meta.aspect_ratio_bucket_indices[bucket] = updated_paths

            # Copy other metadata
            if hasattr(source_meta, "image_metadata") and source_meta.image_metadata:
                target_meta.image_metadata = {}
                for path, metadata in source_meta.image_metadata.items():
                    # Update paths in image metadata too
                    if os.path.isabs(path):
                        rel_path = os.path.relpath(path, source_dir)
                        new_path = os.path.join(target_dir, rel_path)
                    else:
                        new_path = os.path.join(target_dir, os.path.basename(path))
                    target_meta.image_metadata[new_path] = metadata

            # Copy any other attributes that need to be preserved
            for attr in ["metadata_update_interval", "cache_file_suffix"]:
                if hasattr(source_meta, attr):
                    setattr(target_meta, attr, getattr(source_meta, attr))

        else:
            # Regular copy without path translation
            logger.info("Copying metadata without path translation")
            target_meta.set_metadata(metadata_backend=source_meta, update_json=True)

        # Save the updated metadata
        target_meta.save_cache()

        logger.info("Metadata copied successfully.")
        source_meta.print_debug_info()
        target_meta.print_debug_info()
        target_meta.set_readonly()

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

        # Set core fields
        target_cfg["auto_generated"] = True
        target_cfg["source_dataset_id"] = source_id
        target_cfg["dataset_type"] = "conditioning"
        target_cfg["conditioning_config"] = cond_cfg
        target_cfg["conditioning_type"] = cond_cfg.get(
            "conditioning_type", "reference_strict"
        )

        # Override for controlnet
        if global_cfg.controlnet:
            target_cfg["conditioning_type"] = "controlnet"

        # Set VAE cache directory
        source_vae_path = source_cfg.get("cache_dir_vae", None)
        if source_vae_path is not None:
            target_vae_path = os.path.join(source_vae_path, target_cfg["id"])
        else:
            target_vae_path = os.path.join(
                global_cfg.cache_dir, "vae", target_cfg["id"]
            )
        target_cfg["cache_dir_vae"] = target_vae_path

        # Create directories and set absolute paths for local backends
        if target_cfg.get("type", "local") == "local":
            os.makedirs(target_cfg["instance_data_dir"], exist_ok=True)
            os.makedirs(target_cfg["cache_dir_vae"], exist_ok=True)
            target_cfg["cache_dir_vae"] = os.path.abspath(target_cfg["cache_dir_vae"])
            target_cfg["instance_data_dir"] = os.path.abspath(
                target_cfg["instance_data_dir"]
            )

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
            target_cfg["instance_prompt"] = cond_cfg.get(
                "captions", None
            ) or cond_cfg.get("instance_prompt", None)

        return target_cfg
