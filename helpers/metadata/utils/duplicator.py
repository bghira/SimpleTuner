import logging
import os
from helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if _get_rank() == 0 else logging.WARNING)


class DatasetDuplicator:
    @staticmethod
    def copy_metadata(source_backend, target_backend):
        """Copy metadata from source backend to target backend."""
        source_meta = source_backend.get("metadata_backend", None)
        target_meta = target_backend.get("metadata_backend", None)

        if source_meta is None or target_meta is None:
            raise ValueError(
                "Both source and target backends must have metadata_backend defined."
                f" Received {type(source_meta)} and {type(target_meta)}."
            )

        logger.info("Reloading source metadata caches...")
        source_meta.reload_cache(set_config=False)
        logger.info("Reloading target metadata caches...")
        target_meta.reload_cache(set_config=False)

        logger.info("Copying metadata from source to target backend...")
        target_meta.set_metadata(metadata_backend=source_meta, update_json=True)

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

            target_backend_configs.append(target_cfg)
            target_backend_ids.append(target_cfg["id"])

        # Remove the conditioning config from the source backend config
        source_backend_config.pop("conditioning", None)
        # Link first conditioning dataset to source (current limitation)
        source_backend_config["conditioning_data"] = target_backend_ids[0]

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
            target_cfg["instance_prompt"] = cond_cfg["captions"]

        return target_cfg
