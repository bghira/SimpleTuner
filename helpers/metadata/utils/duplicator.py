import logging, os
from helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if _get_rank() == 0 else logging.WARNING)


class DatasetDuplicator:
    @staticmethod
    def copy_metadata(source_backend, target_backend):
        """
        Copies metadata from source backend to target backend.

        Args:
            source_backend: The backend from which to copy metadata.
            target_backend: The backend to which metadata will be copied.
        """
        source_metadata = source_backend.get("metadata_backend", None)
        target_metadata = target_backend.get("metadata_backend", None)
        if source_metadata is None or target_metadata is None:
            raise ValueError(
                "Both source and target backends must have metadata_backend defined."
            )

        logger.info("Reloading source metadata caches...")
        source_metadata.reload_cache(set_config=False)
        logger.info("Reloading target metadata caches...")
        target_metadata.reload_cache(set_config=False)

        # Write the new metadata to the target backend
        logger.info("Copying metadata from source to target backend...")
        target_metadata.set_metadata(
            metadata_backend=source_metadata,
            update_json=True,
        )
        logger.info("Metadata copied successfully.")
        source_metadata.print_debug_info()
        target_metadata.print_debug_info()
        target_metadata.set_readonly()

    @staticmethod
    def generate_conditioning_datasets(global_config, source_backend_config):
        """
        Generates one or more conditioning dataset config from a source dataset.

        Attaches the conditioning dataset IDs to the source backend config.

        Args:
            global_config: Global configuration object from commandline/config file.
            source_backend_config: Configuration of the source backend.

        Returns:
            dict: source_backend_config (dict), target_backend_configs (list[dict])
        """
        source_dataset_id = source_backend_config.get("id")
        source_conditioning_config = source_backend_config.get("conditioning", None)
        if source_conditioning_config is None:
            return []
        if type(source_conditioning_config) is dict:
            source_conditioning_config = [source_conditioning_config]
        elif type(source_conditioning_config) is not list:
            raise ValueError("Conditioning config must be a dict or a list of dicts.")

        target_backend_configs = []
        target_backend_ids = []
        for conditioning_config in source_conditioning_config:
            target_backend_config = source_backend_config.copy()
            target_dataset_path = conditioning_config.get("instance_data_dir", None)
            conditioning_data_type = conditioning_config.get("type", None)
            target_backend_config["id"] = (
                f"{source_dataset_id}_conditioning_{conditioning_data_type}"
            )
            if conditioning_data_type is None:
                raise ValueError(
                    "Conditioning config must have a 'type' field containing a value like 'canny', 'depth_midas', etc."
                )
            if target_dataset_path is None:
                target_dataset_path = os.path.join(
                    global_config.cache_dir,
                    "conditioning_data",
                    target_backend_config["id"],
                )
            del target_backend_config["conditioning"]
            target_backend_config["auto_generated"] = True
            target_backend_config["source_dataset_id"] = source_dataset_id
            target_backend_config["dataset_type"] = "conditioning"
            target_backend_config["conditioning_config"] = conditioning_config
            target_backend_config["instance_data_dir"] = (
                target_dataset_path  # where the generated conditioning data will be stored
            )
            target_backend_config["conditioning_type"] = conditioning_config.get(
                "conditioning_type", "reference_strict"
            )

            if global_config.controlnet:
                target_backend_config["conditioning_type"] = "controlnet"
            source_vae_path = source_backend_config.get("cache_dir_vae", None)
            if source_vae_path is not None:
                target_vae_path = os.path.join(
                    source_vae_path, target_backend_config["id"]
                )
            else:
                target_vae_path = os.path.join(
                    global_config.cache_dir, "vae", target_backend_config["id"]
                )
            target_backend_config["cache_dir_vae"] = target_vae_path
            if target_backend_config.get("type", "local") == "local":
                # Ensure the directory exists
                os.makedirs(target_backend_config["instance_data_dir"], exist_ok=True)
                os.makedirs(target_backend_config["cache_dir_vae"], exist_ok=True)
                target_backend_config["cache_dir_vae"] = os.path.abspath(
                    target_backend_config["cache_dir_vae"]
                )
                target_backend_config["instance_data_dir"] = os.path.abspath(
                    target_backend_config["instance_data_dir"]
                )
            target_backend_config["caption_strategy"] = conditioning_config.get(
                "caption_strategy", "instanceprompt"
            )
            target_backend_config["instance_prompt"] = None
            if conditioning_config.get(
                "captions", False
            ) is not False and conditioning_config.get("caption_strategy", None) in [
                None,
                "instanceprompt",
            ]:
                # if there's some captions defined, use them as instance prompts
                target_backend_config["instance_prompt"] = conditioning_config[
                    "captions"
                ]

            target_backend_configs.append(target_backend_config)
            target_backend_ids.append(target_backend_config["id"])

        # remove the conditioning config from the source backend config
        source_backend_config.pop("conditioning", None)
        # source_backend_config["conditioning_data"] = target_backend_ids
        source_backend_config["conditioning_data"] = target_backend_ids[
            0
        ]  # for now, we can only do one.

        return source_backend_config, target_backend_configs
