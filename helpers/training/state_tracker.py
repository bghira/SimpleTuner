from os import environ
from pathlib import Path
import json
import time, os
import logging
from helpers.models.all import model_families

logger = logging.getLogger("StateTracker")
logger.setLevel(environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

filename_mapping = {
    "all_image_files": "image",
    "all_vae_cache_files": "vae",
    "all_text_cache_files": "text",
}


class StateTracker:
    config_path = None
    # Class variables
    model_type = ""
    model = None
    # Job ID for FastAPI. None if local.
    job_id = None

    ## Training state
    global_step = 0
    global_resume_step = None
    epoch_step = 0
    epoch_micro_step = 0
    epoch = 1

    ## Caches
    all_image_files = {}
    all_vae_cache_files = {}
    all_text_cache_files = {}
    all_caption_files = None

    ## Backend entities for retrieval
    default_text_embed_cache = None
    _is_sdxl_refiner = False
    accelerator = None
    data_backends = {}
    parquet_databases = {}
    # A list of backend IDs to exhaust.
    exhausted_backends = []
    # A dict of backend IDs to the number of times they have been repeated.
    repeats = {}
    # The images we'll use for upscaling at validation time. Stored at startup.
    validation_sample_images = []
    vae = None
    vae_dtype = None
    weight_dtype = None
    args = None
    # Aspect to resolution map, we'll store once generated for consistency.
    aspect_resolution_map = {}

    # for schedulefree
    last_lr = 0.0

    # hugging face hub user details
    hf_user = None

    webhook_handler = None

    @classmethod
    def delete_cache_files(
        cls, data_backend_id: str = None, preserve_data_backend_cache=False
    ):
        for cache_name in [
            "all_image_files",
            "all_vae_cache_files",
            "all_text_cache_files",
        ]:
            if filename_mapping[cache_name] in str(preserve_data_backend_cache):
                continue
            data_backend_id_suffix = ""
            if data_backend_id:
                data_backend_id_suffix = f"_{data_backend_id}"
            cache_path = (
                Path(cls.args.output_dir) / f"{cache_name}{data_backend_id_suffix}.json"
            )
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    logger.warning(
                        f"(rank={os.environ.get('RANK')}) Deleted cache file: {cache_path}"
                    )
                except:
                    pass

    @classmethod
    def _load_from_disk(cls, cache_name, retry_limit: int = 0):
        cache_path = Path(cls.args.output_dir) / f"{cache_name}.json"
        retry_count = 0
        results = None
        while retry_count < retry_limit and (
            not cache_path.exists() or results is None
        ):
            if cache_path.exists():
                try:
                    with cache_path.open("r") as f:
                        results = json.load(f)
                except Exception as e:
                    logger.error(
                        f"Invalidating cache: error loading {cache_name} from disk. {e}"
                    )
                    return None
            else:
                retry_count += 1
                if retry_count < retry_limit:
                    logger.debug(
                        f"Cache file {cache_name} does not exist. Retry {retry_count}/{retry_limit}."
                    )
                    time.sleep(1)
                else:
                    logger.warning(f"No cache file was found: {cache_path}")
        logger.debug(f"Returning: {type(results)}")
        return results

    @classmethod
    def _save_to_disk(cls, cache_name, data):
        cache_path = Path(cls.args.output_dir) / f"{cache_name}.json"
        logger.debug(
            f"(rank={os.environ.get('RANK')}) Saving {cache_name} to disk: {cache_path}"
        )
        with cache_path.open("w") as f:
            json.dump(data, f)
        logger.debug(
            f"(rank={os.environ.get('RANK')}) Save complete {cache_name} to disk: {cache_path}"
        )

    @classmethod
    def set_config_path(cls, config_path: str):
        cls.config_path = config_path

    @classmethod
    def get_config_path(cls):
        return cls.config_path

    @classmethod
    def set_model_family(cls, model_type: str):
        if model_type not in model_families.keys():
            raise ValueError(f"Unknown model type: {model_type}")
        cls.model_type = model_type

    @classmethod
    def get_model_family(cls):
        return cls.model_type

    @classmethod
    def set_model(cls, model):
        cls.model = model

    @classmethod
    def get_model(cls):
        return cls.model

    @classmethod
    def get_hf_user(cls):
        return cls.hf_user

    @classmethod
    def set_hf_user(cls, hf_user):
        cls.hf_user = hf_user

    @classmethod
    def get_hf_username(cls):
        if cls.hf_user is not None and "name" in cls.hf_user:
            return cls.hf_user["name"]
        return None

    @classmethod
    def is_sdxl_refiner(cls, set_value=None):
        if set_value is not None:
            cls._is_sdxl_refiner = set_value
        return cls._is_sdxl_refiner

    @classmethod
    def set_parquet_database(cls, data_backend_id: str, parquet_database: tuple):
        """parquet_database is a tuple (dataframe, filename_column, caption_column, fallback_caption_column)"""
        cls.parquet_databases[data_backend_id] = parquet_database

    @classmethod
    def get_parquet_database(cls, data_backend_id: str):
        return cls.parquet_databases.get(data_backend_id, (None, None, None, None))

    @classmethod
    def set_image_files(cls, raw_file_list: list, data_backend_id: str):
        if cls.all_image_files[data_backend_id] is not None:
            cls.all_image_files[data_backend_id].clear()
        else:
            cls.all_image_files[data_backend_id] = {}
        for subdirectory_list in raw_file_list:
            _, _, files = subdirectory_list
            for image in files:
                cls.all_image_files[data_backend_id][image] = False
        cls._save_to_disk(
            "all_image_files_{}".format(data_backend_id),
            cls.all_image_files[data_backend_id],
        )
        logger.debug(
            f"set_image_files found {len(cls.all_image_files[data_backend_id])} images."
        )
        return cls.all_image_files[data_backend_id]

    @classmethod
    def get_image_files(cls, data_backend_id: str, retry_limit: int = 0):
        if (
            data_backend_id in cls.all_image_files
            and cls.all_image_files[data_backend_id] is None
        ):
            # we should probaby try to reload it from disk if it failed earlier.
            logger.debug(
                f"(rank={os.environ.get('RANK')}) Clearing out invalid pre-loaded cache entry for {data_backend_id}"
            )
            del cls.all_image_files[data_backend_id]
        if data_backend_id not in cls.all_image_files:
            logger.debug(
                f"(rank={os.environ.get('RANK')}) Attempting to load from disk: {data_backend_id}"
            )
            cls.all_image_files[data_backend_id] = cls._load_from_disk(
                "all_image_files_{}".format(data_backend_id), retry_limit=retry_limit
            )
            logger.debug(
                f"(rank={os.environ.get('RANK')}) Completed load from disk: {data_backend_id}: {type(cls.all_image_files[data_backend_id])}"
            )
        else:
            logger.debug(f"()")
        logger.debug(
            f"(rank={os.environ.get('RANK')}) Returning {type(cls.all_image_files[data_backend_id])} for {data_backend_id}"
        )
        return cls.all_image_files[data_backend_id]

    @classmethod
    def get_global_resume_step(cls):
        return cls.global_resume_step

    @classmethod
    def set_global_resume_step(cls, global_resume_step: int):
        cls.global_resume_step = global_resume_step

    @classmethod
    def get_global_step(cls):
        return cls.global_step

    @classmethod
    def set_global_step(cls, global_step: int):
        cls.global_step = global_step

    @classmethod
    def get_epoch(cls):
        return cls.epoch

    @classmethod
    def set_epoch(cls, epoch: int):
        logger.debug(f"Current training state: {cls.get_training_state()}")
        cls.epoch = epoch

    @classmethod
    def get_epoch_step(cls):
        return cls.epoch_step

    @classmethod
    def set_epoch_step(cls, epoch_step: int):
        cls.epoch_step = epoch_step

    @classmethod
    def set_repeats(cls, repeats: dict):
        cls.repeats = repeats

    @classmethod
    def load_training_state(cls, state_path: str):
        try:
            with open(state_path, "r") as f:
                training_state = json.load(f)
        except OSError as e:
            logger.error(f"Error loading training state: {e}")
            training_state = {}
        except Exception as e:
            logger.error(f"Error loading training state: {e}")
            training_state = {}
        cls.set_global_step(training_state.get("global_step", 0))
        cls.set_epoch_step(training_state.get("epoch_step", 0))
        cls.set_epoch(training_state.get("epoch", 1))
        cls.set_exhausted_backends(training_state.get("exhausted_backends", []))
        cls.init_repeats(training_state.get("repeats", {}))
        logging.debug(f"Training state loaded: {cls.get_training_state()}")

    @classmethod
    def save_training_state(cls, state_path: str):
        training_state = {
            "global_step": cls.global_step,
            "epoch_step": cls.epoch_step,
            "epoch": cls.epoch,
            "exhausted_backends": cls.exhausted_backends,
            "repeats": cls.repeats,
        }
        logger.debug(f"Saving training state: {training_state}")
        with open(state_path, "w") as f:
            json.dump(training_state, f)

    @classmethod
    def get_training_state(cls):
        return {
            "global_step": cls.global_step,
            "epoch_step": cls.epoch_step,
            "epoch": cls.epoch,
            "exhausted_backends": cls.exhausted_backends,
            "repeats": cls.repeats,
        }

    @classmethod
    def set_repeats(cls, repeats: int, data_backend_id: str = None):
        if data_backend_id is None:
            # set every entry in repeats to zero
            for key in cls.repeats.keys():
                cls.repeats[key] = repeats
        else:
            cls.repeats[data_backend_id] = repeats

    @classmethod
    def init_repeats(cls, repeats: int):
        cls.repeats = repeats

    @classmethod
    def get_repeats(cls, data_backend_id: str):
        if data_backend_id not in cls.repeats:
            return 0
        return cls.repeats[data_backend_id]

    @classmethod
    def increment_repeats(cls, data_backend_id: str):
        cls.set_repeats(
            data_backend_id=data_backend_id,
            repeats=cls.get_repeats(data_backend_id) + 1,
        )

    @classmethod
    def backend_status(cls, data_backend_id: str):
        return data_backend_id in cls.exhausted_backends

    @classmethod
    def backend_exhausted(cls, data_backend_id: str):
        cls.exhausted_backends.append(data_backend_id)

    @classmethod
    def backend_enable(cls, data_backend_id: str):
        cls.exhausted_backends.remove(data_backend_id)

    @classmethod
    def set_exhausted_backends(cls, exhausted_backends: list):
        cls.exhausted_backends = exhausted_backends

    @classmethod
    def clear_exhausted_buckets(cls):
        cls.exhausted_backends = []

    @classmethod
    def set_vae_cache_files(cls, raw_file_list: list, data_backend_id: str):
        if cls.all_vae_cache_files.get(data_backend_id) is not None:
            cls.all_vae_cache_files[data_backend_id].clear()
        else:
            cls.all_vae_cache_files[data_backend_id] = {}
        for subdirectory_list in raw_file_list:
            _, _, files = subdirectory_list
            for image in files:
                cls.all_vae_cache_files[data_backend_id][image] = False
        cls._save_to_disk(
            "all_vae_cache_files_{}".format(data_backend_id),
            cls.all_vae_cache_files[data_backend_id],
        )
        logger.debug(
            f"set_vae_cache_files found {len(cls.all_vae_cache_files[data_backend_id])} images."
        )

    @classmethod
    def get_vae_cache_files(cls: list, data_backend_id: str, retry_limit: int = 0):
        if (
            data_backend_id not in cls.all_vae_cache_files
            or cls.all_vae_cache_files.get(data_backend_id) is None
        ):
            cls.all_vae_cache_files[data_backend_id] = cls._load_from_disk(
                "all_vae_cache_files_{}".format(data_backend_id)
            )
        return cls.all_vae_cache_files[data_backend_id] or {}

    @classmethod
    def set_text_cache_files(cls, raw_file_list: list, data_backend_id: str):
        if cls.all_text_cache_files[data_backend_id] is not None:
            cls.all_text_cache_files[data_backend_id].clear()
        else:
            cls.all_text_cache_files[data_backend_id] = {}
        for subdirectory_list in raw_file_list:
            _, _, files = subdirectory_list
            for text_embed_path in files:
                cls.all_text_cache_files[data_backend_id][text_embed_path] = False
        cls._save_to_disk(
            "all_text_cache_files_{}".format(data_backend_id),
            cls.all_text_cache_files[data_backend_id],
        )
        logger.debug(
            f"set_text_cache_files found {len(cls.all_text_cache_files[data_backend_id])} images."
        )

    @classmethod
    def get_text_cache_files(cls: list, data_backend_id: str, retry_limit: int = 0):
        if data_backend_id not in cls.all_text_cache_files:
            cls.all_text_cache_files[data_backend_id] = cls._load_from_disk(
                "all_text_cache_files_{}".format(data_backend_id),
                retry_limit=retry_limit,
            )
        return cls.all_text_cache_files[data_backend_id]

    @classmethod
    def set_caption_files(cls, caption_files):
        cls.all_caption_files = caption_files
        cls._save_to_disk("all_caption_files", cls.all_caption_files)

    @classmethod
    def get_caption_files(cls, retry_limit: int = 0):
        if not cls.all_caption_files:
            cls.all_caption_files = cls._load_from_disk(
                "all_caption_files", retry_limit=retry_limit
            )
        return cls.all_caption_files

    @classmethod
    def get_validation_sample_images(cls):
        return cls.validation_sample_images

    @classmethod
    def set_validation_sample_images(cls, validation_sample_images):
        cls.validation_sample_images = validation_sample_images

    @classmethod
    def register_data_backend(cls, data_backend):
        cls.data_backends[data_backend["id"]] = data_backend

    @classmethod
    def get_data_backend(cls, id: str):
        return cls.data_backends[id]

    @classmethod
    def get_dataset_size(cls, data_backend_id: str):
        if "sampler" in cls.data_backends[data_backend_id]:
            return len(cls.data_backends[data_backend_id]["sampler"])
        return 0

    @classmethod
    def set_conditioning_dataset(
        cls, data_backend_id: str, conditioning_backend_id: str
    ):
        cls.data_backends[data_backend_id]["conditioning_data"] = cls.data_backends[
            conditioning_backend_id
        ]

    @classmethod
    def get_conditioning_dataset(cls, data_backend_id: str):
        return cls.data_backends[data_backend_id].get("conditioning_data", None)

    @classmethod
    def get_data_backend_config(cls, data_backend_id: str):
        return cls.data_backends.get(data_backend_id, {}).get("config", {})

    @classmethod
    def set_data_backend_config(cls, data_backend_id: str, config: dict):
        if data_backend_id not in cls.data_backends:
            cls.data_backends[data_backend_id] = {}
        cls.data_backends[data_backend_id]["config"] = config

    @classmethod
    def get_conditioning_mappings(cls):
        conditioning_mappings = {}
        for data_backend_id, data_backend in cls.data_backends.items():
            if "conditioning_data" in data_backend:
                conditioning_mappings[data_backend_id] = data_backend[
                    "conditioning_data"
                ]["id"]
        return conditioning_mappings

    @classmethod
    def clear_data_backends(cls):
        cls.data_backends = {}

    @classmethod
    def get_data_backends(cls, _type="image", _types=["image", "video"]):
        output = {}
        for backend_id, backend in dict(cls.data_backends).items():
            if backend.get("dataset_type", "image") == _type or (
                type(_types) is list and backend.get("dataset_type", "image") in _types
            ):
                output[backend_id] = backend
        return output

    @classmethod
    def set_accelerator(cls, accelerator):
        cls.accelerator = accelerator

    @classmethod
    def get_accelerator(cls):
        return cls.accelerator

    @classmethod
    def get_webhook_handler(cls):
        return cls.webhook_handler

    @classmethod
    def set_webhook_handler(cls, webhook_handler):
        cls.webhook_handler = webhook_handler

    @classmethod
    def set_job_id(cls, job_id: str):
        cls.job_id = job_id

    @classmethod
    def get_job_id(cls):
        return cls.job_id

    @classmethod
    def set_vae(cls, vae):
        cls.vae = vae

    @classmethod
    def get_vae(cls):
        return cls.vae

    @classmethod
    def set_vae_dtype(cls, vae_dtype):
        cls.vae_dtype = vae_dtype

    @classmethod
    def get_vae_dtype(cls):
        return cls.vae_dtype

    @classmethod
    def set_weight_dtype(cls, weight_dtype):
        cls.weight_dtype = weight_dtype

    @classmethod
    def get_weight_dtype(cls):
        return cls.weight_dtype

    @classmethod
    def set_args(cls, args):
        cls.args = args

    @classmethod
    def get_args(cls):
        return cls.args

    @classmethod
    def get_vaecache(cls, id: str):
        return cls.data_backends[id].get("vaecache", None)

    @classmethod
    def set_default_text_embed_cache(cls, default_text_embed_cache):
        cls.default_text_embed_cache = default_text_embed_cache

    @classmethod
    def get_default_text_embed_cache(cls):
        return cls.default_text_embed_cache

    @classmethod
    def get_embedcache(cls, data_backend_id: str):
        return cls.data_backends[data_backend_id]["text_embed_cache"]

    @classmethod
    def get_metadata_by_filepath(
        cls,
        filepath,
        data_backend_id: str,
        search_dataset_types: list = ["image", "video"],
    ):
        for _, data_backend in cls.get_data_backends(
            _types=search_dataset_types
        ).items():
            if "metadata_backend" not in data_backend:
                continue
            if data_backend_id != data_backend["metadata_backend"].id:
                continue
            metadata = data_backend["metadata_backend"].get_metadata_by_filepath(
                filepath
            )
            if metadata is not None:
                return metadata
        return None

    @classmethod
    def get_resolution_by_aspect(cls, dataloader_resolution: float, aspect: float):
        return cls.aspect_resolution_map.get(dataloader_resolution, {}).get(
            str(aspect), None
        )

    @classmethod
    def set_resolution_by_aspect(
        cls, dataloader_resolution: float, aspect: float, resolution: int
    ):
        if dataloader_resolution not in cls.aspect_resolution_map:
            cls.aspect_resolution_map[dataloader_resolution] = {}
        cls.aspect_resolution_map[dataloader_resolution][str(aspect)] = resolution
        cls._save_to_disk(
            f"aspect_resolution_map-{dataloader_resolution}",
            cls.aspect_resolution_map[dataloader_resolution],
        )
        logger.debug(
            f"Aspect resolution map: {cls.aspect_resolution_map[dataloader_resolution]}"
        )

    @classmethod
    def save_aspect_resolution_map(cls, dataloader_resolution: float):
        cls._save_to_disk(
            f"aspect_resolution_map-{dataloader_resolution}",
            cls.aspect_resolution_map[dataloader_resolution],
        )

    @classmethod
    def load_aspect_resolution_map(
        cls, dataloader_resolution: float, retry_limit: int = 0
    ):
        if dataloader_resolution not in cls.aspect_resolution_map:
            cls.aspect_resolution_map = {dataloader_resolution: {}}

        cls.aspect_resolution_map[dataloader_resolution] = (
            cls._load_from_disk(
                f"aspect_resolution_map-{dataloader_resolution}",
                retry_limit=retry_limit,
            )
            or {}
        )
        logger.debug(
            f"Aspect resolution map: {cls.aspect_resolution_map[dataloader_resolution]}"
        )

    @classmethod
    def get_last_lr(cls):
        return cls.last_lr

    @classmethod
    def set_last_lr(cls, last_lr: float):
        cls.last_lr = float(last_lr)
