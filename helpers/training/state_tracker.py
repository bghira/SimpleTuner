from multiprocessing import Manager
from os import environ, path
from pathlib import Path
import json, logging

logger = logging.getLogger("StateTracker")
logger.setLevel(environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class StateTracker:
    # Class variables
    global_step = 0
    epoch_step = 0
    has_training_started = False
    calculate_luminance = False
    all_image_files = {}
    all_vae_cache_files = {}
    all_caption_files = None

    # Backend entities for retrieval
    data_backend = None
    vaecache = None
    embedcache = None
    accelerator = None
    bucket_managers = []
    data_backends = {}
    vae = None
    vae_dtype = None
    weight_dtype = None
    args = None

    @classmethod
    def delete_cache_files(cls):
        for cache_name in ["all_image_files", "all_vae_cache_files"]:
            cache_path = Path(cls.args.output_dir) / f"{cache_name}.json"
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except:
                    pass

        # Glob the directory for "all_image_files.*.json" and "all_vae_cache_files.*.json", and delete those too
        # This is a workaround for the fact that the cache files are named with the data_backend_id
        filelist = Path(cls.args.output_dir).glob("all_image_files_*.json")
        for file in filelist:
            try:
                file.unlink()
            except:
                pass

        filelist = Path(cls.args.output_dir).glob("all_vae_cache_files_*.json")
        for file in filelist:
            try:
                file.unlink()
            except:
                pass

    @classmethod
    def _load_from_disk(cls, cache_name):
        cache_path = Path(cls.args.output_dir) / f"{cache_name}.json"
        if cache_path.exists():
            with cache_path.open("r") as f:
                return json.load(f)
        return None

    @classmethod
    def _save_to_disk(cls, cache_name, data):
        cache_path = Path(cls.args.output_dir) / f"{cache_name}.json"
        with cache_path.open("w") as f:
            json.dump(data, f)

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
    def get_image_files(cls, data_backend_id: str):
        if data_backend_id not in cls.all_image_files:
            cls.all_image_files[data_backend_id] = cls._load_from_disk(
                "all_image_files_{}".format(data_backend_id)
            )
        return cls.all_image_files[data_backend_id]

    @classmethod
    def get_global_step(cls):
        return cls.global_step

    @classmethod
    def set_global_step(cls, global_step: int):
        cls.global_step = global_step

    @classmethod
    def get_epoch_step(cls):
        return cls.epoch_step

    @classmethod
    def set_epoch_step(cls, epoch_step: int):
        cls.epoch_step = epoch_step

    @classmethod
    def set_vae_cache_files(cls, raw_file_list: list, data_backend_id: str):
        if cls.all_vae_cache_files[data_backend_id] is not None:
            cls.all_vae_cache_files[data_backend_id].clear()
        else:
            cls.all_vae_cache_files[data_backend_id] = {}
        for subdirectory_list in raw_file_list:
            _, _, files = subdirectory_list
            for image in files:
                cls.all_vae_cache_files[data_backend_id][path.basename(image)] = False
        cls._save_to_disk(
            "all_vae_cache_files_{}".format(data_backend_id),
            cls.all_vae_cache_files[data_backend_id],
        )
        logger.debug(
            f"set_vae_cache_files found {len(cls.all_vae_cache_files[data_backend_id])} images."
        )

    @classmethod
    def get_vae_cache_files(cls: list, data_backend_id: str):
        if data_backend_id not in cls.all_vae_cache_files:
            cls.all_vae_cache_files[data_backend_id] = cls._load_from_disk(
                "all_vae_cache_files_{}".format(data_backend_id)
            )
        return cls.all_vae_cache_files[data_backend_id]

    @classmethod
    def set_caption_files(cls, caption_files):
        cls.all_caption_files = caption_files
        cls._save_to_disk("all_caption_files", cls.all_caption_files)

    @classmethod
    def get_caption_files(cls):
        if not cls.all_caption_files:
            cls.all_caption_files = cls._load_from_disk("all_caption_files")
        return cls.all_caption_files

    @classmethod
    def register_data_backend(cls, data_backend):
        cls.data_backends[data_backend["id"]] = data_backend

    @classmethod
    def get_data_backend(cls, id: str):
        return cls.data_backends[id]

    @classmethod
    def get_data_backend_config(cls, data_backend_id: str):
        return cls.data_backends.get(data_backend_id, {}).get("config", {})

    @classmethod
    def set_data_backend_config(cls, data_backend_id: str, config: dict):
        if data_backend_id not in cls.data_backends:
            cls.data_backends[data_backend_id] = {}
        cls.data_backends[data_backend_id]["config"] = config

    @classmethod
    def get_data_backends(cls):
        return cls.data_backends

    @classmethod
    def set_accelerator(cls, accelerator):
        cls.accelerator = accelerator

    @classmethod
    def get_accelerator(cls):
        return cls.accelerator

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
    def get_bucket_managers(cls):
        return cls.bucket_managers

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
    def set_vaecache(cls, vaecache):
        cls.vaecache = vaecache

    @classmethod
    def get_vaecache(cls, id: str):
        return cls.data_backends[id]["vaecache"]

    @classmethod
    def set_embedcache(cls, embedcache):
        cls.embedcache = embedcache

    @classmethod
    def get_embedcache(cls):
        return cls.embedcache

    @classmethod
    def get_metadata_by_filepath(cls, filepath):
        for bucket_manager in cls.get_bucket_managers():
            metadata = bucket_manager.get_metadata_by_filepath(filepath)
            if metadata is not None:
                return metadata
        return None
