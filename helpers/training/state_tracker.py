from multiprocessing import Manager
from os import environ, path
from pathlib import Path
import json, logging

logger = logging.getLogger("StateTracker")
logger.setLevel(environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class StateTracker:
    # Class variables
    has_training_started = False
    calculate_luminance = False
    all_image_files = None
    all_vae_cache_files = None
    all_caption_files = None

    # Backend entities for retrieval
    data_backend = None
    vaecache = None
    embedcache = None
    accelerator = None
    bucket_manager = None
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
    def set_image_files(cls, raw_file_list):
        if cls.all_image_files is not None:
            cls.all_image_files.clear()
        else:
            cls.all_image_files = {}
        for subdirectory_list in raw_file_list:
            _, _, files = subdirectory_list
            for image in files:
                cls.all_image_files[image] = False
        cls._save_to_disk("all_image_files", cls.all_image_files)
        logger.debug(f"set_image_files found {len(cls.all_image_files)} images.")
        return cls.all_image_files

    @classmethod
    def get_image_files(cls):
        if not cls.all_image_files:
            cls.all_image_files = cls._load_from_disk("all_image_files")
        return cls.all_image_files

    @classmethod
    def set_vae_cache_files(cls, raw_file_list):
        if cls.all_vae_cache_files is not None:
            cls.all_vae_cache_files.clear()
        else:
            cls.all_vae_cache_files = {}
        for subdirectory_list in raw_file_list:
            _, _, files = subdirectory_list
            for image in files:
                cls.all_vae_cache_files[path.basename(image)] = False
        cls._save_to_disk("all_vae_cache_files", cls.all_vae_cache_files)
        logger.debug(
            f"set_vae_cache_files found {len(cls.all_vae_cache_files)} images."
        )

    @classmethod
    def get_vae_cache_files(cls):
        if not cls.all_vae_cache_files:
            cls.all_vae_cache_files = cls._load_from_disk("all_vae_cache_files")
        return cls.all_vae_cache_files

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
    def has_image_files_loaded(cls):
        return len(list(cls.all_image_files.keys())) > 0

    @classmethod
    def has_vae_cache_files_loaded(cls):
        return len(list(cls.all_vae_cache_files.keys())) > 0

    @classmethod
    def has_caption_files_loaded(cls):
        return len(list(cls.all_caption_files.keys())) > 0

    @classmethod
    def set_data_backend(cls, data_backend):
        cls.data_backend = data_backend

    @classmethod
    def get_data_backend(cls):
        return cls.data_backend

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
    def set_bucket_manager(cls, bucket_manager):
        cls.bucket_manager = bucket_manager

    @classmethod
    def get_bucket_manager(cls):
        return cls.bucket_manager

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
    def get_vaecache(cls):
        return cls.vaecache

    @classmethod
    def set_embedcache(cls, embedcache):
        cls.embedcache = embedcache

    @classmethod
    def get_embedcache(cls):
        return cls.embedcache
