from multiprocessing import Manager
from os import environ

manager = Manager()
all_image_files = manager.dict()
all_vae_cache_files = manager.dict()
all_caption_files = manager.list()

import logging

logger = logging.getLogger("StateTracker")
logger.setLevel(environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class StateTracker:
    # Class variables
    has_training_started = False
    calculate_luminance = False

    # Backend entities for retrieval
    data_backend = None
    vaecache = None
    embedcache = None
    accelerator = None
    vae = None
    vae_dtype = None
    args = None

    @classmethod
    def start_training(cls):
        cls.has_training_started = True

    @classmethod
    def status_training(cls):
        return cls.has_training_started

    @classmethod
    def enable_luminance(cls):
        cls.calculate_luminance = True

    @classmethod
    def tracking_luminance(cls):
        return cls.calculate_luminance

    @classmethod
    def set_image_files(cls, raw_file_list):
        all_image_files.clear()
        for subdirectory_list in raw_file_list:
            _, _, files = subdirectory_list
            for image in files:
                all_image_files[image] = False
        logger.debug(f'set_image_files found {len(all_image_files)} images.')
        return all_image_files

    @classmethod
    def get_image_files(cls):
        return all_image_files

    @classmethod
    def has_image_files_loaded(cls):
        return len(list(all_image_files.keys())) > 0

    @classmethod
    def set_vae_cache_files(cls, raw_file_list):
        all_vae_cache_files.clear()
        for subdirectory_list in raw_file_list:
            _, _, files = subdirectory_list
            for image in files:
                all_vae_cache_files[image] = False
        logger.debug(f'set_vae_cache_files found {len(all_vae_cache_files)} images.')
        return all_vae_cache_files

    @classmethod
    def get_vae_cache_files(cls):
        return all_vae_cache_files

    @classmethod
    def has_vae_cache_files_loaded(cls):
        return len(list(all_vae_cache_files.keys())) > 0

    @classmethod
    def set_caption_files(cls, caption_files):
        all_caption_files[:] = caption_files
        return all_caption_files

    @classmethod
    def get_caption_files(cls):
        return cls.all_caption_files

    @classmethod
    def has_caption_files_loaded(cls):
        return len(list(all_caption_files.keys())) > 0

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
