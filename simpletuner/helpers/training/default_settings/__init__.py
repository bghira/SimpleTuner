CURRENT_VERSION = 2

# hash_filenames was removed in version 2 as it is now always enabled
LATEST_DEFAULTS = {1: {}, 2: {}}


def default(setting: str, current_version: int = None, default_value=None):
    if current_version <= 0 or current_version is None:
        current_version = CURRENT_VERSION
    if current_version in LATEST_DEFAULTS:
        return LATEST_DEFAULTS[current_version].get(setting, default_value)
    return default_value


def latest_config_version():
    return CURRENT_VERSION
