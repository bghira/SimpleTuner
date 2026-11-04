"""Private pinned FA4 helpers and independent compilation caches."""

import hashlib
import importlib
from pathlib import Path
from types import FunctionType, SimpleNamespace

_EXPECTED_SOURCES = {
    "flash_bwd_postprocess": "1a8bf9f93c9bc755ae443cedde2225942822a428eccc9b8c9b8e0c1a5be637a7",
    "interface": "7890aa821fe751fc2e088a2b7a2a2c3c4fdaa3832ab2a4e13c6b143f566ba496",
    "softmax": "aaddf3030dba7b9809c4fefae3caab22d0f22b9ecf5050601d6646aefb611a49",
    "flash_fwd": "be6e88c7d9f122aa5ab29ff5946f6270af0b580e4403351b869f1330b71ca53c",
    "flash_fwd_sm90": "c6da4474257654e9889c4ed33e1e0f9afa841c7ce2c1fe537ba437b1ed93829a",
    "flash_bwd_sm90": "0414a9f5d3a93f4c79fd6093b7d7c83bce1f20dd343804a48317a05d763d5aff",
    "flash_bwd_preprocess": "f31922a9ecb817923038db3addcd83a9486a8bfba2b0d74be6ee23f42256e704",
    "tile_scheduler": "3f3840cce88f02e7a518f93160fc8d5ebf32c2a9e61d71e33953fc61ca4955d0",
    "config": "624855aca54f0bd455914ac1caa30855257421889c6b0dfb2d26ec843da55917",
    "utils": "7344b808aa4eb6ebc5c48ab821485e32feea1397a14a540433191e08895d9c33",
    "pipeline": "16a276487d7d0cb2c90a2f451ad482939b9faaaaa1453e43a1fcc7b73a187337",
    "mask": "7ec7481d186ba9408a44d89e8827e4717add6baf259d843e9b1f26949b600fc4",
}


def interface_matches():
    for name, expected in _EXPECTED_SOURCES.items():
        module = importlib.import_module("flash_attn.cute." + name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            return False
    return True


def make_private_api(**overrides):
    from flash_attn.cute import interface

    if not interface_matches():
        raise ValueError("The optional Hopper schedule requires the pinned FA4 sources.")
    scope = dict(vars(interface))
    for name, original in vars(interface).items():
        if not isinstance(original, FunctionType) or original.__module__ != interface.__name__:
            continue
        clone = FunctionType(original.__code__, scope, original.__name__, original.__defaults__, original.__closure__)
        clone.__kwdefaults__ = original.__kwdefaults__
        clone.__annotations__ = dict(original.__annotations__)
        clone.__dict__.update(vars(original))
        if hasattr(original, "compile_cache"):
            clone.compile_cache = {}
        scope[name] = clone
    scope.update(overrides)
    return SimpleNamespace(**scope)
