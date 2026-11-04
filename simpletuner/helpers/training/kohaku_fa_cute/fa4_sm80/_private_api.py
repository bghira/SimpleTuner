"""Private pinned FA4 helpers and independent compilation caches."""

import hashlib
import importlib
from pathlib import Path
from types import FunctionType, SimpleNamespace

_EXPECTED_SOURCES = {
    "interface": "7890aa821fe751fc2e088a2b7a2a2c3c4fdaa3832ab2a4e13c6b143f566ba496",
    "softmax": "aaddf3030dba7b9809c4fefae3caab22d0f22b9ecf5050601d6646aefb611a49",
    "flash_fwd": "be6e88c7d9f122aa5ab29ff5946f6270af0b580e4403351b869f1330b71ca53c",
    "flash_bwd": "80cb6bfb436160d73b7214529f94d28b7333c98dff75a53d20552eb33645fea2",
    "flash_bwd_preprocess": "f31922a9ecb817923038db3addcd83a9486a8bfba2b0d74be6ee23f42256e704",
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
        raise ValueError("The optional SM80 schedule requires the pinned FA4 sources.")
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
