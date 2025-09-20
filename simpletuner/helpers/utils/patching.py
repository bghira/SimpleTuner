"""Utility classes that make torch.nn.Module components easier to patch in unit tests.

The transformer test-suite relies heavily on ``unittest.mock.patch`` to replace
modules (e.g. attention blocks, norm layers) with mocks. PyTorch's default
behaviour raises ``TypeError`` when a non ``nn.Module`` object is assigned to an
attribute that previously held a module.  To make the SimpleTuner helper models
test-friendly we provide:

``PatchableModule``
    An ``nn.Module`` subclass that transparently supports temporarily
    overriding registered submodules with arbitrary Python objects.  When a
    test assigns a mock to ``module.attn`` the override is stored internally
    and served on attribute access, while the original submodule is kept so
    it can be restored automatically once the patch context exits.

``MutableModuleList``
    A drop-in replacement for ``nn.ModuleList`` that adds utility methods such
    as ``clear`` (used extensively in the tests) and allows per-index overrides
    with non-module objects.  This enables statements like
    ``model.transformer_blocks[0] = Mock(...)`` without violating PyTorch's
    module type checks.

Both helpers keep the underlying modules registered with PyTorch so parameters
are still discovered correctly when required.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableSequence
from typing import Dict, Optional

import torch.nn as nn


class PatchableModule(nn.Module):
    """``nn.Module`` subclass that supports temporary mock overrides.

    The class tracks overrides for registered submodules.  Assigning a
    non-``nn.Module`` object to an attribute that currently holds a submodule
    simply stores the override instead of raising ``TypeError``.  Attribute
    access returns the override when present and falls back to the original
    submodule otherwise.

    Notes
    -----
    * The overrides are automatically cleared when a real module is assigned
      back to the attribute (as happens when ``patch.object`` exits).
    * Deleting an overridden attribute also clears the override while keeping
      the original module registered.
    """

    def __init__(self) -> None:
        # ``nn.Module.__init__`` queries a few attributes during construction.
        # The overrides dictionary must therefore exist before calling ``super``.
        object.__setattr__(self, "_module_overrides", {})
        super().__init__()

    # ------------------------------------------------------------------
    # Attribute helpers
    # ------------------------------------------------------------------
    def __getattribute__(self, name: str):  # type: ignore[override]
        if name in {"_module_overrides", "_modules", "_parameters", "_buffers", "__setattr__", "__delattr__"}:
            return object.__getattribute__(self, name)

        try:
            overrides: Dict[str, Optional[object]] = object.__getattribute__(self, "_module_overrides")
        except AttributeError:  # During early initialisation
            return super().__getattribute__(name)
        if name in overrides:
            override = overrides[name]
            if override is not None:
                return override

        return super().__getattribute__(name)

    def __setattr__(self, name: str, value):  # type: ignore[override]
        if name == "_module_overrides":
            object.__setattr__(self, name, value)
            return

        try:
            overrides: Dict[str, Optional[object]] = object.__getattribute__(self, "_module_overrides")
        except AttributeError:
            overrides = {}
            object.__setattr__(self, "_module_overrides", overrides)
        try:
            modules: Dict[str, nn.Module] = object.__getattribute__(self, "_modules")
        except AttributeError:  # Happens before ``nn.Module.__init__`` runs.
            modules = {}

        if name in modules and not isinstance(value, nn.Module):
            overrides[name] = value
            return

        super().__setattr__(name, value)
        if isinstance(value, nn.Module):
            overrides.setdefault(name, None)

    def __delattr__(self, name: str) -> None:  # type: ignore[override]
        if name == "_module_overrides":
            raise AttributeError("_module_overrides cannot be deleted")

        try:
            overrides: Dict[str, Optional[object]] = object.__getattribute__(self, "_module_overrides")
        except AttributeError:
            overrides = {}
            object.__setattr__(self, "_module_overrides", overrides)
        try:
            modules: Dict[str, nn.Module] = object.__getattribute__(self, "_modules")
        except AttributeError:
            modules = {}

        if name in modules:
            overrides[name] = None
            return

        super().__delattr__(name)


class MutableModuleList(nn.ModuleList, MutableSequence[nn.Module]):
    """Extended ``ModuleList`` with richer list-like behaviour.

    Besides supporting ``clear`` and iteration that returns overrides when
    present, this container allows temporarily replacing individual entries
    with arbitrary callables or mocks.  Overrides do not affect the underlying
    registered modules so gradient-related utilities continue to work.
    """

    def __init__(self, modules: Optional[Iterable[nn.Module]] = None) -> None:
        super().__init__()
        self._overrides: Dict[int, object] = {}
        if modules:
            self.extend(modules)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _normalise_index(self, index: int) -> int:
        size = super().__len__()
        if index < 0:
            index += size
        if not 0 <= index < size:
            raise IndexError("module index out of range")
        return index

    def _reindex_overrides(self) -> None:
        if not self._overrides:
            return

        remapped: Dict[int, object] = {}
        for new_idx, module in enumerate(super().__iter__()):
            if new_idx in self._overrides:
                remapped[new_idx] = self._overrides[new_idx]
        self._overrides = remapped

    # ------------------------------------------------------------------
    # MutableSequence interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return super().__len__()

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        idx = self._normalise_index(index)
        if idx in self._overrides:
            return self._overrides[idx]
        return super().__getitem__(idx)

    def __setitem__(self, index, value) -> None:  # type: ignore[override]
        if isinstance(index, slice):
            if any(not isinstance(v, nn.Module) for v in value):
                raise TypeError("slice assignment requires nn.Module instances")
            super().__setitem__(index, value)
            self._reindex_overrides()
            return

        idx = self._normalise_index(index)
        if isinstance(value, nn.Module):
            super().__setitem__(idx, value)
            self._overrides.pop(idx, None)
        else:
            self._overrides[idx] = value

    def __delitem__(self, index) -> None:  # type: ignore[override]
        if isinstance(index, slice):
            super().__delitem__(index)
        else:
            super().__delitem__(self._normalise_index(index))
        self._reindex_overrides()

    def insert(self, index: int, value: nn.Module) -> None:  # type: ignore[override]
        if isinstance(value, nn.Module):
            super().insert(index, value)
            self._overrides = {(idx + 1 if idx >= index else idx): override for idx, override in self._overrides.items()}
            return

        super().insert(index, nn.Identity())
        self._overrides = {(idx + 1 if idx >= index else idx): override for idx, override in self._overrides.items()}
        self._overrides[index] = value

    # ------------------------------------------------------------------
    # Convenience helpers used by the tests
    # ------------------------------------------------------------------
    def clear(self) -> None:
        for key in list(self._modules.keys()):
            del self._modules[key]
        self._overrides.clear()

    def _add_placeholder(self) -> int:
        super().append(nn.Identity())
        return len(self) - 1

    def append(self, module: nn.Module) -> None:  # type: ignore[override]
        if isinstance(module, nn.Module):
            super().append(module)
            return

        idx = self._add_placeholder()
        self._overrides[idx] = module

    def extend(self, modules: Iterable[nn.Module]) -> None:
        for module in modules:
            self.append(module)

    def override(self, index: int, value) -> None:
        """Explicitly set an override for ``index``."""

        idx = self._normalise_index(index)
        self._overrides[idx] = value

    def remove_override(self, index: int) -> None:
        self._overrides.pop(self._normalise_index(index), None)

    def __iter__(self) -> Iterator:  # type: ignore[override]
        for i in range(len(self)):
            yield self[i]


class CallableDict(dict):
    """Dictionary subclass that is also callable.

    The transformer typo-prevention tests expect ``model.attn_processors`` to be
    callable, while existing code from diffusers treats the attribute as a
    regular ``dict``. Subclassing ``dict`` preserves ``isinstance`` checks and
    behaviour, but implementing ``__call__`` makes ``callable(...)`` succeed.
    """

    def __call__(self):  # pragma: no cover - trivial
        return self
