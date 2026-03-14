# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/pipelines/anima/options.py
# Adapted for SimpleTuner local imports.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnimaComponents:
    """Sources for an Anima single-file load.

    All auxiliary-component sources (text encoder, VAE, tokenizers) are
    resolved inside ``loading.py`` using the hardcoded Anima defaults.
    Only the transformer checkpoint path is caller-supplied.
    """

    model_path: str


@dataclass(frozen=True)
class AnimaLoaderOptions:
    local_files_only: bool
    cache_dir: str | None = None
    force_download: bool = False
    token: str | bool | None = None
    revision: str | None = None
    proxies: dict[str, str] | None = None


@dataclass(frozen=True)
class AnimaRuntimeOptions:
    device: str = "auto"
    dtype: str = "auto"
    text_encoder_dtype: str = "auto"
