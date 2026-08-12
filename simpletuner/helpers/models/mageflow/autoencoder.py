import os
from types import SimpleNamespace
from typing import Optional

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from huggingface_hub import snapshot_download

from simpletuner.helpers.models.mageflow.vendor.models.modules.mage_vae import MageVAE as _VendorMageVAE


def _resolve_repo_dir(repo_id_or_path: str, *, revision: Optional[str] = None, local_files_only: bool = False) -> str:
    if os.path.isdir(repo_id_or_path):
        return os.path.abspath(repo_id_or_path)
    return snapshot_download(repo_id=repo_id_or_path, revision=revision, local_files_only=local_files_only)


class MageVAE(_VendorMageVAE, ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        ckpt_path: str,
        sample_posterior: bool = False,
        latent_channels: int = 128,
        downsample_factor: int = 16,
    ):
        del latent_channels, downsample_factor
        _VendorMageVAE.__init__(self, ckpt_path=ckpt_path, sample_posterior=sample_posterior)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: Optional[str] = "vae", **kwargs):
        revision = kwargs.pop("revision", None)
        local_files_only = bool(kwargs.pop("local_files_only", False))
        torch_dtype = kwargs.pop("torch_dtype", None)
        repo_dir = _resolve_repo_dir(
            pretrained_model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
        )
        vae_dir = os.path.join(repo_dir, subfolder) if subfolder else repo_dir
        checkpoint_path = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
        sample_posterior = bool(kwargs.pop("sample_posterior", False))
        vae = cls(ckpt_path=checkpoint_path, sample_posterior=sample_posterior)
        if torch_dtype is not None:
            vae.to(dtype=torch_dtype)
        return vae

    def decode_to_tensor(self, z: torch.Tensor) -> torch.Tensor:
        return _VendorMageVAE.decode(self, z)

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        sample = self.decode_to_tensor(z)
        if return_dict:
            return SimpleNamespace(sample=sample)
        return (sample,)
