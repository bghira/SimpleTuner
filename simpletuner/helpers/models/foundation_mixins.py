import inspect
import logging

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from simpletuner.helpers.data_backend.dataset_types import DatasetType

logger = logging.getLogger(__name__)


class PipelineSupportMixin:
    @classmethod
    def _iter_pipeline_classes(cls):
        pipelines = getattr(cls, "PIPELINE_CLASSES", {})
        if not isinstance(pipelines, dict):
            return []
        pipeline_classes = []
        for pipeline_cls in pipelines.values():
            if inspect.isclass(pipeline_cls):
                pipeline_classes.append(pipeline_cls)
        return pipeline_classes

    @staticmethod
    def _pipeline_has_lora_loader(pipeline_cls) -> bool:
        if not inspect.isclass(pipeline_cls):
            return False
        for base in inspect.getmro(pipeline_cls):
            if base.__name__.endswith("LoraLoaderMixin"):
                return True
        return False

    @classmethod
    def supports_lora(cls) -> bool:
        if cls.SUPPORTS_LORA is not None:
            return bool(cls.SUPPORTS_LORA)

        for pipeline_cls in cls._iter_pipeline_classes():
            if cls._pipeline_has_lora_loader(pipeline_cls):
                return True
        return False

    @classmethod
    def supports_controlnet(cls) -> bool:
        if cls.SUPPORTS_CONTROLNET is not None:
            return bool(cls.SUPPORTS_CONTROLNET)

        pipelines = getattr(cls, "PIPELINE_CLASSES", {})
        if not isinstance(pipelines, dict):
            return False

        from simpletuner.helpers.models.common import PipelineTypes

        for pipeline_type, pipeline_cls in pipelines.items():
            if pipeline_cls is None:
                continue
            if isinstance(pipeline_type, PipelineTypes):
                if pipeline_type in (PipelineTypes.CONTROLNET, PipelineTypes.CONTROL):
                    return True
            elif isinstance(pipeline_type, str) and pipeline_type.lower() in {"controlnet", "control"}:
                return True
        return False


class VaeLatentScalingMixin:
    def scale_vae_latents_for_cache(self, latents, vae):
        if vae is None or not hasattr(vae, "config") or latents is None:
            return latents
        shift_factor = getattr(vae.config, "shift_factor", None)
        scaling_factor = getattr(self, "AUTOENCODER_SCALING_FACTOR", getattr(vae.config, "scaling_factor", 1.0))
        if shift_factor is not None:
            return (latents - shift_factor) * scaling_factor
        if isinstance(latents, torch.Tensor) and hasattr(vae.config, "scaling_factor"):
            scaled = latents * scaling_factor
            logger.debug("Latents shape after scaling: %s", scaled.shape)
            return scaled
        return latents


class VideoToTensor:
    def __call__(self, video):
        """
        Converts a video (numpy array of shape (num_frames, height, width, channels))
        to a tensor of shape (num_frames, channels, height, width) by applying the
        standard ToTensor conversion to each frame.
        """
        if isinstance(video, np.ndarray):
            frames = []
            for frame in video:
                # Convert frame to PIL Image if not already.
                if not isinstance(frame, Image.Image):
                    frame = Image.fromarray(frame)
                frame_tensor = transforms.functional.to_tensor(frame)
                frames.append(frame_tensor)
            return torch.stack(frames)
        elif isinstance(video, list):
            frames = []
            for frame in video:
                if not isinstance(frame, Image.Image):
                    frame = Image.fromarray(frame)
                frames.append(transforms.functional.to_tensor(frame))
            return torch.stack(frames)
        else:
            raise TypeError("Input video must be a numpy array or a list of frames.")

    def __repr__(self):
        return self.__class__.__name__ + "()"


class VideoTransformMixin:
    @classmethod
    def adjust_video_frames(cls, num_frames: int) -> int:
        """
        Calculate nearest valid frame count at or below the given count.
        Default implementation returns the input unchanged.
        Subclasses override to enforce model-specific constraints.

        Args:
            num_frames: The desired number of frames

        Returns:
            Adjusted frame count (always >= 1)
        """
        return num_frames

    def get_transforms(self, dataset_type: str = "image"):
        if dataset_type == DatasetType.AUDIO.value or dataset_type == "audio":
            if self.uses_audio_latents():

                def _audio_transform(sample):
                    waveform = sample
                    if isinstance(sample, dict):
                        waveform = sample.get("waveform")
                    if waveform is None:
                        raise ValueError("Audio transform expected a waveform tensor in the sample payload.")
                    if isinstance(waveform, np.ndarray):
                        waveform = torch.from_numpy(waveform)
                    if not torch.is_tensor(waveform):
                        raise ValueError(f"Unsupported audio payload type: {type(waveform)}")
                    waveform = waveform.detach().clone()
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0)
                    return waveform

                return _audio_transform
        return transforms.Compose(
            [
                VideoToTensor() if dataset_type == "video" else transforms.ToTensor(),
            ]
        )

    def apply_i2v_augmentation(self, batch):
        pass

    def prepare_5d_inputs(self, tensor):
        """
        Example method to handle default 5D shape. The typical shape might be:
        (batch_size, frames, channels, height, width).

        You can reshape or permute as needed for the underlying model.
        """
        return tensor


class AudioTransformMixin:
    def supports_audio_inputs(self) -> bool:
        return True

    def uses_audio_latents(self) -> bool:
        return True

    def get_transforms(self, dataset_type: str = "image"):
        if dataset_type == DatasetType.AUDIO.value or dataset_type == "audio":

            def _audio_transform(sample):
                waveform = sample
                if isinstance(sample, dict):
                    waveform = sample.get("waveform")
                if waveform is None:
                    raise ValueError("Audio transform expected a waveform tensor in the sample payload.")
                if isinstance(waveform, np.ndarray):
                    waveform = torch.from_numpy(waveform)
                if not torch.is_tensor(waveform):
                    raise ValueError(f"Unsupported audio payload type: {type(waveform)}")
                waveform = waveform.detach().clone()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                return waveform

            return _audio_transform
        return super().get_transforms(dataset_type=dataset_type)

    @torch.no_grad()
    def encode_with_vae(self, vae, samples):
        """
        Music-focused autoencoders often return both latents and accompanying
        sequence lengths. Wrap both into a dict so downstream caches retain the metadata.
        """
        if samples is None:
            raise ValueError("Audio VAE received no samples to encode.")
        audio = samples
        if not torch.is_tensor(audio):
            raise ValueError(f"Audio encoder expected a Tensor input, received {type(audio)}.")
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        audio = audio.to(device=self.accelerator.device, dtype=torch.float32)
        result = vae.encode(audio)
        latent_lengths = None
        latents = result
        if isinstance(result, tuple):
            latents, *extras = result
            latent_lengths = extras[0] if extras else None
        elif isinstance(result, dict):
            latents = result.get("latents")
            latent_lengths = result.get("latent_lengths")
        payload = {"latents": latents}
        if latent_lengths is not None:
            payload["latent_lengths"] = latent_lengths
        return payload
