import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type, Tuple
from PIL import Image, ImageFilter
import numpy as np
import torch
import io
from torchvision import transforms
import random

logger = logging.getLogger("SampleGenerator")


class SampleGenerator(ABC):
    """Base class for sample generators that create conditioning data from source images."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conditioning_type = config.get("type", "unknown")

    @abstractmethod
    def transform_batch(
        self,
        images: List[Image.Image],
        source_paths: List[str],
        metadata_list: List[Dict],
        accelerator,
    ) -> List[Image.Image]:
        """Transform a batch of images.

        Args:
            images: List of PIL Images
            source_paths: List of source file paths
            metadata_list: List of metadata dicts for each image
            accelerator: Accelerator instance

        Returns:
            List of transformed PIL Images
        """
        pass

    @staticmethod
    def from_backend(backend_config: Dict[str, Any]) -> "SampleGenerator":
        """Factory method to create appropriate SampleGenerator from backend config.

        Args:
            backend_config: Backend configuration dict containing conditioning config

        Returns:
            SampleGenerator instance
        """
        conditioning_config = backend_config.get("conditioning_config", {})
        conditioning_type = conditioning_config.get("type", None)

        if conditioning_type is None:
            raise ValueError(
                "Backend config must have conditioning_config with 'type' field."
                "\n Current config: {}".format(backend_config)
            )

        # Get the appropriate generator class
        generator_class = GENERATOR_REGISTRY.get(conditioning_type, None)
        if generator_class is None:
            raise ValueError(
                f"Unknown conditioning type: {conditioning_type}. "
                f"Available types: {list(GENERATOR_REGISTRY.keys())}"
            )

        # Create and return instance
        return generator_class(conditioning_config)

    def image_to_bytes(self, img: Image.Image, format: Optional[str] = None) -> bytes:
        """Convert PIL Image to bytes."""
        if format is None:
            format = img.format if img.format else "PNG"
        with io.BytesIO() as buffer:
            img.save(buffer, format=format)
            return buffer.getvalue()

    def get_target_format(self, img_path: str) -> str:
        """Determine the target format for saving images based on file extension."""
        if img_path.lower().endswith((".jpg", ".jpeg")):
            return "JPEG"
        elif img_path.lower().endswith(".png"):
            return "PNG"
        elif img_path.lower().endswith(".webp"):
            return "WEBP"
        elif img_path.lower().endswith((".bmp", ".tiff", ".tif")):
            return "BMP"
        elif img_path.lower().endswith(".jxl"):
            return "JPEG XL"
        else:
            raise ValueError(
                f"Unsupported image format for path: {img_path}. Supported formats are JPG, PNG, WEBP, BMP, TIFF, JPEG XL."
            )


class SuperResolutionSampleGenerator(SampleGenerator):
    """
    Creates low-resolution blurry versions of images for super-resolution training.
    Maintains original dimensions but reduces quality through blur and optional noise.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Configuration for degradation
        self.blur_radius = config.get("blur_radius", 2.0)
        self.blur_type = config.get("blur_type", "gaussian")  # gaussian, box, or both
        self.add_noise = config.get("add_noise", True)
        self.noise_level = config.get(
            "noise_level", 0.02
        )  # Standard deviation for gaussian noise
        self.jpeg_quality = config.get(
            "jpeg_quality", None
        )  # Optional JPEG compression
        self.downscale_factor = config.get(
            "downscale_factor", None
        )  # Optional temp downscale

    def transform_batch(
        self,
        images: List[Image.Image],
        source_paths: List[str],
        metadata_list: List[Dict],
        accelerator,
    ) -> List[Image.Image]:
        """Create low-quality versions of input images."""

        transformed_images = []

        for img, path, metadata in zip(images, source_paths, metadata_list):
            try:
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                degraded_img = self._degrade_image(img)
                transformed_images.append(degraded_img)

            except Exception as e:
                logger.error(f"Error transforming image {path}: {e}")
                raise

        return transformed_images

    def _degrade_image(self, img: Image.Image) -> Image.Image:
        """Apply degradation to create low-quality version."""

        result = img.copy()
        original_size = result.size

        # Optional: Temporary downscale and upscale to lose detail
        if self.downscale_factor is not None and self.downscale_factor > 1:
            small_size = (
                int(original_size[0] / self.downscale_factor),
                int(original_size[1] / self.downscale_factor),
            )
            result = result.resize(small_size, Image.Resampling.BILINEAR)
            result = result.resize(original_size, Image.Resampling.BILINEAR)

        # Apply blur
        if self.blur_type == "gaussian":
            result = result.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        elif self.blur_type == "box":
            result = result.filter(ImageFilter.BoxBlur(radius=self.blur_radius))
        elif self.blur_type == "both":
            # Apply both for stronger degradation
            result = result.filter(ImageFilter.BoxBlur(radius=self.blur_radius * 0.7))
            result = result.filter(
                ImageFilter.GaussianBlur(radius=self.blur_radius * 0.7)
            )

        # Add noise if requested
        if self.add_noise and self.noise_level > 0:
            result = self._add_gaussian_noise(result, self.noise_level)

        # Optional JPEG compression artifacts
        if self.jpeg_quality is not None:
            import io

            buffer = io.BytesIO()
            result.save(buffer, format="JPEG", quality=self.jpeg_quality)
            buffer.seek(0)
            result = Image.open(buffer)
            # Convert back to RGB after JPEG
            result = result.convert("RGB")

        return result

    def _add_gaussian_noise(self, img: Image.Image, noise_level: float) -> Image.Image:
        """Add Gaussian noise to image."""
        # Convert to numpy
        img_array = np.array(img).astype(np.float32) / 255.0

        # Generate noise
        noise = np.random.normal(0, noise_level, img_array.shape)

        # Add noise and clip
        noisy_array = img_array + noise
        noisy_array = np.clip(noisy_array, 0, 1)

        # Convert back to PIL Image
        noisy_array = (noisy_array * 255).astype(np.uint8)
        return Image.fromarray(noisy_array)


class JPEGArtifactsSampleGenerator(SampleGenerator):
    """
    Creates images with JPEG compression artifacts for artifact removal training.
    Supports various compression strategies including multiple compression rounds,
    quality variation, and different subsampling modes.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Quality settings
        self.quality_mode = config.get("quality_mode", "fixed")  # fixed, random, range
        self.quality = config.get("quality", 25)  # For fixed mode
        self.quality_range = config.get("quality_range", (10, 50))  # For range mode
        self.quality_list = config.get(
            "quality_list", [10, 20, 30, 40]
        )  # For random mode

        # Compression rounds (recompression artifacts)
        self.compression_rounds = config.get("compression_rounds", 1)
        self.round_quality_decay = config.get(
            "round_quality_decay", 0.9
        )  # Quality multiplier per round

        # Advanced JPEG settings
        self.subsampling = config.get(
            "subsampling", None
        )  # None, '4:4:4', '4:2:2', '4:2:0'
        self.optimize = config.get("optimize", False)
        self.progressive = config.get("progressive", False)

        # Pre/post processing
        self.pre_blur = config.get("pre_blur", False)  # Blur before compression
        self.pre_blur_radius = config.get("pre_blur_radius", 0.5)
        self.add_noise = config.get("add_noise", False)
        self.noise_level = config.get("noise_level", 0.01)

        # Block artifact enhancement (optional)
        self.enhance_blocks = config.get("enhance_blocks", False)
        self.block_noise_level = config.get("block_noise_level", 0.02)

    def transform_batch(
        self,
        images: List[Image.Image],
        source_paths: List[str],
        metadata_list: List[Dict],
        accelerator,
    ) -> List[Image.Image]:
        """Create JPEG artifact versions of input images."""

        transformed_images = []

        for img, path, metadata in zip(images, source_paths, metadata_list):
            try:
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                artifact_img = self._apply_jpeg_artifacts(img)
                transformed_images.append(artifact_img)

            except Exception as e:
                logger.error(f"Error applying JPEG artifacts to {path}: {e}")
                raise

        return transformed_images

    def _apply_jpeg_artifacts(self, img: Image.Image) -> Image.Image:
        """Apply JPEG compression artifacts to image."""

        result = img.copy()

        # Pre-blur if requested (simulates camera/scanner blur)
        if self.pre_blur:
            result = result.filter(
                ImageFilter.GaussianBlur(radius=self.pre_blur_radius)
            )

        # Pre-noise if requested
        if self.add_noise and self.noise_level > 0:
            result = self._add_gaussian_noise(result, self.noise_level)

        # Apply compression rounds
        for round_idx in range(self.compression_rounds):
            quality = self._get_quality_for_round(round_idx)
            result = self._compress_jpeg(result, quality)

        # Enhance block artifacts if requested
        if self.enhance_blocks:
            result = self._enhance_block_artifacts(result)

        return result

    def _get_quality_for_round(self, round_idx: int) -> int:
        """Get JPEG quality for specific compression round."""

        base_quality = self._get_base_quality()

        # Apply decay for multiple rounds
        if round_idx > 0:
            quality = int(base_quality * (self.round_quality_decay**round_idx))
            quality = max(1, min(95, quality))  # Clamp to valid range
        else:
            quality = base_quality

        return quality

    def _get_base_quality(self) -> int:
        """Get base JPEG quality based on mode."""

        if self.quality_mode == "fixed":
            return self.quality
        elif self.quality_mode == "range":
            return random.randint(self.quality_range[0], self.quality_range[1])
        elif self.quality_mode == "random":
            return random.choice(self.quality_list)
        else:
            return self.quality

    def _compress_jpeg(self, img: Image.Image, quality: int) -> Image.Image:
        """Apply JPEG compression with specified quality."""

        buffer = io.BytesIO()

        # Prepare save kwargs
        save_kwargs = {
            "format": "JPEG",
            "quality": quality,
            "optimize": self.optimize,
            "progressive": self.progressive,
        }

        # Add subsampling if specified
        if self.subsampling:
            if self.subsampling == "4:4:4":
                save_kwargs["subsampling"] = 0
            elif self.subsampling == "4:2:2":
                save_kwargs["subsampling"] = 1
            elif self.subsampling == "4:2:0":
                save_kwargs["subsampling"] = 2

        # Save and reload
        img.save(buffer, **save_kwargs)
        buffer.seek(0)
        compressed = Image.open(buffer)

        # Ensure RGB mode
        return compressed.convert("RGB")

    def _add_gaussian_noise(self, img: Image.Image, noise_level: float) -> Image.Image:
        """Add Gaussian noise to image."""
        img_array = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, noise_level, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy_array * 255).astype(np.uint8))

    def _enhance_block_artifacts(self, img: Image.Image) -> Image.Image:
        """Enhance JPEG block artifacts by adding noise to block boundaries."""

        img_array = np.array(img).astype(np.float32) / 255.0
        h, w, c = img_array.shape

        # Create block mask (8x8 JPEG blocks)
        block_mask = np.zeros((h, w), dtype=np.float32)
        for i in range(0, h, 8):
            block_mask[i, :] = 1.0
        for j in range(0, w, 8):
            block_mask[:, j] = 1.0

        # Add noise to block boundaries
        block_noise = np.random.normal(0, self.block_noise_level, (h, w, c))
        block_noise *= block_mask[:, :, np.newaxis]

        # Apply noise
        enhanced = img_array + block_noise
        enhanced = np.clip(enhanced, 0, 1)

        return Image.fromarray((enhanced * 255).astype(np.uint8))


class CannyEdgeSampleGenerator(SampleGenerator):
    """Creates Canny edge detection images for ControlNet training."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.low_threshold = config.get("low_threshold", 100)
        self.high_threshold = config.get("high_threshold", 200)

    def transform_batch(
        self,
        images: List[Image.Image],
        source_paths: List[str],
        metadata_list: List[Dict],
        accelerator,
    ) -> List[Image.Image]:
        """Create Canny edge images."""
        import cv2

        transformed_images = []

        for img, path, metadata in zip(images, source_paths, metadata_list):
            try:
                # Convert to grayscale numpy array
                if img.mode != "RGB":
                    img = img.convert("RGB")

                img_array = np.array(img)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                # Apply Canny edge detection
                edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)

                # Convert to 3-channel for consistency
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

                # Convert back to PIL
                result = Image.fromarray(edges_rgb)
                transformed_images.append(result)

            except Exception as e:
                logger.error(f"Error creating Canny edges for {path}: {e}")
                # Return black image on error
                transformed_images.append(Image.new("RGB", img.size, (0, 0, 0)))

        return transformed_images


class DepthMapSampleGenerator(SampleGenerator):
    """Creates depth map images for ControlNet training."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = config.get("model_type", "DPT")  # DPT or MiDaS
        self._model = None
        self._transform = None

    def _load_model(self, device):
        """Lazy load the depth estimation model."""
        if self._model is None:
            if self.model_type == "DPT":
                from transformers import DPTForDepthEstimation, DPTImageProcessor

                self._model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
                self._transform = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            else:
                # MiDaS implementation
                import torchvision.transforms as transforms
                from midas.dpt_depth import DPTDepthModel

                self._model = DPTDepthModel(path="weights/dpt_large-midas-2f21e586.pt")
                self._transform = transforms.Compose(
                    [
                        transforms.Resize((384, 384)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

            self._model = self._model.to(device)
            self._model.eval()

    def transform_batch(
        self,
        images: List[Image.Image],
        source_paths: List[str],
        metadata_list: List[Dict],
        accelerator,
    ) -> List[Image.Image]:
        """Create depth map images."""

        device = (
            accelerator.device
            if accelerator
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._load_model(device)

        transformed_images = []

        with torch.no_grad():
            for img, path, metadata in zip(images, source_paths, metadata_list):
                try:
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    original_size = img.size

                    if self.model_type == "DPT":
                        # DPT processing
                        inputs = self._transform(images=img, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = self._model(**inputs)
                        depth = outputs.predicted_depth
                    else:
                        # MiDaS processing
                        input_tensor = self._transform(img).unsqueeze(0).to(device)
                        depth = self._model(input_tensor)

                    # Normalize depth map
                    depth = depth.squeeze().cpu().numpy()
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    depth = (depth * 255).astype(np.uint8)

                    # Convert to PIL and resize back to original
                    depth_img = Image.fromarray(depth)
                    depth_img = depth_img.resize(
                        original_size, Image.Resampling.BILINEAR
                    )

                    # Convert to RGB
                    depth_rgb = depth_img.convert("RGB")
                    transformed_images.append(depth_rgb)

                except Exception as e:
                    logger.error(f"Error creating depth map for {path}: {e}")
                    # Return black image on error
                    transformed_images.append(Image.new("RGB", img.size, (0, 0, 0)))

        return transformed_images


# Registry of available generators
GENERATOR_REGISTRY: Dict[str, Type[SampleGenerator]] = {
    "superresolution": SuperResolutionSampleGenerator,
    "super_resolution": SuperResolutionSampleGenerator,  # Alias
    "lowres": SuperResolutionSampleGenerator,  # Alias
    "jpeg_artifacts": JPEGArtifactsSampleGenerator,
    "jpeg": JPEGArtifactsSampleGenerator,  # Alias
    "compression": JPEGArtifactsSampleGenerator,  # Alias
    "canny": CannyEdgeSampleGenerator,
    "edges": CannyEdgeSampleGenerator,  # Alias
    "depth": DepthMapSampleGenerator,
    "depth_map": DepthMapSampleGenerator,  # Alias
    "depth_midas": DepthMapSampleGenerator,  # Alias
}


def register_generator(name: str, generator_class: Type[SampleGenerator]):
    """Register a new generator type.

    Args:
        name: String identifier for the generator
        generator_class: Class that inherits from SampleGenerator
    """
    if not issubclass(generator_class, SampleGenerator):
        raise ValueError(f"{generator_class} must inherit from SampleGenerator")

    GENERATOR_REGISTRY[name] = generator_class


def list_available_generators() -> List[str]:
    """Get list of available generator types."""
    return list(GENERATOR_REGISTRY.keys())
