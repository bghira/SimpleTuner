import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from PIL import Image, ImageFilter
import numpy as np
import torch
import io
from torchvision import transforms

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
