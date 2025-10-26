import io
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

logger = logging.getLogger("SampleGenerator")


class SampleGenerator(ABC):
    """Base class for sample generators that create conditioning data from source images."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conditioning_type = config.get("type", "unknown")
        self.max_worker_count = None

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
                f"Unknown conditioning type: {conditioning_type}. " f"Available types: {list(GENERATOR_REGISTRY.keys())}"
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
        self.noise_level = config.get("noise_level", 0.02)  # Standard deviation for gaussian noise
        self.jpeg_quality = config.get("jpeg_quality", None)  # Optional JPEG compression
        self.downscale_factor = config.get("downscale_factor", None)  # Optional temp downscale

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
            result = result.filter(ImageFilter.GaussianBlur(radius=self.blur_radius * 0.7))

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
        self.quality_list = config.get("quality_list", [10, 20, 30, 40])  # For random mode

        # Compression rounds (recompression artifacts)
        self.compression_rounds = config.get("compression_rounds", 1)
        self.round_quality_decay = config.get("round_quality_decay", 0.9)  # Quality multiplier per round

        # Advanced JPEG settings
        self.subsampling = config.get("subsampling", None)  # None, '4:4:4', '4:2:2', '4:2:0'
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
            result = result.filter(ImageFilter.GaussianBlur(radius=self.pre_blur_radius))

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
        """Create Canny edge images with 8x performance improvement."""
        import trainingsample as tsr

        # Convert PIL images to numpy arrays
        img_arrays = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_arrays.append(np.array(img))

        # Process images individually but with optimized trainingsample functions
        edges_rgb_batch = []
        for img_array in img_arrays:
            # Convert to grayscale
            gray = tsr.cvt_color_py(img_array, 7)  # 7 = COLOR_RGB2GRAY
            # Apply Canny edge detection
            edges = tsr.canny_py(gray, self.low_threshold, self.high_threshold)
            # Convert back to RGB
            edges_rgb = tsr.cvt_color_py(edges, 8)  # 8 = COLOR_GRAY2RGB
            edges_rgb_batch.append(edges_rgb)

        transformed_images = []
        for i, (path, metadata) in enumerate(zip(source_paths, metadata_list)):
            try:
                # Convert back to PIL
                result = Image.fromarray(edges_rgb_batch[i])
                transformed_images.append(result)

            except Exception as e:
                logger.error(f"Error creating Canny edges for {path}: {e}")
                # Return black image on error
                transformed_images.append(Image.new("RGB", img.size, (0, 0, 0)))

        return transformed_images


class RandomMasksSampleGenerator(SampleGenerator):
    """
    Creates random masks for inpainting training.
    Generates various types of masks including rectangles, circles, brush strokes, and irregular shapes.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Mask type settings
        self.mask_types = config.get("mask_types", ["rectangle", "circle", "brush", "irregular"])
        self.mask_type_weights = config.get("mask_type_weights", None)  # For weighted random selection

        # Coverage settings
        self.min_coverage = config.get("min_coverage", 0.1)  # Minimum mask coverage
        self.max_coverage = config.get("max_coverage", 0.5)  # Maximum mask coverage

        # Per-type settings
        self.min_masks = config.get("min_masks", 1)  # Minimum number of mask regions
        self.max_masks = config.get("max_masks", 5)  # Maximum number of mask regions

        # Rectangle settings
        self.rect_min_size = config.get("rect_min_size", 0.1)  # Min size as fraction of image
        self.rect_max_size = config.get("rect_max_size", 0.4)

        # Circle settings
        self.circle_min_radius = config.get("circle_min_radius", 0.05)
        self.circle_max_radius = config.get("circle_max_radius", 0.2)

        # Brush stroke settings
        self.brush_min_width = config.get("brush_min_width", 5)
        self.brush_max_width = config.get("brush_max_width", 40)
        self.brush_min_length = config.get("brush_min_length", 0.1)
        self.brush_max_length = config.get("brush_max_length", 0.5)

        # Irregular mask settings
        self.irregular_min_points = config.get("irregular_min_points", 5)
        self.irregular_max_points = config.get("irregular_max_points", 20)

        # Edge settings
        self.edge_blur = config.get("edge_blur", True)
        self.edge_blur_radius = config.get("edge_blur_radius", 2.0)
        self.edge_feather = config.get("edge_feather", True)

        # Output settings
        self.output_mode = config.get("output_mode", "mask")  # "mask" or "masked_image"
        self.mask_value = config.get("mask_value", 255)  # Value for masked areas
        self.invert_mask = config.get("invert_mask", False)

    def transform_batch(
        self,
        images: List[Image.Image],
        source_paths: List[str],
        metadata_list: List[Dict],
        accelerator,
    ) -> List[Image.Image]:
        """Create mask images or masked versions of input images."""

        transformed_images = []

        for img, path, metadata in zip(images, source_paths, metadata_list):
            try:
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Generate mask
                mask = self._generate_random_mask(img.size)

                # Apply based on output mode
                if self.output_mode == "mask":
                    # Return mask as RGB image
                    mask_rgb = mask.convert("RGB")
                    transformed_images.append(mask_rgb)
                else:
                    # Return masked image
                    masked_img = self._apply_mask_to_image(img, mask)
                    transformed_images.append(masked_img)

            except Exception as e:
                logger.error(f"Error creating mask for {path}: {e}")
                raise

        return transformed_images

    def _generate_random_mask(self, size: Tuple[int, int]) -> Image.Image:
        """Generate a random mask of specified size."""

        width, height = size
        mask = Image.new("L", size, 0)

        # Determine number of mask regions
        num_masks = random.randint(self.min_masks, self.max_masks)

        # Track coverage
        mask_array = np.array(mask)
        target_coverage = random.uniform(self.min_coverage, self.max_coverage)

        for _ in range(num_masks):
            # Check current coverage
            current_coverage = np.sum(mask_array > 0) / (width * height)
            if current_coverage >= target_coverage:
                break

            # Select mask type
            if self.mask_type_weights:
                mask_type = random.choices(self.mask_types, weights=self.mask_type_weights)[0]
            else:
                mask_type = random.choice(self.mask_types)

            # Generate mask region
            if mask_type == "rectangle":
                mask = self._add_rectangle_mask(mask)
            elif mask_type == "circle":
                mask = self._add_circle_mask(mask)
            elif mask_type == "brush":
                mask = self._add_brush_stroke(mask)
            elif mask_type == "irregular":
                mask = self._add_irregular_mask(mask)

            mask_array = np.array(mask)

        # Post-process mask
        if self.edge_blur:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=self.edge_blur_radius))

        if self.edge_feather:
            mask = self._feather_edges(mask)

        if self.invert_mask:
            mask = Image.eval(mask, lambda x: 255 - x)

        return mask

    def _add_rectangle_mask(self, mask: Image.Image) -> Image.Image:
        """Add a rectangular mask region."""
        from PIL import ImageDraw

        draw = ImageDraw.Draw(mask)
        width, height = mask.size

        # Random rectangle size
        rect_width = int(width * random.uniform(self.rect_min_size, self.rect_max_size))
        rect_height = int(height * random.uniform(self.rect_min_size, self.rect_max_size))

        # Random position
        x = random.randint(0, width - rect_width)
        y = random.randint(0, height - rect_height)

        # Draw rectangle
        draw.rectangle([x, y, x + rect_width, y + rect_height], fill=self.mask_value)

        return mask

    def _add_circle_mask(self, mask: Image.Image) -> Image.Image:
        """Add a circular mask region."""
        from PIL import ImageDraw

        draw = ImageDraw.Draw(mask)
        width, height = mask.size

        # Random radius
        min_dim = min(width, height)
        radius = int(min_dim * random.uniform(self.circle_min_radius, self.circle_max_radius))

        # Random center
        cx = random.randint(radius, width - radius)
        cy = random.randint(radius, height - radius)

        # Draw circle
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=self.mask_value)

        return mask

    def _add_brush_stroke(self, mask: Image.Image) -> Image.Image:
        """Add a brush stroke-like mask."""
        from PIL import ImageDraw

        draw = ImageDraw.Draw(mask)
        width, height = mask.size

        # Random brush width
        brush_width = random.randint(self.brush_min_width, self.brush_max_width)

        # Random path length
        path_length = int(min(width, height) * random.uniform(self.brush_min_length, self.brush_max_length))
        num_points = max(2, path_length // 20)

        # Generate random path
        points = []
        start_x = random.randint(brush_width, width - brush_width)
        start_y = random.randint(brush_width, height - brush_width)
        points.append((start_x, start_y))

        for _ in range(num_points - 1):
            # Random walk
            prev_x, prev_y = points[-1]
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(20, 50)

            new_x = int(prev_x + distance * np.cos(angle))
            new_y = int(prev_y + distance * np.sin(angle))

            # Keep within bounds
            new_x = max(brush_width, min(width - brush_width, new_x))
            new_y = max(brush_width, min(height - brush_width, new_y))

            points.append((new_x, new_y))

        # Draw thick line along path
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=self.mask_value, width=brush_width)

        # Draw circles at joints for smoother appearance
        for point in points:
            x, y = point
            draw.ellipse(
                [
                    x - brush_width // 2,
                    y - brush_width // 2,
                    x + brush_width // 2,
                    y + brush_width // 2,
                ],
                fill=self.mask_value,
            )

        return mask

    def _add_irregular_mask(self, mask: Image.Image) -> Image.Image:
        """Add an irregular polygon mask."""
        from PIL import ImageDraw

        draw = ImageDraw.Draw(mask)
        width, height = mask.size

        # Generate random polygon points
        num_points = random.randint(self.irregular_min_points, self.irregular_max_points)

        # Start from random center
        cx = random.randint(width // 4, 3 * width // 4)
        cy = random.randint(height // 4, 3 * height // 4)

        # Generate points in polar coordinates for smoother shape
        points = []
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points + random.uniform(-0.3, 0.3)
            radius = random.uniform(0.1, 0.3) * min(width, height)

            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))

            # Keep within bounds
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))

            points.append((x, y))

        # Draw polygon
        draw.polygon(points, fill=self.mask_value)

        return mask

    def _feather_edges(self, mask: Image.Image) -> Image.Image:
        """Feather the edges of the mask for smoother blending."""
        try:
            from scipy import ndimage

            # Convert to numpy for processing
            mask_array = np.array(mask).astype(np.float32) / 255.0

            # Binary mask
            binary_mask = mask_array > 0.5

            # Distance from edge
            dist_outside = ndimage.distance_transform_edt(~binary_mask)
            dist_inside = ndimage.distance_transform_edt(binary_mask)

            # Feather width
            feather_width = 10

            # Create feathered mask
            feathered = np.ones_like(mask_array)
            feathered[~binary_mask] = np.clip(dist_outside[~binary_mask] / feather_width, 0, 1)
            feathered[binary_mask] = np.clip(1 - (dist_inside[binary_mask] - feather_width) / feather_width, 0, 1)

            # Convert back to PIL
            feathered = (feathered * 255).astype(np.uint8)
            return Image.fromarray(feathered, mode="L")

        except ImportError:
            # Fallback to simple Gaussian blur if scipy not available
            return mask.filter(ImageFilter.GaussianBlur(radius=self.edge_blur_radius * 2))

    def _apply_mask_to_image(self, img: Image.Image, mask: Image.Image) -> Image.Image:
        """Apply mask to image (set masked areas to white or black)."""
        img_array = np.array(img)
        mask_array = np.array(mask)

        # Normalize mask
        mask_normalized = mask_array.astype(np.float32) / 255.0

        # Apply mask (white fill for masked areas)
        masked_array = img_array.copy()
        for c in range(3):
            masked_array[:, :, c] = (img_array[:, :, c] * (1 - mask_normalized) + 255 * mask_normalized).astype(np.uint8)

        return Image.fromarray(masked_array)


class DepthMapSampleGenerator(SampleGenerator):
    """Creates depth map images for ControlNet training."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = config.get("model_type", "DPT")
        self._model = None
        self._transform = None
        self.max_worker_count = 1

    @property
    def requires_gpu(self) -> bool:
        """Depth models require GPU and must run in main process."""
        return True

    @property
    def is_thread_safe(self) -> bool:
        """GPU models are not thread-safe across processes."""
        return False

    def _load_model(self, device):
        """Lazy load the depth estimation model."""
        try:
            if self._model is None:
                logger.info("Loading DPT depth estimation model")
                from transformers import DPTImageProcessor, pipeline

                self._model = pipeline(task="depth-estimation", model="Intel/dpt-large", device=device)
                logger.info("Loaded DPT model successfully")
        except Exception as e:
            logger.error(f"Error loading depth estimation model: {e}")
            raise

    def transform_batch(
        self,
        images: List[Image.Image],
        source_paths: List[str],
        metadata_list: List[Dict],
        accelerator,
    ) -> List[Image.Image]:
        """Create depth map images with minimal processing."""

        device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(device)

        # Ensure all images are RGB
        rgb_images = []
        original_sizes = []

        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            rgb_images.append(img)
            original_sizes.append(img.size)

        try:
            # Process entire batch at once
            with torch.no_grad():
                depth_outputs = self._model(rgb_images)

            # Extract and format depth maps
            depth_images = []
            for depth_output, original_size in zip(depth_outputs, original_sizes):
                depth_img = depth_output["depth"] if isinstance(depth_output, dict) else depth_output

                # Resize if needed
                if depth_img.size != original_size:
                    depth_img = depth_img.resize(original_size, Image.Resampling.LANCZOS)

                # Ensure RGB
                if depth_img.mode != "RGB":
                    depth_img = depth_img.convert("RGB")

                depth_images.append(depth_img)

            return depth_images

        except Exception as e:
            logger.error(f"Batch processing failed: {e}, falling back to black images")
            # Return all black images on complete failure
            return [Image.new("RGB", size, (0, 0, 0)) for size in original_sizes]


class SegmentationSampleGenerator(SampleGenerator):
    """
    Example of a GPU-enabled generator that creates segmentation masks.
    This demonstrates proper GPU handling in the DataGenerator pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "facebook/mask2former-swin-large-cityscapes-semantic")
        self._model = None
        self._processor = None
        self.max_worker_count = 1  # Signal single-threaded execution

    @property
    def requires_gpu(self) -> bool:
        """This generator requires GPU for the segmentation model."""
        return True

    @property
    def is_thread_safe(self) -> bool:
        """GPU models are not safe to use across process boundaries."""
        return False

    def _load_model(self, device):
        """Lazy load the segmentation model on the specified device."""
        if self._model is None:
            try:
                from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

                logger.info(f"Loading segmentation model {self.model_name} on {device}")

                # Load processor and model
                self._processor = AutoImageProcessor.from_pretrained(self.model_name)
                self._model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name).to(device)

                # Set to eval mode
                self._model.eval()

                logger.info("Segmentation model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load segmentation model: {e}")
                raise

    def transform_batch(
        self,
        images: List[Image.Image],
        source_paths: List[str],
        metadata_list: List[Dict],
        accelerator,
    ) -> List[Image.Image]:
        """Create segmentation mask images."""

        # Determine device
        device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model on first use
        self._load_model(device)

        transformed_images = []

        # Process in eval mode with no grad
        with torch.no_grad():
            for img, path, metadata in zip(images, source_paths, metadata_list):
                try:
                    # Ensure RGB
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    original_size = img.size

                    # Preprocess image
                    inputs = self._processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Run inference
                    outputs = self._model(**inputs)

                    # Post-process to get segmentation mask
                    predicted_segmentation = self._processor.post_process_semantic_segmentation(
                        outputs,
                        target_sizes=[original_size[::-1]],  # (height, width)
                    )[0]

                    # Convert to numpy and normalize
                    seg_array = predicted_segmentation.cpu().numpy()

                    # Convert to color map or grayscale
                    seg_image = self._segmentation_to_image(seg_array)

                    # Resize back to original if needed
                    if seg_image.size != original_size:
                        seg_image = seg_image.resize(original_size, Image.Resampling.NEAREST)

                    transformed_images.append(seg_image)

                except Exception as e:
                    logger.error(f"Error creating segmentation for {path}: {e}")
                    # Return black image on error
                    transformed_images.append(Image.new("RGB", img.size, (0, 0, 0)))

        return transformed_images

    def _segmentation_to_image(self, seg_array: np.ndarray) -> Image.Image:
        """Convert segmentation array to RGB image."""
        # Simple grayscale visualization
        # You could implement a color map here for better visualization

        # Normalize to 0-255 range
        unique_classes = np.unique(seg_array)
        normalized = np.zeros_like(seg_array, dtype=np.uint8)

        for i, class_id in enumerate(unique_classes):
            # Map each class to a gray level
            gray_value = int((i / len(unique_classes)) * 255)
            normalized[seg_array == class_id] = gray_value

        # Convert to RGB
        seg_rgb = np.stack([normalized] * 3, axis=-1)

        return Image.fromarray(seg_rgb)


class OpticalFlowSampleGenerator(SampleGenerator):
    """
    GPU-enabled generator that creates optical flow visualizations.
    Requires pairs of consecutive frames.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = config.get("model_type", "raft")  # raft, flownet, etc.
        self._model = None
        self.max_worker_count = 1

    @property
    def requires_gpu(self) -> bool:
        return True

    @property
    def is_thread_safe(self) -> bool:
        return False

    def _load_model(self, device):
        """Load optical flow model."""
        if self._model is None:
            try:
                # Example with RAFT model
                from torchvision.models.optical_flow import raft_large

                logger.info(f"Loading RAFT optical flow model on {device}")

                self._model = raft_large(pretrained=True)
                self._model = self._model.to(device)
                self._model.eval()

                logger.info("Optical flow model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load optical flow model: {e}")
                raise

    def transform_batch(
        self,
        images: List[Image.Image],
        source_paths: List[str],
        metadata_list: List[Dict],
        accelerator,
    ) -> List[Image.Image]:
        """Create optical flow visualizations."""

        device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model(device)

        transformed_images = []

        # Process pairs of images
        # For simplicity, we'll create synthetic motion by warping the same image
        # In practice, you'd use consecutive video frames

        with torch.no_grad():
            for img, path, metadata in zip(images, source_paths, metadata_list):
                try:
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    # Create a slightly transformed version for flow calculation
                    img2 = self._create_synthetic_motion(img)

                    # Convert to tensors
                    img1_tensor = self._image_to_tensor(img).to(device)
                    img2_tensor = self._image_to_tensor(img2).to(device)

                    # Calculate optical flow
                    flow = self._model(img1_tensor, img2_tensor)[-1]

                    # Visualize flow
                    flow_vis = self._visualize_flow(flow[0].cpu().numpy())

                    # Convert to PIL Image
                    flow_image = Image.fromarray(flow_vis)

                    # Resize to match original
                    if flow_image.size != img.size:
                        flow_image = flow_image.resize(img.size, Image.Resampling.BILINEAR)

                    transformed_images.append(flow_image)

                except Exception as e:
                    logger.error(f"Error creating optical flow for {path}: {e}")
                    transformed_images.append(Image.new("RGB", img.size, (0, 0, 0)))

        return transformed_images

    def _create_synthetic_motion(self, img: Image.Image) -> Image.Image:
        """Create synthetic motion for demonstration."""
        # Simple translation for demo
        from PIL import ImageChops

        return ImageChops.offset(img, 5, 5)

    def _image_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor."""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform(img).unsqueeze(0)

    def _visualize_flow(self, flow: np.ndarray) -> np.ndarray:
        """Visualize optical flow as RGB image."""
        # Simple flow visualization using HSV color space
        h, w = flow.shape[1:3]

        # Calculate magnitude and angle
        u, v = flow[0], flow[1]
        mag = np.sqrt(u**2 + v**2)
        ang = np.arctan2(v, u)

        # Normalize
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue from angle
        hsv[..., 1] = 255  # Full saturation
        hsv[..., 2] = np.minimum(mag * 20, 255).astype(np.uint8)  # Value from magnitude

        # Convert to RGB using trainingsample
        import trainingsample as tsr

        rgb = tsr.cvt_color_py(hsv, 55)  # 55 = COLOR_HSV2RGB

        return rgb


class I2VFirstFrameSampleGenerator(SampleGenerator):
    """
    Extract the first frame from video samples (or passthrough images) to build
    self-contained I2V conditioning datasets.
    """

    def transform_batch(
        self,
        images: List[Any],
        source_paths: List[str],
        metadata_list: List[Dict],
        accelerator,
    ) -> List[Image.Image]:
        frames: List[Image.Image] = []
        for sample in images:
            frames.append(self._extract_first_frame(sample))
        return frames

    @staticmethod
    def _extract_first_frame(sample: Any) -> Image.Image:
        if isinstance(sample, Image.Image):
            return sample.convert("RGB")

        if torch.is_tensor(sample):
            sample = sample.detach().cpu().numpy()

        if isinstance(sample, np.ndarray):
            data = sample
            if data.ndim == 4:
                data = data[0]
            if data.ndim != 3:
                raise ValueError(f"Unsupported array shape for I2V conditioning: {sample.shape}")

            if data.shape[0] in (1, 3) and data.shape[-1] not in (1, 3):
                data = np.moveaxis(data, 0, -1)

            if data.shape[-1] == 1:
                data = np.repeat(data, 3, axis=-1)

            if data.dtype != np.uint8:
                data = np.clip(data, 0, 255).astype(np.uint8)

            if data.shape[-1] != 3:
                raise ValueError(f"Expected 3 channels for I2V conditioning frame, got shape {data.shape}")

            return Image.fromarray(data, mode="RGB")

        raise ValueError(f"Unsupported sample type for I2V conditioning: {type(sample)}")


# Registry of available generators
GENERATOR_REGISTRY: Dict[str, Type[SampleGenerator]] = {
    "superresolution": SuperResolutionSampleGenerator,
    "super_resolution": SuperResolutionSampleGenerator,  # Alias
    "lowres": SuperResolutionSampleGenerator,  # Alias
    "jpeg_artifacts": JPEGArtifactsSampleGenerator,
    "jpeg": JPEGArtifactsSampleGenerator,  # Alias
    "compression": JPEGArtifactsSampleGenerator,  # Alias
    "random_masks": RandomMasksSampleGenerator,
    "masks": RandomMasksSampleGenerator,  # Alias
    "inpainting": RandomMasksSampleGenerator,  # Alias
    "canny": CannyEdgeSampleGenerator,
    "edges": CannyEdgeSampleGenerator,  # Alias
    "depth": DepthMapSampleGenerator,
    "depth_map": DepthMapSampleGenerator,  # Alias
    "depth_midas": DepthMapSampleGenerator,  # Alias
    "optical_flow": OpticalFlowSampleGenerator,
    "flow": OpticalFlowSampleGenerator,  # Alias
    "segmentation": SegmentationSampleGenerator,
    "semantic_segmentation": SegmentationSampleGenerator,  # Alias
    "i2v_first_frame": I2VFirstFrameSampleGenerator,
    "wan_i2v_first_frame": I2VFirstFrameSampleGenerator,
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
