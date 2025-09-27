"""
Component tests for data backend configuration classes.

Validates configuration classes for handling configuration scenarios and validation rules.
"""

import unittest
from typing import Any, Dict
from unittest.mock import Mock, patch

from simpletuner.helpers.data_backend.config import (
    BaseBackendConfig,
    ImageBackendConfig,
    ImageEmbedBackendConfig,
    TextEmbedBackendConfig,
    create_backend_config,
    validators,
)


class TestBaseBackendConfig(unittest.TestCase):
    """Test the BaseBackendConfig abstract class"""

    def setUp(self):
        """Set up test fixtures"""
        self.args = {
            "resolution": 1.0,
            "resolution_type": "area",
            "caption_strategy": "filename",
            "minimum_image_size": 0.1,
            "maximum_image_size": None,
            "target_downsample_size": None,
            "cache_dir_text": "/tmp/cache/text",
            "cache_dir": "/tmp/cache",
            "model_family": "flux",
            "model_type": "flux",
            "controlnet": False,
            "caption_dropout_probability": 0.1,
        }

    def test_base_config_is_abstract(self):
        """Test that BaseBackendConfig cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            BaseBackendConfig()

    def test_from_dict_method_exists(self):
        """Test that all config classes have from_dict method"""
        for config_class in [ImageBackendConfig, TextEmbedBackendConfig, ImageEmbedBackendConfig]:
            self.assertTrue(hasattr(config_class, "from_dict"))
            self.assertTrue(callable(getattr(config_class, "from_dict")))

    def test_to_dict_method_exists(self):
        """Test that all config classes have to_dict method"""
        # test with concrete implementations
        backend_dict = {"id": "test", "type": "local"}

        text_config = TextEmbedBackendConfig.from_dict(backend_dict, self.args)
        self.assertTrue(hasattr(text_config, "to_dict"))
        self.assertTrue(callable(text_config.to_dict))

        image_config = ImageBackendConfig.from_dict(backend_dict, self.args)
        self.assertTrue(hasattr(image_config, "to_dict"))
        self.assertTrue(callable(image_config.to_dict))

    def test_validate_method_exists(self):
        """Test that all config classes have validate method"""
        backend_dict = {"id": "test", "type": "local"}

        text_config = TextEmbedBackendConfig.from_dict(backend_dict, self.args)
        self.assertTrue(hasattr(text_config, "validate"))
        self.assertTrue(callable(text_config.validate))

        image_config = ImageBackendConfig.from_dict(backend_dict, self.args)
        self.assertTrue(hasattr(image_config, "validate"))
        self.assertTrue(callable(image_config.validate))


class TestTextEmbedBackendConfig(unittest.TestCase):
    """Test TextEmbedBackendConfig class"""

    def setUp(self):
        """Set up test fixtures"""
        self.args = {
            "resolution": 1.0,
            "resolution_type": "area",
            "caption_strategy": "filename",
            "cache_dir_text": "/tmp/cache/text",
            "cache_dir": "/tmp/cache",
            "model_family": "flux",
            "caption_dropout_probability": 0.1,
        }

    def test_from_dict_minimal(self):
        """Test creation from minimal dictionary"""
        backend_dict = {"id": "text_test", "dataset_type": "text_embeds"}

        config = TextEmbedBackendConfig.from_dict(backend_dict, self.args)

        self.assertEqual(config.id, "text_test")
        self.assertEqual(config.dataset_type, "text_embeds")
        self.assertEqual(config.caption_filter_list, [])

    def test_from_dict_with_caption_filter(self):
        """Test creation with caption filter list"""
        backend_dict = {
            "id": "text_test",
            "dataset_type": "text_embeds",
            "caption_filter_list": ["nsfw", "violence", "inappropriate"],
        }

        config = TextEmbedBackendConfig.from_dict(backend_dict, self.args)

        self.assertEqual(config.caption_filter_list, ["nsfw", "violence", "inappropriate"])

    def test_to_dict_output(self):
        """Test to_dict method produces correct output"""
        backend_dict = {"id": "text_test", "dataset_type": "text_embeds", "caption_filter_list": ["test_filter"]}

        config = TextEmbedBackendConfig.from_dict(backend_dict, self.args)
        output = config.to_dict()

        expected = {"id": "text_test", "dataset_type": "text_embeds", "config": {"caption_filter_list": ["test_filter"]}}

        self.assertEqual(output, expected)

    def test_validate_success(self):
        """Test successful validation"""
        backend_dict = {"id": "text_test", "dataset_type": "text_embeds", "caption_filter_list": ["test"]}

        config = TextEmbedBackendConfig.from_dict(backend_dict, self.args)

        # should not raise exceptions
        config.validate(self.args)

    def test_validate_empty_config_success(self):
        """Test validation with empty configuration succeeds"""
        backend_dict = {"id": "text_test", "dataset_type": "text_embeds"}

        config = TextEmbedBackendConfig.from_dict(backend_dict, self.args)

        # should not raise exceptions
        config.validate(self.args)


class TestImageEmbedBackendConfig(unittest.TestCase):
    """Test ImageEmbedBackendConfig class"""

    def setUp(self):
        """Set up test fixtures"""
        self.args = {"resolution": 1.0, "resolution_type": "area", "cache_dir": "/tmp/cache", "model_family": "flux"}

    def test_from_dict_minimal(self):
        """Test creation from minimal dictionary"""
        backend_dict = {"id": "image_embeds_test", "dataset_type": "image_embeds"}

        config = ImageEmbedBackendConfig.from_dict(backend_dict, self.args)

        self.assertEqual(config.id, "image_embeds_test")
        self.assertEqual(config.dataset_type, "image_embeds")

    def test_to_dict_output(self):
        """Test to_dict method produces correct output"""
        backend_dict = {"id": "image_embeds_test", "dataset_type": "image_embeds"}

        config = ImageEmbedBackendConfig.from_dict(backend_dict, self.args)
        output = config.to_dict()

        expected = {"id": "image_embeds_test", "dataset_type": "image_embeds", "config": {}}

        self.assertEqual(output, expected)

    def test_validate_success(self):
        """Test successful validation"""
        backend_dict = {"id": "image_embeds_test", "dataset_type": "image_embeds"}

        config = ImageEmbedBackendConfig.from_dict(backend_dict, self.args)

        # should not raise exceptions
        config.validate(self.args)


class TestImageBackendConfig(unittest.TestCase):
    """Test ImageBackendConfig class"""

    def setUp(self):
        """Set up test fixtures"""
        self.args = {
            "resolution": 1.0,
            "resolution_type": "area",
            "caption_strategy": "filename",
            "minimum_image_size": 0.1,
            "maximum_image_size": None,
            "target_downsample_size": None,
            "cache_dir_text": "/tmp/cache/text",
            "cache_dir": "/tmp/cache",
            "model_family": "flux",
            "model_type": "flux",
            "controlnet": False,
            "caption_dropout_probability": 0.1,
            "compress_disk_cache": False,
            "delete_problematic_images": False,
            "delete_unwanted_images": False,
            "metadata_update_interval": 100,
            "train_batch_size": 4,
            "write_batch_size": 64,
            "skip_file_discovery": [],
            "max_train_steps": 1000,
        }

    def test_from_dict_minimal(self):
        """Test creation from minimal dictionary"""
        backend_dict = {"id": "image_test", "type": "local"}

        config = ImageBackendConfig.from_dict(backend_dict, self.args)

        self.assertEqual(config.id, "image_test")
        self.assertEqual(config.dataset_type, "image")
        self.assertEqual(config.type, "local")
        self.assertFalse(config.crop)
        self.assertEqual(config.crop_aspect, "square")
        self.assertEqual(config.crop_style, "random")
        self.assertEqual(config.resolution, 1.0)
        self.assertEqual(config.resolution_type, "area")

    def test_from_dict_full_config(self):
        """Test creation with full configuration"""
        backend_dict = {
            "id": "image_test",
            "type": "local",
            "dataset_type": "image",
            "crop": True,
            "crop_aspect": "preserve",
            "crop_style": "center",
            "resolution": 2.0,
            "resolution_type": "pixel_area",
            "caption_strategy": "parquet",
            "metadata_backend": "parquet",
            "parquet": {"config": "test"},
            "repeats": 5,
            "probability": 0.8,
            "minimum_image_size": 0.5,
            "maximum_image_size": 10.0,
            "target_downsample_size": 5.0,
        }

        config = ImageBackendConfig.from_dict(backend_dict, self.args)

        self.assertTrue(config.crop)
        self.assertEqual(config.crop_aspect, "preserve")
        self.assertEqual(config.crop_style, "center")
        self.assertAlmostEqual(config.resolution, 4.0)  # 2.0 * 2.0 / 1.0
        self.assertEqual(config.resolution_type, "area")
        self.assertEqual(config.caption_strategy, "parquet")
        self.assertEqual(config.repeats, 5)
        self.assertEqual(config.probability, 0.8)
        self.assertEqual(config.minimum_image_size, 0.5)
        self.assertEqual(config.maximum_image_size, 10.0)
        self.assertEqual(config.target_downsample_size, 5.0)

    def test_crop_aspect_random_with_buckets(self):
        """Test crop_aspect=random with crop_aspect_buckets"""
        backend_dict = {"id": "image_test", "type": "local", "crop_aspect": "random", "crop_aspect_buckets": [1.0, 1.5, 2.0]}

        config = ImageBackendConfig.from_dict(backend_dict, self.args)

        self.assertEqual(config.crop_aspect, "random")
        self.assertEqual(config.crop_aspect_buckets, [1.0, 1.5, 2.0])

    def test_controlnet_conditioning_config(self):
        """Test ControlNet conditioning configuration"""
        backend_dict = {
            "id": "image_test",
            "type": "local",
            "dataset_type": "image",
            "conditioning": {"conditioning_data": "conditioning_dataset"},
        }

        self.args["controlnet"] = True

        with patch("simpletuner.helpers.data_backend.config.validators.StateTracker") as mock_state:
            mock_state.get_args.return_value = Mock(controlnet=True)

            config = ImageBackendConfig.from_dict(backend_dict, self.args)

            self.assertEqual(config.conditioning["conditioning_data"], "conditioning_dataset")

    def test_to_dict_minimal_output(self):
        """Test to_dict method with minimal configuration"""
        backend_dict = {"id": "image_test", "type": "local"}

        config = ImageBackendConfig.from_dict(backend_dict, self.args)
        output = config.to_dict()

        self.assertEqual(output["id"], "image_test")
        self.assertEqual(output["dataset_type"], "image")
        self.assertEqual(output["type"], "local")
        self.assertIn("config", output)
        self.assertFalse(output["config"]["crop"])
        self.assertEqual(output["config"]["crop_aspect"], "square")

    def test_to_dict_full_output(self):
        """Test to_dict method with full configuration"""
        backend_dict = {
            "id": "image_test",
            "type": "local",
            "crop": True,
            "crop_aspect": "preserve",
            "repeats": 3,
            "probability": 0.7,
        }

        config = ImageBackendConfig.from_dict(backend_dict, self.args)
        output = config.to_dict()

        self.assertTrue(output["config"]["crop"])
        self.assertEqual(output["config"]["crop_aspect"], "preserve")
        self.assertEqual(output["config"]["repeats"], 3)
        self.assertEqual(output["config"]["probability"], 0.7)

    def test_validate_success(self):
        """Test successful validation"""
        backend_dict = {"id": "image_test", "type": "local", "crop_aspect": "preserve", "crop_style": "center"}

        config = ImageBackendConfig.from_dict(backend_dict, self.args)

        # should not raise exceptions
        config.validate(self.args)

    def test_validate_invalid_crop_aspect(self):
        """Test validation fails with invalid crop_aspect"""
        backend_dict = {"id": "image_test", "type": "local", "crop_aspect": "invalid_value"}

        config = ImageBackendConfig.from_dict(backend_dict, self.args)

        with self.assertRaises(ValueError) as context:
            config.validate(self.args)

        self.assertIn("crop_aspect must be one of", str(context.exception))

    def test_validate_invalid_crop_style(self):
        """Test validation fails with invalid crop_style"""
        backend_dict = {"id": "image_test", "type": "local", "crop_style": "invalid_value"}

        config = ImageBackendConfig.from_dict(backend_dict, self.args)

        with self.assertRaises(ValueError) as context:
            config.validate(self.args)

        self.assertIn("crop_style must be one of", str(context.exception))

    def test_validate_crop_aspect_random_without_buckets(self):
        """Test validation fails with crop_aspect=random but no buckets"""
        backend_dict = {"id": "image_test", "type": "local", "crop_aspect": "random"}

        config = ImageBackendConfig.from_dict(backend_dict, self.args)

        with self.assertRaises(ValueError) as context:
            config.validate(self.args)

        self.assertIn("crop_aspect_buckets must be provided", str(context.exception))

    def test_validate_invalid_crop_aspect_buckets_type(self):
        """Test validation fails with invalid crop_aspect_buckets type"""
        backend_dict = {
            "id": "image_test",
            "type": "local",
            "crop_aspect": "random",
            "crop_aspect_buckets": ["invalid", "string", "values"],
        }

        config = ImageBackendConfig.from_dict(backend_dict, self.args)

        with self.assertRaises(ValueError) as context:
            config.validate(self.args)

        self.assertIn("crop_aspect_buckets must be a list of float values", str(context.exception))

    def test_validate_maximum_image_size_area_too_large(self):
        """Test validation fails with maximum_image_size too large for area type"""
        backend_dict = {
            "id": "image_test",
            "type": "local",
            "maximum_image_size": 15,  # Too large for area type
            "target_downsample_size": 5,
            "resolution_type": "area",
        }

        config = ImageBackendConfig.from_dict(backend_dict, self.args)

        with self.assertRaises(ValueError) as context:
            config.validate(self.args)

        self.assertIn("maximum_image_size must be less than 10 megapixels", str(context.exception))

    def test_validate_maximum_image_size_pixel_too_small(self):
        """Test validation fails with maximum_image_size too small for pixel type"""
        backend_dict = {
            "id": "image_test",
            "type": "local",
            "maximum_image_size": 256,  # Too small for pixel type
            "target_downsample_size": 128,
            "resolution_type": "pixel",
        }

        # Modify args to have non-deepfloyd model
        args_copy = self.args.copy()
        args_copy["model_type"] = "sdxl"

        config = ImageBackendConfig.from_dict(backend_dict, args_copy)

        with self.assertRaises(ValueError) as context:
            config.validate(args_copy)

        self.assertIn("maximum_image_size must be at least 512 pixels", str(context.exception))

    def test_validate_maximum_size_without_downsample(self):
        """Test validation fails with maximum_image_size but no target_downsample_size"""
        backend_dict = {"id": "image_test", "type": "local", "maximum_image_size": 5}

        config = ImageBackendConfig.from_dict(backend_dict, self.args)

        with self.assertRaises(ValueError) as context:
            config.validate(self.args)

        self.assertIn("you must also provide a value for `target_downsample_size`", str(context.exception))

    def test_validate_parquet_with_json_metadata_error(self):
        """Test validation fails with parquet caption strategy and json metadata backend"""
        backend_dict = {"id": "image_test", "type": "local", "caption_strategy": "parquet", "metadata_backend": "json"}

        config = ImageBackendConfig.from_dict(backend_dict, self.args)

        with self.assertRaises(ValueError) as context:
            config.validate(self.args)

        self.assertIn("Cannot use caption_strategy=parquet with metadata_backend=json", str(context.exception))

    def test_validate_huggingface_caption_strategy(self):
        """Test validation fails with invalid caption strategy for HuggingFace backend"""
        backend_dict = {"id": "image_test", "type": "huggingface", "caption_strategy": "filename"}  # Invalid for HuggingFace

        config = ImageBackendConfig.from_dict(backend_dict, self.args)

        with self.assertRaises(ValueError) as context:
            config.validate(self.args)

        self.assertIn("caption_strategy must be set to 'huggingface'", str(context.exception))

    def test_validate_controlnet_without_conditioning(self):
        """Test validation fails with ControlNet enabled but no conditioning"""
        backend_dict = {"id": "image_test", "type": "local", "dataset_type": "image"}

        self.args["controlnet"] = True

        with patch("simpletuner.helpers.data_backend.config.validators.StateTracker") as mock_state:
            mock_state.get_args.return_value = Mock(controlnet=True)

            config = ImageBackendConfig.from_dict(backend_dict, self.args)

            with self.assertRaises(ValueError) as context:
                config.validate(self.args)

            self.assertIn("conditioning block", str(context.exception))


class TestCreateBackendConfig(unittest.TestCase):
    """Test the create_backend_config factory function"""

    def setUp(self):
        """Set up test fixtures"""
        self.args = {"resolution": 1.0, "resolution_type": "area", "cache_dir": "/tmp/cache", "model_family": "flux"}

    def test_create_text_embeds_config(self):
        """Test creating text embeds configuration"""
        backend_dict = {"id": "text_test", "dataset_type": "text_embeds"}

        config = create_backend_config(backend_dict, self.args)

        self.assertIsInstance(config, TextEmbedBackendConfig)
        self.assertEqual(config.id, "text_test")
        self.assertEqual(config.dataset_type, "text_embeds")

    def test_create_image_embeds_config(self):
        """Test creating image embeds configuration"""
        backend_dict = {"id": "image_embeds_test", "dataset_type": "image_embeds"}

        config = create_backend_config(backend_dict, self.args)

        self.assertIsInstance(config, ImageEmbedBackendConfig)
        self.assertEqual(config.id, "image_embeds_test")
        self.assertEqual(config.dataset_type, "image_embeds")

    def test_create_image_config(self):
        """Test creating image configuration"""
        backend_dict = {"id": "image_test", "type": "local"}

        config = create_backend_config(backend_dict, self.args)

        self.assertIsInstance(config, ImageBackendConfig)
        self.assertEqual(config.id, "image_test")
        self.assertEqual(config.dataset_type, "image")

    def test_create_image_config_explicit_type(self):
        """Test creating image configuration with explicit dataset_type"""
        backend_dict = {"id": "image_test", "type": "local", "dataset_type": "image"}

        config = create_backend_config(backend_dict, self.args)

        self.assertIsInstance(config, ImageBackendConfig)
        self.assertEqual(config.dataset_type, "image")

    def test_create_conditioning_config(self):
        """Test creating conditioning configuration"""
        backend_dict = {"id": "conditioning_test", "type": "local", "dataset_type": "conditioning"}

        config = create_backend_config(backend_dict, self.args)

        self.assertIsInstance(config, ImageBackendConfig)
        self.assertEqual(config.dataset_type, "conditioning")

    def test_create_eval_config(self):
        """Test creating eval configuration"""
        backend_dict = {"id": "eval_test", "type": "local", "dataset_type": "eval"}

        config = create_backend_config(backend_dict, self.args)

        self.assertIsInstance(config, ImageBackendConfig)
        self.assertEqual(config.dataset_type, "eval")

    def test_create_video_config(self):
        """Test creating video configuration"""
        backend_dict = {"id": "video_test", "type": "local", "dataset_type": "video"}

        config = create_backend_config(backend_dict, self.args)

        self.assertIsInstance(config, ImageBackendConfig)
        self.assertEqual(config.dataset_type, "video")

    def test_create_config_invalid_dataset_type(self):
        """Test creating configuration with invalid dataset_type"""
        backend_dict = {"id": "invalid_test", "dataset_type": "invalid_type"}

        with self.assertRaises(ValueError) as context:
            create_backend_config(backend_dict, self.args)

        self.assertIn("Unknown dataset_type: invalid_type", str(context.exception))

    def test_create_config_default_image_type(self):
        """Test creating configuration defaults to image dataset_type"""
        backend_dict = {
            "id": "default_test",
            "type": "local",
            # No dataset_type specified
        }

        config = create_backend_config(backend_dict, self.args)

        self.assertIsInstance(config, ImageBackendConfig)
        self.assertEqual(config.dataset_type, "image")


class TestConfigValidators(unittest.TestCase):
    """Test the validators module functions"""

    def test_validators_module_exists(self):
        """Test that validators module is importable"""
        self.assertTrue(hasattr(validators, "validate_crop_aspect"))
        self.assertTrue(hasattr(validators, "validate_crop_style"))
        self.assertTrue(hasattr(validators, "validate_resolution_constraints"))

    def test_validate_crop_aspect_valid_values(self):
        """Test crop aspect validation with valid values"""
        valid_values = ["square", "preserve", "random"]

        for value in valid_values:
            # should not raise
            validators.validate_crop_aspect(value, [])

    def test_validate_crop_aspect_invalid_value(self):
        """Test crop aspect validation with invalid value"""
        with self.assertRaises(ValueError) as context:
            validators.validate_crop_aspect("invalid", [])

        self.assertIn("crop_aspect must be one of", str(context.exception))

    def test_validate_crop_style_valid_values(self):
        """Test crop style validation with valid values"""
        valid_values = ["random", "center", "face"]

        for value in valid_values:
            # should not raise
            validators.validate_crop_style(value)

    def test_validate_crop_style_invalid_value(self):
        """Test crop style validation with invalid value"""
        with self.assertRaises(ValueError) as context:
            validators.validate_crop_style("invalid")

        self.assertIn("crop_style must be one of", str(context.exception))


class TestConfigRoundTrip(unittest.TestCase):
    """Test round-trip conversion: dict -> config -> dict"""

    def setUp(self):
        """Set up test fixtures"""
        self.args = {
            "resolution": 1.0,
            "resolution_type": "area",
            "caption_strategy": "filename",
            "cache_dir": "/tmp/cache",
            "model_family": "flux",
            "controlnet": False,
        }

    def test_text_embed_config_round_trip(self):
        """Test round-trip conversion for text embed config"""
        original_dict = {"id": "text_test", "dataset_type": "text_embeds", "caption_filter_list": ["nsfw", "violence"]}

        config = TextEmbedBackendConfig.from_dict(original_dict, self.args)
        output_dict = config.to_dict()

        # check essential data preserved
        self.assertEqual(output_dict["id"], original_dict["id"])
        self.assertEqual(output_dict["dataset_type"], original_dict["dataset_type"])
        self.assertEqual(output_dict["config"]["caption_filter_list"], original_dict["caption_filter_list"])

    def test_image_config_round_trip(self):
        """Test round-trip conversion for image config"""
        original_dict = {
            "id": "image_test",
            "type": "local",
            "dataset_type": "image",
            "crop": True,
            "crop_aspect": "preserve",
            "crop_style": "center",
            "repeats": 3,
            "probability": 0.7,
        }

        config = ImageBackendConfig.from_dict(original_dict, self.args)
        output_dict = config.to_dict()

        # check essential data preserved
        self.assertEqual(output_dict["id"], original_dict["id"])
        self.assertEqual(output_dict["type"], original_dict["type"])
        self.assertEqual(output_dict["dataset_type"], original_dict["dataset_type"])
        self.assertEqual(output_dict["config"]["crop"], original_dict["crop"])
        self.assertEqual(output_dict["config"]["crop_aspect"], original_dict["crop_aspect"])
        self.assertEqual(output_dict["config"]["crop_style"], original_dict["crop_style"])
        self.assertEqual(output_dict["config"]["repeats"], original_dict["repeats"])
        self.assertEqual(output_dict["config"]["probability"], original_dict["probability"])

    def test_image_embed_config_round_trip(self):
        """Test round-trip conversion for image embed config"""
        original_dict = {"id": "image_embeds_test", "dataset_type": "image_embeds"}

        config = ImageEmbedBackendConfig.from_dict(original_dict, self.args)
        output_dict = config.to_dict()

        # check essential data preserved
        self.assertEqual(output_dict["id"], original_dict["id"])
        self.assertEqual(output_dict["dataset_type"], original_dict["dataset_type"])
        self.assertEqual(output_dict["config"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
