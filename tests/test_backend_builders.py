"""
Component tests for data backend builder classes.

Validates builder classes for creating backend instances and handling configurations.
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from simpletuner.helpers.data_backend.builders import (
    BaseBackendBuilder,
    LocalBackendBuilder,
    AwsBackendBuilder,
    CsvBackendBuilder,
    HuggingfaceBackendBuilder,
    create_backend_builder,
    build_backend_from_config,
)

from simpletuner.helpers.data_backend.config import ImageBackendConfig, TextEmbedBackendConfig, ImageEmbedBackendConfig


class TestBaseBackendBuilder(unittest.TestCase):
    """Test the BaseBackendBuilder abstract class"""

    def setUp(self):
        """Set up test fixtures"""
        self.accelerator = Mock()
        self.accelerator.is_main_process = True

    def test_base_builder_is_abstract(self):
        """BaseBackendBuilder cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            BaseBackendBuilder(self.accelerator)

    def test_builder_classes_have_required_methods(self):
        """All builder classes implement required methods"""
        builder_classes = [LocalBackendBuilder, AwsBackendBuilder, CsvBackendBuilder, HuggingfaceBackendBuilder]

        for builder_class in builder_classes:
            builder = builder_class(self.accelerator)

            # check required methods exist
            self.assertTrue(hasattr(builder, "build"))
            self.assertTrue(callable(getattr(builder, "build")))

            self.assertTrue(hasattr(builder, "create_metadata_backend"))
            self.assertTrue(callable(getattr(builder, "create_metadata_backend")))

    def test_builder_classes_have_accelerator(self):
        """All builder classes store accelerator"""
        builder_classes = [LocalBackendBuilder, AwsBackendBuilder, CsvBackendBuilder, HuggingfaceBackendBuilder]

        for builder_class in builder_classes:
            builder = builder_class(self.accelerator)
            self.assertEqual(builder.accelerator, self.accelerator)


class TestLocalBackendBuilder(unittest.TestCase):
    """Test LocalBackendBuilder class"""

    def setUp(self):
        """Set up test fixtures"""
        self.accelerator = Mock()
        self.accelerator.is_main_process = True
        self.builder = LocalBackendBuilder(self.accelerator)

        self.args = {"cache_dir": "/tmp/cache", "compress_disk_cache": False, "model_family": "flux"}

    def test_initialization(self):
        """LocalBackendBuilder initialization"""
        self.assertEqual(self.builder.accelerator, self.accelerator)

    @patch("simpletuner.helpers.data_backend.builders.local.LocalDataBackend")
    def test_build_basic_config(self, mock_local_backend_class):
        """Test building with basic configuration"""
        mock_backend = Mock()
        mock_local_backend_class.return_value = mock_backend

        config = ImageBackendConfig.from_dict(
            {"id": "test_local", "type": "local", "instance_data_dir": "/tmp/images", "cache_dir": "/tmp/cache"}, self.args
        )

        result = self.builder.build(config)

        # verify LocalDataBackend called with correct params
        mock_local_backend_class.assert_called_once()
        call_kwargs = mock_local_backend_class.call_args[1]

        self.assertEqual(call_kwargs["id"], "test_local")
        self.assertEqual(call_kwargs["accelerator"], self.accelerator)
        self.assertFalse(call_kwargs["compress_cache"])

        self.assertEqual(result, mock_backend)

    @patch("simpletuner.helpers.data_backend.builders.local.LocalDataBackend")
    def test_build_with_compression(self, mock_local_backend_class):
        """Test building with cache compression enabled"""
        mock_backend = Mock()
        mock_local_backend_class.return_value = mock_backend

        config = ImageBackendConfig.from_dict(
            {"id": "test_local", "type": "local", "instance_data_dir": "/tmp/images", "cache_dir": "/tmp/cache"},
            {**self.args, "compress_disk_cache": True},
        )

        result = self.builder.build(config)

        call_kwargs = mock_local_backend_class.call_args[1]
        self.assertTrue(call_kwargs["compress_cache"])

    @patch("simpletuner.helpers.data_backend.builders.base.ParquetMetadataBackend")
    @patch("simpletuner.helpers.data_backend.builders.base.JsonMetadataBackend")
    def test_create_metadata_backend_json(self, mock_json_backend, mock_parquet_backend):
        """Test creating JSON metadata backend"""
        mock_data_backend = Mock()
        mock_json_instance = Mock()
        mock_json_backend.return_value = mock_json_instance

        config = ImageBackendConfig.from_dict(
            {"id": "test_local", "type": "local", "metadata_backend": "json", "instance_data_dir": "/tmp/images"}, self.args
        )

        result = self.builder.create_metadata_backend(config, mock_data_backend, self.args)

        mock_json_backend.assert_called_once()
        mock_parquet_backend.assert_not_called()
        self.assertEqual(result, mock_json_instance)

    @patch("simpletuner.helpers.data_backend.builders.base.ParquetMetadataBackend")
    @patch("simpletuner.helpers.data_backend.builders.base.JsonMetadataBackend")
    def test_create_metadata_backend_parquet(self, mock_json_backend, mock_parquet_backend):
        """Test creating Parquet metadata backend"""
        mock_data_backend = Mock()
        mock_parquet_instance = Mock()
        mock_parquet_backend.return_value = mock_parquet_instance

        config = ImageBackendConfig.from_dict(
            {"id": "test_local", "type": "local", "metadata_backend": "parquet", "instance_data_dir": "/tmp/images"},
            self.args,
        )

        result = self.builder.create_metadata_backend(config, mock_data_backend, self.args)

        mock_parquet_backend.assert_called_once()
        mock_json_backend.assert_not_called()
        self.assertEqual(result, mock_parquet_instance)

    @patch("simpletuner.helpers.data_backend.builders.local.LocalDataBackend")
    @patch("simpletuner.helpers.data_backend.builders.base.JsonMetadataBackend")
    def test_build_with_metadata_complete(self, mock_json_backend, mock_local_backend):
        """Test complete build_with_metadata workflow"""
        mock_data_backend = Mock()
        mock_local_backend.return_value = mock_data_backend
        mock_metadata_backend = Mock()
        mock_json_backend.return_value = mock_metadata_backend

        config = ImageBackendConfig.from_dict(
            {"id": "test_local", "type": "local", "instance_data_dir": "/tmp/images", "cache_dir": "/tmp/cache"}, self.args
        )

        result = self.builder.build_with_metadata(config, self.args)

        # verify result structure
        self.assertIn("id", result)
        self.assertIn("data_backend", result)
        self.assertIn("metadata_backend", result)
        self.assertIn("instance_data_dir", result)
        self.assertIn("config", result)

        self.assertEqual(result["id"], "test_local")
        self.assertEqual(result["data_backend"], mock_data_backend)
        self.assertEqual(result["metadata_backend"], mock_metadata_backend)
        self.assertEqual(result["instance_data_dir"], "/tmp/images")


class TestAwsBackendBuilder(unittest.TestCase):
    """Test AwsBackendBuilder class"""

    def setUp(self):
        """Set up test fixtures"""
        self.accelerator = Mock()
        self.builder = AwsBackendBuilder(self.accelerator)

        self.args = {
            "cache_dir": "/tmp/cache",
            "compress_disk_cache": False,
            "aws_max_pool_connections": 128,
            "model_family": "flux",
        }

    @patch("simpletuner.helpers.data_backend.builders.aws.S3DataBackend")
    def test_build_basic_config(self, mock_s3_backend_class):
        """Test building with basic AWS configuration"""
        mock_backend = Mock()
        mock_s3_backend_class.return_value = mock_backend

        config = ImageBackendConfig.from_dict(
            {
                "id": "test_aws",
                "type": "aws",
                "cache_dir": "/tmp/cache",
                "aws_bucket_name": "test-bucket",
                "aws_region_name": "us-east-1",
                "aws_endpoint_url": "http://localhost:9000",
                "aws_access_key_id": "test_key",
                "aws_secret_access_key": "test_secret",
            },
            self.args,
        )

        result = self.builder.build(config)

        # verify S3DataBackend called with correct params
        mock_s3_backend_class.assert_called_once()
        call_kwargs = mock_s3_backend_class.call_args[1]

        self.assertEqual(call_kwargs["id"], "test_aws")
        self.assertEqual(call_kwargs["bucket_name"], "test-bucket")
        self.assertEqual(call_kwargs["region_name"], "us-east-1")
        self.assertEqual(call_kwargs["endpoint_url"], "http://localhost:9000")
        self.assertEqual(call_kwargs["aws_access_key_id"], "test_key")
        self.assertEqual(call_kwargs["aws_secret_access_key"], "test_secret")
        self.assertEqual(call_kwargs["accelerator"], self.accelerator)
        self.assertFalse(call_kwargs["compress_cache"])
        self.assertEqual(call_kwargs["max_pool_connections"], 128)

        self.assertEqual(result, mock_backend)

    @patch("simpletuner.helpers.data_backend.builders.aws.S3DataBackend")
    def test_build_with_custom_max_connections(self, mock_s3_backend_class):
        """Test building with custom max pool connections"""
        mock_backend = Mock()
        mock_s3_backend_class.return_value = mock_backend

        config = ImageBackendConfig.from_dict(
            {
                "id": "test_aws",
                "type": "aws",
                "aws_bucket_name": "test-bucket",
                "aws_region_name": "us-east-1",
                "aws_endpoint_url": "http://localhost:9000",
                "aws_access_key_id": "test_key",
                "aws_secret_access_key": "test_secret",
                "aws_max_pool_connections": 256,
            },
            self.args,
        )

        result = self.builder.build(config)

        call_kwargs = mock_s3_backend_class.call_args[1]
        self.assertEqual(call_kwargs["max_pool_connections"], 256)

    def test_validate_aws_config_success(self):
        """Test AWS config validation with valid configuration"""
        config = ImageBackendConfig.from_dict(
            {
                "id": "test_aws",
                "type": "aws",
                "aws_bucket_name": "test-bucket",
                "aws_region_name": "us-east-1",
                "aws_endpoint_url": "http://localhost:9000",
                "aws_access_key_id": "test_key",
                "aws_secret_access_key": "test_secret",
            },
            self.args,
        )

        # Should not raise any exceptions
        self.builder._validate_aws_config(config)

    def test_validate_aws_config_missing_field(self):
        """Test AWS config validation with missing required field"""
        config = ImageBackendConfig.from_dict(
            {
                "id": "test_aws",
                "type": "aws",
                "aws_bucket_name": "test-bucket",
                "aws_region_name": "us-east-1",
                # Missing other required fields
            },
            self.args,
        )

        with self.assertRaises(ValueError) as context:
            self.builder._validate_aws_config(config)

        self.assertIn("Missing required AWS configuration", str(context.exception))


class TestCsvBackendBuilder(unittest.TestCase):
    """Test CsvBackendBuilder class"""

    def setUp(self):
        """Set up test fixtures"""
        self.accelerator = Mock()
        self.builder = CsvBackendBuilder(self.accelerator)

        self.args = {
            "cache_dir": "/tmp/cache",
            "compress_disk_cache": False,
            "caption_strategy": "csv",
            "model_family": "flux",
        }

    @patch("simpletuner.helpers.data_backend.builders.csv.CSVDataBackend")
    def test_build_basic_config(self, mock_csv_backend_class):
        """Test building with basic CSV configuration"""
        mock_backend = Mock()
        mock_csv_backend_class.return_value = mock_backend

        config = ImageBackendConfig.from_dict(
            {
                "id": "test_csv",
                "type": "csv",
                "caption_strategy": "csv",
                "csv_file": "/tmp/data.csv",
                "csv_cache_dir": "/tmp/csv_cache",
                "csv_caption_column": "caption",
                "csv_url_column": "url",
            },
            self.args,
        )

        result = self.builder.build(config)

        # verify CSVDataBackend called with correct params
        mock_csv_backend_class.assert_called_once()
        call_kwargs = mock_csv_backend_class.call_args[1]

        self.assertEqual(call_kwargs["id"], "test_csv")
        self.assertEqual(call_kwargs["csv_file"], "/tmp/data.csv")
        self.assertEqual(call_kwargs["csv_cache_dir"], "/tmp/csv_cache")
        self.assertEqual(call_kwargs["url_column"], "url")
        self.assertEqual(call_kwargs["caption_column"], "caption")
        self.assertEqual(call_kwargs["accelerator"], self.accelerator)
        self.assertFalse(call_kwargs["compress_cache"])

        self.assertEqual(result, mock_backend)

    @patch("simpletuner.helpers.data_backend.builders.csv.CSVDataBackend")
    def test_build_with_optional_params(self, mock_csv_backend_class):
        """Test building with optional CSV parameters"""
        mock_backend = Mock()
        mock_csv_backend_class.return_value = mock_backend

        config = ImageBackendConfig.from_dict(
            {
                "id": "test_csv",
                "type": "csv",
                "caption_strategy": "csv",
                "csv_file": "/tmp/data.csv",
                "csv_cache_dir": "/tmp/csv_cache",
                "csv_caption_column": "caption",
                "csv_url_column": "url",
                "csv_hash_filenames": True,
                "csv_shorten_filenames": False,
            },
            self.args,
        )

        result = self.builder.build(config)

        call_kwargs = mock_csv_backend_class.call_args[1]
        self.assertTrue(call_kwargs["hash_filenames"])
        self.assertFalse(call_kwargs["shorten_filenames"])

    def test_validate_csv_config_success(self):
        """Test CSV config validation with valid configuration"""
        config = ImageBackendConfig.from_dict(
            {
                "id": "test_csv",
                "type": "csv",
                "caption_strategy": "csv",
                "csv_file": "/tmp/data.csv",
                "csv_cache_dir": "/tmp/csv_cache",
                "csv_caption_column": "caption",
                "csv_url_column": "url",
            },
            self.args,
        )

        # Should not raise any exceptions
        self.builder._validate_csv_config(config, self.args)

    def test_validate_csv_config_invalid_strategy(self):
        """Test CSV config validation with invalid caption strategy"""
        config = ImageBackendConfig.from_dict(
            {
                "id": "test_csv",
                "type": "csv",
                "caption_strategy": "filename",  # Invalid for CSV
                "csv_file": "/tmp/data.csv",
                "csv_cache_dir": "/tmp/csv_cache",
                "csv_caption_column": "caption",
                "csv_url_column": "url",
            },
            {**self.args, "caption_strategy": "filename"},
        )

        with self.assertRaises(ValueError) as context:
            self.builder._validate_csv_config(config, {**self.args, "caption_strategy": "filename"})

        self.assertIn("caption_strategy must be 'csv'", str(context.exception))

    def test_validate_csv_config_missing_field(self):
        """Test CSV config validation with missing required field"""
        config = ImageBackendConfig.from_dict(
            {
                "id": "test_csv",
                "type": "csv",
                "caption_strategy": "csv",
                "csv_file": "/tmp/data.csv",
                # Missing other required fields
            },
            self.args,
        )

        with self.assertRaises(ValueError) as context:
            self.builder._validate_csv_config(config, self.args)

        self.assertIn("Missing required CSV configuration", str(context.exception))


class TestHuggingfaceBackendBuilder(unittest.TestCase):
    """Test HuggingfaceBackendBuilder class"""

    def setUp(self):
        """Set up test fixtures"""
        self.accelerator = Mock()
        self.builder = HuggingfaceBackendBuilder(self.accelerator)

        self.args = {"cache_dir": "/tmp/cache", "compress_disk_cache": False, "model_family": "flux"}

    @patch("simpletuner.helpers.data_backend.builders.huggingface.HuggingfaceDatasetsBackend")
    def test_build_basic_config(self, mock_hf_backend_class):
        """Test building with basic HuggingFace configuration"""
        mock_backend = Mock()
        mock_hf_backend_class.return_value = mock_backend

        config = ImageBackendConfig.from_dict(
            {
                "id": "test_hf",
                "type": "huggingface",
                "dataset_name": "test/dataset",
                "caption_strategy": "huggingface",
                "metadata_backend": "huggingface",
                "huggingface": {"cache_dir": "/tmp/hf_cache"},
            },
            self.args,
        )

        result = self.builder.build(config)

        # verify HuggingfaceDatasetsBackend called with correct params
        mock_hf_backend_class.assert_called_once()
        call_kwargs = mock_hf_backend_class.call_args[1]

        self.assertEqual(call_kwargs["identifier"], "test_hf")
        self.assertEqual(call_kwargs["dataset_name"], "test/dataset")
        self.assertEqual(call_kwargs["dataset_type"], "image")
        self.assertEqual(call_kwargs["accelerator"], self.accelerator)
        self.assertFalse(call_kwargs["compress_cache"])
        self.assertFalse(call_kwargs["auto_load"])

        self.assertEqual(result, mock_backend)

    @patch("simpletuner.helpers.data_backend.builders.huggingface.HuggingfaceDatasetsBackend")
    def test_build_with_optional_params(self, mock_hf_backend_class):
        """Test building with optional HuggingFace parameters"""
        mock_backend = Mock()
        mock_hf_backend_class.return_value = mock_backend

        config = ImageBackendConfig.from_dict(
            {
                "id": "test_hf",
                "type": "huggingface",
                "dataset_name": "test/dataset",
                "caption_strategy": "huggingface",
                "metadata_backend": "huggingface",
                "huggingface": {
                    "cache_dir": "/tmp/hf_cache",
                    "split": "validation",
                    "revision": "v1.1",
                    "image_column": "image",
                    "video_column": "video",
                    "streaming": False,
                    "num_proc": 4,
                    "auto_load": True,
                },
            },
            self.args,
        )

        result = self.builder.build(config)

        call_kwargs = mock_hf_backend_class.call_args[1]
        self.assertEqual(call_kwargs["split"], "validation")
        self.assertEqual(call_kwargs["revision"], "v1.1")
        self.assertEqual(call_kwargs["image_column"], "image")
        self.assertEqual(call_kwargs["video_column"], "video")
        self.assertFalse(call_kwargs["streaming"])
        self.assertEqual(call_kwargs["num_proc"], 4)
        self.assertTrue(call_kwargs["auto_load"])

    @patch("simpletuner.helpers.data_backend.builders.huggingface.HuggingfaceDatasetsBackend")
    def test_build_assigns_default_cache_dir_when_missing(self, mock_hf_backend_class):
        """Default cache directory should be derived when none is provided."""
        mock_backend = Mock()
        mock_hf_backend_class.return_value = mock_backend

        config = ImageBackendConfig.from_dict(
            {
                "id": "test_hf",
                "type": "huggingface",
                "dataset_name": "test/dataset",
            },
            self.args,
        )

        builder = HuggingfaceBackendBuilder(self.accelerator, self.args)
        builder.build(config)

        expected_cache = Path(self.args["cache_dir"]) / "huggingface" / "test_hf"

        call_kwargs = mock_hf_backend_class.call_args[1]
        self.assertEqual(Path(call_kwargs["cache_dir"]), expected_cache)
        self.assertEqual(Path(config.huggingface_cache_dir), expected_cache)

    def test_validate_huggingface_config_success(self):
        """Test HuggingFace config validation with valid configuration"""
        config = ImageBackendConfig.from_dict(
            {"id": "test_hf", "type": "huggingface", "dataset_name": "test/dataset", "huggingface": {}}, self.args
        )

        # Should not raise any exceptions
        self.builder._validate_huggingface_config(config)

    def test_validate_huggingface_config_missing_dataset_name(self):
        """Test HuggingFace config validation with missing dataset_name"""
        config = ImageBackendConfig.from_dict({"id": "test_hf", "type": "huggingface", "huggingface": {}}, self.args)

        with self.assertRaises(ValueError) as context:
            self.builder._validate_huggingface_config(config)

        self.assertIn("dataset_name is required", str(context.exception))

    def test_validate_huggingface_config_missing_huggingface_block(self):
        """Test HuggingFace config validation with missing huggingface configuration block"""
        config = ImageBackendConfig.from_dict(
            {"id": "test_hf", "type": "huggingface", "dataset_name": "test/dataset"}, self.args
        )

        try:
            self.builder._validate_huggingface_config(config)
        except ValueError as exc:
            self.fail(f"Legacy configs without a huggingface block should be accepted, but raised: {exc}")


class TestCreateBackendBuilder(unittest.TestCase):
    """Test the create_backend_builder factory function"""

    def setUp(self):
        """Set up test fixtures"""
        self.accelerator = Mock()

    def test_create_local_builder(self):
        """Test creating local backend builder"""
        builder = create_backend_builder("local", self.accelerator)

        self.assertIsInstance(builder, LocalBackendBuilder)
        self.assertEqual(builder.accelerator, self.accelerator)

    def test_create_aws_builder(self):
        """Test creating AWS backend builder"""
        builder = create_backend_builder("aws", self.accelerator)

        self.assertIsInstance(builder, AwsBackendBuilder)
        self.assertEqual(builder.accelerator, self.accelerator)

    def test_create_csv_builder(self):
        """Test creating CSV backend builder"""
        builder = create_backend_builder("csv", self.accelerator)

        self.assertIsInstance(builder, CsvBackendBuilder)
        self.assertEqual(builder.accelerator, self.accelerator)

    def test_create_huggingface_builder(self):
        """Test creating HuggingFace backend builder"""
        builder = create_backend_builder("huggingface", self.accelerator)

        self.assertIsInstance(builder, HuggingfaceBackendBuilder)
        self.assertEqual(builder.accelerator, self.accelerator)

    def test_create_builder_invalid_type(self):
        """Test creating builder with invalid backend type"""
        with self.assertRaises(ValueError) as context:
            create_backend_builder("invalid_type", self.accelerator)

        self.assertIn("Unknown backend type: invalid_type", str(context.exception))
        self.assertIn("Supported types:", str(context.exception))

    def test_supported_backend_types(self):
        """Test that all expected backend types are supported"""
        expected_types = ["local", "aws", "csv", "huggingface"]

        for backend_type in expected_types:
            # Should not raise any exceptions
            builder = create_backend_builder(backend_type, self.accelerator)
            self.assertIsNotNone(builder)


class TestBuildBackendFromConfig(unittest.TestCase):
    """Test the build_backend_from_config convenience function"""

    def setUp(self):
        """Set up test fixtures"""
        self.accelerator = Mock()
        self.args = {"cache_dir": "/tmp/cache", "compress_disk_cache": False, "model_family": "flux"}

    @patch("simpletuner.helpers.data_backend.builders.LocalBackendBuilder")
    def test_build_from_config_local(self, mock_builder_class):
        """Test building backend from local configuration"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        mock_setup = {
            "id": "test_local",
            "data_backend": Mock(),
            "metadata_backend": Mock(),
            "instance_data_dir": "/tmp/images",
            "config": {},
        }
        mock_builder.build_with_metadata.return_value = mock_setup

        config = ImageBackendConfig.from_dict(
            {"id": "test_local", "type": "local", "instance_data_dir": "/tmp/images", "cache_dir": "/tmp/cache"}, self.args
        )

        result = build_backend_from_config(config, self.accelerator, self.args)

        # verify builder created and called correctly
        mock_builder_class.assert_called_once_with(self.accelerator)
        mock_builder.build_with_metadata.assert_called_once_with(config, self.args)

        self.assertEqual(result, mock_setup)

    @patch("simpletuner.helpers.data_backend.builders.AwsBackendBuilder")
    def test_build_from_config_aws(self, mock_builder_class):
        """Test building backend from AWS configuration"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        mock_setup = {
            "id": "test_aws",
            "data_backend": Mock(),
            "metadata_backend": Mock(),
            "instance_data_dir": "",
            "config": {},
        }
        mock_builder.build_with_metadata.return_value = mock_setup

        config = ImageBackendConfig.from_dict(
            {
                "id": "test_aws",
                "type": "aws",
                "aws_bucket_name": "test-bucket",
                "aws_region_name": "us-east-1",
                "aws_endpoint_url": "http://localhost:9000",
                "aws_access_key_id": "test_key",
                "aws_secret_access_key": "test_secret",
            },
            self.args,
        )

        result = build_backend_from_config(config, self.accelerator, self.args)

        mock_builder_class.assert_called_once_with(self.accelerator)
        mock_builder.build_with_metadata.assert_called_once_with(config, self.args)

        self.assertEqual(result, mock_setup)

    def test_build_from_config_validation_error(self):
        """Test that configuration validation errors are propagated"""
        config = ImageBackendConfig.from_dict(
            {"id": "test_invalid", "type": "local", "crop_aspect": "invalid_value"}, self.args
        )

        with self.assertRaises(ValueError) as context:
            build_backend_from_config(config, self.accelerator, self.args)

        self.assertIn("crop_aspect must be one of", str(context.exception))

    @patch("simpletuner.helpers.data_backend.builders.create_backend_builder")
    def test_build_from_config_builder_creation_error(self, mock_create_builder):
        """Test that builder creation errors are propagated"""
        mock_create_builder.side_effect = ValueError("Unknown backend type")

        config = ImageBackendConfig.from_dict(
            {"id": "test_local", "type": "local", "instance_data_dir": "/tmp/images"}, self.args
        )

        with self.assertRaises(ValueError) as context:
            build_backend_from_config(config, self.accelerator, self.args)

        self.assertIn("Unknown backend type", str(context.exception))

    @patch("simpletuner.helpers.data_backend.builders.LocalBackendBuilder")
    def test_build_from_config_fallback_for_simple_builders(self, mock_builder_class):
        """Test fallback mechanism for builders without build_with_metadata"""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        # Remove build_with_metadata method to test fallback
        del mock_builder.build_with_metadata

        mock_data_backend = Mock()
        mock_metadata_backend = Mock()
        mock_builder.build.return_value = mock_data_backend
        mock_builder.create_metadata_backend.return_value = mock_metadata_backend

        config = ImageBackendConfig.from_dict(
            {"id": "test_local", "type": "local", "instance_data_dir": "/tmp/images", "cache_dir": "/tmp/cache"}, self.args
        )

        result = build_backend_from_config(config, self.accelerator, self.args)

        # Check that fallback mechanism was used
        mock_builder.build.assert_called_once_with(config)
        mock_builder.create_metadata_backend.assert_called_once_with(config, mock_data_backend, self.args)

        self.assertEqual(result["id"], "test_local")
        self.assertEqual(result["data_backend"], mock_data_backend)
        self.assertEqual(result["metadata_backend"], mock_metadata_backend)
        self.assertEqual(result["instance_data_dir"], "")


class TestBuilderIntegration(unittest.TestCase):
    """Integration tests for builder components"""

    def setUp(self):
        """Set up test fixtures"""
        self.accelerator = Mock()
        self.args = {
            "cache_dir": "/tmp/cache",
            "compress_disk_cache": False,
            "model_family": "flux",
            "caption_strategy": "filename",
        }

    def test_text_embed_config_integration(self):
        """Test that text embed configs work with builders (should not need builders)"""
        config = TextEmbedBackendConfig.from_dict(
            {"id": "text_test", "dataset_type": "text_embeds", "caption_filter_list": ["nsfw"]}, self.args
        )

        # Text embed configs don't use builders, so this tests the config system
        output = config.to_dict()

        self.assertEqual(output["id"], "text_test")
        self.assertEqual(output["dataset_type"], "text_embeds")
        self.assertEqual(output["config"]["caption_filter_list"], ["nsfw"])

    def test_image_embed_config_integration(self):
        """Test that image embed configs work with builders (should not need builders)"""
        config = ImageEmbedBackendConfig.from_dict({"id": "image_embeds_test", "dataset_type": "image_embeds"}, self.args)

        # Image embed configs don't use builders either
        output = config.to_dict()

        self.assertEqual(output["id"], "image_embeds_test")
        self.assertEqual(output["dataset_type"], "image_embeds")
        self.assertEqual(output["config"], {})

    @patch("simpletuner.helpers.data_backend.builders.LocalBackendBuilder.build_with_metadata")
    def test_config_builder_pipeline(self, mock_build_with_metadata):
        """Test the complete config -> builder pipeline"""
        mock_setup = {
            "id": "pipeline_test",
            "data_backend": Mock(),
            "metadata_backend": Mock(),
            "instance_data_dir": "/tmp/images",
            "config": {},
        }
        mock_build_with_metadata.return_value = mock_setup

        # Create config
        config = ImageBackendConfig.from_dict(
            {
                "id": "pipeline_test",
                "type": "local",
                "instance_data_dir": "/tmp/images",
                "cache_dir": "/tmp/cache",
                "crop": True,
                "crop_aspect": "preserve",
            },
            self.args,
        )

        # Build backend
        result = build_backend_from_config(config, self.accelerator, self.args)

        # Verify the complete pipeline worked
        self.assertEqual(result["id"], "pipeline_test")
        mock_build_with_metadata.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
