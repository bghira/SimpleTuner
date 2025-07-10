import boto3
import os
from os.path import splitext
import time
from botocore.exceptions import (
    NoCredentialsError,
    PartialCredentialsError,
)
import fnmatch
import logging
import torch
from torch import Tensor
import concurrent.futures
from botocore.config import Config
from helpers.data_backend.base import BaseDataBackend
from helpers.training.multi_process import _get_rank as get_rank
from helpers.image_manipulation.load import load_image, load_video
from helpers.training import video_file_extensions, image_file_extensions
from io import BytesIO
from typing import Union

loggers_to_silence = [
    "botocore.hooks",
    "botocore.auth",
    "botocore.httpsession",
    "botocore.parsers",
    "botocore.retryhandler",
    "botocore.loaders",
    "botocore.regions",
    "botocore.utils",
    "botocore.client",
    "botocore.handler",
    "botocore.handlers",
    "botocore.awsrequest",
]

for logger_name in loggers_to_silence:
    logger = logging.getLogger(logger_name)
    logger.setLevel("ERROR")

boto_logger = logging.getLogger("botocore.endpoint")
boto_logger.setLevel(os.environ.get("SIMPLETUNER_AWS_LOG_LEVEL", "ERROR"))

logger = logging.getLogger("S3DataBackend")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class S3DataBackend(BaseDataBackend):
    # Storing the list_files output in a local dict.
    _list_cache: dict = {}

    def __init__(
        self,
        id: str,
        bucket_name,
        accelerator,
        region_name="us-east-1",
        endpoint_url: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        read_retry_limit: int = 5,
        write_retry_limit: int = 5,
        read_retry_interval: int = 5,
        write_retry_interval: int = 5,
        compress_cache: bool = False,
        max_pool_connections: int = 128,
    ):
        self.id = id
        self.accelerator = accelerator
        self.bucket_name = bucket_name
        self.read_retry_limit = read_retry_limit
        self.read_retry_interval = read_retry_interval
        self.write_retry_limit = write_retry_limit
        self.write_retry_interval = write_retry_interval
        self.compress_cache = compress_cache
        self.max_pool_connections = max_pool_connections
        self.type = "aws"

        extra_args = {"region_name": region_name}
        if endpoint_url:
            # If using endpoint_url, we do not use region
            extra_args = {"endpoint_url": endpoint_url}

        s3_config = Config(max_pool_connections=self.max_pool_connections)
        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=s3_config,
            **extra_args,
        )

    def get_instance_representation(self) -> dict:
        """Get a serializable representation of this backend instance."""
        return {
            "backend_type": "aws",
            "id": self.id,
            "bucket_name": self.bucket_name,
            "region_name": self.client.meta.region_name,
            "endpoint_url": self.client.meta.endpoint_url,
            "read_retry_limit": self.read_retry_limit,
            "write_retry_limit": self.write_retry_limit,
            "read_retry_interval": self.read_retry_interval,
            "write_retry_interval": self.write_retry_interval,
            "compress_cache": self.compress_cache,
            "max_pool_connections": self.max_pool_connections,
            # Note: accelerator and credentials are not serializable
        }

    @staticmethod
    def from_instance_representation(representation: dict) -> "S3DataBackend":
        """Create a new S3DataBackend instance from a serialized representation."""
        if representation.get("backend_type") != "aws":
            raise ValueError(
                f"Expected backend_type 'aws', got {representation.get('backend_type')}"
            )

        # Extract parameters from representation
        return S3DataBackend(
            id=representation["id"],
            bucket_name=representation["bucket_name"],
            accelerator=None,  # Will be set by subprocess if needed
            region_name=representation.get("region_name", "us-east-1"),
            endpoint_url=representation.get("endpoint_url"),
            aws_access_key_id=None,  # Credentials should be re-established in subprocess
            aws_secret_access_key=None,  # Credentials should be re-established in subprocess
            read_retry_limit=representation.get("read_retry_limit", 5),
            write_retry_limit=representation.get("write_retry_limit", 5),
            read_retry_interval=representation.get("read_retry_interval", 5),
            write_retry_interval=representation.get("write_retry_interval", 5),
            compress_cache=representation.get("compress_cache", False),
            max_pool_connections=representation.get("max_pool_connections", 128),
        )

    def get_abs_path(self, sample_path: str) -> tuple:
        """
        Given a relative path of a sample, return the absolute path.
        For S3, this is just the S3 key.
        """
        if not sample_path:
            return (self.bucket_name, "")
        if sample_path.startswith("/"):
            sample_path = sample_path[1:]
        return (self.bucket_name, sample_path)

    def exists(self, s3_key):
        """Check if the key exists in S3, with retries for transient errors."""
        for i in range(self.read_retry_limit):
            try:
                self.client.head_object(Bucket=self.bucket_name, Key=str(s3_key))
                return True
            except self.client.exceptions.NoSuchKey:
                logger.debug(
                    f"File {s3_key} does not exist in S3 bucket ({self.bucket_name})"
                )
                return False
            except (NoCredentialsError, PartialCredentialsError) as e:
                raise e  # Raise credential errors to the caller
            except Exception as e:
                if (
                    "An error occurred (404) when calling the HeadObject operation: Not Found"
                    in str(e)
                ):
                    return False
                logger.error(f'Error checking existence of S3 key "{s3_key}": {e}')
                if i == self.read_retry_limit - 1:
                    raise e
                else:
                    time.sleep(self.read_retry_interval)

    def read(self, s3_key):
        """Retrieve and return the content of the file from S3."""
        for i in range(self.read_retry_limit):
            try:
                response = self.client.get_object(
                    Bucket=self.bucket_name, Key=str(s3_key)
                )
                return response["Body"].read()
            except self.client.exceptions.NoSuchKey:
                logger.debug(
                    f"File {s3_key} does not exist in S3 bucket ({self.bucket_name})"
                )
                return None
            except (NoCredentialsError, PartialCredentialsError) as e:
                raise e  # Raise credential errors to the caller
            except Exception as e:
                logger.error(f'Error reading S3 key "{s3_key}": {e}')
                if i == self.read_retry_limit - 1:
                    raise e
                else:
                    time.sleep(self.read_retry_interval)

    def open_file(self, s3_key, mode):
        """Open the file in the specified mode."""
        return self.read(s3_key)

    def write(self, s3_key, data):
        """
        Upload data to the specified S3 key.

        """
        real_key = str(s3_key)
        for i in range(self.write_retry_limit):
            try:
                if isinstance(data, (dict, torch.Tensor)):
                    return self.torch_save(data, real_key)
                response = self.client.put_object(
                    Body=data, Bucket=self.bucket_name, Key=real_key
                )
                return response
            except Exception as e:
                logger.error(f'Error writing S3 key "{real_key}": {e}')
                if i == self.write_retry_limit - 1:
                    raise e
                else:
                    time.sleep(self.write_retry_interval)

    def delete(self, s3_key):
        """Delete the specified file from S3."""
        for i in range(self.write_retry_limit):
            try:
                logger.debug(f'Deleting S3 key "{s3_key}"')
                response = self.client.delete_object(
                    Bucket=self.bucket_name, Key=str(s3_key)
                )
                return response
            except Exception as e:
                logger.error(f'Error deleting S3 key "{s3_key}": {e}')
                if i == self.write_retry_limit - 1:
                    raise e
                else:
                    time.sleep(self.write_retry_interval)

    def list_by_prefix(self, prefix=""):
        """List all files under a specific path (prefix) in the S3 bucket."""
        response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        bucket_prefix = f"{self.bucket_name}/"

        return [
            (
                item["Key"][len(bucket_prefix) :]
                if item["Key"].startswith(bucket_prefix)
                else item["Key"]
            )
            for item in response.get("Contents", [])
        ]

    def list_files(self, file_extensions: list, instance_data_dir: str = None):
        results = []

        def splitext_(path):
            ext = splitext(path)[1].lower()
            return ext[1:] if ext else ext

        start_time = time.time()
        paginator = self.client.get_paginator("list_objects_v2")
        prefix_dict = {}

        logger.debug(
            f"Listing files in S3 bucket {self.bucket_name} in prefix {instance_data_dir} with extensions: {file_extensions}"
        )

        for page in paginator.paginate(Bucket=self.bucket_name, MaxKeys=1000):
            for obj in page.get("Contents", []):
                ext = splitext_(obj["Key"])
                if file_extensions and ext not in file_extensions:
                    continue
                parts = obj["Key"].split("/")
                subdir = "/".join(parts[:-1])
                filename = parts[-1]

                if subdir not in prefix_dict:
                    prefix_dict[subdir] = []
                prefix_dict[subdir].append(obj["Key"])

        for subdir, files in prefix_dict.items():
            results.append((subdir, [], files))

        end_time = time.time()
        total_time = end_time - start_time
        if total_time > 120:
            logger.debug(f"Completed file list in {total_time/60} minutes.")
        elif total_time < 60:
            logger.debug(f"Completed file list in {total_time} seconds.")
        return results

    def read_image(self, s3_key):
        """
        Read an image OR a video from S3.
        """
        data = self.read(s3_key)
        if data is None:
            return None
        buffer = BytesIO(data)

        # Check extension
        ext = s3_key.rsplit(".", 1)[-1].lower() if "." in s3_key else ""
        if ext in video_file_extensions:
            return load_video(buffer)
        else:
            return load_image(buffer)

    def read_image_batch(self, s3_keys: list, delete_problematic_images: bool = False):
        """
        Return a list of Image (or video) objects, given a list of S3 keys.
        This uses read_batch(...) for potential concurrency.
        """
        batch = self.read_batch(s3_keys)
        output_images = []
        available_keys = []

        for s3_key, data in zip(s3_keys, batch):
            if data is None:
                logger.warning(f"Unable to load image '{s3_key}', skipping (no data).")
                continue
            try:
                # Check extension to decide loader
                ext = s3_key.rsplit(".", 1)[-1].lower() if "." in s3_key else ""
                buffer = BytesIO(data)
                if ext in video_file_extensions:
                    image_data = load_video(buffer)
                else:
                    image_data = load_image(buffer)

                if image_data is None:
                    logger.warning(f"Unable to load image '{s3_key}', skipping.")
                    continue
                output_images.append(image_data)
                available_keys.append(s3_key)
            except Exception as e:
                if delete_problematic_images:
                    logger.warning(f"Deleting image '{s3_key}' due to error: {e}")
                    self.delete(s3_key)
                else:
                    logger.warning(f"Problematic image/video {s3_key} encountered: {e}")

        return (available_keys, output_images)

    def create_directory(self, directory_path):
        """
        S3 doesn't have a directory structure, so this is a no-op.
        """
        pass

    def _detect_file_format(self, fileobj):
        fileobj.seek(0)
        magic_number = fileobj.read(4)
        fileobj.seek(0)
        logger.debug(f"Magic number: {magic_number}")

        if magic_number[:2] == b"\x80\x04" or b"PK" in magic_number:
            try:
                obj = torch.load(fileobj, map_location="cpu")
                if isinstance(obj, bytes):
                    return "incorrect"
                else:
                    return "correct_uncompressed"
            except Exception:
                return "correct_compressed"
        elif magic_number[:2] == b"\x1f\x8b":
            return "correct_compressed"
        else:
            return "unknown"

    def torch_load(self, s3_key):
        for i in range(self.read_retry_limit):
            try:
                data = self.read(s3_key)
                if data is None:
                    return None
                stored_data = BytesIO(data)
                stored_data.seek(0)

                file_format = self._detect_file_format(stored_data)
                logger.debug(f"File format: {file_format}")
                if file_format == "incorrect":
                    stored_data.seek(0)
                    compressed_data = BytesIO(
                        torch.load(stored_data, map_location="cpu")
                    )
                    stored_tensor = self._decompress_torch(compressed_data)
                elif file_format == "correct_compressed":
                    stored_tensor = self._decompress_torch(data)
                else:
                    stored_tensor = stored_data

                if hasattr(stored_tensor, "seek"):
                    stored_tensor.seek(0)
                obj = torch.load(stored_tensor, map_location="cpu")

                if isinstance(obj, tuple):
                    obj = tuple(o.to(torch.float32) for o in obj)
                elif isinstance(obj, torch.Tensor):
                    obj = obj.to(torch.float32)

                return obj
            except Exception as e:
                logging.error(f"Failed to load tensor from {s3_key}: {e}")
                if i == self.read_retry_limit - 1:
                    raise
                else:
                    logging.info(f"Retrying... ({i+1}/{self.read_retry_limit})")

    def torch_save(self, data, s3_key):
        """
        Save a torch object (tensor or dict) to S3 using possible compression.
        """
        import torch
        from io import BytesIO

        for i in range(self.write_retry_limit):
            try:
                buffer = BytesIO()
                if self.compress_cache:
                    compressed_data = self._compress_torch(data)
                    buffer.write(compressed_data)
                else:
                    torch.save(data, buffer)
                buffer.seek(0)
                logger.debug(f"Writing torch file: {s3_key}")
                result = self.write(s3_key, buffer.getvalue())
                logger.debug(f"Write completed: {s3_key}")
                return result
            except Exception as e:
                logger.error(f"Could not torch save to backend: {e}")
                if i == self.write_retry_limit - 1:
                    raise e
                else:
                    time.sleep(self.write_retry_interval)

    def write_batch(self, s3_keys, data_list):
        """Write a batch of files to the specified S3 keys concurrently."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.write, s3_keys, data_list)

    def read_batch(self, s3_keys):
        """Read a batch of files from the specified S3 keys concurrently."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.read, s3_keys))

    def bulk_exists(self, s3_keys, prefix=""):
        """Check the existence of a list of S3 keys in bulk."""
        objects = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        existing_keys = set(obj["Key"] for obj in objects.get("Contents", []))

        return [key in existing_keys for key in s3_keys]
