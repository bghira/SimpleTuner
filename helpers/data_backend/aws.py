import boto3, os, time
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import fnmatch, logging
from torch import Tensor
from pathlib import PosixPath
import concurrent.futures
from botocore.config import Config
from helpers.data_backend.base import BaseDataBackend
from PIL import Image
from io import BytesIO

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

# Arguably, the most interesting one:
boto_logger = logging.getLogger("botocore.endpoint")
boto_logger.setLevel(os.environ.get("SIMPLETUNER_AWS_LOG_LEVEL", "ERROR"))

logger = logging.getLogger("S3DataBackend")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING"))


class S3DataBackend(BaseDataBackend):
    # Storing the list_files output in a local dict.
    _list_cache = {}

    def __init__(
        self,
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
    ):
        self.accelerator = accelerator
        self.bucket_name = bucket_name
        self.read_retry_limit = read_retry_limit
        self.read_retry_interval = read_retry_interval
        self.write_retry_limit = write_retry_limit
        self.write_retry_interval = write_retry_interval
        # AWS buckets might use a region.
        extra_args = {
            "region_name": region_name,
        }
        # If using an endpoint_url, we do not use the region.
        if endpoint_url:
            extra_args = {
                "endpoint_url": endpoint_url,
            }
        s3_config = Config(max_pool_connections=100)
        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=s3_config,
            **extra_args,
        )

    def exists(self, s3_key) -> bool:
        """Determine whether a file exists in S3."""
        try:
            self.client.head_object(
                Bucket=self.bucket_name, Key=self._convert_path_to_key(str(s3_key))
            )
            return True
        except:
            return False

    def read(self, s3_key):
        """Retrieve and return the content of the file from S3."""
        for i in range(self.read_retry_limit):
            try:
                response = self.client.get_object(
                    Bucket=self.bucket_name, Key=self._convert_path_to_key(str(s3_key))
                )
                return response["Body"].read()
            except self.client.exceptions.NoSuchKey:
                logger.debug(f'File "{s3_key}" does not exist in S3 bucket.')
                return None
            except (NoCredentialsError, PartialCredentialsError) as e:
                raise e  # Raise credential errors to the caller
            except Exception as e:
                logger.error(f'Error reading S3 bucket key "{s3_key}": {e}')
                if i == self.read_retry_limit - 1:
                    # We have reached our maximum retry count.
                    raise e
                else:
                    # Sleep for a bit before retrying.
                    time.sleep(self.read_retry_interval)

    def open_file(self, s3_key, mode):
        """Open the file in the specified mode."""
        return self.read(s3_key)

    def write(self, s3_key, data):
        """Upload data to the specified S3 key."""
        real_key = self._convert_path_to_key(str(s3_key))
        for i in range(self.write_retry_limit):
            try:
                if type(data) == Tensor:
                    return self.torch_save(data, real_key)
                logger.debug(f"Writing to S3 key {real_key}")
                response = self.client.put_object(
                    Body=data,
                    Bucket=self.bucket_name,
                    Key=real_key,
                )
                logger.debug(f"S3-Key {s3_key} Response: {response}")
                return response
            except Exception as e:
                logger.error(f'Error writing S3 bucket key "{real_key}": {e}')
                if i == self.write_retry_limit - 1:
                    # We have reached our maximum retry count.
                    raise e
                else:
                    # Sleep for a bit before retrying.
                    time.sleep(self.write_retry_interval)

    def delete(self, s3_key):
        """Delete the specified file from S3."""
        for i in range(self.write_retry_limit):
            try:
                logger.debug(f'Deleting S3 key "{s3_key}"')
                response = self.client.delete_object(
                    Bucket=self.bucket_name, Key=self._convert_path_to_key(str(s3_key))
                )
                return response
            except Exception as e:
                logger.error(f'Error deleting S3 bucket key "{s3_key}": {e}')
                if i == self.write_retry_limit - 1:
                    # We have reached our maximum retry count.
                    raise e
                else:
                    # Sleep for a bit before retrying.
                    time.sleep(self.write_retry_interval)

    def list_by_prefix(self, prefix=""):
        """List all files under a specific path (prefix) in the S3 bucket."""
        response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        bucket_prefix = f"{self.bucket_name}/"

        return [
            item["Key"][len(bucket_prefix) :]
            if item["Key"].startswith(bucket_prefix)
            else item["Key"]
            for item in response.get("Contents", [])
        ]

    def list_files(self, str_pattern: str, instance_data_root: str = None):
        # Initialize the results list
        results = []

        # Grab a timestamp for our start time.
        start_time = time.time()

        # Temporarily, we do not use prefixes in S3.
        instance_data_root = None

        # Using paginator to handle potential large number of objects
        paginator = self.client.get_paginator("list_objects_v2")

        # We'll use fnmatch to filter based on the provided pattern.
        pattern = os.path.join(instance_data_root or "", str_pattern)

        # Using a dictionary to hold files based on their prefixes (subdirectories)
        prefix_dict = {}
        # Log the first few items, alphabetically sorted:
        logger.debug(
            f"Listing files in S3 bucket {self.bucket_name} with search pattern: {pattern}"
        )

        # Paginating over the entire bucket objects
        for page in paginator.paginate(Bucket=self.bucket_name, MaxKeys=1000):
            for obj in page.get("Contents", []):
                # Filter based on the provided pattern
                if fnmatch.fnmatch(obj["Key"], pattern):
                    # Split the S3 key to determine the directory and file structure
                    parts = obj["Key"].split("/")
                    subdir = "/".join(
                        parts[:-1]
                    )  # Get the directory excluding the file
                    filename = parts[-1]  # Get the file name

                    # Storing filenames under their respective subdirectories
                    if subdir not in prefix_dict:
                        prefix_dict[subdir] = []
                    prefix_dict[subdir].append(filename)

        # Transforming the prefix_dict into the desired results format
        for subdir, files in prefix_dict.items():
            results.append((subdir, [], files))

        end_time = time.time()
        total_time = end_time - start_time
        # Log the output in n automatically human friendly manner, eg. "x minutes" or "x seconds"
        if total_time > 120:
            logger.debug(f"Completed file list in {total_time/60} minutes.")
        elif total_time < 60:
            logger.debug(f"Completed file list in {total_time} seconds.")
        return results

    def _convert_path_to_key(self, path: str) -> str:
        """
        Turn a /path/to/img.png into img.png

        Args:
            path (str): Full path, or just the base name.

        Returns:
            str: extracted basename, or input filename if already stripped.
        """
        return path.split("/")[-1]

    def read_image(self, s3_key):
        return Image.open(BytesIO(self.read(s3_key)))

    def read_image_batch(self, s3_keys: list, delete_problematic_images: bool = False):
        """
        Return a list of Image objects, given a list of S3 keys.
        This makes use of read_batch for efficiency.
        Args:
            s3_keys (list): List of S3 keys to read. May not be included in the output, if it does not exist, or had an error.
            delete_problematic_images (bool, optional): Whether to delete problematic images. Defaults to False.

        Returns:
            tuple(list, list): (available_keys, output_images)
        """
        batch = self.read_batch(s3_keys)
        output_images = []
        available_keys = []
        for s3_key, data in zip(s3_keys, batch):
            try:
                image_data = Image.open(BytesIO(data))
                if image_data is None:
                    logger.warning(f"Unable to load image '{s3_key}', skipping.")
                    continue
                output_images.append(image_data)
                available_keys.append(s3_key)
            except Exception as e:
                if delete_problematic_images:
                    logger.warning(
                        f"Deleting image '{s3_key}', because --delete_problematic_images is provided. Error: {e}"
                    )
                    self.delete(s3_key)
                else:
                    logger.warning(
                        f"A problematic image {s3_key} is detected, but we are not allowed to remove it, because --delete_problematic_image is not provided."
                        f" Please correct this manually. Error: {e}"
                    )
        return (available_keys, output_images)

    def create_directory(self, directory_path):
        # Since S3 doesn't have a traditional directory structure, this is just a pass-through
        logger.debug(f"Not creating directory {directory_path} on S3 backend.")
        pass

    def torch_load(self, s3_key):
        import torch
        from io import BytesIO

        return torch.load(
            BytesIO(self.read(s3_key)), map_location=self.accelerator.device
        )

    def torch_save(self, data, s3_key):
        import torch
        from io import BytesIO

        try:
            buffer = BytesIO()
            torch.save(data, buffer)
            return self.write(s3_key, buffer.getvalue())
        except Exception as e:
            logger.error(f"Could not torch save to backend: {e}")

    def write_batch(self, s3_keys, data_list):
        """Write a batch of files to the specified S3 keys concurrently."""
        # Use ThreadPoolExecutor for concurrent uploads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.write, s3_keys, data_list)

    def read_batch(self, s3_keys):
        """Read a batch of files from the specified S3 keys concurrently."""

        def read_from_s3(s3_key):
            """Helper function to read data from S3."""
            response = self.client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response["Body"].read()

        # Use ThreadPoolExecutor for concurrent reads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(read_from_s3, s3_keys))

    def bulk_exists(self, s3_keys, prefix=""):
        """Check the existence of a list of S3 keys in bulk."""

        # List all objects with the given prefix
        objects = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        existing_keys = set(obj["Key"] for obj in objects.get("Contents", []))

        # Check existence for each key
        return [key in existing_keys for key in s3_keys]
