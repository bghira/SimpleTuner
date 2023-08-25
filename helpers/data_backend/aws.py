import boto3, os
import fnmatch, logging
from pathlib import PosixPath
from helpers.data_backend.base import BaseDataBackend

boto_logger = logging.getLogger("botocore.hooks")
boto_logger.setLevel("WARNING")
boto_logger = logging.getLogger("botocore.auth")
boto_logger.setLevel("WARNING")
boto_logger = logging.getLogger("botocore.httpsession")
boto_logger.setLevel("WARNING")
boto_logger = logging.getLogger("botocore.parsers")
boto_logger.setLevel("WARNING")
boto_logger = logging.getLogger("botocore.retryhandler")
boto_logger.setLevel("WARNING")
boto_logger = logging.getLogger("botocore.loaders")
boto_logger.setLevel("WARNING")
boto_logger = logging.getLogger("botocore.regions")
boto_logger.setLevel("WARNING")
boto_logger = logging.getLogger("botocore.utils")
boto_logger.setLevel("WARNING")
boto_logger = logging.getLogger("botocore.client")
boto_logger.setLevel("WARNING")


class S3DataBackend(BaseDataBackend):
    def __init__(
        self,
        bucket_name,
        region_name="us-east-1",
        endpoint_url: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
    ):
        self.bucket_name = bucket_name
        # AWS buckets might use a region.
        extra_args = {
            "region_name": region_name,
        }
        # If using an endpoint_url, we do not use the region.
        if endpoint_url:
            extra_args = {
                "endpoint_url": endpoint_url,
            }
        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            **extra_args
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
        response = self.client.get_object(
            Bucket=self.bucket_name, Key=self._convert_path_to_key(str(s3_key))
        )
        return response["Body"].read()

    def open_file(self, s3_key, mode):
        """Open the file in the specified mode."""
        return self.read(s3_key)

    def write(self, s3_key, data):
        """Upload data to the specified S3 key."""
        self.client.put_object(
            Body=data,
            Bucket=self.bucket_name,
            Key=self._convert_path_to_key(str(s3_key)),
        )

    def delete(self, s3_key):
        """Delete the specified file from S3."""
        self.client.delete_object(
            Bucket=self.bucket_name, Key=self._convert_path_to_key(str(s3_key))
        )

    def list_by_prefix(self, prefix=""):
        """List all files under a specific path (prefix) in the S3 bucket."""
        response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        return [item["Key"] for item in response.get("Contents", [])]

    def list_files(self, str_pattern: str, instance_data_root: str = None):
        # Initialize the results list
        results = []

        # Using paginator to handle potential large number of objects
        paginator = self.client.get_paginator("list_objects_v2")

        # We'll use fnmatch to filter based on the provided pattern.
        pattern = os.path.join(instance_data_root or "", str_pattern)

        # Using a dictionary to hold files based on their prefixes (subdirectories)
        prefix_dict = {}

        # Paginating over the entire bucket objects
        for page in paginator.paginate(Bucket=self.bucket_name):
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
        from PIL import Image
        from io import BytesIO

        return Image.open(BytesIO(self.read(s3_key)))

    def create_directory(self, directory_path):
        # Since S3 doesn't have a traditional directory structure, this is just a pass-through
        pass

    def torch_load(self, s3_key):
        import torch
        from io import BytesIO

        return torch.load(BytesIO(self.read(s3_key)))

    def torch_save(self, data, s3_key):
        import torch
        from io import BytesIO

        buffer = BytesIO()
        torch.save(data, buffer)
        self.write(s3_key, buffer.getvalue())
