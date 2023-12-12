from helpers.data_backend.local import LocalDataBackend
from helpers.data_backend.aws import S3DataBackend

import json, os


def configure_multi_databackend(args: dict, accelerator):
    """
    Configure a multiple dataloaders based on the provided commandline args.
    """
    if args.data_backend_config is None:
        raise ValueError(
            "Must provide a data backend config file via --data_backend_config"
        )
    if not os.path.exists(args.data_backend_config):
        raise FileNotFoundError(
            f"Data backend config file {args.data_backend_config} not found."
        )
    with open(args.data_backend_config, "r") as f:
        data_backend_config = json.load(f)
    if len(data_backend_config) == 0:
        raise ValueError(
            "Must provide at least one data backend in the data backend config file."
        )
    data_backends = []
    for backend in data_backend_config:
        if backend["type"] == "local":
            data_backends.append(get_local_backend(accelerator))
        elif backend["type"] == "aws":
            check_aws_config(backend)
            data_backends.append(
                get_aws_backend(
                    aws_bucket_name=backend["aws_bucket_name"],
                    aws_region_name=backend["aws_region_name"],
                    aws_endpoint_url=backend["aws_endpoint_url"],
                    aws_access_key_id=backend["aws_access_key_id"],
                    aws_secret_access_key=backend["aws_secret_access_key"],
                    accelerator=accelerator,
                )
            )
        else:
            raise ValueError(f"Unknown data backend type: {backend['type']}")
    if len(data_backends) == 0:
        raise ValueError(
            "Must provide at least one data backend in the data backend config file."
        )
    return data_backends


def get_local_backend(accelerator) -> LocalDataBackend:
    """
    Get a local disk backend.

    Args:
        accelerator (Accelerator): A Huggingface Accelerate object.
    Returns:
        LocalDataBackend: A LocalDataBackend object.
    """
    return LocalDataBackend(accelerator=accelerator)


def check_aws_config(backend: dict) -> None:
    """
    Check the configuration for an AWS backend.

    Args:
        backend (dict): A dictionary of the backend configuration.
    Returns:
        None
    """
    required_keys = [
        "aws_bucket_name",
        "aws_region_name",
        "aws_endpoint_url",
        "aws_access_key_id",
        "aws_secret_access_key",
    ]
    for key in required_keys:
        if key not in backend:
            raise ValueError(f"Missing required key {key} in AWS backend config.")


def get_aws_backend(
    aws_bucket_name: str,
    aws_region_name: str,
    aws_endpoint_url: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    accelerator,
) -> S3DataBackend:
    return S3DataBackend(
        bucket_name=aws_bucket_name,
        accelerator=accelerator,
        region_name=aws_region_name,
        endpoint_url=aws_endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def get_dataset(args: dict, accelerator) -> list:
    """Retrieve a dataset based on the provided commandline args.

    Args:
        args (dict): A dictionary from parseargs.
        accelerator (Accelerator): A Huggingface Accelerate object.
    Returns:
        list: A list of DataBackend objects.
    """
    if args.data_backend == "multi":
        return configure_multi_databackend(args)
    elif args.data_backend == "local":
        if not os.path.exists(args.instance_data_dir):
            raise FileNotFoundError(
                f"Instance {args.instance_data_root} images root doesn't exist. Cannot continue."
            )
        return [get_local_backend(args, accelerator)]
    elif args.data_backend == "aws":
        return [
            get_aws_backend(
                aws_bucket_name=args.aws_bucket_name,
                aws_region_name=args.aws_region_name,
                aws_endpoint_url=args.aws_endpoint_url,
                aws_access_key_id=args.aws_access_key_id,
                aws_secret_access_key=args.aws_secret_access_key,
                accelerator=accelerator,
            )
        ]
