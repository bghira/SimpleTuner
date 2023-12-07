import boto3, os, logging, argparse, datetime
from botocore.config import Config

# Set up logging
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def initialize_s3_client(args):
    """Initialize the boto3 S3 client using the provided AWS credentials and settings."""
    s3_config = Config(max_pool_connections=100)

    s3_client = boto3.client(
        "s3",
        endpoint_url=args.aws_endpoint_url,
        region_name=args.aws_region_name,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        config=s3_config,
    )
    return s3_client


from concurrent.futures import ThreadPoolExecutor


def delete_object(s3_client, bucket_name, object_key):
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=object_key)
        logger.info(f"Deleted: {object_key}")
    except Exception as e:
        logger.error(f"Error deleting {object_key} in bucket {bucket_name}: {e}")


def clear_s3_bucket(
    s3_client,
    bucket_name,
    num_workers=10,
    search_pattern: str = None,
    older_than_date: str = None,
):
    try:
        logger.info(f"Clearing out bucket {bucket_name}")

        # Convert the date string to a datetime object
        if older_than_date:
            target_date = datetime.datetime.strptime(older_than_date, "%Y-%m-%d")
        else:
            target_date = None

        # Initialize paginator
        paginator = s3_client.get_paginator("list_objects_v2")

        # Create a PageIterator from the Paginator
        page_iterator = paginator.paginate(Bucket=bucket_name)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for page in page_iterator:
                if "Contents" not in page:
                    logger.info(f"No more items in bucket {bucket_name}")
                    break

                # Filter by the older_than_date if provided
                if target_date:
                    filtered_objects = [
                        s3_object
                        for s3_object in page["Contents"]
                        if s3_object["LastModified"].replace(tzinfo=None) < target_date
                    ]
                else:
                    filtered_objects = page["Contents"]

                if search_pattern is not None:
                    keys_to_delete = [
                        s3_object["Key"]
                        for s3_object in filtered_objects
                        if search_pattern in s3_object["Key"]
                    ]
                else:
                    keys_to_delete = [
                        s3_object["Key"] for s3_object in filtered_objects
                    ]

                executor.map(
                    delete_object,
                    [s3_client] * len(keys_to_delete),
                    [bucket_name] * len(keys_to_delete),
                    keys_to_delete,
                )

        logger.info(f"Cleared out bucket {bucket_name}")

    except Exception as e:
        logger.error(f"Error clearing out bucket {bucket_name}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Clear out an S3 bucket.")
    parser.add_argument(
        "--aws_bucket_name",
        type=str,
        required=True,
        help="The AWS bucket name to clear.",
    )
    parser.add_argument("--aws_endpoint_url", type=str, help="The AWS server to use.")
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers to use for clearing.",
        default=10,
    )
    parser.add_argument(
        "--search_pattern",
        type=str,
        help="If provided, files with this in their Content key will be removed only.",
        default=None,
    )
    parser.add_argument("--aws_region_name", type=str, help="The AWS region to use.")
    parser.add_argument("--aws_access_key_id", type=str, help="AWS access key ID.")
    parser.add_argument(
        "--aws_secret_access_key", type=str, help="AWS secret access key."
    )
    parser.add_argument(
        "--older_than_date",
        type=str,
        help="If provided, only files older than this date (format: YYYY-MM-DD) will be cleared.",
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    s3_client = initialize_s3_client(args)
    clear_s3_bucket(
        s3_client,
        args.aws_bucket_name,
        num_workers=args.num_workers,
        search_pattern=args.search_pattern,
        older_than_date=args.older_than_date,
    )


if __name__ == "__main__":
    main()
