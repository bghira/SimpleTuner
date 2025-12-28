#!/usr/bin/env python3
"""
Script to generate Parquet metadata from an S3 video dataset.

This dramatically speeds up SimpleTuner startup for massive datasets
by avoiding individual file scanning.

Usage:
    python generate_metadata_parquet.py \
        --bucket YOUR_BUCKET_NAME \
        --prefix "videos/" \
        --output metadata.parquet \
        --region us-east-1 \
        --workers 32
"""

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import cv2
import pandas as pd
from tqdm import tqdm


def get_s3_client(region: str, endpoint_url: str = None):
    """Create an S3 client."""
    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def list_s3_videos(client, bucket: str, prefix: str) -> list[str]:
    """List all video files (.mp4, .mov, etc.) in the S3 bucket."""
    videos = []
    paginator = client.get_paginator("list_objects_v2")

    print(f"Listing videos in s3://{bucket}/{prefix}...")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
                videos.append(key)

    print(f"Found {len(videos)} videos")
    return videos


def get_video_metadata(client, bucket: str, video_key: str) -> dict | None:
    """Extract metadata from a video (downloads temporarily)."""
    try:
        # Download video to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            client.download_file(bucket, video_key, tmp.name)

            # Open with OpenCV to extract metadata
            cap = cv2.VideoCapture(tmp.name)
            if not cap.isOpened():
                return None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

        # Try to read caption from corresponding .txt file
        caption = ""
        txt_key = Path(video_key).with_suffix(".txt").as_posix()
        try:
            response = client.get_object(Bucket=bucket, Key=txt_key)
            caption = response["Body"].read().decode("utf-8").strip()
            # Take only the first line if there are multiple
            caption = caption.split("\n")[0].strip()
        except Exception:
            # If .txt not found, use filename as caption
            caption = Path(video_key).stem.replace("_", " ").replace("-", " ")

        return {
            "filename": Path(video_key).name,
            "s3_key": video_key,
            "caption": caption,
            "width": width,
            "height": height,
            "fps": fps,
            "num_frames": frame_count,
            "duration_seconds": round(duration, 2),
        }

    except Exception as e:
        print(f"Error processing {video_key}: {e}")
        return None


def process_video_batch(args):
    """Process a batch of videos (for parallelization)."""
    client, bucket, video_keys = args
    results = []
    for key in video_keys:
        meta = get_video_metadata(client, bucket, key)
        if meta:
            results.append(meta)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate Parquet metadata for an S3 video dataset"
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="", help="S3 prefix/folder")
    parser.add_argument("--output", default="metadata.parquet", help="Output file path")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--endpoint-url", default=None, help="Endpoint URL (for R2, MinIO, etc.)")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size per worker")
    parser.add_argument("--sample", type=int, default=None, help="Process only N videos (for testing)")

    args = parser.parse_args()

    # Create S3 client
    client = get_s3_client(args.region, args.endpoint_url)

    # List all videos
    videos = list_s3_videos(client, args.bucket, args.prefix)

    if args.sample:
        import random
        videos = random.sample(videos, min(args.sample, len(videos)))
        print(f"Sampling {len(videos)} videos for testing")

    # Split into batches
    batches = []
    for i in range(0, len(videos), args.batch_size):
        batch = videos[i : i + args.batch_size]
        # Each worker needs its own S3 client
        batches.append((get_s3_client(args.region, args.endpoint_url), args.bucket, batch))

    # Process in parallel
    all_metadata = []

    print(f"Processing {len(videos)} videos with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_video_batch, batch) for batch in batches]

        with tqdm(total=len(videos), desc="Extracting metadata") as pbar:
            for future in as_completed(futures):
                results = future.result()
                all_metadata.extend(results)
                pbar.update(len(results))

    # Create DataFrame and save
    df = pd.DataFrame(all_metadata)

    print(f"\nDataset statistics:")
    print(f"  Total videos processed: {len(df)}")
    print(f"  Unique resolutions: {df.groupby(['width', 'height']).size().shape[0]}")
    print(f"  Average duration: {df['duration_seconds'].mean():.2f}s")
    print(f"  Average frames: {df['num_frames'].mean():.0f}")
    print(f"  Average FPS: {df['fps'].mean():.1f}")

    # Filter corrupted or very short videos (< 10 frames)
    # Note: with uniform frame sampling in VAE cache, videos with varying FPS are accepted
    min_frames = 10  # Minimum to filter corrupted files
    df_valid = df[df["num_frames"] >= min_frames].copy()
    print(f"  Valid videos (>= {min_frames} frames): {len(df_valid)}")

    # Save Parquet
    df_valid.to_parquet(args.output, index=False)
    print(f"\nParquet saved to: {args.output}")

    # Also save a CSV sample for manual inspection
    csv_path = args.output.replace(".parquet", ".csv")
    df_valid.head(1000).to_csv(csv_path, index=False)
    print(f"CSV sample saved to: {csv_path}")


if __name__ == "__main__":
    main()
