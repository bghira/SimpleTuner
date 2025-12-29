"""Cloud storage endpoints for training output uploads.

Provides S3-compatible local upload endpoints for receiving training outputs
from cloud providers like Replicate.

NOTE: This module was renamed from s3.py to storage.py for clarity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, status

from ._shared import get_client_ip, get_job_store, get_local_upload_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/storage", tags=["storage"])


@router.put("/{bucket}/{key:path}")
async def put_object(bucket: str, key: str, request: Request) -> Dict[str, Any]:
    """S3-compatible PUT Object endpoint for receiving file uploads from the Cog.

    Requires authentication via per-job upload token.
    """
    store = get_job_store()
    client_ip = get_client_ip(request)

    # Note: Rate limiting is handled by RateLimitMiddleware

    # Extract per-job upload token from headers
    upload_token = request.headers.get("X-SimpleTuner-Secret", "") or request.headers.get("X-Upload-Token", "")

    if not upload_token:
        logger.warning("Storage upload without authentication from IP: %s", client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide per-job upload token via X-Upload-Token header.",
        )

    # Validate the token against job records
    authenticated_job = await store.get_job_by_upload_token(upload_token)
    if not authenticated_job:
        logger.warning("Storage upload with invalid token from IP: %s", client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid upload token. Token must match an active job.",
        )

    logger.debug("Storage upload authenticated for job %s from IP: %s", authenticated_job.job_id, client_ip)

    try:
        content = await request.body()

        # Validate bucket and key to prevent path traversal
        if ".." in bucket or ".." in key:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid bucket or key path")

        upload_dir = get_local_upload_dir()
        file_path = upload_dir / bucket / key

        # Verify the path is still within upload_dir
        try:
            file_path.resolve().relative_to(upload_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path")

        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        file_path.write_bytes(content)

        logger.info("Storage PUT: %s/%s (%d bytes)", bucket, key, len(content))

        return {
            "ETag": f'"{hash(content) & 0xFFFFFFFF:08x}"',
            "Key": key,
            "Bucket": bucket,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Storage PUT failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/{bucket}/{key:path}")
async def get_object(bucket: str, key: str) -> bytes:
    """S3-compatible GET Object endpoint."""
    if ".." in bucket or ".." in key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid bucket or key path")

    upload_dir = get_local_upload_dir()
    file_path = upload_dir / bucket / key

    try:
        file_path.resolve().relative_to(upload_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path")

    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Object not found")

    return file_path.read_bytes()


@router.get("")
async def list_buckets() -> Dict[str, Any]:
    """List all upload buckets (job output directories)."""
    upload_dir = get_local_upload_dir()

    if not upload_dir.exists():
        return {"Buckets": [], "total_size": 0}

    buckets = []
    total_size = 0

    for bucket_path in upload_dir.iterdir():
        if bucket_path.is_dir():
            # Calculate bucket size
            bucket_size = sum(f.stat().st_size for f in bucket_path.rglob("*") if f.is_file())
            file_count = sum(1 for f in bucket_path.rglob("*") if f.is_file())
            total_size += bucket_size

            # Get last modified time
            last_modified = None
            for f in bucket_path.rglob("*"):
                if f.is_file():
                    mtime = f.stat().st_mtime
                    if last_modified is None or mtime > last_modified:
                        last_modified = mtime

            buckets.append(
                {
                    "Name": bucket_path.name,
                    "Size": bucket_size,
                    "FileCount": file_count,
                    "LastModified": last_modified,
                }
            )

    # Sort by last modified, newest first
    buckets.sort(key=lambda b: b.get("LastModified") or 0, reverse=True)

    return {"Buckets": buckets, "total_size": total_size}


@router.get("/{bucket}")
async def list_objects(bucket: str) -> Dict[str, Any]:
    """S3-compatible list objects in bucket."""
    # Validate bucket name to prevent path traversal
    if ".." in bucket or "/" in bucket or "\\" in bucket:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid bucket name")

    upload_dir = get_local_upload_dir()
    bucket_path = upload_dir / bucket

    # Verify the resolved path is still within upload_dir
    try:
        bucket_path.resolve().relative_to(upload_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid bucket path")

    if not bucket_path.exists():
        return {"Contents": [], "Name": bucket}

    objects = []
    for file_path in bucket_path.rglob("*"):
        if file_path.is_file():
            rel_key = file_path.relative_to(bucket_path).as_posix()
            stat = file_path.stat()
            objects.append(
                {
                    "Key": rel_key,
                    "Size": stat.st_size,
                    "LastModified": stat.st_mtime,
                }
            )

    return {"Contents": objects, "Name": bucket}
