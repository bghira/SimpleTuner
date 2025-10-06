"""Service for handling HuggingFace checkpoint uploads via futures."""

from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import status

from simpletuner.helpers.publishing.huggingface import HubManager, LORA_SAFETENSORS_FILENAME
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.publishing_service import PublishingService
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore

logger = logging.getLogger(__name__)


class HuggingfaceServiceError(Exception):
    """Domain error raised when HuggingFace upload operations fail."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class UploadTask:
    """Represents a checkpoint upload task."""

    def __init__(
        self,
        task_id: str,
        environment_id: str,
        checkpoint_name: str,
        repo_id: str,
        branch: Optional[str] = None,
        subfolder: Optional[str] = None,
        callback_url: Optional[str] = None,
    ):
        self.task_id = task_id
        self.environment_id = environment_id
        self.checkpoint_name = checkpoint_name
        self.repo_id = repo_id
        self.branch = branch
        self.subfolder = subfolder
        self.callback_url = callback_url
        self.status = "pending"
        self.progress = 0
        self.message = "Upload queued"
        self.error: Optional[str] = None
        self.future: Optional[Future] = None


class HuggingfaceService:
    """Service for managing HuggingFace checkpoint uploads."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hf_upload_")
        self._tasks: Dict[str, UploadTask] = {}
        self._publishing_service = PublishingService()

    def _get_config_store(self) -> ConfigStore:
        """Return a ConfigStore using user defaults when available."""
        try:
            defaults = WebUIStateStore().load_defaults()
            if defaults.configs_dir:
                expanded_dir = Path(defaults.configs_dir).expanduser()
                return ConfigStore(config_dir=expanded_dir, config_type="model")
        except Exception:
            pass
        return ConfigStore(config_type="model")

    def _get_config(self, environment_id: str) -> Dict[str, Any]:
        """Load configuration for an environment."""
        store = self._get_config_store()
        try:
            config, _ = store.load_config(environment_id)
            return config
        except FileNotFoundError as exc:
            raise HuggingfaceServiceError(
                f"Environment '{environment_id}' not found",
                status.HTTP_404_NOT_FOUND,
            ) from exc

    def _validate_upload_config(self, config: Dict[str, Any], repo_id: str) -> None:
        """Validate that the configuration allows uploads."""
        # Check if push_to_hub is enabled
        push_to_hub = config.get("--push_to_hub") or config.get("push_to_hub", False)
        if not push_to_hub:
            raise HuggingfaceServiceError(
                "push_to_hub must be enabled in the configuration",
                status.HTTP_400_BAD_REQUEST,
            )

        # Check if we have a target repository
        if not repo_id:
            hub_model_id = config.get("--hub_model_id") or config.get("hub_model_id")
            if not hub_model_id:
                raise HuggingfaceServiceError(
                    "No target repository specified. Configure hub_model_id or provide repo_id",
                    status.HTTP_400_BAD_REQUEST,
                )

    def _validate_auth(self) -> None:
        """Validate HuggingFace authentication."""
        token_status = self._publishing_service.validate_token()
        if not token_status.get("valid"):
            raise HuggingfaceServiceError(
                "Not authenticated with HuggingFace. Please login on the Publishing tab.",
                status.HTTP_401_UNAUTHORIZED,
            )

    def upload_checkpoint(
        self,
        environment_id: str,
        checkpoint_name: str,
        repo_id: Optional[str] = None,
        branch: Optional[str] = None,
        subfolder: Optional[str] = None,
        callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload a single checkpoint to HuggingFace Hub.

        Args:
            environment_id: Environment (config) name
            checkpoint_name: Name of the checkpoint to upload
            repo_id: Target repository ID (overrides config)
            branch: Target branch (None for main)
            subfolder: Subfolder path in repo
            callback_url: Webhook URL for progress callbacks

        Returns:
            Dictionary with task information
        """
        # Validate authentication
        self._validate_auth()

        # Load config and validate
        config = self._get_config(environment_id)

        # Use provided repo_id or fall back to config
        if not repo_id:
            repo_id = config.get("--hub_model_id") or config.get("hub_model_id")

        self._validate_upload_config(config, repo_id)

        # Create task
        task_id = str(uuid4())
        task = UploadTask(
            task_id=task_id,
            environment_id=environment_id,
            checkpoint_name=checkpoint_name,
            repo_id=repo_id,
            branch=branch,
            subfolder=subfolder,
            callback_url=callback_url,
        )

        # Store task
        self._tasks[task_id] = task

        # Submit to executor
        future = self._executor.submit(self._upload_checkpoint_worker, task, config)
        task.future = future

        return {
            "task_id": task_id,
            "status": task.status,
            "message": task.message,
            "checkpoint": checkpoint_name,
            "repo_id": repo_id,
            "branch": branch,
            "subfolder": subfolder,
        }

    def upload_checkpoints(
        self,
        environment_id: str,
        checkpoint_names: List[str],
        repo_id: Optional[str] = None,
        upload_mode: str = "single_commit",
        callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload multiple checkpoints to HuggingFace Hub.

        Args:
            environment_id: Environment (config) name
            checkpoint_names: List of checkpoint names to upload
            repo_id: Target repository ID (overrides config)
            upload_mode: "single_commit" or "separate_branches"
            callback_url: Webhook URL for progress callbacks

        Returns:
            Dictionary with task information
        """
        # Validate authentication
        self._validate_auth()

        # Load config and validate
        config = self._get_config(environment_id)

        # Use provided repo_id or fall back to config
        if not repo_id:
            repo_id = config.get("--hub_model_id") or config.get("hub_model_id")

        self._validate_upload_config(config, repo_id)

        if upload_mode == "single_commit":
            # Create a single task for all checkpoints
            task_id = str(uuid4())
            task = UploadTask(
                task_id=task_id,
                environment_id=environment_id,
                checkpoint_name="batch_upload",  # Special name for batch
                repo_id=repo_id,
                branch=None,
                subfolder=None,
                callback_url=callback_url,
            )
            # Store checkpoint names in task for batch processing
            task.checkpoint_names = checkpoint_names
            self._tasks[task_id] = task

            # Submit batch upload task
            future = self._executor.submit(self._upload_batch_worker, task, config)
            task.future = future

            return {
                "tasks": [{
                    "task_id": task_id,
                    "checkpoints": checkpoint_names,
                    "mode": "batch"
                }],
                "count": 1,
                "repo_id": repo_id,
                "upload_mode": upload_mode,
            }

        else:  # separate_branches
            # Upload each checkpoint to its own branch
            tasks = []
            for checkpoint_name in checkpoint_names:
                task_id = str(uuid4())
                task = UploadTask(
                    task_id=task_id,
                    environment_id=environment_id,
                    checkpoint_name=checkpoint_name,
                    repo_id=repo_id,
                    branch=checkpoint_name,  # Use checkpoint name as branch
                    subfolder=None,  # No subfolder
                    callback_url=callback_url,
                )
                self._tasks[task_id] = task
                future = self._executor.submit(self._upload_checkpoint_worker, task, config)
                task.future = future
                tasks.append({
                    "task_id": task_id,
                    "checkpoint": checkpoint_name,
                    "branch": checkpoint_name,
                })

            return {
                "tasks": tasks,
                "count": len(tasks),
                "repo_id": repo_id,
                "upload_mode": upload_mode,
            }

    def _upload_batch_worker(self, task: UploadTask, config: Dict[str, Any]) -> None:
        """Worker function to upload multiple checkpoints in a single commit."""
        try:
            task.status = "running"
            task.message = f"Starting batch upload of {len(task.checkpoint_names)} checkpoints..."
            self._send_callback(task, "started")

            # Get output directory
            output_dir = config.get("--output_dir") or config.get("output_dir")
            if not output_dir:
                raise ValueError("No output_dir in configuration")

            # Check if this is a LoRA model
            is_lora = "lora" in config.get("--model_type", config.get("model_type", "")).lower()

            from huggingface_hub import HfApi, CommitOperationAdd, create_repo
            api = HfApi()

            # Ensure the repository exists
            try:
                api.repo_info(repo_id=task.repo_id, repo_type="model")
                logger.info(f"Repository {task.repo_id} already exists")
            except Exception:
                try:
                    logger.info(f"Creating repository {task.repo_id}")
                    create_repo(
                        repo_id=task.repo_id,
                        repo_type="model",
                        exist_ok=True,
                        private=config.get("--hub_private_repo", config.get("hub_private_repo", False))
                    )
                    task.message = f"Created repository {task.repo_id}"
                    self._send_callback(task, "progress")
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to create repository: {error_msg}")
                    # Check if it's a permission issue
                    if "403" in error_msg or "forbidden" in error_msg.lower():
                        raise Exception(f"Permission denied: Cannot create repository {task.repo_id}. Please check your HuggingFace token permissions.")
                    elif "401" in error_msg or "unauthorized" in error_msg.lower():
                        raise Exception(f"Authentication failed: Please check your HuggingFace token is valid.")
                    # For other errors, log but continue - the repo might already exist

            # Collect all operations
            operations = []
            total_checkpoints = len(task.checkpoint_names)

            for idx, checkpoint_name in enumerate(task.checkpoint_names):
                checkpoint_path = Path(output_dir).expanduser() / checkpoint_name
                if not checkpoint_path.exists():
                    logger.warning(f"Checkpoint not found: {checkpoint_path}")
                    continue

                # Update progress
                progress = int((idx / total_checkpoints) * 50)
                task.progress = progress
                task.message = f"Preparing {checkpoint_name}..."
                self._send_callback(task, "progress")

                # Add operations for this checkpoint
                if is_lora:
                    # LoRA files
                    lora_path = checkpoint_path / LORA_SAFETENSORS_FILENAME
                    if lora_path.exists():
                        operations.append(
                            CommitOperationAdd(
                                path_in_repo=f"{checkpoint_name}/{LORA_SAFETENSORS_FILENAME}",
                                path_or_fileobj=str(lora_path),
                            )
                        )

                    readme_path = checkpoint_path / "README.md"
                    if readme_path.exists():
                        operations.append(
                            CommitOperationAdd(
                                path_in_repo=f"{checkpoint_name}/README.md",
                                path_or_fileobj=str(readme_path),
                            )
                        )

                    # EMA folders
                    for ema_name in ["transformer_ema", "unet_ema", "controlnet_ema", "ema"]:
                        ema_path = checkpoint_path / ema_name
                        if ema_path.exists() and ema_path.is_dir():
                            for file_path in ema_path.rglob("*"):
                                if file_path.is_file():
                                    relative_path = file_path.relative_to(ema_path)
                                    operations.append(
                                        CommitOperationAdd(
                                            path_in_repo=f"{checkpoint_name}/ema/{relative_path}",
                                            path_or_fileobj=str(file_path),
                                        )
                                    )
                else:
                    # Full model - add all files
                    for file_path in checkpoint_path.rglob("*"):
                        if file_path.is_file():
                            relative_path = file_path.relative_to(checkpoint_path)
                            operations.append(
                                CommitOperationAdd(
                                    path_in_repo=f"{checkpoint_name}/{relative_path}",
                                    path_or_fileobj=str(file_path),
                                )
                            )

            if not operations:
                raise ValueError("No files found to upload")

            # Upload all files in a single commit
            task.progress = 50
            task.message = f"Uploading {len(operations)} files..."
            self._send_callback(task, "progress")

            api.create_commit(
                repo_id=task.repo_id,
                operations=operations,
                commit_message=f"Upload {total_checkpoints} checkpoints",
                parent_commit=None,  # Use latest commit
            )

            task.status = "completed"
            task.progress = 100
            task.message = f"Successfully uploaded {total_checkpoints} checkpoints"
            self._send_callback(task, "completed")

        except Exception as e:
            logger.error(f"Batch upload failed for task {task.task_id}: {str(e)}")
            task.status = "failed"
            task.error = str(e)
            task.message = f"Batch upload failed: {str(e)}"
            self._send_callback(task, "failed")

    def _upload_checkpoint_worker(self, task: UploadTask, config: Dict[str, Any]) -> None:
        """Worker function to upload a checkpoint."""
        try:
            task.status = "running"
            task.message = "Starting upload..."
            self._send_callback(task, "started")

            # Get checkpoint path
            output_dir = config.get("--output_dir") or config.get("output_dir")
            if not output_dir:
                raise ValueError("No output_dir in configuration")

            checkpoint_path = Path(output_dir).expanduser() / task.checkpoint_name
            if not checkpoint_path.exists():
                raise ValueError(f"Checkpoint not found: {checkpoint_path}")

            # Upload directly - subfolder handling is done in _do_upload
            self._do_upload(task, config, str(checkpoint_path))

            task.status = "completed"
            task.progress = 100
            task.message = "Upload completed successfully"
            self._send_callback(task, "completed")

        except Exception as e:
            logger.error(f"Upload failed for task {task.task_id}: {str(e)}")
            task.status = "failed"
            task.error = str(e)
            task.message = f"Upload failed: {str(e)}"
            self._send_callback(task, "failed")

    def _do_upload(self, task: UploadTask, config: Dict[str, Any], upload_path: str) -> None:
        """Perform the actual upload using HubManager."""
        # For manual checkpoint uploads, always use direct upload without model card
        # This avoids all the complexity of mocking the training environment
        from huggingface_hub import upload_folder, create_branch, create_repo, HfApi

        # First, ensure the repository exists
        api = HfApi()
        try:
            # Try to get repo info first
            api.repo_info(repo_id=task.repo_id, repo_type="model")
            logger.info(f"Repository {task.repo_id} already exists")
        except Exception:
            # Repository doesn't exist, create it
            try:
                logger.info(f"Creating repository {task.repo_id}")
                create_repo(
                    repo_id=task.repo_id,
                    repo_type="model",
                    exist_ok=True,
                    private=config.get("--hub_private_repo", config.get("hub_private_repo", False))
                )
                task.message = f"Created repository {task.repo_id}"
                self._send_callback(task, "progress")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to create repository: {error_msg}")
                # Check if it's a permission issue
                if "403" in error_msg or "forbidden" in error_msg.lower():
                    raise Exception(f"Permission denied: Cannot create repository {task.repo_id}. Please check your HuggingFace token permissions.")
                elif "401" in error_msg or "unauthorized" in error_msg.lower():
                    raise Exception(f"Authentication failed: Please check your HuggingFace token is valid.")
                # For other errors, log but continue - the repo might already exist

        # Check if this is a LoRA model
        is_lora = "lora" in config.get("--model_type", config.get("model_type", "")).lower()

        # For subfolder uploads with LoRA models, we need to handle them specially
        if task.subfolder and is_lora:
            # For LoRA models with subfolders, we upload files directly
            from huggingface_hub import HfApi, upload_file
            api = HfApi()

            task.progress = 25
            task.message = "Preparing LoRA upload..."
            self._send_callback(task, "progress")

            # Get the files to upload
            lora_weights_path = Path(upload_path) / LORA_SAFETENSORS_FILENAME
            readme_path = Path(upload_path) / "README.md"

            if not lora_weights_path.exists():
                raise FileNotFoundError(f"LoRA weights not found: {lora_weights_path}")

            task.progress = 50
            task.message = "Uploading LoRA weights..."
            self._send_callback(task, "progress")

            # Create branch if needed
            if task.branch:
                from huggingface_hub import create_branch
                try:
                    create_branch(
                        repo_id=task.repo_id,
                        branch=task.branch,
                        token=None,  # Use default token
                    )
                except Exception as e:
                    # Branch might already exist, which is fine
                    logger.info(f"Branch creation result: {str(e)}")

            # Upload to subfolder
            try:
                upload_file(
                    repo_id=task.repo_id,
                    path_in_repo=f"{task.subfolder}/{LORA_SAFETENSORS_FILENAME}",
                    path_or_fileobj=str(lora_weights_path),
                    commit_message=f"Upload {task.checkpoint_name} checkpoint",
                    revision=task.branch if task.branch else None,
                )

                if readme_path.exists():
                    upload_file(
                        repo_id=task.repo_id,
                        path_in_repo=f"{task.subfolder}/README.md",
                        path_or_fileobj=str(readme_path),
                        commit_message=f"Upload {task.checkpoint_name} model card",
                        revision=task.branch if task.branch else None,
                    )

                # Upload EMA if exists
                for ema_name in ["transformer_ema", "unet_ema", "controlnet_ema", "ema"]:
                    ema_path = Path(upload_path) / ema_name
                    if ema_path.exists() and ema_path.is_dir():
                        from huggingface_hub import upload_folder

                        task.progress = 75
                        task.message = "Uploading EMA weights..."
                        self._send_callback(task, "progress")

                        upload_folder(
                            repo_id=task.repo_id,
                            folder_path=str(ema_path),
                            path_in_repo=f"{task.subfolder}/ema",
                            commit_message=f"Upload {task.checkpoint_name} EMA weights",
                            revision=task.branch if task.branch else None,
                        )
                        break

            except Exception as e:
                raise Exception(f"Failed to upload LoRA model: {str(e)}")

        else:
            # For all other uploads, use direct upload_folder without model card generation
            # This avoids the complexity of mocking the entire training environment

            task.progress = 25
            task.message = "Preparing checkpoint upload..."
            self._send_callback(task, "progress")

            # Create branch if needed
            if task.branch:
                try:
                    create_branch(
                        repo_id=task.repo_id,
                        branch=task.branch,
                        token=None,  # Use default token
                    )
                except Exception as e:
                    # Branch might already exist, which is fine
                    logger.info(f"Branch creation result: {str(e)}")

            task.progress = 50
            task.message = "Uploading checkpoint files..."
            self._send_callback(task, "progress")

            # Determine the path in repo
            path_in_repo = "."
            if task.subfolder:
                path_in_repo = task.subfolder

            # Upload the entire checkpoint folder with progress capture
            try:
                logger.info(f"Uploading {upload_path} to {task.repo_id}/{path_in_repo}")

                # Monkey-patch tqdm to capture progress
                import tqdm
                original_tqdm = tqdm.tqdm
                parent_service = self
                captured_bars = []

                class ProgressCaptureTqdm(original_tqdm):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        captured_bars.append(self)
                        self.last_n = 0

                    def update(self, n=1):
                        result = super().update(n)

                        # Calculate and send progress
                        if self.total and self.total > 0:
                            progress_pct = self.n / self.total
                            # Map to our 50-90% range
                            task_progress = int(50 + (40 * progress_pct))

                            # Update task with progress
                            task.progress = task_progress
                            task.message = f"Uploading... ({self.n}/{self.total} files)"
                            parent_service._send_callback(task, "progress")

                        self.last_n = self.n
                        return result

                    def close(self):
                        super().close()
                        if self in captured_bars:
                            captured_bars.remove(self)

                # Replace tqdm temporarily
                tqdm.tqdm = ProgressCaptureTqdm
                if hasattr(tqdm, 'std'):
                    tqdm.std.tqdm = ProgressCaptureTqdm
                # Also patch auto module if it exists
                if hasattr(tqdm, 'auto'):
                    tqdm.auto.tqdm = ProgressCaptureTqdm

                try:
                    # Count files for initial message
                    from pathlib import Path
                    upload_path_obj = Path(upload_path)
                    file_count = sum(1 for f in upload_path_obj.rglob("*") if f.is_file() and not any(f.match(p) for p in ["*.pyc", "__pycache__", "*.tmp", "*.log", ".git"]))
                    total_size = sum(f.stat().st_size for f in upload_path_obj.rglob("*") if f.is_file() and not any(f.match(p) for p in ["*.pyc", "__pycache__", "*.tmp", "*.log", ".git"]))

                    task.progress = 50
                    task.message = f"Starting upload of {file_count} files ({self._format_size(total_size)})..."
                    self._send_callback(task, "progress")

                    # Start the upload
                    upload_folder(
                        repo_id=task.repo_id,
                        folder_path=upload_path,
                        path_in_repo=path_in_repo,
                        commit_message=f"Upload {task.checkpoint_name} checkpoint",
                        revision=task.branch if task.branch else None,
                        ignore_patterns=["*.pyc", "__pycache__", "*.tmp", "*.log", ".git"]
                    )

                finally:
                    # Restore original tqdm
                    tqdm.tqdm = original_tqdm
                    if hasattr(tqdm, 'std'):
                        tqdm.std.tqdm = original_tqdm
                    if hasattr(tqdm, 'auto'):
                        tqdm.auto.tqdm = original_tqdm

            except Exception as e:
                raise Exception(f"Failed to upload checkpoint: {str(e)}")

        task.progress = 90
        task.message = "Finalizing upload..."
        self._send_callback(task, "progress")

    def _send_callback(self, task: UploadTask, event: str) -> None:
        """Send webhook callback for task progress."""
        if not task.callback_url:
            return

        try:
            import requests

            data = {
                "message_type": "checkpoint_upload",
                "task_id": task.task_id,
                "event": event,
                "status": task.status,
                "progress": task.progress,
                "message": task.message,
                "checkpoint": task.checkpoint_name,
                "repo_id": task.repo_id,
                "environment": task.environment_id,
            }

            if task.error:
                data["error"] = task.error

            if task.branch:
                data["branch"] = task.branch
            if task.subfolder:
                data["subfolder"] = task.subfolder

            # Send callback synchronously in thread pool
            # Since we're already in a thread executor, we can make the request directly
            try:
                response = requests.post(task.callback_url, json=data, timeout=5)
                if response.status_code >= 400:
                    logger.warning(f"Callback failed with status {response.status_code}")
            except requests.exceptions.Timeout:
                logger.warning("Callback request timed out")
            except Exception as e:
                logger.error(f"Callback request failed: {str(e)}")

        except Exception as e:
            logger.error(f"Failed to send callback: {str(e)}")


    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of an upload task."""
        task = self._tasks.get(task_id)
        if not task:
            raise HuggingfaceServiceError(
                f"Task '{task_id}' not found",
                status.HTTP_404_NOT_FOUND,
            )

        return {
            "task_id": task.task_id,
            "status": task.status,
            "progress": task.progress,
            "message": task.message,
            "error": task.error,
            "checkpoint": task.checkpoint_name,
            "repo_id": task.repo_id,
            "branch": task.branch,
            "subfolder": task.subfolder,
        }

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel an upload task."""
        task = self._tasks.get(task_id)
        if not task:
            raise HuggingfaceServiceError(
                f"Task '{task_id}' not found",
                status.HTTP_404_NOT_FOUND,
            )

        if task.future and not task.future.done():
            task.future.cancel()
            task.status = "cancelled"
            task.message = "Upload cancelled"
            self._send_callback(task, "cancelled")

        return {"task_id": task_id, "status": "cancelled"}

    def cleanup_completed_tasks(self) -> Dict[str, Any]:
        """Remove completed tasks from memory."""
        removed = 0
        for task_id in list(self._tasks.keys()):
            task = self._tasks[task_id]
            if task.status in ["completed", "failed", "cancelled"]:
                del self._tasks[task_id]
                removed += 1

        return {"removed": removed}

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"


# Singleton instance
HUGGINGFACE_SERVICE = HuggingfaceService()