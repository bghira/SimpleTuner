"""Service helpers for checkpoint management operations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import status

from simpletuner.helpers.utils.checkpoint_manager import CheckpointManager
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore


class CheckpointsServiceError(Exception):
    """Domain error raised when checkpoint service operations fail."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class CheckpointsService:
    """Coordinator for checkpoint-related operations."""

    def __init__(self) -> None:
        self._managers: Dict[str, CheckpointManager] = {}

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

    def _get_output_dir(self, environment_id: str) -> str:
        """
        Get the output directory for a given environment.

        Args:
            environment_id: The environment (config) name.

        Returns:
            The output directory path.

        Raises:
            CheckpointsServiceError: If environment not found or output_dir not configured.
        """
        store = self._get_config_store()

        try:
            config, metadata = store.load_config(environment_id)
        except FileNotFoundError as exc:
            raise CheckpointsServiceError(
                f"Environment '{environment_id}' not found",
                status.HTTP_404_NOT_FOUND,
            ) from exc

        # Check for output_dir in config
        output_dir = config.get("--output_dir") or config.get("output_dir")

        if not output_dir:
            raise CheckpointsServiceError(
                f"Environment '{environment_id}' does not have an output_dir configured",
                status.HTTP_400_BAD_REQUEST,
            )

        # Expand user paths and resolve
        expanded_dir = os.path.expanduser(str(output_dir))
        return expanded_dir

    def _get_checkpoint_manager(self, environment_id: str) -> CheckpointManager:
        """
        Get or create a CheckpointManager for an environment.

        Args:
            environment_id: The environment (config) name.

        Returns:
            CheckpointManager instance.
        """
        if environment_id not in self._managers:
            output_dir = self._get_output_dir(environment_id)
            self._managers[environment_id] = CheckpointManager(output_dir)

        return self._managers[environment_id]

    def list_checkpoints(self, environment_id: str, sort_by: str = "step-desc") -> Dict[str, Any]:
        """
        List checkpoints from environment's output directory.

        Args:
            environment_id: The environment (config) name.
            sort_by: Sort order - one of "step-desc", "step-asc", "size-desc".

        Returns:
            Dictionary with checkpoint list and metadata.
        """
        try:
            manager = self._get_checkpoint_manager(environment_id)
            checkpoints = manager.list_checkpoints(include_metadata=True)

            # Apply sorting
            if sort_by == "step-asc":
                checkpoints.sort(key=lambda x: x.get("step", 0), reverse=False)
            elif sort_by == "step-desc":
                checkpoints.sort(key=lambda x: x.get("step", 0), reverse=True)
            elif sort_by == "size-desc":
                # Calculate size for each checkpoint
                for ckpt in checkpoints:
                    ckpt_path = ckpt.get("path")
                    if ckpt_path and os.path.exists(ckpt_path):
                        try:
                            size = sum(
                                os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, dirnames, filenames in os.walk(ckpt_path)
                                for filename in filenames
                            )
                            ckpt["size_bytes"] = size
                        except Exception:
                            ckpt["size_bytes"] = 0
                    else:
                        ckpt["size_bytes"] = 0
                checkpoints.sort(key=lambda x: x.get("size_bytes", 0), reverse=True)

            return {
                "environment": environment_id,
                "checkpoints": checkpoints,
                "count": len(checkpoints),
                "sort_by": sort_by,
            }

        except CheckpointsServiceError:
            raise
        except Exception as e:
            raise CheckpointsServiceError(
                f"Failed to list checkpoints: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e

    def validate_checkpoint(self, environment_id: str, checkpoint_name: str) -> Dict[str, Any]:
        """
        Validate a checkpoint for resuming training.

        Simplified validation: check if has .safetensors OR pytorch_model.bin + training_state.json.

        Args:
            environment_id: The environment (config) name.
            checkpoint_name: Name of the checkpoint (e.g., "checkpoint-1000").

        Returns:
            Dictionary with validation results.
        """
        try:
            manager = self._get_checkpoint_manager(environment_id)
            is_valid, error_message = manager.validate_checkpoint(checkpoint_name)

            result = {
                "environment": environment_id,
                "checkpoint": checkpoint_name,
                "valid": is_valid,
            }

            if error_message:
                result["error"] = error_message
            else:
                result["message"] = "Checkpoint is valid for resuming training"

            return result

        except CheckpointsServiceError:
            raise
        except Exception as e:
            raise CheckpointsServiceError(
                f"Failed to validate checkpoint: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e

    def preview_cleanup(self, environment_id: str, limit: int) -> Dict[str, Any]:
        """
        Preview what checkpoints would be deleted by cleanup operation.

        Args:
            environment_id: The environment (config) name.
            limit: Maximum number of checkpoints to keep.

        Returns:
            Dictionary with checkpoints that would be removed.
        """
        try:
            manager = self._get_checkpoint_manager(environment_id)
            checkpoints = manager.list_checkpoints(include_metadata=False)

            # Sort by step number
            checkpoints.sort(key=lambda x: x.get("step", 0), reverse=False)

            # Determine which checkpoints would be removed
            to_remove = []
            if len(checkpoints) >= limit:
                num_to_remove = len(checkpoints) - limit + 1
                to_remove = checkpoints[:num_to_remove]

            return {
                "environment": environment_id,
                "limit": limit,
                "total_checkpoints": len(checkpoints),
                "checkpoints_to_remove": to_remove,
                "count_to_remove": len(to_remove),
                "checkpoints_to_keep": len(checkpoints) - len(to_remove),
            }

        except CheckpointsServiceError:
            raise
        except Exception as e:
            raise CheckpointsServiceError(
                f"Failed to preview cleanup: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e

    def execute_cleanup(self, environment_id: str, limit: int) -> Dict[str, Any]:
        """
        Execute cleanup operation to remove old checkpoints.

        Args:
            environment_id: The environment (config) name.
            limit: Maximum number of checkpoints to keep.

        Returns:
            Dictionary with cleanup results.
        """
        try:
            manager = self._get_checkpoint_manager(environment_id)

            # Get preview first
            preview = self.preview_cleanup(environment_id, limit)
            to_remove = preview["checkpoints_to_remove"]

            # Execute cleanup
            manager.cleanup_checkpoints(limit=limit)

            return {
                "environment": environment_id,
                "limit": limit,
                "removed_checkpoints": to_remove,
                "count_removed": len(to_remove),
                "message": f"Successfully removed {len(to_remove)} checkpoint(s)",
            }

        except CheckpointsServiceError:
            raise
        except Exception as e:
            raise CheckpointsServiceError(
                f"Failed to execute cleanup: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e

    def get_checkpoints_for_resume(self, environment_id: str) -> Dict[str, Any]:
        """
        Get checkpoints formatted for resume dropdown.

        Returns simplified list with just name and step for UI dropdown.

        Args:
            environment_id: The environment (config) name.

        Returns:
            Dictionary with simplified checkpoint list.
        """
        try:
            manager = self._get_checkpoint_manager(environment_id)
            checkpoints = manager.list_checkpoints(include_metadata=False)

            # Sort by step number (descending - most recent first)
            checkpoints.sort(key=lambda x: x.get("step", 0), reverse=True)

            # Simplify for dropdown
            simplified = [
                {
                    "name": ckpt["name"],
                    "step": ckpt["step"],
                    "label": f"{ckpt['name']} (step {ckpt['step']})",
                }
                for ckpt in checkpoints
            ]

            return {
                "environment": environment_id,
                "checkpoints": simplified,
                "count": len(simplified),
            }

        except CheckpointsServiceError:
            raise
        except Exception as e:
            raise CheckpointsServiceError(
                f"Failed to get checkpoints for resume: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e

    def get_retention_config(self, environment_id: str) -> Dict[str, Any]:
        """
        Get checkpoint retention configuration for an environment.

        Args:
            environment_id: The environment (config) name.

        Returns:
            Dictionary with retention configuration.
        """
        try:
            store = self._get_config_store()
            config, metadata = store.load_config(environment_id)

            # Get retention limit from config, default to 10
            retention_limit = config.get("--checkpoints_total_limit") or config.get("checkpoints_total_limit") or 10

            return {
                "environment": environment_id,
                "retention_limit": int(retention_limit),
            }

        except FileNotFoundError as exc:
            raise CheckpointsServiceError(
                f"Environment '{environment_id}' not found",
                status.HTTP_404_NOT_FOUND,
            ) from exc
        except Exception as e:
            raise CheckpointsServiceError(
                f"Failed to get retention config: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e

    def update_retention_config(self, environment_id: str, retention_limit: int) -> Dict[str, Any]:
        """
        Update checkpoint retention configuration for an environment.

        Args:
            environment_id: The environment (config) name.
            retention_limit: Maximum number of checkpoints to keep.

        Returns:
            Dictionary with updated retention configuration.
        """
        if retention_limit < 1:
            raise CheckpointsServiceError(
                "Retention limit must be at least 1",
                status.HTTP_400_BAD_REQUEST,
            )

        try:
            store = self._get_config_store()
            config, metadata = store.load_config(environment_id)

            # Update the config
            config["--checkpoints_total_limit"] = retention_limit

            # Save back to store
            store.save_config(environment_id, config, metadata)

            return {
                "environment": environment_id,
                "retention_limit": retention_limit,
                "message": "Retention configuration updated successfully",
            }

        except FileNotFoundError as exc:
            raise CheckpointsServiceError(
                f"Environment '{environment_id}' not found",
                status.HTTP_404_NOT_FOUND,
            ) from exc
        except Exception as e:
            raise CheckpointsServiceError(
                f"Failed to update retention config: {str(e)}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from e


# Singleton instance used by routes
CHECKPOINTS_SERVICE = CheckpointsService()
