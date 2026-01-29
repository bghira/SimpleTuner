"""Service helpers for checkpoint management operations."""

from __future__ import annotations

import base64
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
        self._preview_cache: Dict[str, Dict[str, Any]] = {}

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

            # Always calculate size for each checkpoint
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

            # Apply sorting
            if sort_by == "step-asc":
                checkpoints.sort(key=lambda x: x.get("step", 0), reverse=False)
            elif sort_by == "step-desc":
                checkpoints.sort(key=lambda x: x.get("step", 0), reverse=True)
            elif sort_by == "size-desc":
                checkpoints.sort(key=lambda x: x.get("size_bytes", 0), reverse=True)

            previews = [self._build_checkpoint_preview(environment_id, manager, ckpt) for ckpt in checkpoints]

            return {
                "environment": environment_id,
                "checkpoints": previews,
                "count": len(previews),
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
            if len(checkpoints) > limit:
                num_to_remove = len(checkpoints) - limit
                to_remove = checkpoints[:num_to_remove]

            removal_previews = [self._build_checkpoint_preview(environment_id, manager, ckpt) for ckpt in to_remove]

            return {
                "environment": environment_id,
                "limit": limit,
                "total_checkpoints": len(checkpoints),
                "checkpoints_to_remove": removal_previews,
                "count_to_remove": len(removal_previews),
                "checkpoints_to_keep": len(checkpoints) - len(removal_previews),
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

            for removed in to_remove:
                cache_key = f"{environment_id}:{removed.get('name')}"
                self._preview_cache.pop(cache_key, None)

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

    def delete_checkpoint(self, environment_id: str, checkpoint_name: str) -> Dict[str, Any]:
        """Remove a specific checkpoint directory."""
        try:
            manager = self._get_checkpoint_manager(environment_id)
            checkpoint_path = os.path.join(manager.output_dir, checkpoint_name)

            if not os.path.exists(checkpoint_path):
                raise CheckpointsServiceError(
                    f"Checkpoint '{checkpoint_name}' not found",
                    status.HTTP_404_NOT_FOUND,
                )

            manager.remove_checkpoint(checkpoint_name)

            return {
                "environment": environment_id,
                "checkpoint": checkpoint_name,
                "message": f"Checkpoint '{checkpoint_name}' deleted",
            }

        except CheckpointsServiceError:
            raise
        except Exception as e:
            raise CheckpointsServiceError(
                f"Failed to delete checkpoint: {str(e)}",
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

    def _build_checkpoint_preview(
        self,
        environment_id: str,
        manager: CheckpointManager,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        name = checkpoint.get("name")
        checkpoint_path = checkpoint.get("path")
        cache_key = f"{environment_id}:{name}"
        mtime = None
        assets_mtime = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                mtime = os.path.getmtime(checkpoint_path)
                assets_dir = os.path.join(checkpoint_path, "assets")
                if os.path.exists(assets_dir):
                    assets_mtime = os.path.getmtime(assets_dir)
            except OSError:
                mtime = None

        cached = self._preview_cache.get(cache_key)
        if cached and cached.get("mtime") == mtime and cached.get("assets_mtime") == assets_mtime:
            cached_data = cached.get("data")
            if cached_data:
                return dict(cached_data)

        preview = dict(checkpoint)

        is_valid = False
        error_message: Optional[str] = None
        if name:
            try:
                is_valid, error_message = manager.validate_checkpoint(name)
            except Exception:
                is_valid = False
                error_message = "Failed to validate checkpoint"

        if checkpoint_path:
            preview.setdefault("files", self._list_checkpoint_files(checkpoint_path))

        assets = self._load_checkpoint_assets(checkpoint_path)
        asset_map = {asset["name"]: asset.get("data") for asset in assets if asset.get("data")}

        readme = self._load_checkpoint_readme(checkpoint_path, asset_map)

        preview["validation"] = {
            "valid": is_valid,
            "error": error_message,
        }
        preview["assets"] = assets
        if readme is not None:
            preview["readme"] = readme
        if mtime is not None:
            preview["modified_at"] = mtime

        # Ensure size field is available (alias for size_bytes)
        if "size_bytes" in preview:
            preview["size"] = preview["size_bytes"]

        self._preview_cache[cache_key] = {"mtime": mtime, "assets_mtime": assets_mtime, "data": dict(preview)}

        if len(self._preview_cache) > 128:
            excess = len(self._preview_cache) - 128
            for key in list(self._preview_cache.keys())[:excess]:
                self._preview_cache.pop(key, None)

        return preview

    def _load_checkpoint_assets(self, checkpoint_path: Optional[str], limit: int = 4) -> List[Dict[str, Any]]:
        assets: List[Dict[str, Any]] = []
        if not checkpoint_path:
            return assets

        assets_dir = Path(checkpoint_path) / "assets"
        if not assets_dir.exists() or not assets_dir.is_dir():
            return assets

        supported_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".avi", ".mov", ".webm"}
        candidates = sorted(
            (path for path in assets_dir.iterdir() if path.suffix.lower() in supported_suffixes),
            key=lambda p: p.name,
        )

        for asset_path in candidates[:limit]:
            encoded = self._encode_asset(asset_path)
            if encoded:
                assets.append(encoded)

        return assets

    def _encode_asset(self, asset_path: Path) -> Optional[Dict[str, Any]]:
        try:
            data = asset_path.read_bytes()
            mime = self._guess_mime(asset_path.suffix)
            encoded = base64.b64encode(data).decode("utf-8")
            return {
                "name": asset_path.name,
                "mime": mime,
                "data": f"data:{mime};base64,{encoded}",
                "size": asset_path.stat().st_size,
            }
        except Exception:
            return None

    @staticmethod
    def _guess_mime(suffix: str) -> str:
        suffix = suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        if suffix == ".gif":
            return "image/gif"
        if suffix == ".mp4":
            return "video/mp4"
        if suffix == ".avi":
            return "video/x-msvideo"
        if suffix == ".mov":
            return "video/quicktime"
        if suffix == ".webm":
            return "video/webm"
        return "image/png"

    def _load_checkpoint_readme(
        self,
        checkpoint_path: Optional[str],
        asset_map: Dict[str, Optional[str]],
    ) -> Optional[Dict[str, Any]]:
        if not checkpoint_path:
            return None

        readme_path = Path(checkpoint_path) / "README.md"
        if not readme_path.exists():
            return None

        try:
            raw_content = readme_path.read_text(encoding="utf-8")
        except Exception:
            return None

        front_matter, body = self._split_front_matter(raw_content)
        if body and asset_map:
            body = self._replace_asset_paths(body, asset_map)

        return {
            "front_matter": front_matter,
            "body": body,
        }

    @staticmethod
    def _split_front_matter(content: str) -> Tuple[Optional[str], str]:
        if not content:
            return None, ""

        stripped = content.lstrip()
        if not stripped.startswith("---"):
            return None, content

        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return None, content

        closing_index = None
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                closing_index = idx
                break

        if closing_index is None:
            return None, content

        front_matter_lines = lines[1:closing_index]
        body_lines = lines[closing_index + 1 :]

        front_matter = "\n".join(front_matter_lines).strip() or None
        body = "\n".join(body_lines).lstrip("\n")

        return front_matter, body

    @staticmethod
    def _replace_asset_paths(text: str, asset_map: Dict[str, Optional[str]]) -> str:
        result = text
        for name, data_uri in asset_map.items():
            if not data_uri:
                continue
            result = result.replace(f"(assets/{name})", f"({data_uri})")
            result = result.replace(f"(./assets/{name})", f"({data_uri})")
            result = result.replace(f"./assets/{name}", data_uri)
            result = result.replace(f"assets/{name}", data_uri)
        return result

    def _list_checkpoint_files(self, checkpoint_path: Optional[str], limit: int = 25) -> List[str]:
        if not checkpoint_path:
            return []

        path_obj = Path(checkpoint_path)
        if not path_obj.exists() or not path_obj.is_dir():
            return []

        files: List[str] = []
        for file_path in sorted(path_obj.iterdir(), key=lambda p: p.name):
            if file_path.is_file():
                files.append(file_path.name)
            if len(files) >= limit:
                break

        return files

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
