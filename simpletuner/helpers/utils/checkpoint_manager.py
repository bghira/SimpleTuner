"""Checkpoint management utilities for SimpleTuner."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"


class CheckpointManager:
    """Manages checkpoint operations including listing, validation, and cleanup."""

    def __init__(self, output_dir: str):
        """Initialize CheckpointManager with output directory.

        Args:
            output_dir: Directory where checkpoints are stored
        """
        self.output_dir = output_dir

    def list_checkpoints(self, include_metadata: bool = True) -> List[Dict[str, any]]:
        """List all available checkpoints in the output directory.

        Args:
            include_metadata: Whether to include checkpoint metadata

        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []

        if not os.path.exists(self.output_dir):
            return checkpoints

        dirs = os.listdir(self.output_dir)
        checkpoint_dirs = [d for d in dirs if d.startswith("checkpoint-") and not d.endswith("-tmp")]

        for checkpoint_dir in checkpoint_dirs:
            checkpoint_path = os.path.join(self.output_dir, checkpoint_dir)
            info = {
                "name": checkpoint_dir,
                "path": checkpoint_path,
                "step": self._extract_step_from_name(checkpoint_dir),
            }

            if include_metadata:
                metadata = self._load_checkpoint_metadata(checkpoint_path)
                if metadata:
                    info.update(metadata)

            checkpoints.append(info)

        # Sort by step number (descending)
        checkpoints.sort(key=lambda x: x.get("step", 0), reverse=True)
        return checkpoints

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint.

        Returns:
            Checkpoint directory name or None if no checkpoints exist
        """
        try:
            dirs = os.listdir(self.output_dir) if os.path.exists(self.output_dir) else []
            checkpoint_dirs = [d for d in dirs if d.startswith("checkpoint") and not d.endswith("tmp")]
            checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
            return checkpoint_dirs[-1] if checkpoint_dirs else None
        except Exception as e:
            logger.debug(f"Error getting latest checkpoint: {e}")
            return None

    def validate_checkpoint(self, checkpoint_name: str) -> Tuple[bool, Optional[str]]:
        """Validate a checkpoint for resuming training.

        Args:
            checkpoint_name: Name of the checkpoint (e.g., "checkpoint-1000")

        Returns:
            Tuple of (is_valid, error_message)
        """
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            return False, f"Checkpoint '{checkpoint_name}' does not exist"

        # Check for required files
        required_files = ["training_state.json"]
        missing_files = []

        for file in required_files:
            file_path = os.path.join(checkpoint_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)

        if missing_files:
            return False, f"Checkpoint missing required files: {', '.join(missing_files)}"

        # Ensure at least one set of weights exists (full-model or adapter)
        weight_candidates = [
            "pytorch_model.bin",
            "model.safetensors",
            "diffusion_pytorch_model.safetensors",
            "pytorch_lora_weights.safetensors",
            "adapter_model.safetensors",
        ]

        if not any(os.path.exists(os.path.join(checkpoint_path, candidate)) for candidate in weight_candidates):
            return False, "Checkpoint is missing model or adapter weight files"

        # Validate training state
        try:
            training_state_path = os.path.join(checkpoint_path, "training_state.json")
            with open(training_state_path, "r") as f:
                training_state = json.load(f)

            if "global_step" not in training_state:
                return False, "Invalid training state: missing global_step"

        except (json.JSONDecodeError, IOError) as e:
            return False, f"Failed to read training state: {str(e)}"

        return True, None

    def write_manifest(self, checkpoint_path: str, metadata: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """Write a manifest file listing checkpoint contents for remote resume."""
        root = Path(checkpoint_path)
        if not root.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
        files = [
            str(path.relative_to(root).as_posix())
            for path in root.rglob("*")
            if path.is_file() and path.name != CHECKPOINT_MANIFEST_FILENAME
        ]
        files.sort()
        manifest: Dict[str, object] = {
            "version": 1,
            "checkpoint": root.name,
            "files": files,
            "metadata": metadata or {},
        }
        manifest_path = root / CHECKPOINT_MANIFEST_FILENAME
        with manifest_path.open("w") as handle:
            json.dump(manifest, handle, indent=2)
        return manifest

    def load_manifest(self, checkpoint_path: str) -> Optional[Dict[str, object]]:
        """Load a checkpoint manifest if present."""
        manifest_path = Path(checkpoint_path) / CHECKPOINT_MANIFEST_FILENAME
        if not manifest_path.exists():
            return None
        with manifest_path.open("r") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return None
        return data

    def cleanup_checkpoints(self, limit: int, suffix: Optional[str] = None):
        """Clean up old checkpoints, keeping only the most recent ones.

        Args:
            limit: Maximum number of checkpoints to keep
            suffix: Optional suffix to filter checkpoints
        """
        # Remove temp checkpoints first
        self._remove_temp_checkpoints()

        # Get filtered checkpoints
        checkpoints = self._filter_checkpoints(suffix)
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # Remove old checkpoints if we exceed the limit
        if len(checkpoints) > limit:
            num_to_remove = len(checkpoints) - limit
            removing_checkpoints = checkpoints[:num_to_remove]

            logger.debug(f"{len(checkpoints)} checkpoints exist, removing {len(removing_checkpoints)} checkpoints")
            logger.debug(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

            for checkpoint in removing_checkpoints:
                self.remove_checkpoint(checkpoint)

    def remove_checkpoint(self, checkpoint_name: str):
        """Remove a specific checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint to remove
        """
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        try:
            logger.debug(f"Removing {checkpoint_path}")
            shutil.rmtree(checkpoint_path, ignore_errors=True)
        except Exception as e:
            logger.error(f"Failed to remove directory: {checkpoint_path}")
            logger.error(str(e))

    def _filter_checkpoints(self, suffix: Optional[str] = None) -> List[str]:
        """Filter checkpoints by suffix.

        Args:
            suffix: Optional suffix to filter by

        Returns:
            List of checkpoint names
        """
        checkpoints_keep = []
        checkpoints = os.listdir(self.output_dir) if os.path.exists(self.output_dir) else []

        for checkpoint in checkpoints:
            parts = checkpoint.split("-")
            base = parts[0]
            checkpoint_suffix = None

            if len(parts) < 2:
                continue
            elif len(parts) > 2:
                checkpoint_suffix = parts[2]

            if base != "checkpoint":
                continue
            if suffix and checkpoint_suffix and suffix != checkpoint_suffix:
                continue
            if (suffix and not checkpoint_suffix) or (checkpoint_suffix and not suffix):
                continue

            checkpoints_keep.append(checkpoint)

        return checkpoints_keep

    def _remove_temp_checkpoints(self):
        """Remove any temporary checkpoints."""
        temp_checkpoints = self._filter_checkpoints("tmp")
        for checkpoint in temp_checkpoints:
            self.remove_checkpoint(checkpoint)

    def _extract_step_from_name(self, checkpoint_name: str) -> int:
        """Extract step number from checkpoint name.

        Args:
            checkpoint_name: Name like "checkpoint-1000" or "checkpoint-1000-rolling"

        Returns:
            Step number or 0 if extraction fails
        """
        try:
            parts = checkpoint_name.split("-")
            if len(parts) >= 2:
                return int(parts[1])
        except (ValueError, IndexError):
            pass
        return 0

    def _load_checkpoint_metadata(self, checkpoint_path: str) -> Optional[Dict[str, any]]:
        """Load metadata from checkpoint training state.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary with metadata or None
        """
        training_state_path = os.path.join(checkpoint_path, "training_state.json")

        if not os.path.exists(training_state_path):
            return None

        try:
            with open(training_state_path, "r") as f:
                state = json.load(f)

            return {
                "global_step": state.get("global_step", 0),
                "epoch": state.get("epoch", 0),
                "loss": state.get("loss", 0.0),
                "learning_rate": state.get("learning_rate", 0.0),
                "timestamp": os.path.getmtime(training_state_path),
            }
        except (json.JSONDecodeError, IOError):
            return None
