import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

from simpletuner.simpletuner_sdk.server.services.checkpoints_service import CheckpointsService, CheckpointsServiceError


class TestCheckpointsService(unittest.TestCase):
    """Test cases for CheckpointsService."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.service = CheckpointsService()
        self.environment_id = "test-env"
        self.output_dir = "/fake/output/dir"

        # Mock ConfigStore and WebUIStateStore
        self.config_store_patcher = patch("simpletuner.simpletuner_sdk.server.services.checkpoints_service.ConfigStore")
        self.webui_state_patcher = patch("simpletuner.simpletuner_sdk.server.services.checkpoints_service.WebUIStateStore")

        self.mock_config_store_class = self.config_store_patcher.start()
        self.mock_webui_state_class = self.webui_state_patcher.start()

        # Set up default mocks
        mock_defaults = MagicMock()
        mock_defaults.configs_dir = None
        self.mock_webui_state_class.return_value.load_defaults.return_value = mock_defaults

        self.mock_config_store = MagicMock()
        self.mock_config_store_class.return_value = self.mock_config_store

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.config_store_patcher.stop()
        self.webui_state_patcher.stop()

    def _setup_config_mock(self, output_dir: str = None) -> None:
        """Helper to set up config store mock with output_dir.

        Args:
            output_dir: Output directory to return, defaults to self.output_dir
        """
        if output_dir is None:
            output_dir = self.output_dir

        config = {"--output_dir": output_dir}
        metadata = {"name": self.environment_id}
        self.mock_config_store.load_config.return_value = (config, metadata)

    def _create_mock_checkpoint_manager(self) -> MagicMock:
        """Helper to create a mock CheckpointManager.

        Returns:
            Mock CheckpointManager instance
        """
        mock_manager = MagicMock()
        return mock_manager

    def test_list_checkpoints(self) -> None:
        """Test listing checkpoints from output_dir."""
        self._setup_config_mock()

        mock_checkpoints = [
            {"name": "checkpoint-1000", "step": 1000, "path": "/fake/output/dir/checkpoint-1000"},
            {"name": "checkpoint-2000", "step": 2000, "path": "/fake/output/dir/checkpoint-2000"},
            {"name": "checkpoint-500", "step": 500, "path": "/fake/output/dir/checkpoint-500"},
        ]

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = mock_checkpoints.copy()
            mock_manager_class.return_value = mock_manager

            result = self.service.list_checkpoints(self.environment_id)

            self.assertEqual(result["environment"], self.environment_id)
            self.assertEqual(result["count"], 3)
            self.assertEqual(result["sort_by"], "step-desc")

            # Should be sorted by step descending
            checkpoints = result["checkpoints"]
            self.assertEqual(checkpoints[0]["step"], 2000)
            self.assertEqual(checkpoints[1]["step"], 1000)
            self.assertEqual(checkpoints[2]["step"], 500)

            mock_manager.list_checkpoints.assert_called_once_with(include_metadata=True)

    def test_list_checkpoints_with_sorting_step_asc(self) -> None:
        """Test listing checkpoints with step-asc sorting."""
        self._setup_config_mock()

        mock_checkpoints = [
            {"name": "checkpoint-1000", "step": 1000},
            {"name": "checkpoint-2000", "step": 2000},
            {"name": "checkpoint-500", "step": 500},
        ]

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = mock_checkpoints.copy()
            mock_manager_class.return_value = mock_manager

            result = self.service.list_checkpoints(self.environment_id, sort_by="step-asc")

            checkpoints = result["checkpoints"]
            self.assertEqual(checkpoints[0]["step"], 500)
            self.assertEqual(checkpoints[1]["step"], 1000)
            self.assertEqual(checkpoints[2]["step"], 2000)
            self.assertEqual(result["sort_by"], "step-asc")

    def test_list_checkpoints_with_sorting_size_desc(self) -> None:
        """Test listing checkpoints with size-desc sorting."""
        self._setup_config_mock()

        mock_checkpoints = [
            {"name": "checkpoint-1000", "step": 1000, "path": "/fake/output/dir/checkpoint-1000"},
            {"name": "checkpoint-2000", "step": 2000, "path": "/fake/output/dir/checkpoint-2000"},
            {"name": "checkpoint-500", "step": 500, "path": "/fake/output/dir/checkpoint-500"},
        ]

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = mock_checkpoints.copy()
            mock_manager_class.return_value = mock_manager

            # Mock os.walk to return different sizes for each checkpoint
            def mock_walk_side_effect(path):
                if "checkpoint-1000" in path:
                    return [("/fake/path", [], ["file1.bin"])]
                elif "checkpoint-2000" in path:
                    return [("/fake/path", [], ["file1.bin", "file2.bin", "file3.bin"])]
                elif "checkpoint-500" in path:
                    return [("/fake/path", [], ["file1.bin", "file2.bin"])]
                return []

            with (
                patch("os.walk", side_effect=mock_walk_side_effect),
                patch("os.path.exists", return_value=True),
                patch("os.path.getsize", return_value=1000),
            ):
                result = self.service.list_checkpoints(self.environment_id, sort_by="size-desc")

                checkpoints = result["checkpoints"]
                # checkpoint-2000 should be first (3 files = 3000 bytes)
                # checkpoint-500 should be second (2 files = 2000 bytes)
                # checkpoint-1000 should be last (1 file = 1000 bytes)
                self.assertEqual(checkpoints[0]["name"], "checkpoint-2000")
                self.assertEqual(checkpoints[0]["size_bytes"], 3000)
                self.assertEqual(checkpoints[1]["name"], "checkpoint-500")
                self.assertEqual(checkpoints[1]["size_bytes"], 2000)
                self.assertEqual(checkpoints[2]["name"], "checkpoint-1000")
                self.assertEqual(checkpoints[2]["size_bytes"], 1000)

    def test_list_checkpoints_empty_dir(self) -> None:
        """Test when no checkpoints exist."""
        self._setup_config_mock()

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = []
            mock_manager_class.return_value = mock_manager

            result = self.service.list_checkpoints(self.environment_id)

            self.assertEqual(result["environment"], self.environment_id)
            self.assertEqual(result["count"], 0)
            self.assertEqual(result["checkpoints"], [])

    def test_list_checkpoints_environment_not_found(self) -> None:
        """Test listing checkpoints when environment doesn't exist."""
        self.mock_config_store.load_config.side_effect = FileNotFoundError("Config not found")

        with self.assertRaises(CheckpointsServiceError) as ctx:
            self.service.list_checkpoints("nonexistent-env")

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("not found", ctx.exception.message)

    def test_list_checkpoints_no_output_dir(self) -> None:
        """Test listing checkpoints when output_dir is not configured."""
        config = {}  # No output_dir
        metadata = {"name": self.environment_id}
        self.mock_config_store.load_config.return_value = (config, metadata)

        with self.assertRaises(CheckpointsServiceError) as ctx:
            self.service.list_checkpoints(self.environment_id)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("does not have an output_dir configured", ctx.exception.message)

    def test_validate_checkpoint_valid(self) -> None:
        """Test validating a valid checkpoint."""
        self._setup_config_mock()

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.validate_checkpoint.return_value = (True, None)
            mock_manager_class.return_value = mock_manager

            result = self.service.validate_checkpoint(self.environment_id, "checkpoint-1000")

            self.assertEqual(result["environment"], self.environment_id)
            self.assertEqual(result["checkpoint"], "checkpoint-1000")
            self.assertTrue(result["valid"])
            self.assertIn("valid for resuming training", result["message"])
            self.assertNotIn("error", result)

            mock_manager.validate_checkpoint.assert_called_once_with("checkpoint-1000")

    def test_validate_checkpoint_missing_files(self) -> None:
        """Test when checkpoint is missing required files."""
        self._setup_config_mock()

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            error_msg = "Checkpoint missing required files: pytorch_model.bin"
            mock_manager.validate_checkpoint.return_value = (False, error_msg)
            mock_manager_class.return_value = mock_manager

            result = self.service.validate_checkpoint(self.environment_id, "checkpoint-1000")

            self.assertEqual(result["environment"], self.environment_id)
            self.assertEqual(result["checkpoint"], "checkpoint-1000")
            self.assertFalse(result["valid"])
            self.assertEqual(result["error"], error_msg)
            self.assertNotIn("message", result)

    def test_validate_checkpoint_invalid_training_state(self) -> None:
        """Test when checkpoint has invalid training state."""
        self._setup_config_mock()

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            error_msg = "Invalid training state: missing global_step"
            mock_manager.validate_checkpoint.return_value = (False, error_msg)
            mock_manager_class.return_value = mock_manager

            result = self.service.validate_checkpoint(self.environment_id, "checkpoint-1000")

            self.assertFalse(result["valid"])
            self.assertEqual(result["error"], error_msg)

    def test_preview_cleanup(self) -> None:
        """Test cleanup preview logic."""
        self._setup_config_mock()

        mock_checkpoints = [
            {"name": "checkpoint-500", "step": 500},
            {"name": "checkpoint-1000", "step": 1000},
            {"name": "checkpoint-1500", "step": 1500},
            {"name": "checkpoint-2000", "step": 2000},
            {"name": "checkpoint-2500", "step": 2500},
        ]

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = mock_checkpoints.copy()
            mock_manager_class.return_value = mock_manager

            result = self.service.preview_cleanup(self.environment_id, limit=3)

            self.assertEqual(result["environment"], self.environment_id)
            self.assertEqual(result["limit"], 3)
            self.assertEqual(result["total_checkpoints"], 5)
            self.assertEqual(result["count_to_remove"], 3)
            self.assertEqual(result["checkpoints_to_keep"], 2)

            # Should remove oldest checkpoints (500, 1000, 1500)
            to_remove = result["checkpoints_to_remove"]
            self.assertEqual(len(to_remove), 3)
            self.assertEqual(to_remove[0]["name"], "checkpoint-500")
            self.assertEqual(to_remove[1]["name"], "checkpoint-1000")
            self.assertEqual(to_remove[2]["name"], "checkpoint-1500")

    def test_preview_cleanup_no_removal_needed(self) -> None:
        """Test cleanup preview when no removal is needed."""
        self._setup_config_mock()

        mock_checkpoints = [
            {"name": "checkpoint-1000", "step": 1000},
            {"name": "checkpoint-2000", "step": 2000},
        ]

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = mock_checkpoints.copy()
            mock_manager_class.return_value = mock_manager

            result = self.service.preview_cleanup(self.environment_id, limit=5)

            self.assertEqual(result["total_checkpoints"], 2)
            self.assertEqual(result["count_to_remove"], 0)
            self.assertEqual(result["checkpoints_to_keep"], 2)
            self.assertEqual(result["checkpoints_to_remove"], [])

    def test_execute_cleanup(self) -> None:
        """Test actual cleanup execution."""
        self._setup_config_mock()

        mock_checkpoints = [
            {"name": "checkpoint-500", "step": 500},
            {"name": "checkpoint-1000", "step": 1000},
            {"name": "checkpoint-1500", "step": 1500},
            {"name": "checkpoint-2000", "step": 2000},
        ]

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = mock_checkpoints.copy()
            mock_manager_class.return_value = mock_manager

            result = self.service.execute_cleanup(self.environment_id, limit=2)

            self.assertEqual(result["environment"], self.environment_id)
            self.assertEqual(result["limit"], 2)
            self.assertEqual(result["count_removed"], 3)
            self.assertIn("Successfully removed 3 checkpoint(s)", result["message"])

            # Verify removed checkpoints
            removed = result["removed_checkpoints"]
            self.assertEqual(len(removed), 3)
            self.assertEqual(removed[0]["name"], "checkpoint-500")
            self.assertEqual(removed[1]["name"], "checkpoint-1000")
            self.assertEqual(removed[2]["name"], "checkpoint-1500")

            # Verify cleanup_checkpoints was called
            mock_manager.cleanup_checkpoints.assert_called_once_with(limit=2)

    def test_execute_cleanup_error_handling(self) -> None:
        """Test cleanup execution error handling."""
        self._setup_config_mock()

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.side_effect = Exception("Disk error")
            mock_manager_class.return_value = mock_manager

            with self.assertRaises(CheckpointsServiceError) as ctx:
                self.service.execute_cleanup(self.environment_id, limit=2)

            self.assertEqual(ctx.exception.status_code, 500)
            # Error occurs during preview phase which is called by execute_cleanup
            self.assertIn("Failed to preview cleanup", ctx.exception.message)

    def test_get_checkpoints_for_resume(self) -> None:
        """Test resume dropdown endpoint."""
        self._setup_config_mock()

        mock_checkpoints = [
            {"name": "checkpoint-500", "step": 500},
            {"name": "checkpoint-1000", "step": 1000},
            {"name": "checkpoint-2000", "step": 2000},
        ]

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = mock_checkpoints.copy()
            mock_manager_class.return_value = mock_manager

            result = self.service.get_checkpoints_for_resume(self.environment_id)

            self.assertEqual(result["environment"], self.environment_id)
            self.assertEqual(result["count"], 3)

            # Should be sorted by step descending (most recent first)
            checkpoints = result["checkpoints"]
            self.assertEqual(checkpoints[0]["name"], "checkpoint-2000")
            self.assertEqual(checkpoints[0]["step"], 2000)
            self.assertEqual(checkpoints[0]["label"], "checkpoint-2000 (step 2000)")

            self.assertEqual(checkpoints[1]["name"], "checkpoint-1000")
            self.assertEqual(checkpoints[2]["name"], "checkpoint-500")

            # Should call list_checkpoints without metadata
            mock_manager.list_checkpoints.assert_called_once_with(include_metadata=False)

    def test_get_checkpoints_for_resume_empty(self) -> None:
        """Test resume dropdown with no checkpoints."""
        self._setup_config_mock()

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = []
            mock_manager_class.return_value = mock_manager

            result = self.service.get_checkpoints_for_resume(self.environment_id)

            self.assertEqual(result["environment"], self.environment_id)
            self.assertEqual(result["count"], 0)
            self.assertEqual(result["checkpoints"], [])

    def test_checkpoint_manager_caching(self) -> None:
        """Test that CheckpointManager instances are cached per environment."""
        self._setup_config_mock()

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = []
            mock_manager_class.return_value = mock_manager

            # Call twice with same environment
            self.service.list_checkpoints(self.environment_id)
            self.service.list_checkpoints(self.environment_id)

            # CheckpointManager should only be instantiated once
            mock_manager_class.assert_called_once_with(self.output_dir)

    def test_output_dir_with_alternative_key(self) -> None:
        """Test that output_dir can be specified without -- prefix."""
        config = {"output_dir": self.output_dir}  # Without --
        metadata = {"name": self.environment_id}
        self.mock_config_store.load_config.return_value = (config, metadata)

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = []
            mock_manager_class.return_value = mock_manager

            result = self.service.list_checkpoints(self.environment_id)

            # Should work with either key format
            self.assertEqual(result["environment"], self.environment_id)
            mock_manager_class.assert_called_once_with(self.output_dir)

    def test_output_dir_expansion(self) -> None:
        """Test that output_dir paths are expanded (e.g., ~ to home directory)."""
        config = {"--output_dir": "~/output"}
        metadata = {"name": self.environment_id}
        self.mock_config_store.load_config.return_value = (config, metadata)

        with (
            patch("simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager") as mock_manager_class,
            patch("os.path.expanduser") as mock_expand,
        ):
            mock_expand.return_value = "/home/user/output"
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = []
            mock_manager_class.return_value = mock_manager

            self.service.list_checkpoints(self.environment_id)

            # Should expand path
            mock_expand.assert_called_once_with("~/output")
            mock_manager_class.assert_called_once_with("/home/user/output")

    def test_config_store_with_custom_configs_dir(self) -> None:
        """Test that ConfigStore respects custom configs_dir from WebUIStateStore."""
        mock_defaults = MagicMock()
        mock_defaults.configs_dir = "/custom/configs"
        self.mock_webui_state_class.return_value.load_defaults.return_value = mock_defaults

        # Create new service instance to trigger config store initialization
        service = CheckpointsService()

        # Trigger _get_config_store by attempting an operation
        config = {"--output_dir": self.output_dir}
        metadata = {"name": self.environment_id}
        self.mock_config_store.load_config.return_value = (config, metadata)

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.return_value = []
            mock_manager_class.return_value = mock_manager

            service.list_checkpoints(self.environment_id)

            # ConfigStore should be initialized with custom directory
            self.mock_config_store_class.assert_called()

    def test_list_checkpoints_general_exception(self) -> None:
        """Test that general exceptions are wrapped in CheckpointsServiceError."""
        self._setup_config_mock()

        with patch(
            "simpletuner.simpletuner_sdk.server.services.checkpoints_service.CheckpointManager"
        ) as mock_manager_class:
            mock_manager = self._create_mock_checkpoint_manager()
            mock_manager.list_checkpoints.side_effect = RuntimeError("Unexpected error")
            mock_manager_class.return_value = mock_manager

            with self.assertRaises(CheckpointsServiceError) as ctx:
                self.service.list_checkpoints(self.environment_id)

            self.assertEqual(ctx.exception.status_code, 500)
            self.assertIn("Failed to list checkpoints", ctx.exception.message)
            self.assertIn("Unexpected error", ctx.exception.message)


if __name__ == "__main__":
    unittest.main()
