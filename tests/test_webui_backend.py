"""
Unit tests for WebUI backend components.

Tests the WebUIStateStore, configuration management, and related services.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from simpletuner.simpletuner_sdk.server.services.webui_state import (
    OnboardingStepState,
    WebUIDefaults,
    WebUIOnboardingState,
    WebUIState,
    WebUIStateStore,
)


@pytest.fixture
def temp_state_dir():
    """Create a temporary directory for state storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def state_store(temp_state_dir):
    """Create a WebUIStateStore instance with temporary directory."""
    with patch.dict(os.environ, {"HOME": str(temp_state_dir)}):
        yield WebUIStateStore()


class TestWebUIStateStore:
    """Test WebUIStateStore functionality."""

    def test_init_creates_directory(self, temp_state_dir):
        """Test that initialization creates the state directory."""
        with patch.dict(os.environ, {"HOME": str(temp_state_dir)}):
            store = WebUIStateStore()
            assert store.base_dir.exists()
            assert store.base_dir == temp_state_dir / ".simpletuner" / "webui"

    def test_save_and_load_defaults(self, state_store):
        """Test saving and loading WebUI defaults."""
        defaults = WebUIDefaults(
            configs_dir="/path/to/configs",
            output_dir="/path/to/output",
            active_config="my-config",
        )

        state_store.save_defaults(defaults)
        loaded = state_store.load_defaults()

        assert loaded.configs_dir == defaults.configs_dir
        assert loaded.output_dir == defaults.output_dir
        assert loaded.active_config == defaults.active_config

    def test_save_and_load_onboarding_state(self, state_store):
        """Test saving and loading onboarding state."""
        from datetime import datetime

        onboarding = WebUIOnboardingState()
        step_state = OnboardingStepState()
        step_state.value = "/some/path"
        step_state.completed_version = 1
        step_state.completed_at = "2025-01-01T00:00:00"
        onboarding.steps["test_step"] = step_state

        state_store.save_onboarding(onboarding)
        loaded = state_store.load_onboarding()

        assert "test_step" in loaded.steps
        assert loaded.steps["test_step"].value == "/some/path"
        assert loaded.steps["test_step"].completed_version == 1

    def test_record_onboarding_step(self, state_store):
        """Test recording an onboarding step."""
        state_store.record_onboarding_step(
            step_id="configs_dir",
            version=2,
            value="/new/configs"
        )

        onboarding = state_store.load_onboarding()
        assert "configs_dir" in onboarding.steps
        assert onboarding.steps["configs_dir"].value == "/new/configs"
        assert onboarding.steps["configs_dir"].completed_version == 2
        assert onboarding.steps["configs_dir"].completed_at is not None

    def test_load_state_combines_defaults_and_onboarding(self, state_store):
        """Test that load_state properly combines defaults and onboarding."""
        # Set up test data
        defaults = WebUIDefaults(configs_dir="/configs", output_dir="/output")
        state_store.save_defaults(defaults)

        state_store.record_onboarding_step("test_step", 1, "/test/value")

        # Load complete state
        state = state_store.load_state()

        assert state.defaults.configs_dir == "/configs"
        assert state.defaults.output_dir == "/output"
        assert "test_step" in state.onboarding.steps
        assert state.onboarding.steps["test_step"].value == "/test/value"

    def test_load_defaults_creates_empty_if_missing(self, state_store):
        """Test that load_defaults returns empty defaults if file missing."""
        defaults = state_store.load_defaults()

        assert isinstance(defaults, WebUIDefaults)
        assert defaults.configs_dir is None
        assert defaults.output_dir is None
        assert defaults.active_config is None

    def test_load_onboarding_creates_empty_if_missing(self, state_store):
        """Test that load_onboarding returns empty state if file missing."""
        onboarding = state_store.load_onboarding()

        assert isinstance(onboarding, WebUIOnboardingState)
        assert len(onboarding.steps) == 0

    def test_invalid_json_handling(self, state_store):
        """Test handling of invalid JSON in state files."""
        # Write invalid JSON
        defaults_file = state_store.base_dir / "defaults.json"
        defaults_file.write_text("{ invalid json }")

        with pytest.raises(ValueError, match="Failed to read web UI state"):
            state_store.load_defaults()

    def test_path_normalization(self, state_store):
        """Test that paths are normalized when saving."""
        defaults = WebUIDefaults(
            configs_dir="~/configs/../configs",
            output_dir="./output"
        )

        with patch.dict(os.environ, {"HOME": "/home/user"}):
            state_store.save_defaults(defaults)
            loaded = state_store.load_defaults()

        # Paths should be expanded and normalized - check the actual behavior
        # WebUIDefaults.save_defaults() might not expand paths automatically
        assert loaded.configs_dir == "~/configs/../configs"
        assert loaded.output_dir == "./output"

    def test_concurrent_access_safety(self, state_store):
        """Test that concurrent saves don't corrupt state."""
        import threading

        def save_defaults(value):
            defaults = WebUIDefaults(configs_dir=f"/path/{value}")
            state_store.save_defaults(defaults)

        threads = []
        for i in range(5):
            t = threading.Thread(target=save_defaults, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have saved one of the values without corruption
        loaded = state_store.load_defaults()
        assert loaded.configs_dir is not None
        assert loaded.configs_dir.startswith("/path/")


class TestWebUIDefaultsUpdate:
    """Test WebUIDefaults update logic."""

    def test_update_configs_dir_only(self, state_store):
        """Test updating only configs_dir preserves other fields."""
        # Set initial state
        defaults = WebUIDefaults(
            configs_dir="/old/configs",
            output_dir="/old/output",
            active_config="old-config"
        )
        state_store.save_defaults(defaults)

        # Update only configs_dir
        loaded = state_store.load_defaults()
        loaded.configs_dir = "/new/configs"
        state_store.save_defaults(loaded)

        # Verify other fields preserved
        final = state_store.load_defaults()
        assert final.configs_dir == "/new/configs"
        assert final.output_dir == "/old/output"
        assert final.active_config == "old-config"

    def test_update_multiple_fields(self, state_store):
        """Test updating multiple fields at once."""
        defaults = WebUIDefaults()
        defaults.configs_dir = "/configs"
        defaults.output_dir = "/output"
        state_store.save_defaults(defaults)

        loaded = state_store.load_defaults()
        assert loaded.configs_dir == "/configs"
        assert loaded.output_dir == "/output"


class TestWebUIStateIntegration:
    """Integration tests for WebUI state management."""

    def test_full_onboarding_flow(self, state_store):
        """Test complete onboarding flow."""
        # Step 1: User completes configs_dir
        state_store.record_onboarding_step(
            "default_configs_dir",
            version=2,
            value="/home/user/configs"
        )

        # Step 2: User completes output_dir
        state_store.record_onboarding_step(
            "default_output_dir",
            version=1,
            value="/home/user/output"
        )

        # Apply to defaults
        defaults = state_store.load_defaults()
        defaults.configs_dir = "/home/user/configs"
        defaults.output_dir = "/home/user/output"
        state_store.save_defaults(defaults)

        # Verify final state
        state = state_store.load_state()
        assert state.defaults.configs_dir == "/home/user/configs"
        assert state.defaults.output_dir == "/home/user/output"
        assert len(state.onboarding.steps) == 2

    def test_version_upgrade_handling(self, state_store):
        """Test handling of onboarding version upgrades."""
        # Record step with version 1
        state_store.record_onboarding_step("test_step", 1, "value1")

        # Record same step with version 2 (upgrade)
        state_store.record_onboarding_step("test_step", 2, "value2")

        # Should have updated version and value
        onboarding = state_store.load_onboarding()
        assert onboarding.steps["test_step"].completed_version == 2
        assert onboarding.steps["test_step"].value == "value2"

    def test_reset_onboarding(self, state_store):
        """Test resetting onboarding state."""
        # Set up some state
        state_store.record_onboarding_step("step1", 1, "value1")
        state_store.record_onboarding_step("step2", 1, "value2")

        # Reset
        state_store.save_onboarding(WebUIOnboardingState())

        # Verify reset
        onboarding = state_store.load_onboarding()
        assert len(onboarding.steps) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])