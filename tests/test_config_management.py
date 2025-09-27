#!/usr/bin/env python
"""Test script for configuration management system."""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simpletuner.simpletuner_sdk.server.services.config_store import ConfigMetadata, ConfigStore


def test_config_store():
    """Test ConfigStore functionality."""
    print("Testing ConfigStore...")

    # Use temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConfigStore(config_dir=tmpdir)

        # Test 1: Create a configuration
        print("✓ Creating test configuration...")
        config = {
            "--model_type": "lora",
            "--model_family": "flux",
            "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
            "--output_dir": "/tmp/test_output",
            "--train_batch_size": 2,
            "--learning_rate": 0.0002,
            "--max_train_steps": 500,
            "--resolution": 1024,
        }

        metadata = store.save_config("test_config", config)
        assert metadata.name == "test_config"
        print(f"  Created config: {metadata.name}")

        # Test 2: List configurations
        print("✓ Listing configurations...")
        configs = store.list_configs()
        assert len(configs) >= 2  # default and test_config
        config_names = [c.get("name") for c in configs]
        assert "default" in config_names
        assert "test_config" in config_names
        print(f"  Found {len(configs)} configs: {', '.join(config_names)}")

        # Test 3: Load configuration
        print("✓ Loading configuration...")
        loaded_config, loaded_metadata = store.load_config("test_config")
        assert loaded_config["--model_type"] == "lora"
        assert loaded_metadata.name == "test_config"
        print(f"  Loaded config with {len(loaded_config)} fields")

        # Test 4: Validate configuration
        print("✓ Validating configuration...")
        validation = store.validate_config(config)
        assert validation.is_valid
        print(f"  Config is valid: {validation.is_valid}")
        if validation.warnings:
            print(f"  Warnings: {validation.warnings}")
        if validation.suggestions:
            print(f"  Suggestions: {validation.suggestions}")

        # Test 5: Copy configuration
        print("✓ Copying configuration...")
        copy_metadata = store.copy_config("test_config", "test_copy")
        assert copy_metadata.name == "test_copy"
        copy_config, _ = store.load_config("test_copy")
        assert copy_config["--model_type"] == config["--model_type"]
        print(f"  Created copy: {copy_metadata.name}")

        # Test 6: Rename configuration
        print("✓ Renaming configuration...")
        rename_metadata = store.rename_config("test_copy", "test_renamed")
        assert rename_metadata.name == "test_renamed"
        configs = store.list_configs()
        config_names = [c.get("name") for c in configs]
        assert "test_renamed" in config_names
        assert "test_copy" not in config_names
        print(f"  Renamed to: {rename_metadata.name}")

        # Test 7: Set active configuration
        print("✓ Setting active configuration...")
        store.set_active_config("test_config")
        active = store.get_active_config()
        assert active == "test_config"
        print(f"  Active config: {active}")

        # Test 8: Export configuration
        print("✓ Exporting configuration...")
        exported = store.export_config("test_config")
        assert "_metadata" in exported
        assert "config" in exported
        assert exported["config"]["--model_type"] == "lora"
        print(f"  Exported config with metadata")

        # Test 9: Delete configuration
        print("✓ Deleting configuration...")
        deleted = store.delete_config("test_renamed")
        assert deleted
        configs = store.list_configs()
        config_names = [c.get("name") for c in configs]
        assert "test_renamed" not in config_names
        print(f"  Deleted test_renamed")

        # Test 10: Invalid config validation
        print("✓ Testing invalid configuration...")
        invalid_config = {
            "--model_type": "invalid_type",
            "--resolution": 100,  # Too small
        }
        validation = store.validate_config(invalid_config)
        assert not validation.is_valid
        assert len(validation.errors) > 0
        print(f"  Found {len(validation.errors)} errors in invalid config")
        for error in validation.errors:
            print(f"    - {error}")

    print("\n✅ All tests passed!")


def test_api_routes():
    """Test API routes (requires server running)."""
    try:
        import requests
    except ImportError:
        print("⚠️  Skipping API tests (requests not installed)")
        return

    print("\nTesting API Routes...")
    base_url = "http://localhost:8001/api/configs"

    try:
        # Test list configs
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Listed {data['count']} configs via API")
            print(f"  Active config: {data.get('active', 'none')}")
        else:
            print(f"⚠️  API not available (status {response.status_code})")
            return
    except requests.ConnectionError:
        print("⚠️  API server not running (start with: python -m simpletuner.simpletuner_sdk)")
        return

    print("✅ API tests completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("SimpleTuner Configuration Management Test Suite")
    print("=" * 60)
    print()

    test_config_store()
    test_api_routes()

    print("\n" + "=" * 60)
    print("Test suite completed successfully!")
    print("=" * 60)
