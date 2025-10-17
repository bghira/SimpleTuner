"""
Helper functions and base classes for SimpleTuner unittest tests.
"""

import json
import logging
import multiprocessing
import os
import tempfile
import time
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, Mock

# Disable logging by default
logging.disable(logging.WARNING)
os.environ["SIMPLETUNER_LOG_LEVEL"] = "ERROR"
os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class SimpleTunerTestCase(unittest.TestCase):
    """Base test case with common fixtures and utilities."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        # Silence loggers
        logging.disable(logging.WARNING)

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dirs = []
        self.temp_files = []

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp files
        for path in self.temp_files:
            if os.path.exists(path):
                os.unlink(path)

        # Clean up temp dirs
        for path in self.temp_dirs:
            if os.path.exists(path):
                import shutil

                shutil.rmtree(path)

    def create_temp_dir(self):
        """Create a temporary directory that will be cleaned up."""
        path = tempfile.mkdtemp()
        self.temp_dirs.append(path)
        return path

    def create_temp_file(self, content="", suffix=".txt"):
        """Create a temporary file that will be cleaned up."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        if content:
            with open(path, "w") as f:
                f.write(content)

        self.temp_files.append(path)
        return path

    def create_valid_config(self) -> Dict[str, Any]:
        """Create a valid trainer configuration dict."""
        return {
            "model_type": "lora",
            "model_family": "sdxl",
            "model_flavour": "base-1.0",
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "output_dir": self.create_temp_dir(),
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "max_train_steps": 100,
            "checkpointing_steps": 50,
            "validation_steps": 25,
            "tracker_project_name": "test_project",
            "tracker_run_name": "test_run",
            "mixed_precision": "no",
            "resolution": 1024,
            "dataloader_num_workers": 0,
            "train_text_encoder": False,
        }

    def create_mock_trainer(self):
        """Create a mock trainer object."""
        trainer = MagicMock()
        trainer.run.return_value = None
        trainer.abort.return_value = None
        trainer.state = MagicMock()
        trainer.state.global_step = 0
        trainer.state.current_epoch = 0
        trainer.state.loss = 0.5
        return trainer

    def wait_for_condition(self, condition_func, timeout=5.0, interval=0.1):
        """Wait for a condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False


class AsyncTestCase(SimpleTunerTestCase):
    """Base test case for async tests."""

    def run_async(self, coro):
        """Run an async coroutine in the test."""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


class ProcessTestCase(SimpleTunerTestCase):
    """Base test case for process-related tests."""

    def setUp(self):
        """Set up process test fixtures."""
        super().setUp()
        self.test_processes = []

    def tearDown(self):
        """Clean up test processes."""
        for proc in self.test_processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)
                if proc.is_alive():
                    proc.kill()
        super().tearDown()

    def create_test_process(self, target, args=()):
        """Create a test process that will be cleaned up."""
        proc = multiprocessing.Process(target=target, args=args)
        self.test_processes.append(proc)
        return proc


class MockStore:
    """Mock event store for testing."""

    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events.append(event)
        return len(self.events) - 1

    def get_events_since(self, index):
        return self.events[index + 1 :]

    def get_all_events(self):
        return self.events

    def clear(self):
        self.events = []

    def get_event_count(self):
        return len(self.events)

    def get_latest_event(self):
        return self.events[-1] if self.events else None
