import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from simpletuner.helpers.training.dynamo import dynamo_config_context


class CompileDiskSpaceTests(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        self.callbacks = list(torch._dynamo.callback_handler.start_callbacks)
        self.config = SimpleNamespace(dynamo_backend="inductor", disk_low_threshold="160M", disk_low_action="wait")
        self.threshold = 160 * 1024**2

    def tearDown(self):
        self.assertEqual(torch._dynamo.callback_handler.start_callbacks, self.callbacks)
        torch._dynamo.reset()

    def test_wait_before_initial_compile_and_recompile(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = str(Path(directory) / "inductor")
            triton = str(Path(directory) / "triton")
            with (
                patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": cache, "TRITON_CACHE_DIR": triton}),
                patch(
                    "simpletuner.helpers.training.disk_space.get_available_disk_space",
                    side_effect=[0, self.threshold, self.threshold] * 3,
                ) as free,
                patch("simpletuner.helpers.training.disk_space.time") as clock,
                dynamo_config_context(self.config),
            ):
                compiled = torch.compile(lambda x: x.sin() + 1, backend="inductor", dynamic=False)
                for size in (3, 5):
                    value = torch.arange(size, dtype=torch.float32)
                    torch.testing.assert_close(compiled(value), value.sin() + 1)
                # Reusing an already compiled shape does not check disk again.
                compiled(value)
                self.assertEqual(clock.sleep.call_count, 3)
                self.assertEqual([call.args[0] for call in free.call_args_list], [cache, cache, triton] * 3)

    def test_stop_before_compilation_and_remove_callback(self):
        self.config.disk_low_action = "stop"
        with tempfile.TemporaryDirectory() as directory:
            cache = str(Path(directory) / "not-created")
            with (
                patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": cache}),
                patch("simpletuner.helpers.training.disk_space.get_available_disk_space", return_value=0),
                self.assertRaisesRegex(RuntimeError, "Disk space critically low"),
                dynamo_config_context(self.config),
            ):
                torch.compile(lambda x: x.cos(), backend="inductor")(torch.ones(2))

    def test_default_cache_and_cleanup_script(self):
        self.config.disk_low_action = "script"
        self.config.disk_low_script = "cleanup-cache"
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("torch._inductor.runtime.runtime_utils.default_cache_dir", return_value="compiler-cache"),
            patch(
                "simpletuner.helpers.training.disk_space.get_available_disk_space",
                side_effect=[0, self.threshold, self.threshold],
            ) as free,
            patch("simpletuner.helpers.training.disk_space.subprocess.run") as script,
            dynamo_config_context(self.config),
        ):
            callback = torch._dynamo.callback_handler.start_callbacks[-1]
            callback(None)
            self.assertEqual([call.args[0] for call in free.call_args_list], ["compiler-cache"] * 3)
            script.assert_called_once_with(["cleanup-cache"], check=True)

    def test_disabled_threshold_or_backend_does_not_register(self):
        for backend, threshold in (("inductor", None), ("eager", "160M")):
            with self.subTest(backend=backend), patch.dict(os.environ, {}, clear=True):
                config = SimpleNamespace(dynamo_backend=backend, disk_low_threshold=threshold)
                with dynamo_config_context(config):
                    self.assertEqual(torch._dynamo.callback_handler.start_callbacks, self.callbacks)

    def test_callback_does_not_create_missing_cache_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = str(Path(directory) / "missing")
            with (
                patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": cache}),
                patch(
                    "simpletuner.helpers.training.disk_space.shutil.disk_usage",
                    return_value=SimpleNamespace(free=self.threshold),
                ) as usage,
                dynamo_config_context(self.config),
            ):
                torch._dynamo.callback_handler.start_callbacks[-1](None)
                self.assertEqual(usage.call_args_list[0].args[0], str(Path(directory).resolve()))
                self.assertFalse(Path(cache).exists())

    def test_exceptions_propagate_and_callback_is_removed(self):
        for error in (ValueError("compile failed"), torch.OutOfMemoryError("GPU allocation failed")):
            with self.subTest(error=type(error).__name__):
                with (
                    patch("simpletuner.helpers.training.disk_space.get_available_disk_space", return_value=self.threshold),
                    self.assertRaises(type(error)) as raised,
                ):
                    with dynamo_config_context(self.config):
                        raise error
                self.assertIs(raised.exception, error)
                self.assertEqual(torch._dynamo.callback_handler.start_callbacks, self.callbacks)

    def test_stop_on_later_compile_and_remove_callback(self):
        self.config.disk_low_action = "stop"
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("simpletuner.helpers.training.disk_space.get_available_disk_space", side_effect=[self.threshold, 0]),
            self.assertRaisesRegex(RuntimeError, "Disk space critically low"),
            dynamo_config_context(self.config),
        ):
            torch.compile(lambda x: x.cos(), backend="inductor")(torch.ones(2))

    def test_backend_failure_is_not_retried(self):
        calls = []

        def failing_backend(graph, inputs):
            calls.append(graph)
            raise RuntimeError("compiler failure")

        with (
            patch("simpletuner.helpers.training.disk_space.get_available_disk_space", return_value=self.threshold),
            self.assertRaisesRegex(torch._dynamo.exc.BackendCompilerFailed, "compiler failure"),
            dynamo_config_context(self.config),
        ):
            torch.compile(lambda x: x.cos(), backend=failing_backend)(torch.ones(2))
        self.assertEqual(len(calls), 1)
