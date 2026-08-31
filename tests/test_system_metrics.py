from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from simpletuner.helpers.training.system_metrics import (
    SystemMetricsSampler,
    log_system_metrics_to_trackers,
    should_collect_manual_system_metrics,
)


class SystemMetricsSamplerTests(unittest.TestCase):
    def test_should_collect_manual_system_metrics_skips_native_wandb_only(self):
        self.assertFalse(should_collect_manual_system_metrics("wandb"))
        self.assertFalse(should_collect_manual_system_metrics(["none"]))
        self.assertTrue(should_collect_manual_system_metrics("simpletuner"))
        self.assertTrue(should_collect_manual_system_metrics(["wandb", "simpletuner"]))
        self.assertTrue(should_collect_manual_system_metrics(["tensorboard"]))

    def test_samples_system_metrics_and_network_rates(self):
        times = iter([10.0, 15.0])
        counters = [
            SimpleNamespace(bytes_sent=1_000, bytes_recv=2_000),
            SimpleNamespace(bytes_sent=2_000, bytes_recv=4_500),
        ]
        with tempfile.TemporaryDirectory() as directory:
            sampler = SystemMetricsSampler(output_dir=directory, min_interval_seconds=0, time_source=lambda: next(times))
            sampler._sample_gpu = lambda metrics: metrics.update({"system/gpu/0/utilization_percent": 42.0})
            with (
                mock.patch("simpletuner.helpers.training.system_metrics.psutil.cpu_percent", return_value=25.5),
                mock.patch(
                    "simpletuner.helpers.training.system_metrics.psutil.virtual_memory",
                    return_value=SimpleNamespace(percent=63.2, available=4 * 1024**3),
                ),
                mock.patch(
                    "simpletuner.helpers.training.system_metrics.psutil.net_io_counters",
                    side_effect=counters,
                ),
                mock.patch(
                    "simpletuner.helpers.training.system_metrics.shutil.disk_usage",
                    return_value=SimpleNamespace(total=100, used=25, free=75),
                ),
            ):
                first = sampler.sample(force=True)
                second = sampler.sample(force=True)

        self.assertEqual(first["system/cpu_percent"], 25.5)
        self.assertEqual(first["system/memory_percent"], 63.2)
        self.assertEqual(first["system/disk_percent"], 25.0)
        self.assertEqual(first["system/gpu/0/utilization_percent"], 42.0)
        self.assertNotIn("system/network_sent_mbps", first)
        self.assertEqual(second["system/network_sent_mbps"], 0.002)
        self.assertEqual(second["system/network_recv_mbps"], 0.004)

    def test_sampler_throttles_between_intervals(self):
        times = iter([10.0, 11.0])
        with tempfile.TemporaryDirectory() as directory:
            sampler = SystemMetricsSampler(output_dir=directory, min_interval_seconds=5, time_source=lambda: next(times))
            sampler._sample_system = lambda metrics, _now: metrics.update({"system/cpu_percent": 1.0})
            sampler._sample_gpu = lambda _metrics: None

            self.assertEqual(sampler.sample(), {"system/cpu_percent": 1.0})
            self.assertEqual(sampler.sample(), {})

    def test_rocm_json_metrics_are_flattened(self):
        payload = {
            "card0": {
                "GPU use (%)": "76%",
                "GPU Memory Allocated (VRAM%)": "41%",
                "Temperature (Sensor edge) (C)": "62.5c",
                "Fan Speed (%)": "35%",
                "Average Graphics Package Power (W)": "180.5 W",
            }
        }
        completed = SimpleNamespace(stdout=json.dumps(payload))
        sampler = SystemMetricsSampler(output_dir=".", min_interval_seconds=0)
        metrics: dict[str, float] = {}
        with (
            mock.patch("simpletuner.helpers.training.system_metrics.shutil.which", return_value="/usr/bin/rocm-smi"),
            mock.patch("simpletuner.helpers.training.system_metrics.subprocess.run", return_value=completed),
        ):
            sampler._sample_rocm(metrics)

        self.assertEqual(metrics["system/gpu/0/utilization_percent"], 76.0)
        self.assertEqual(metrics["system/gpu/0/memory_percent"], 41.0)
        self.assertEqual(metrics["system/gpu/0/temperature_celsius"], 62.5)
        self.assertEqual(metrics["system/gpu/0/fan_speed_percent"], 35.0)
        self.assertEqual(metrics["system/gpu/0/power_usage_watts"], 180.5)


class SystemMetricsTrackerRoutingTests(unittest.TestCase):
    def test_logs_to_non_native_trackers_only(self):
        wandb = SimpleNamespace(name="wandb", log=mock.Mock())
        simpletuner = SimpleNamespace(name="simpletuner", log=mock.Mock())
        tensorboard = SimpleNamespace(name="tensorboard", log=mock.Mock())

        log_system_metrics_to_trackers(
            [wandb, simpletuner, tensorboard],
            {"system/cpu_percent": 33.0},
            step=12,
        )

        wandb.log.assert_not_called()
        simpletuner.log.assert_called_once_with({"system/cpu_percent": 33.0}, step=12)
        tensorboard.log.assert_called_once_with({"system/cpu_percent": 33.0}, step=12)


if __name__ == "__main__":
    unittest.main()
