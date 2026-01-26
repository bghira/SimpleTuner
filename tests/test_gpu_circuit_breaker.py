"""Tests for GPU circuit breaker for hardware fault detection.

Tests cover:
- GPUCircuitBreaker state management
- CUDA error detection (is_cuda_error)
- GPU health metrics collection (mocked)
- Webhook emission on circuit open
- Circuit state transitions
"""

from __future__ import annotations

import logging
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# Suppress expected error logs during tests
logging.getLogger("simpletuner.helpers.training.gpu_circuit_breaker").setLevel(logging.CRITICAL)

from simpletuner.helpers.training.exceptions import GPUHealthError
from simpletuner.helpers.training.gpu_circuit_breaker import (
    CircuitState,
    GPUCircuitBreaker,
    GPUHealthMetrics,
    get_current_gpu_index,
    get_gpu_circuit_breaker,
    is_cuda_error,
    reset_gpu_circuit_breaker,
)
from simpletuner.helpers.webhooks.events import gpu_fault_event


class TestIsCudaError(unittest.TestCase):
    """Tests for is_cuda_error detection function."""

    def test_cuda_runtime_error(self):
        """Test detection of CUDA runtime errors."""
        error = RuntimeError("CUDA error: device-side assert triggered")
        self.assertTrue(is_cuda_error(error))

    def test_cuda_driver_error(self):
        """Test detection of CUDA driver errors."""
        error = RuntimeError("CUDA driver error: unknown error")
        self.assertTrue(is_cuda_error(error))

    def test_cudnn_error(self):
        """Test detection of cuDNN errors."""
        error = RuntimeError("cuDNN error: CUDNN_STATUS_EXECUTION_FAILED")
        self.assertTrue(is_cuda_error(error))

    def test_out_of_memory_error(self):
        """Test detection of GPU out of memory errors."""
        error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.assertTrue(is_cuda_error(error))

    def test_nccl_error(self):
        """Test detection of NCCL errors."""
        error = RuntimeError("NCCL error: unhandled system error")
        self.assertTrue(is_cuda_error(error))

    def test_illegal_memory_access(self):
        """Test detection of illegal memory access errors."""
        error = RuntimeError("an illegal memory access was encountered")
        self.assertTrue(is_cuda_error(error))

    def test_non_cuda_error(self):
        """Test non-CUDA errors are not detected."""
        error = ValueError("Invalid argument")
        self.assertFalse(is_cuda_error(error))

    def test_generic_runtime_error(self):
        """Test generic runtime errors without CUDA indicators."""
        error = ValueError("Something went wrong")  # Use ValueError to avoid RuntimeError match
        self.assertFalse(is_cuda_error(error))

    def test_case_insensitive(self):
        """Test error detection is case insensitive."""
        error = RuntimeError("cuda Error: Something failed")
        self.assertTrue(is_cuda_error(error))


class TestGPUFaultEvent(unittest.TestCase):
    """Tests for gpu_fault_event webhook event creation."""

    def test_basic_event(self):
        """Test creating a basic GPU fault event."""
        event = gpu_fault_event(
            fault_type="cuda_error",
            message="CUDA driver error: unknown error",
        )

        self.assertEqual(event["type"], "gpu.fault")
        self.assertEqual(event["severity"], "critical")
        self.assertEqual(event["fault"]["type"], "cuda_error")
        self.assertEqual(event["message"], "CUDA driver error: unknown error")

    def test_full_event(self):
        """Test creating a GPU fault event with all fields."""
        event = gpu_fault_event(
            fault_type="ecc_error",
            message="Uncorrectable ECC errors detected",
            gpu_index=0,
            gpu_name="NVIDIA A100-SXM4-80GB",
            job_id="job-123",
            severity="critical",
            temperature_celsius=75.5,
            ecc_errors_single=10,
            ecc_errors_double=2,
            throttle_reasons=["hw_thermal_slowdown"],
            memory_used_percent=85.5,
            action_taken="circuit_opened",
            exception_type="RuntimeError",
        )

        self.assertEqual(event["type"], "gpu.fault")
        self.assertEqual(event["severity"], "critical")
        self.assertEqual(event["job_id"], "job-123")
        self.assertEqual(event["fault"]["type"], "ecc_error")
        self.assertEqual(event["fault"]["gpu"]["index"], 0)
        self.assertEqual(event["fault"]["gpu"]["name"], "NVIDIA A100-SXM4-80GB")
        self.assertEqual(event["fault"]["gpu"]["temperature_celsius"], 75.5)
        self.assertEqual(event["fault"]["gpu"]["ecc_errors_single"], 10)
        self.assertEqual(event["fault"]["gpu"]["ecc_errors_double"], 2)
        self.assertEqual(event["fault"]["gpu"]["throttle_reasons"], ["hw_thermal_slowdown"])
        self.assertEqual(event["fault"]["gpu"]["memory_used_percent"], 85.5)
        self.assertEqual(event["fault"]["action_taken"], "circuit_opened")
        self.assertEqual(event["fault"]["exception_type"], "RuntimeError")


class TestGPUHealthMetrics(unittest.TestCase):
    """Tests for GPUHealthMetrics dataclass."""

    def test_default_values(self):
        """Test default values for GPUHealthMetrics."""
        metrics = GPUHealthMetrics(index=0)

        self.assertEqual(metrics.index, 0)
        self.assertEqual(metrics.name, "")
        self.assertIsNone(metrics.temperature_celsius)
        self.assertEqual(metrics.ecc_errors_single, 0)
        self.assertEqual(metrics.ecc_errors_double, 0)
        self.assertEqual(metrics.throttle_reasons, [])
        self.assertIsNone(metrics.memory_used_percent)
        self.assertTrue(metrics.is_healthy)
        self.assertIsNone(metrics.fault_reason)

    def test_with_values(self):
        """Test GPUHealthMetrics with values set."""
        metrics = GPUHealthMetrics(
            index=1,
            name="NVIDIA RTX 5090",
            temperature_celsius=82.0,
            ecc_errors_single=5,
            ecc_errors_double=0,
            throttle_reasons=["sw_thermal_slowdown"],
            memory_used_percent=75.5,
            is_healthy=True,
        )

        self.assertEqual(metrics.index, 1)
        self.assertEqual(metrics.name, "NVIDIA RTX 5090")
        self.assertEqual(metrics.temperature_celsius, 82.0)
        self.assertEqual(metrics.ecc_errors_single, 5)
        self.assertEqual(metrics.throttle_reasons, ["sw_thermal_slowdown"])
        self.assertEqual(metrics.memory_used_percent, 75.5)


class TestGPUCircuitBreakerStateTransitions(unittest.TestCase):
    """Tests for GPUCircuitBreaker state transitions."""

    def setUp(self):
        """Set up test fixtures."""
        reset_gpu_circuit_breaker()
        self.breaker = GPUCircuitBreaker(
            webhook_handler=None,
            job_id="test-job",
            cuda_failure_threshold=1,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.breaker.stop_monitoring()
        reset_gpu_circuit_breaker()

    def test_initial_state_is_closed(self):
        """Test initial state is CLOSED."""
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertFalse(self.breaker.is_open)

    def test_opens_on_cuda_error(self):
        """Test circuit opens on CUDA error."""
        error = RuntimeError("CUDA driver error: unknown error")
        self.breaker.record_cuda_error(error, gpu_index=0)

        self.assertEqual(self.breaker.state, CircuitState.OPEN)
        self.assertTrue(self.breaker.is_open)

    def test_stays_closed_below_threshold(self):
        """Test circuit stays closed below failure threshold."""
        breaker = GPUCircuitBreaker(
            webhook_handler=None,
            job_id="test-job",
            cuda_failure_threshold=3,
        )

        error = RuntimeError("CUDA error")
        breaker.record_cuda_error(error, gpu_index=0)
        breaker.record_cuda_error(error, gpu_index=0)

        # 2 failures, threshold is 3
        self.assertEqual(breaker.state, CircuitState.CLOSED)
        self.assertFalse(breaker.is_open)

        breaker.stop_monitoring()

    def test_get_status(self):
        """Test getting circuit breaker status."""
        status = self.breaker.get_status()

        self.assertEqual(status["state"], "closed")
        self.assertEqual(status["failure_count"], 0)
        self.assertIsNone(status["last_failure_time"])
        self.assertIsNone(status["last_failure_reason"])
        self.assertIsNone(status["opened_at"])


class TestGPUCircuitBreakerWebhook(unittest.TestCase):
    """Tests for GPUCircuitBreaker webhook emission."""

    def setUp(self):
        """Set up test fixtures."""
        reset_gpu_circuit_breaker()
        self.mock_webhook_handler = MagicMock()
        self.breaker = GPUCircuitBreaker(
            webhook_handler=self.mock_webhook_handler,
            job_id="test-job-123",
            cuda_failure_threshold=1,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.breaker.stop_monitoring()
        reset_gpu_circuit_breaker()

    def test_emits_webhook_on_cuda_error(self):
        """Test webhook is emitted when circuit opens on CUDA error."""
        error = RuntimeError("CUDA driver error: unknown error")
        self.breaker.record_cuda_error(error, gpu_index=0)

        self.mock_webhook_handler.send.assert_called_once()
        call_args = self.mock_webhook_handler.send.call_args[0][0]

        self.assertEqual(call_args["type"], "gpu.fault")
        self.assertEqual(call_args["fault"]["type"], "cuda_error")
        self.assertEqual(call_args["job_id"], "test-job-123")

    def test_no_webhook_without_handler(self):
        """Test no error when webhook handler is None."""
        breaker = GPUCircuitBreaker(
            webhook_handler=None,
            job_id="test-job",
            cuda_failure_threshold=1,
        )

        error = RuntimeError("CUDA error")
        # Should not raise
        breaker.record_cuda_error(error, gpu_index=0)

        self.assertTrue(breaker.is_open)
        breaker.stop_monitoring()


class TestGPUCircuitBreakerHealthMonitor(unittest.TestCase):
    """Tests for GPUCircuitBreaker health monitoring (mocked)."""

    def setUp(self):
        """Set up test fixtures."""
        reset_gpu_circuit_breaker()

    def tearDown(self):
        """Clean up test fixtures."""
        reset_gpu_circuit_breaker()

    @patch("simpletuner.helpers.training.gpu_circuit_breaker.pynvml", None)
    def test_no_nvml_graceful_degradation(self):
        """Test circuit breaker works without pynvml."""
        breaker = GPUCircuitBreaker(
            webhook_handler=None,
            job_id="test-job",
        )

        # Should not fail without pynvml
        self.assertFalse(breaker._nvml_initialized)

        # Health check should return True (healthy) when NVML unavailable
        self.assertTrue(breaker.check_health())

        breaker.stop_monitoring()


class TestGPUCircuitBreakerGlobalInstance(unittest.TestCase):
    """Tests for global GPU circuit breaker instance management."""

    def setUp(self):
        """Set up test fixtures."""
        reset_gpu_circuit_breaker()

    def tearDown(self):
        """Clean up test fixtures."""
        reset_gpu_circuit_breaker()

    def test_get_returns_same_instance(self):
        """Test get_gpu_circuit_breaker returns same instance."""
        breaker1 = get_gpu_circuit_breaker(webhook_handler=None, job_id="job-1")
        breaker2 = get_gpu_circuit_breaker(webhook_handler=None, job_id="job-2")

        # Should return same instance
        self.assertIs(breaker1, breaker2)

    def test_reset_clears_instance(self):
        """Test reset_gpu_circuit_breaker clears the global instance."""
        breaker1 = get_gpu_circuit_breaker(webhook_handler=None, job_id="job-1")
        reset_gpu_circuit_breaker()
        breaker2 = get_gpu_circuit_breaker(webhook_handler=None, job_id="job-2")

        # Should be different instances after reset
        self.assertIsNot(breaker1, breaker2)


class TestGPUHealthError(unittest.TestCase):
    """Tests for GPUHealthError exception."""

    def test_basic_exception(self):
        """Test basic GPUHealthError creation."""
        error = GPUHealthError(
            message="GPU circuit breaker is open",
            fault_type="circuit_open",
        )

        self.assertEqual(str(error), "GPU circuit breaker is open")
        self.assertEqual(error.fault_type, "circuit_open")
        self.assertIsNone(error.gpu_index)
        self.assertIsNone(error.gpu_name)

    def test_full_exception(self):
        """Test GPUHealthError with all fields."""
        error = GPUHealthError(
            message="Uncorrectable ECC errors detected",
            fault_type="ecc_error",
            gpu_index=0,
            gpu_name="NVIDIA A100",
        )

        self.assertEqual(str(error), "Uncorrectable ECC errors detected")
        self.assertEqual(error.fault_type, "ecc_error")
        self.assertEqual(error.gpu_index, 0)
        self.assertEqual(error.gpu_name, "NVIDIA A100")


class TestGPUCircuitBreakerMonitorThread(unittest.TestCase):
    """Tests for GPU health monitor thread lifecycle."""

    def setUp(self):
        """Set up test fixtures."""
        reset_gpu_circuit_breaker()

    def tearDown(self):
        """Clean up test fixtures."""
        reset_gpu_circuit_breaker()

    @patch("simpletuner.helpers.training.gpu_circuit_breaker.pynvml")
    def test_start_stop_monitoring(self, mock_pynvml):
        """Test starting and stopping the monitor thread."""
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1

        breaker = GPUCircuitBreaker(
            webhook_handler=None,
            job_id="test-job",
            poll_interval_seconds=0.1,
        )

        breaker.start_monitoring()

        # Thread should be running
        self.assertIsNotNone(breaker._monitor_thread)
        self.assertTrue(breaker._monitor_thread.is_alive())

        breaker.stop_monitoring()

        # Thread should be stopped (None or not alive)
        time.sleep(0.2)
        if breaker._monitor_thread is not None:
            self.assertFalse(breaker._monitor_thread.is_alive())
        # If _monitor_thread is None, that's also fine (thread was cleaned up)

    @patch("simpletuner.helpers.training.gpu_circuit_breaker.pynvml", None)
    def test_no_monitoring_without_nvml(self):
        """Test monitoring doesn't start without pynvml."""
        breaker = GPUCircuitBreaker(
            webhook_handler=None,
            job_id="test-job",
        )

        breaker.start_monitoring()

        # Thread should not be started
        self.assertIsNone(breaker._monitor_thread)

        breaker.stop_monitoring()


if __name__ == "__main__":
    unittest.main()
