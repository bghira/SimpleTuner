"""GPU circuit breaker for detecting hardware faults and notifying orchestrators.

This module provides:
1. Background health monitoring thread (ECC errors, temperature, throttling)
2. Circuit breaker that trips on CUDA errors or health warnings
3. Webhook emission for orchestrator notification

Zero-config: always enabled when GPUs are present.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from simpletuner.helpers.webhooks.handler import WebhookHandler

logger = logging.getLogger(__name__)

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    import torch
except ImportError:
    torch = None


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Fault detected, reject operations


@dataclass
class GPUHealthMetrics:
    """Health metrics for a single GPU."""

    index: int
    name: str = ""
    temperature_celsius: Optional[float] = None
    temperature_threshold_slowdown: Optional[float] = None
    temperature_threshold_shutdown: Optional[float] = None
    ecc_errors_single: int = 0
    ecc_errors_double: int = 0
    throttle_reasons: List[str] = field(default_factory=list)
    memory_used_percent: Optional[float] = None
    memory_used_bytes: Optional[int] = None
    memory_total_bytes: Optional[int] = None
    gpu_utilization_percent: Optional[float] = None
    memory_utilization_percent: Optional[float] = None
    fan_speed_percent: Optional[float] = None
    power_usage_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    clock_graphics_mhz: Optional[int] = None
    clock_memory_mhz: Optional[int] = None
    is_healthy: bool = True
    fault_reason: Optional[str] = None


@dataclass
class CircuitBreakerState:
    """State for the GPU circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    last_failure_reason: Optional[str] = None
    last_failure_gpu_index: Optional[int] = None
    opened_at: Optional[float] = None


# Throttle reason bit masks from NVML
THROTTLE_REASONS = {
    0x0000000000000001: "gpu_idle",
    0x0000000000000002: "applications_clocks_setting",
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000010: "sync_boost",
    0x0000000000000020: "sw_thermal_slowdown",
    0x0000000000000040: "hw_thermal_slowdown",
    0x0000000000000080: "hw_power_brake_slowdown",
    0x0000000000000100: "display_clocks_setting",
}

# Reasons that indicate potential hardware issues (fatal - opens circuit)
CRITICAL_THROTTLE_REASONS = {
    "hw_slowdown",
    "hw_power_brake_slowdown",
}

# Reasons that indicate thermal throttling (warning only - does not open circuit)
WARNING_THROTTLE_REASONS = {
    "hw_thermal_slowdown",
}


class GPUCircuitBreaker:
    """Circuit breaker for GPU operations with health monitoring.

    Detects GPU faults through:
    1. Background health monitoring (ECC errors, temperature, throttling)
    2. CUDA exception catching during training

    When a fault is detected:
    1. Circuit opens (blocks further operations)
    2. Webhook is emitted to notify orchestrator
    3. Training should exit cleanly
    """

    def __init__(
        self,
        webhook_handler: Optional["WebhookHandler"] = None,
        job_id: Optional[str] = None,
        poll_interval_seconds: float = 5.0,
        ecc_error_threshold: int = 10,
        cuda_failure_threshold: int = 1,
    ):
        """Initialize the GPU circuit breaker.

        Args:
            webhook_handler: Handler for sending webhook notifications
            job_id: Training job identifier for webhook payloads
            poll_interval_seconds: How often to poll GPU health
            ecc_error_threshold: Number of ECC errors before opening circuit
            cuda_failure_threshold: Number of CUDA failures before opening circuit
        """
        self.webhook_handler = webhook_handler
        self.job_id = job_id
        self.poll_interval = poll_interval_seconds
        self.ecc_error_threshold = ecc_error_threshold
        self.cuda_failure_threshold = cuda_failure_threshold

        self._state = CircuitBreakerState()
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._nvml_initialized = False
        self._gpu_count = 0
        self._baseline_ecc_errors: Dict[int, int] = {}

        # Track cumulative ECC errors per GPU (baseline established at start)
        # Track thermal warnings per GPU (non-fatal, for WebUI display)
        self._thermal_warnings: Dict[int, GPUHealthMetrics] = {}
        self._thermal_warning_lock = threading.Lock()

        self._initialize_nvml()

    def _initialize_nvml(self) -> bool:
        """Initialize NVML for GPU monitoring."""
        if pynvml is None:
            logger.debug("pynvml not available, GPU health monitoring disabled")
            return False

        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"GPU circuit breaker initialized with {self._gpu_count} GPU(s)")
            # Establish baseline ECC error counts
            for i in range(self._gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    errors = self._get_ecc_errors(handle)
                    self._baseline_ecc_errors[i] = errors
                except Exception:
                    self._baseline_ecc_errors[i] = 0
            return True
        except Exception as e:
            logger.debug(f"Failed to initialize NVML: {e}")
            return False

    def _shutdown_nvml(self) -> None:
        """Shutdown NVML."""
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    def _get_ecc_errors(self, handle) -> int:
        """Get total ECC error count for a GPU handle."""
        if not self._nvml_initialized:
            return 0

        total_errors = 0
        try:
            # Try to get volatile double-bit (uncorrectable) errors first
            try:
                errors = pynvml.nvmlDeviceGetTotalEccErrors(
                    handle,
                    pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    pynvml.NVML_VOLATILE_ECC,
                )
                total_errors += errors
            except (pynvml.NVMLError, AttributeError):
                pass

            # Also get aggregate errors
            try:
                errors = pynvml.nvmlDeviceGetTotalEccErrors(
                    handle,
                    pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    pynvml.NVML_AGGREGATE_ECC,
                )
                total_errors += errors
            except (pynvml.NVMLError, AttributeError):
                pass

        except Exception:
            pass

        return total_errors

    def _get_gpu_metrics(self, gpu_index: int) -> Optional[GPUHealthMetrics]:
        """Get health metrics for a specific GPU."""
        if not self._nvml_initialized:
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception as e:
            logger.debug(f"Failed to get handle for GPU {gpu_index}: {e}")
            return None

        metrics = GPUHealthMetrics(index=gpu_index)

        # Get GPU name
        try:
            metrics.name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(metrics.name, bytes):
                metrics.name = metrics.name.decode("utf-8")
        except Exception:
            metrics.name = f"GPU {gpu_index}"

        # Get temperature
        try:
            metrics.temperature_celsius = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass

        # Get temperature thresholds
        try:
            metrics.temperature_threshold_slowdown = pynvml.nvmlDeviceGetTemperatureThreshold(
                handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN
            )
            metrics.temperature_threshold_shutdown = pynvml.nvmlDeviceGetTemperatureThreshold(
                handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN
            )
        except Exception:
            pass

        # Get ECC errors
        try:
            # Single-bit (correctable) errors
            try:
                metrics.ecc_errors_single = pynvml.nvmlDeviceGetTotalEccErrors(
                    handle,
                    pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    pynvml.NVML_VOLATILE_ECC,
                )
            except (pynvml.NVMLError, AttributeError):
                pass

            # Double-bit (uncorrectable) errors
            try:
                metrics.ecc_errors_double = pynvml.nvmlDeviceGetTotalEccErrors(
                    handle,
                    pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    pynvml.NVML_VOLATILE_ECC,
                )
            except (pynvml.NVMLError, AttributeError):
                pass
        except Exception:
            pass

        # Get throttle reasons
        try:
            throttle_mask = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            for mask, reason in THROTTLE_REASONS.items():
                if throttle_mask & mask:
                    metrics.throttle_reasons.append(reason)
        except Exception:
            pass

        # Get memory info
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics.memory_used_bytes = mem_info.used
            metrics.memory_total_bytes = mem_info.total
            if mem_info.total > 0:
                metrics.memory_used_percent = (mem_info.used / mem_info.total) * 100
        except Exception:
            pass

        # Get GPU utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics.gpu_utilization_percent = util.gpu
            metrics.memory_utilization_percent = util.memory
        except Exception:
            pass

        # Get fan speed
        try:
            metrics.fan_speed_percent = pynvml.nvmlDeviceGetFanSpeed(handle)
        except Exception:
            pass

        # Get power usage
        try:
            # Power is returned in milliwatts
            metrics.power_usage_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            pass

        try:
            metrics.power_limit_watts = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except Exception:
            pass

        # Get clock speeds
        try:
            metrics.clock_graphics_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        except Exception:
            pass

        try:
            metrics.clock_memory_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except Exception:
            pass

        # Evaluate health
        metrics.is_healthy = True

        # Check for uncorrectable ECC errors above baseline
        baseline = self._baseline_ecc_errors.get(gpu_index, 0)
        new_ecc_errors = metrics.ecc_errors_double - baseline
        if new_ecc_errors >= self.ecc_error_threshold:
            metrics.is_healthy = False
            metrics.fault_reason = f"Uncorrectable ECC errors detected: {new_ecc_errors} new errors"

        # Check for critical throttling
        critical_throttles = set(metrics.throttle_reasons) & CRITICAL_THROTTLE_REASONS
        if critical_throttles:
            metrics.is_healthy = False
            metrics.fault_reason = f"Critical throttling detected: {', '.join(critical_throttles)}"

        # Check temperature approaching shutdown
        if metrics.temperature_celsius is not None and metrics.temperature_threshold_shutdown is not None:
            temp_margin = metrics.temperature_threshold_shutdown - metrics.temperature_celsius
            if temp_margin < 5:  # Within 5C of shutdown
                metrics.is_healthy = False
                metrics.fault_reason = (
                    f"Temperature critical: {metrics.temperature_celsius}C "
                    f"(shutdown at {metrics.temperature_threshold_shutdown}C)"
                )

        return metrics

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        logger.debug("GPU health monitor started")
        while not self._stop_event.wait(self.poll_interval):
            if self._state.state == CircuitState.OPEN:
                continue  # Already tripped, no need to monitor

            for gpu_index in range(self._gpu_count):
                metrics = self._get_gpu_metrics(gpu_index)
                if metrics is None:
                    continue

                # Check for thermal throttling (non-fatal warning)
                thermal_throttles = set(metrics.throttle_reasons) & WARNING_THROTTLE_REASONS
                if thermal_throttles:
                    self._handle_thermal_warning(gpu_index, metrics, thermal_throttles)
                else:
                    # Clear thermal warning if no longer throttling
                    self._clear_thermal_warning(gpu_index)

                # Check for critical issues (fatal - opens circuit)
                if not metrics.is_healthy:
                    logger.warning(f"GPU {gpu_index} ({metrics.name}) unhealthy: {metrics.fault_reason}")
                    self._open_circuit(
                        fault_type="health_warning",
                        reason=metrics.fault_reason or "Unknown health issue",
                        gpu_index=gpu_index,
                        metrics=metrics,
                    )
                    break

        logger.debug("GPU health monitor stopped")

    def start_monitoring(self) -> None:
        """Start the background health monitoring thread."""
        if not self._nvml_initialized:
            logger.debug("NVML not initialized, skipping GPU health monitoring")
            return

        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True, name="gpu-health-monitor")
        self._monitor_thread.start()
        atexit.register(self.stop_monitoring)

    def stop_monitoring(self) -> None:
        """Stop the background health monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        self._shutdown_nvml()

    def _open_circuit(
        self,
        fault_type: str,
        reason: str,
        gpu_index: Optional[int] = None,
        metrics: Optional[GPUHealthMetrics] = None,
        exception: Optional[Exception] = None,
    ) -> None:
        """Open the circuit breaker and emit webhook."""
        with self._lock:
            if self._state.state == CircuitState.OPEN:
                return  # Already open

            self._state.state = CircuitState.OPEN
            self._state.opened_at = time.time()
            self._state.last_failure_time = time.time()
            self._state.last_failure_reason = reason
            self._state.last_failure_gpu_index = gpu_index

        logger.error(
            f"GPU circuit breaker OPEN: {fault_type} - {reason} "
            f"(GPU {gpu_index if gpu_index is not None else 'unknown'})"
        )

        # Emit webhook
        self._emit_fault_webhook(
            fault_type=fault_type,
            reason=reason,
            gpu_index=gpu_index,
            metrics=metrics,
            exception=exception,
        )

    def _emit_fault_webhook(
        self,
        fault_type: str,
        reason: str,
        gpu_index: Optional[int] = None,
        metrics: Optional[GPUHealthMetrics] = None,
        exception: Optional[Exception] = None,
    ) -> None:
        """Emit a GPU fault webhook."""
        if self.webhook_handler is None:
            logger.debug("No webhook handler configured, skipping webhook emission")
            return

        try:
            from simpletuner.helpers.webhooks.events import attach_timestamp, gpu_fault_event

            event = gpu_fault_event(
                fault_type=fault_type,
                message=reason,
                gpu_index=gpu_index,
                gpu_name=metrics.name if metrics else None,
                job_id=self.job_id,
                severity="critical",
                temperature_celsius=metrics.temperature_celsius if metrics else None,
                ecc_errors_single=metrics.ecc_errors_single if metrics else None,
                ecc_errors_double=metrics.ecc_errors_double if metrics else None,
                throttle_reasons=metrics.throttle_reasons if metrics else None,
                memory_used_percent=metrics.memory_used_percent if metrics else None,
                action_taken="circuit_opened",
                exception_type=type(exception).__name__ if exception else None,
            )
            event = attach_timestamp(event)

            # Send webhook synchronously to ensure delivery before exit
            self.webhook_handler.send(event)
            logger.info("GPU fault webhook sent successfully")
        except Exception as e:
            logger.error(f"Failed to send GPU fault webhook: {e}")

    def _handle_thermal_warning(
        self,
        gpu_index: int,
        metrics: GPUHealthMetrics,
        thermal_throttles: set,
    ) -> None:
        """Handle thermal throttling warning (non-fatal)."""
        with self._thermal_warning_lock:
            was_warning = gpu_index in self._thermal_warnings
            self._thermal_warnings[gpu_index] = metrics

        # Log warning (only on first detection or if new)
        if not was_warning:
            logger.warning(
                f"GPU {gpu_index} ({metrics.name}) thermal throttling: "
                f"{', '.join(thermal_throttles)} at {metrics.temperature_celsius}C"
            )
            # Emit webhook event (non-fatal, just for monitoring)
            self._emit_thermal_warning_webhook(gpu_index, metrics, thermal_throttles)

    def _clear_thermal_warning(self, gpu_index: int) -> None:
        """Clear thermal warning when GPU is no longer throttling."""
        with self._thermal_warning_lock:
            if gpu_index in self._thermal_warnings:
                del self._thermal_warnings[gpu_index]
                logger.info(f"GPU {gpu_index} thermal throttling resolved")

    def _emit_thermal_warning_webhook(
        self,
        gpu_index: int,
        metrics: GPUHealthMetrics,
        thermal_throttles: set,
    ) -> None:
        """Emit a thermal warning webhook (non-fatal, informational)."""
        if self.webhook_handler is None:
            logger.debug("No webhook handler configured, skipping thermal warning webhook")
            return

        try:
            from simpletuner.helpers.webhooks.events import attach_timestamp, gpu_fault_event

            event = gpu_fault_event(
                fault_type="thermal",
                message=f"GPU thermal throttling: {', '.join(thermal_throttles)} at {metrics.temperature_celsius}C",
                gpu_index=gpu_index,
                gpu_name=metrics.name,
                job_id=self.job_id,
                severity="warning",  # Not critical - just a warning
                temperature_celsius=metrics.temperature_celsius,
                throttle_reasons=list(thermal_throttles),
                memory_used_percent=metrics.memory_used_percent,
                action_taken=None,  # No action taken - training continues
            )
            event = attach_timestamp(event)

            self.webhook_handler.send(event)
            logger.info("GPU thermal warning webhook sent")
        except Exception as e:
            logger.error(f"Failed to send GPU thermal warning webhook: {e}")

    def get_gpu_thermal_status(self) -> List[Dict[str, Any]]:
        """Get full status for all GPUs (for WebUI display).

        Returns:
            List of GPU status dicts with all metrics for dashboard display
        """
        gpu_statuses = []
        with self._thermal_warning_lock:
            thermal_warnings = dict(self._thermal_warnings)

        for gpu_index in range(self._gpu_count):
            metrics = self._get_gpu_metrics(gpu_index)
            if metrics is None:
                continue

            status = {
                "index": gpu_index,
                "name": metrics.name,
                # Temperature
                "temperature_celsius": metrics.temperature_celsius,
                "temperature_threshold_slowdown": metrics.temperature_threshold_slowdown,
                "temperature_threshold_shutdown": metrics.temperature_threshold_shutdown,
                "is_thermal_throttling": gpu_index in thermal_warnings,
                "throttle_reasons": metrics.throttle_reasons,
                # VRAM
                "memory_used_bytes": metrics.memory_used_bytes,
                "memory_total_bytes": metrics.memory_total_bytes,
                "memory_used_percent": metrics.memory_used_percent,
                # Utilization
                "gpu_utilization_percent": metrics.gpu_utilization_percent,
                "memory_utilization_percent": metrics.memory_utilization_percent,
                # Fan
                "fan_speed_percent": metrics.fan_speed_percent,
                # Power
                "power_usage_watts": metrics.power_usage_watts,
                "power_limit_watts": metrics.power_limit_watts,
                # Clocks
                "clock_graphics_mhz": metrics.clock_graphics_mhz,
                "clock_memory_mhz": metrics.clock_memory_mhz,
            }
            gpu_statuses.append(status)

        return gpu_statuses

    @property
    def is_open(self) -> bool:
        """Check if the circuit is open (fault detected)."""
        with self._lock:
            return self._state.state == CircuitState.OPEN

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state.state

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        with self._lock:
            return {
                "state": self._state.state.value,
                "failure_count": self._state.failure_count,
                "last_failure_time": self._state.last_failure_time,
                "last_failure_reason": self._state.last_failure_reason,
                "last_failure_gpu_index": self._state.last_failure_gpu_index,
                "opened_at": self._state.opened_at,
                "gpu_count": self._gpu_count,
                "nvml_initialized": self._nvml_initialized,
            }

    def record_cuda_error(self, error: Exception, gpu_index: Optional[int] = None) -> None:
        """Record a CUDA error and potentially open the circuit.

        Args:
            error: The CUDA exception that occurred
            gpu_index: GPU index if known
        """
        should_open = False
        with self._lock:
            self._state.failure_count += 1
            self._state.last_failure_time = time.time()

            if self._state.failure_count >= self.cuda_failure_threshold:
                should_open = True

        if should_open:
            # Get current metrics if possible (outside lock to avoid deadlock)
            metrics = None
            if gpu_index is not None:
                metrics = self._get_gpu_metrics(gpu_index)

            self._open_circuit(
                fault_type="cuda_error",
                reason=str(error),
                gpu_index=gpu_index,
                metrics=metrics,
                exception=error,
            )

    def check_health(self) -> bool:
        """Check GPU health synchronously.

        Returns:
            True if all GPUs are healthy, False if circuit should open.
        """
        if not self._nvml_initialized:
            return True

        for gpu_index in range(self._gpu_count):
            metrics = self._get_gpu_metrics(gpu_index)
            if metrics and not metrics.is_healthy:
                self._open_circuit(
                    fault_type="health_check",
                    reason=metrics.fault_reason or "Health check failed",
                    gpu_index=gpu_index,
                    metrics=metrics,
                )
                return False

        return True


# Global circuit breaker instance
_circuit_breaker: Optional[GPUCircuitBreaker] = None
_circuit_breaker_lock = threading.Lock()


def get_gpu_circuit_breaker(
    webhook_handler: Optional["WebhookHandler"] = None,
    job_id: Optional[str] = None,
) -> GPUCircuitBreaker:
    """Get or create the global GPU circuit breaker instance.

    Args:
        webhook_handler: Handler for webhook notifications (only used on creation)
        job_id: Training job ID (only used on creation)

    Returns:
        The global GPUCircuitBreaker instance
    """
    global _circuit_breaker
    with _circuit_breaker_lock:
        if _circuit_breaker is None:
            _circuit_breaker = GPUCircuitBreaker(
                webhook_handler=webhook_handler,
                job_id=job_id,
            )
        return _circuit_breaker


def reset_gpu_circuit_breaker() -> None:
    """Reset the global circuit breaker (for testing)."""
    global _circuit_breaker
    with _circuit_breaker_lock:
        if _circuit_breaker is not None:
            _circuit_breaker.stop_monitoring()
            _circuit_breaker = None


def is_cuda_error(error: Exception) -> bool:
    """Check if an exception is a CUDA-related error.

    Args:
        error: Exception to check

    Returns:
        True if this is a CUDA error that should trigger the circuit breaker
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    cuda_indicators = [
        "cuda",
        "cudnn",
        "cublas",
        "nccl",
        "device-side assert",
        "out of memory",
        "illegal memory access",
    ]

    # Check exception type - only "cuda" in type name is definitive
    if "cuda" in error_type:
        return True

    # Check error message for CUDA indicators
    for indicator in cuda_indicators:
        if indicator in error_str:
            return True

    return False


def get_current_gpu_index() -> Optional[int]:
    """Get the current GPU index being used.

    Returns:
        GPU index or None if not determinable
    """
    if torch is None:
        return None

    try:
        if torch.cuda.is_available():
            return torch.cuda.current_device()
    except Exception:
        pass

    return None
