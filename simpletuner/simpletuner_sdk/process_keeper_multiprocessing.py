"""
Process keeper for managing training subprocesses with proper termination support.
Uses multiprocessing for true process isolation and termination capability.
"""

import json
import logging
import multiprocessing
import os
import signal
import threading
import time
from datetime import datetime
from multiprocessing import Pipe, Process, Queue
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ProcessKeeper")

# Registry of active processes
process_registry: Dict[str, Dict[str, Any]] = {}
# Lock for thread-safe access to registry
lock = threading.Lock()


class TrainerProcess:
    """Wrapper for a training subprocess with IPC communication."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.process: Optional[Process] = None
        self.command_queue = Queue()  # Commands to trainer
        self.event_pipe_recv, self.event_pipe_send = Pipe(duplex=False)  # Events from trainer
        self.status = "pending"
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.last_heartbeat = time.time()
        self.stop_event = None  # Will be created after process starts
        self.event_thread = None

    def start(self, target_func, config: Dict[str, Any]):
        """Start the trainer subprocess."""
        # Serialize config for passing to subprocess
        serialized_config = json.dumps(config)

        # Create stop event after process creation to avoid pickling issues
        self.stop_event = threading.Event()

        self.process = Process(
            target=_run_trainer_process,
            args=(target_func, serialized_config, self.command_queue, self.event_pipe_send),
            name=f"trainer_{self.job_id}",
        )
        self.process.start()
        self.status = "running"

        # Start event listener thread
        self.event_thread = threading.Thread(target=self._event_listener, daemon=True)
        self.event_thread.start()

    def _event_listener(self):
        """Listen for events from the subprocess."""
        while self.stop_event is None or not self.stop_event.is_set():
            try:
                # Check if process is still alive
                if self.process and not self.process.is_alive():
                    break

                # Poll with timeout to allow checking stop_event
                if self.event_pipe_recv.poll(timeout=0.1):
                    event = self.event_pipe_recv.recv()
                    self._handle_event(event)
            except (EOFError, BrokenPipeError):
                # Pipe closed, exit gracefully
                break
            except Exception as e:
                if not (self.stop_event and self.stop_event.is_set()):
                    logger.error(f"Event listener error: {e}")
                break

    def _handle_event(self, event: Dict[str, Any]):
        """Handle an event from the trainer subprocess."""
        self.last_heartbeat = time.time()

        # Store event for retrieval
        with lock:
            if self.job_id in process_registry:
                if "events" not in process_registry[self.job_id]:
                    process_registry[self.job_id]["events"] = []
                process_registry[self.job_id]["events"].append(event)

                # Update status from events (but not if we're terminating)
                if event.get("type") == "state":
                    state_data = event.get("data", {})
                    if "status" in state_data:
                        # Don't override terminated status with aborting
                        if not (self.status == "terminated" and state_data["status"] == "aborting"):
                            self.status = state_data["status"]
                            process_registry[self.job_id]["status"] = self.status

    def send_command(self, command: str, data: Optional[Dict] = None):
        """Send a command to the trainer subprocess."""
        if not self.process or not self.process.is_alive():
            raise Exception(f"Process {self.job_id} is not running")

        message = {"command": command}
        if data:
            message["data"] = data

        self.command_queue.put(message)

    def terminate(self, timeout: int = 5) -> bool:
        """Terminate the subprocess, first gracefully then forcefully."""
        if not self.process:
            return True

        try:
            # Signal the event listener to stop
            if self.stop_event:
                self.stop_event.set()

            # Try to send abort command first (may fail if process not responsive)
            try:
                self.send_command("abort")
            except:
                pass  # Process might not be responsive

            # Wait for graceful shutdown
            self.process.join(timeout=timeout)

            if self.process.is_alive():
                # Force terminate if still running
                logger.warning(f"Force terminating process {self.job_id}")
                self.process.terminate()
                self.process.join(timeout=2)

                if self.process.is_alive():
                    # Kill as last resort
                    logger.warning(f"Force killing process {self.job_id}")
                    self.process.kill()
                    self.process.join(timeout=1)

            # Clean up resources
            self._cleanup_resources()

            self.status = "terminated"
            self.end_time = datetime.now().isoformat()

            # Update registry immediately
            with lock:
                if self.job_id in process_registry:
                    process_registry[self.job_id]["status"] = "terminated"

            return True

        except Exception as e:
            logger.error(f"Error terminating process {self.job_id}: {e}")
            return False

    def _cleanup_resources(self):
        """Clean up all resources associated with this process."""
        # Stop event listener thread
        if self.event_thread and self.event_thread.is_alive():
            if self.stop_event:
                self.stop_event.set()
            self.event_thread.join(timeout=1)

        # Close pipes
        try:
            if hasattr(self, "event_pipe_recv"):
                self.event_pipe_recv.close()
            if hasattr(self, "event_pipe_send"):
                self.event_pipe_send.close()
        except:
            pass

        # Clear command queue
        try:
            while not self.command_queue.empty():
                self.command_queue.get_nowait()
        except:
            pass

        # Clear process reference to allow GC
        self.process = None

    def is_alive(self) -> bool:
        """Check if the process is still running."""
        return self.process and self.process.is_alive()


def _run_trainer_process(target_func, serialized_config: str, command_queue: Queue, event_pipe: Any):
    """Run the trainer in subprocess context (standalone function for pickling)."""
    # Force colors to be enabled in subprocess (stdout is piped so TTY detection fails)
    # Remove SIMPLETUNER_WEB_MODE so subprocess can use colors even when launched from web UI
    os.environ.pop("SIMPLETUNER_WEB_MODE", None)
    os.environ.pop("SIMPLETUNER_DISABLE_COLORS", None)
    os.environ["FORCE_COLOR"] = "1"
    os.environ["CLICOLOR_FORCE"] = "1"

    try:
        # Deserialize config
        config = json.loads(serialized_config)

        # Create a wrapper that handles commands and sends events
        from simpletuner.simpletuner_sdk.subprocess_wrapper import SubprocessTrainerWrapper

        wrapper = SubprocessTrainerWrapper(command_queue, event_pipe)

        # Run the actual trainer with wrapper hooks
        wrapper.run_trainer(target_func, config)

    except Exception as e:
        # Send error event
        event_pipe.send({"type": "error", "data": {"message": str(e), "traceback": str(e.__traceback__)}})
        raise


def submit_job(job_id: str, func, config: Dict[str, Any]) -> TrainerProcess:
    """Submit a new training job as a subprocess."""
    with lock:
        # Check if job already exists and is running
        if job_id in process_registry:
            existing = process_registry[job_id].get("process")
            if existing and existing.is_alive():
                raise Exception(f"Job {job_id} is already running")

        # Create new process wrapper
        trainer_process = TrainerProcess(job_id)

        # Register it
        process_registry[job_id] = {
            "process": trainer_process,
            "status": "pending",
            "start_time": trainer_process.start_time,
            "events": [],
        }

        # Start the process
        trainer_process.start(func, config)

        return trainer_process


def get_process_status(job_id: str) -> str:
    """Get the status of a process."""
    with lock:
        if job_id not in process_registry:
            return "not_found"

        entry = process_registry[job_id]
        process = entry.get("process")

        if not process:
            return "no_process"

        # Check if process is actually alive
        if not process.is_alive():
            if process.status == "running":
                # Process died unexpectedly
                process.status = "crashed"
                entry["status"] = "crashed"

        return process.status


def terminate_process(job_id: str) -> bool:
    """Terminate a running process."""
    with lock:
        if job_id not in process_registry:
            return False

        process = process_registry[job_id].get("process")
        if not process:
            return False

        success = process.terminate()

        if success:
            # Update registry
            process_registry[job_id]["status"] = "terminated"
            process_registry[job_id]["end_time"] = process.end_time

        return success


def send_process_command(job_id: str, command: str, data: Optional[Dict] = None):
    """Send a command to a running process."""
    with lock:
        if job_id not in process_registry:
            raise Exception(f"Job {job_id} not found")

        process = process_registry[job_id].get("process")
        if not process:
            raise Exception(f"No process for job {job_id}")

        process.send_command(command, data)


def get_process_events(job_id: str, since_index: int = 0) -> List[Dict[str, Any]]:
    """Get events from a process since a given index."""
    with lock:
        if job_id not in process_registry:
            return []

        events = process_registry[job_id].get("events", [])
        return events[since_index:]


def list_processes() -> Dict[str, Dict[str, Any]]:
    """List all registered processes and their status."""
    with lock:
        result = {}
        for job_id, entry in process_registry.items():
            process = entry.get("process")
            result[job_id] = {
                "status": process.status if process else "unknown",
                "start_time": entry.get("start_time"),
                "end_time": entry.get("end_time"),
                "alive": process.is_alive() if process else False,
            }
        return result


def cleanup_dead_processes():
    """Clean up entries for dead processes."""
    with lock:
        dead_jobs = []
        for job_id, entry in process_registry.items():
            process = entry.get("process")
            if process and not process.is_alive():
                if process.status == "running":
                    process.status = "crashed"
                    entry["status"] = "crashed"
                if time.time() - process.last_heartbeat > 300:  # 5 minutes
                    dead_jobs.append(job_id)

        for job_id in dead_jobs:
            logger.info(f"Cleaning up dead process {job_id}")
            del process_registry[job_id]


# Set up periodic cleanup
def start_cleanup_thread():
    """Start a background thread to clean up dead processes."""

    def cleanup_loop():
        while True:
            time.sleep(60)  # Check every minute
            try:
                cleanup_dead_processes()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()


# Start cleanup on import
start_cleanup_thread()
