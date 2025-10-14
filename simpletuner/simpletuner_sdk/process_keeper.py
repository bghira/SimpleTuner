"""
Process keeper for managing training subprocesses with proper termination support.
Uses subprocess module for true process isolation and termination capability.
"""

import json
import logging
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Optional dependency; used for robust process tree termination
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional
    psutil = None

logger = logging.getLogger("ProcessKeeper")

# Registry of active processes
process_registry: Dict[str, Dict[str, Any]] = {}
# Lock for thread-safe access to registry
lock = threading.Lock()


class TrainerProcess:
    """Wrapper for a training subprocess with IPC communication."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.process: Optional[subprocess.Popen] = None
        self.status = "pending"
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.last_heartbeat = time.time()
        self.stop_event = threading.Event()
        self.event_thread = None
        self.events = []
        self.events_lock = threading.Lock()
        self.output_thread = None
        # Create temp directory for IPC
        self.ipc_dir = tempfile.mkdtemp(prefix=f"trainer_{job_id}_")
        logger.info(f"IPC dir {self.ipc_dir}")
        self.command_file = os.path.join(self.ipc_dir, "commands.json")
        self.event_file = os.path.join(self.ipc_dir, "events.json")
        self.func_file = os.path.join(self.ipc_dir, "func.pkl")

        # Initialize command file
        with open(self.command_file, "w") as f:
            json.dump([], f)

        # Initialize event file
        with open(self.event_file, "w") as f:
            json.dump([], f)

    def start(self, target_func, config: Dict[str, Any]):
        """Start the trainer subprocess."""
        # Get function module and name for import
        func_module = target_func.__module__
        func_name = target_func.__name__

        # Save function info as JSON instead of pickling
        func_info = {"module": func_module, "name": func_name}
        with open(self.func_file, "w") as f:
            json.dump(func_info, f)

        # Create the subprocess runner script inline
        runner_code = f'''
import sys
import os
import json
import signal
import time
import threading
import logging

# Add parent directories to path for imports
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SubprocessRunner")

# IPC paths
command_file = r"{self.command_file}"
event_file = r"{self.event_file}"
func_file = r"{self.func_file}"

# State
should_abort = False
command_check_thread = None

def send_event(event_type, data=None):
    """Send event back to parent process."""
    event = {{
        "type": event_type,
        "timestamp": time.time(),
        "data": data or {{}}
    }}

    # Read existing events
    try:
        with open(event_file, 'r') as f:
            events = json.load(f)
    except:
        events = []

    # Append new event
    events.append(event)

    # Write back
    try:
        with open(event_file, 'w') as f:
            json.dump(events, f)
    except Exception as e:
        logger.error(f"Failed to write event: {{e}}")

# Load function info
with open(func_file, 'r') as f:
    func_info = json.load(f)

# Import the function
try:
    import importlib
    logger.info(f"Importing module: {func_info['module']}")
    module = importlib.import_module(func_info['module'])
    logger.info(f"Getting function: {func_info['name']}")
    target_func = getattr(module, func_info['name'])
    logger.info("Function loaded successfully")
except Exception as e:
    logger.error(f"Failed to import function: {{e}}")
    send_event("error", {{"message": f"Import error: {{e}}"}})
    sys.exit(1)

# Load config from JSON string
config_json = {repr(json.dumps(config))}
config = json.loads(config_json)

def check_commands():
    """Check for commands from parent process."""
    global should_abort

    while not should_abort:
        try:
            with open(command_file, 'r') as f:
                commands = json.load(f)

            if commands:
                # Process the first command
                cmd = commands.pop(0)

                if cmd.get("command") == "abort":
                    logger.info("Received abort command")
                    should_abort = True
                    send_event("state", {{"status": "aborting"}})

                # Write back remaining commands
                with open(command_file, 'w') as f:
                    json.dump(commands, f)

        except Exception as e:
            logger.error(f"Command check error: {{e}}")

        time.sleep(0.1)  # Check every 100ms

def handle_sigterm(signum, frame):
    """Handle SIGTERM gracefully."""
    global should_abort
    logger.info("Received SIGTERM")
    should_abort = True
    send_event("state", {{"status": "terminated"}})

# Set up signal handler
signal.signal(signal.SIGTERM, handle_sigterm)

# Start command checker thread
command_check_thread = threading.Thread(target=check_commands, daemon=True)
command_check_thread.start()

# Send starting event
send_event("state", {{"status": "running"}})

# Create wrapper that can check should_abort
class ConfigWrapper:
    def __init__(self, config):
        self.__dict__.update(config)
        self.should_abort = lambda: should_abort

wrapped_config = ConfigWrapper(config)

# Run the target function
try:
    logger.info("Starting target function")
    result = target_func(wrapped_config)
    send_event("state", {{"status": "completed", "result": str(result)}})
except Exception as e:
    logger.error(f"Function error: {{e}}")
    import traceback
    traceback_str = traceback.format_exc()
    logger.error(traceback_str)
    send_event("error", {{"message": str(e)}})
    send_event("state", {{"status": "failed"}})

logger.info("Subprocess exiting")
'''

        env = os.environ.copy()
        env.setdefault("WANDB_CONSOLE", "off")
        env.setdefault("WANDB_SILENT", "true")
        env.setdefault("WANDB_DISABLE_SERVICE", "true")
        env.setdefault("WANDB_STDOUT_CAPTURE", "off")
        env.setdefault("WANDB_STDERR_CAPTURE", "off")

        # Start the subprocess
        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "env": env,
        }

        if os.name != "nt":
            popen_kwargs["preexec_fn"] = os.setsid
        else:  # pragma: no cover - Windows specific
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        self.process = subprocess.Popen([sys.executable, "-c", runner_code], **popen_kwargs)

        self.status = "running"

        # Start event listener thread
        self.event_thread = threading.Thread(target=self._event_listener, daemon=True)
        self.event_thread.start()

        # Stream stdout to avoid blocking pipes and preserve logs
        if self.process.stdout is not None:
            self.output_thread = threading.Thread(target=self._stream_output, daemon=True)
            self.output_thread.start()

    def _event_listener(self):
        """Listen for events from the subprocess."""
        while not self.stop_event.is_set():
            try:
                # Check if process is still alive
                if self.process and self.process.poll() is not None:
                    # Process has exited - do one final check for events
                    try:
                        with open(self.event_file, "r") as f:
                            all_events = json.load(f)

                        # Process any remaining events
                        if len(all_events) > len(self.events):
                            new_events = all_events[len(self.events) :]
                            with self.events_lock:
                                self.events.extend(new_events)

                            # Handle each new event
                            for event in new_events:
                                self._handle_event(event)
                    except Exception as e:
                        logger.debug(f"Final event read error: {e}")

                    if self.status == "running":
                        # Only update if no status was set via events
                        self.status = "completed" if self.process.returncode == 0 else "failed"
                    break

                # Read events file
                try:
                    with open(self.event_file, "r") as f:
                        all_events = json.load(f)

                    # Process new events
                    if len(all_events) > len(self.events):
                        new_events = all_events[len(self.events) :]
                        with self.events_lock:
                            self.events.extend(new_events)

                        # Handle each new event
                        for event in new_events:
                            self._handle_event(event)

                except Exception as e:
                    if not self.stop_event.is_set():
                        logger.debug(f"Event read error: {e}")

                time.sleep(0.1)  # Poll every 100ms

            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"Event listener error: {e}")
                break

    def _stream_output(self):
        """Continuously read subprocess stdout to prevent pipe backpressure."""
        if not self.process or self.process.stdout is None:
            return

        try:
            for line in iter(self.process.stdout.readline, ""):
                if line == "":
                    break
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except Exception:
                    # If stdout is unavailable just log the line
                    logger.info(line.rstrip())
        except Exception as exc:
            if not self.stop_event.is_set():
                logger.debug(f"stdout streaming error for {self.job_id}: {exc}")

    def _handle_event(self, event: Dict[str, Any]):
        """Handle an event from the trainer subprocess."""
        self.last_heartbeat = time.time()

        # Store event for retrieval
        with lock:
            if self.job_id in process_registry:
                if "events" not in process_registry[self.job_id]:
                    process_registry[self.job_id]["events"] = []
                process_registry[self.job_id]["events"].append(event)

                # Update status from events
                if event.get("type") == "state":
                    state_data = event.get("data", {})
                    if "status" in state_data:
                        # Don't override terminated status with aborting
                        if not (self.status == "terminated" and state_data["status"] == "aborting"):
                            self.status = state_data["status"]
                            process_registry[self.job_id]["status"] = self.status

    def send_command(self, command: str, data: Optional[Dict] = None):
        """Send a command to the trainer subprocess."""
        if not self.process or self.process.poll() is not None:
            raise Exception(f"Process {self.job_id} is not running")

        message = {"command": command, "timestamp": time.time()}
        if data:
            message["data"] = data

        # Read existing commands
        try:
            with open(self.command_file, "r") as f:
                commands = json.load(f)
        except:
            commands = []

        # Append new command
        commands.append(message)

        # Write back
        with open(self.command_file, "w") as f:
            json.dump(commands, f)

    def _append_event(self, event_type: str, data: Optional[Dict] = None):
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data or {},
        }

        try:
            with open(self.event_file, "r") as f:
                events = json.load(f)
        except Exception:
            events = []

        events.append(event)

        try:
            with open(self.event_file, "w") as f:
                json.dump(events, f)
        except Exception as exc:
            logger.debug(f"Failed to append event for {self.job_id}: {exc}")

    def terminate(self, timeout: int = 5) -> bool:
        """Terminate the subprocess, first gracefully then forcefully."""
        if not self.process:
            return True

        try:
            # Signal the event listener to stop
            self.stop_event.set()

            # Only try to send abort if process is still running
            if self.process.poll() is None:
                try:
                    self.send_command("abort")
                except Exception:
                    pass  # Process might not be responsive

                self._force_kill_process_tree(timeout)

            # Clean up resources
            self._cleanup_resources()

            self.status = "terminated"
            self.end_time = datetime.now().isoformat()
            self._append_event("state", {"status": "terminated"})

            # Note: Registry update is handled by the caller (terminate_process)
            # since it already holds the lock. Attempting to acquire lock here
            # would cause a deadlock.

            return True

        except Exception as e:
            logger.error(f"Error terminating process {self.job_id}: {e}")
        return False

    def _force_kill_process_tree(self, timeout: int) -> None:
        """Forcefully terminate the subprocess and any children."""

        if not self.process or self.process.poll() is not None:
            return

        pid = self.process.pid

        def _kill_with_psutil(proc_pid: int) -> None:
            if not psutil:
                return

            try:
                parent = psutil.Process(proc_pid)
            except Exception:
                return

            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass

            try:
                parent.kill()
            except Exception:
                pass

        try:
            if os.name != "nt" and hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception as exc:
                    logger.debug(f"killpg failed for {self.job_id}: {exc}")
            else:  # pragma: no cover - Windows specific
                try:
                    self.process.kill()
                except Exception as exc:
                    logger.debug(f"kill() failed for {self.job_id}: {exc}")

            _kill_with_psutil(pid)

            try:
                self.process.wait(timeout=timeout)
            except Exception:
                pass
        except Exception as exc:
            logger.debug(f"Error forcing kill for {self.job_id}: {exc}")

    def _cleanup_resources(self):
        """Clean up all resources associated with this process."""
        # Stop event listener thread with shorter timeout to prevent hanging
        if self.event_thread and self.event_thread.is_alive():
            self.stop_event.set()
            # Use shorter timeout - thread is daemon anyway
            self.event_thread.join(timeout=0.5)
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=0.5)
        self.output_thread = None

        # Clean up IPC directory
        try:
            import shutil

            if os.path.exists(self.ipc_dir):
                shutil.rmtree(self.ipc_dir)
        except Exception as e:
            logger.debug(f"Failed to clean up IPC dir: {e}")

        # Close subprocess pipes safely
        if self.process:
            self._safe_close_pipes()

        # Clear process reference
        self.process = None

    def _safe_close_pipes(self):
        """Safely close subprocess pipes without hanging."""
        if not self.process:
            return

        pipes_to_close = [("stdout", self.process.stdout), ("stderr", self.process.stderr), ("stdin", self.process.stdin)]

        for pipe_name, pipe in pipes_to_close:
            if pipe is None:
                continue
            try:
                # For stdout/stderr, try to drain remaining data first
                if pipe_name in ["stdout", "stderr"] and hasattr(pipe, "read"):
                    try:
                        import select

                        if hasattr(select, "select"):
                            # Quick non-blocking check
                            readable, _, _ = select.select([pipe], [], [], 0)
                            if readable:
                                # Drain available data
                                pipe.read()
                    except Exception:
                        pass

                # Close the pipe
                pipe.close()
            except Exception as e:
                logger.debug(f"Error closing {pipe_name}: {e}")
                # Continue with other pipes

    def is_alive(self) -> bool:
        """Check if the process is still running."""
        return self.process and self.process.poll() is None

    def get_output(self, timeout=0.1):
        """Get any available output from the process (for debugging)."""
        if not self.process:
            return None, None

        # Don't try to read from completed processes to avoid hanging
        if self.process.poll() is not None:
            return None, None

        try:
            # Non-blocking read of available output
            import select

            if hasattr(select, "select"):
                # Unix-like systems
                if self.process.stdout:
                    readable, _, _ = select.select([self.process.stdout], [], [], timeout)
                    if readable:
                        # Use limited read to avoid blocking
                        data = self.process.stdout.read(1024)
                        return data, None
                return None, None
            else:
                # Windows - avoid communicate on live process
                return None, None
        except:
            return None, None


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
            # Don't override completed/failed/terminated status with crashed
            if process.status == "running":
                # Give the event thread a moment to process final events
                if process.event_thread and process.event_thread.is_alive():
                    # Event thread is still running, wait briefly for it to finish
                    # Release lock while waiting to avoid deadlock
                    event_thread = process.event_thread
                    lock.release()
                    try:
                        event_thread.join(timeout=0.3)
                    finally:
                        lock.acquire()

                    # Re-check status after waiting
                    if process.status == "running":
                        # Event thread finished but status still running = crashed
                        process.status = "crashed"
                        entry["status"] = "crashed"
                else:
                    # Event thread finished, if still running then it crashed
                    process.status = "crashed"
                    entry["status"] = "crashed"
            elif process.status not in ["completed", "failed", "terminated"]:
                # Update entry status to match process status
                entry["status"] = process.status

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
