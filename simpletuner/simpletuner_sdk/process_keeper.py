"""
Process keeper for managing training subprocesses with proper termination support.
Uses subprocess module for true process isolation and termination capability.
"""

import json
import logging
import marshal
import os
import pickle
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from simpletuner.helpers.log_format import strip_ansi

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
        self._relayed_failure = False
        self._relayed_completion = False
        # IPC paths are initialized when the subprocess starts (needs config)
        self.ipc_dir: Optional[str] = None
        self.command_file: Optional[str] = None
        self.event_file: Optional[str] = None
        self.func_file: Optional[str] = None
        self.log_file: Optional[str] = None

    def _build_callable_payload(self, target_func) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "mode": "callable",
            "name": target_func.__name__,
            "code": marshal.dumps(target_func.__code__),
            "defaults": target_func.__defaults__,
            "kwdefaults": target_func.__kwdefaults__,
            "globals": {
                "modules": {},
                "values": {},
            },
        }

        global_modules: Dict[str, str] = payload["globals"]["modules"]
        global_values: Dict[str, Any] = payload["globals"]["values"]

        func_globals = getattr(target_func, "__globals__", {}) or {}

        for name in set(target_func.__code__.co_names or ()):  # type: ignore[attr-defined]
            if name not in func_globals:
                continue

            value = func_globals[name]

            if isinstance(value, types.ModuleType):
                global_modules[name] = value.__name__
                continue

            try:
                pickle.dumps(value)
            except Exception:
                continue

            global_values[name] = value

        return payload

    def _resolve_runtime_base(self, config: Optional[Any]) -> Path:
        """Determine a writable base directory for IPC files."""
        candidates: List[Path] = []

        def _coerce_path(value: Optional[Any]) -> Optional[Path]:
            if not value:
                return None
            try:
                return Path(str(value)).expanduser().resolve()
            except Exception:
                return None

        output_dir = None
        if isinstance(config, Mapping):
            output_dir = config.get("output_dir") or config.get("--output_dir")
        else:
            output_dir = getattr(config, "output_dir", None)
        output_path = _coerce_path(output_dir)
        if output_path is not None:
            candidates.append(output_path / ".simpletuner_runtime")

        env_runtime = _coerce_path(os.environ.get("SIMPLETUNER_RUNTIME_DIR"))
        if env_runtime is not None:
            candidates.append(env_runtime)

        # Always fall back to the system temporary directory
        candidates.append(Path(tempfile.gettempdir()))

        for base in candidates:
            try:
                base.mkdir(parents=True, exist_ok=True)
                ipc_path = Path(tempfile.mkdtemp(prefix=f"trainer_{self.job_id}_", dir=str(base)))
                return ipc_path
            except Exception as exc:  # pragma: no cover - best effort, continue to fallback
                logger.debug(f"Failed to create IPC dir in {base}: {exc}")

        # Final fallback - let mkdtemp choose location
        return Path(tempfile.mkdtemp(prefix=f"trainer_{self.job_id}_"))

    def _initialize_ipc_paths(self, config: Optional[Any]) -> None:
        if self.ipc_dir is not None:
            return

        ipc_path = self._resolve_runtime_base(config)
        self.ipc_dir = str(ipc_path)
        logger.info(f"IPC dir {self.ipc_dir}")
        self.command_file = os.path.join(self.ipc_dir, "commands.json")
        self.event_file = os.path.join(self.ipc_dir, "events.json")
        self.func_file = os.path.join(self.ipc_dir, "func.pkl")
        self.log_file = os.path.join(self.ipc_dir, "stdout.log")

        # Initialize command and event files
        with open(self.command_file, "w") as f:
            json.dump([], f)
        with open(self.event_file, "w") as f:
            json.dump([], f)
        if self.log_file:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("")

    def start(self, target_func, config: Dict[str, Any]):
        """Start the trainer subprocess."""
        self._initialize_ipc_paths(config)

        # Prepare callable payload
        func_module = target_func.__module__
        func_name = target_func.__name__

        try:
            func_payload = self._build_callable_payload(target_func)
        except Exception as exc:
            logger.debug(f"Falling back to module import for {func_module}.{func_name}: {exc}")
            func_payload = {
                "mode": "module",
                "module": func_module,
                "name": func_name,
            }

        with open(self.func_file, "wb") as f:
            pickle.dump(func_payload, f)

        # Create the subprocess runner script inline
        runner_code = f'''
import sys
import os
import json
import pickle
import marshal
import signal
import time
import threading
import logging
import types
import importlib
from collections.abc import Mapping

# Add parent directories to path for imports
import os

# Wrap stdout to make it appear as a TTY for tqdm
class TTYWrapper:
    def __init__(self, stream):
        self._stream = stream

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def isatty(self):
        return True

    def write(self, data):
        return self._stream.write(data)

    def flush(self):
        return self._stream.flush()

    def fileno(self):
        # Return underlying fileno if available
        if hasattr(self._stream, 'fileno'):
            try:
                return self._stream.fileno()
            except Exception:
                pass
        return 1  # stdout fileno

# Replace stdout with TTY wrapper so tqdm uses dynamic updates
sys.stdout = TTYWrapper(sys.stdout)
sys.stderr = TTYWrapper(sys.stderr)

# Also set environment variables that tqdm checks
os.environ['TERM'] = 'xterm-256color'

# Monkey-patch tqdm to force dynamic mode after it's imported
_original_tqdm = None
_tqdm_patched = False

def _patch_tqdm():
    """Patch tqdm to force it to use dynamic updates even when piped."""
    global _tqdm_patched
    if _tqdm_patched:
        return

    try:
        import tqdm.std

        # Store original
        original_init = tqdm.std.tqdm.__init__

        def patched_init(self, *args, **kwargs):
            # Force file to use our wrapped stdout
            if 'file' not in kwargs:
                kwargs['file'] = sys.stdout

            # Call original init
            original_init(self, *args, **kwargs)

            # Override TTY detection - force dynamic mode
            if hasattr(self, 'fp') and hasattr(self.fp, 'isatty'):
                # Create a wrapper that always returns True for isatty
                original_fp = self.fp

                class ForceTTY:
                    def __getattr__(self, name):
                        if name == 'isatty':
                            return lambda: True
                        return getattr(original_fp, name)
                    def write(self, *args, **kwargs):
                        return original_fp.write(*args, **kwargs)
                    def flush(self, *args, **kwargs):
                        return original_fp.flush(*args, **kwargs) if hasattr(original_fp, 'flush') else None

                self.fp = ForceTTY()

        tqdm.std.tqdm.__init__ = patched_init
        _tqdm_patched = True
    except Exception as e:
        # If patching fails, just continue - the TTYWrapper might still work
        logger.debug(f"Failed to patch tqdm: {{e}}")
        pass

# Set up logging
log_level_name = os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, log_level_name, logging.WARNING)
logging.basicConfig(level=log_level)
logger = logging.getLogger("SubprocessRunner")
logger.setLevel(log_level)

# IPC paths
command_file = r"{self.command_file}"
event_file = r"{self.event_file}"
func_file = r"{self.func_file}"

# State
should_abort = False
command_check_thread = None
manual_validation_event = threading.Event()
manual_checkpoint_event = threading.Event()


def consume_manual_validation_request():
    """Return True if a manual validation trigger was pending and clear it."""
    if manual_validation_event.is_set():
        manual_validation_event.clear()
        return True
    return False


def consume_manual_checkpoint_request():
    """Return True if a manual checkpoint trigger was pending and clear it."""
    if manual_checkpoint_event.is_set():
        manual_checkpoint_event.clear()
        return True
    return False


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
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Failed to write event: {{e}}")

# Load function payload
try:
    with open(func_file, 'rb') as f:
        func_payload = pickle.load(f)
except Exception as e:
    logger.error(f"Failed to load function payload: {{e}}")
    payload_error = f"Payload error: {{e}}"
    send_event("error", {{"message": payload_error}})
    send_event("state", {{"status": "failed", "message": payload_error}})
    time.sleep(0.2)
    sys.exit(1)

mode = func_payload.get("mode")

if mode == "callable":
    try:
        globals_dict = {{"__builtins__": __builtins__}}
        globals_meta = func_payload.get("globals", {{}}) or {{}}
        modules_meta = globals_meta.get("modules", {{}}) or {{}}
        values_meta = globals_meta.get("values", {{}}) or {{}}

        for name, module_name in modules_meta.items():
            globals_dict[name] = importlib.import_module(module_name)
        for name, value in values_meta.items():
            globals_dict[name] = value

        code_obj = marshal.loads(func_payload["code"])
        func_name = func_payload.get("name") or "target_func"
        target_func = types.FunctionType(code_obj, globals_dict, func_name)
        target_func.__defaults__ = func_payload.get("defaults")
        target_func.__kwdefaults__ = func_payload.get("kwdefaults")
        logger.info("Callable payload reconstructed successfully")
    except Exception as e:
        logger.error(f"Failed to reconstruct function: {{e}}")
        import_error = f"Import error: {{e}}"
        send_event("error", {{"message": import_error}})
        send_event("state", {{"status": "failed", "message": import_error}})
        time.sleep(0.2)
        sys.exit(1)
else:
    try:
        logger.info(f"Importing module: {{func_payload['module']}}")
        module = importlib.import_module(func_payload["module"])
        logger.info(f"Getting function: {{func_payload['name']}}")
        target_func = getattr(module, func_payload["name"])
        logger.info("Function loaded successfully")
    except Exception as e:
        logger.error(f"Failed to import function: {{e}}")
        import_error = f"Import error: {{e}}"
        send_event("error", {{"message": import_error}})
        send_event("state", {{"status": "failed", "message": import_error}})
        time.sleep(0.2)
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
                command_name = str(cmd.get("command") or "").lower()

                if command_name == "abort":
                    logger.info("Received abort command")
                    should_abort = True
                    send_event("state", {{"status": "aborting"}})
                elif command_name == "trigger_validation":
                    logger.info("Received manual validation trigger")
                    manual_validation_event.set()
                elif command_name == "trigger_checkpoint":
                    logger.info("Received manual checkpoint trigger")
                    manual_checkpoint_event.set()

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

# Detect parent exit and abort if this process is orphaned
parent_pid_env = os.environ.get("SIMPLETUNER_PARENT_PID")
try:
    _parent_pid = int(parent_pid_env) if parent_pid_env else None
except Exception:
    _parent_pid = None

def _monitor_parent():
    global should_abort
    if _parent_pid is None or _parent_pid <= 1:
        return
    while not should_abort:
        try:
            current_ppid = os.getppid()
            if current_ppid != _parent_pid:
                logger.info("Parent process disappeared; aborting trainer subprocess")
                should_abort = True
                send_event("state", {{"status": "aborting", "reason": "parent_exit"}})
                try:
                    if hasattr(os, "getpgid") and hasattr(os, "killpg"):
                        os.killpg(os.getpgid(0), signal.SIGTERM)
                    else:
                        os.kill(os.getpid(), signal.SIGTERM)
                except Exception:
                    pass
                break
        except Exception:
            break
        time.sleep(1.0)

parent_monitor_thread = threading.Thread(target=_monitor_parent, daemon=True)
parent_monitor_thread.start()

# Send starting event
send_event("state", {{"status": "running"}})

# Create wrapper that can check should_abort
class ConfigWrapper:
    def __init__(self, config):
        self.__dict__.update(config)
        self.should_abort = lambda: should_abort
        self.consume_manual_validation_request = consume_manual_validation_request
        self.consume_manual_checkpoint_request = consume_manual_checkpoint_request

wrapped_config = ConfigWrapper(config)

# Patch tqdm to force dynamic mode before running target function
_patch_tqdm()

# Run the target function
try:
    logger.info("Starting target function")
    result = target_func(wrapped_config)
    send_event("state", {{"status": "completed", "result": str(result)}})
except SystemExit as exc:
    exit_code = getattr(exc, "code", None)
    payload = {{}}

    if isinstance(exit_code, Mapping):
        payload = dict(exit_code)
        exit_message = str(payload.get("message") or payload.get("error") or "").strip()
    elif isinstance(exit_code, str) and exit_code.strip():
        exit_message = exit_code.strip()
    elif isinstance(exit_code, int):
        exit_message = f"Training process exited during configuration (exit code {{exit_code}}). Check the logs above for details."
        payload["exit_code"] = exit_code
    else:
        exit_message = "Training process exited during configuration. Check the logs above for details."
        if exit_code is not None:
            payload["exit_code"] = exit_code

    if not exit_message:
        exit_message = "Training process exited during configuration. Check the logs above for details."

    logger.error(f"Training terminated via SystemExit: {{exit_message}}")

    error_event = {{"message": exit_message}}
    error_event.update(payload)
    send_event("error", error_event)

    state_event = {{"status": "failed", "message": exit_message}}
    state_event.update(payload)
    send_event("state", state_event)
    time.sleep(0.2)
    raise
except Exception as e:
    error_message = str(e)
    log_level = logging.ERROR
    if "Training configuration could not be parsed" in error_message:
        log_level = logging.INFO
    logger.log(log_level, f"Function error: {{error_message}}")
    import traceback
    traceback_str = traceback.format_exc() if log_level == logging.ERROR else ""
    if log_level == logging.ERROR:
        logger.error(traceback_str)
    # Extract log excerpt if available (attached by run_trainer_job error handling)
    log_excerpt = getattr(e, "_simpletuner_log_excerpt", None)
    recent_lines = getattr(e, "_simpletuner_recent_log_lines", None)
    error_payload = {{"message": error_message, "traceback": traceback_str}}
    if log_excerpt:
        error_payload["log_excerpt"] = log_excerpt
    if recent_lines:
        error_payload["recent_lines"] = recent_lines
    send_event("error", error_payload)
    send_event("state", {{"status": "failed", "message": error_message, "log_excerpt": log_excerpt}})
    time.sleep(0.2)

logger.info("Subprocess exiting")
'''

        env = os.environ.copy()

        # Force colors to be enabled in subprocess (stdout is piped so TTY detection fails)
        # Remove SIMPLETUNER_WEB_MODE so subprocess can use colors even when launched from web UI
        env.pop("SIMPLETUNER_WEB_MODE", None)
        env.pop("SIMPLETUNER_DISABLE_COLORS", None)
        # env["FORCE_COLOR"] = "1"
        # env["CLICOLOR_FORCE"] = "1"

        env.setdefault("WANDB_CONSOLE", "off")
        env.setdefault("WANDB_SILENT", "true")
        env.setdefault("WANDB_DISABLE_SERVICE", "true")
        env.setdefault("WANDB_STDOUT_CAPTURE", "off")
        env.setdefault("WANDB_STDERR_CAPTURE", "off")
        env.setdefault("SIMPLETUNER_PARENT_PID", str(os.getpid()))

        # Start the subprocess
        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "env": env,
            "bufsize": 1,  # Line buffered
        }

        if os.name != "nt":
            popen_kwargs["preexec_fn"] = os.setsid
        else:  # pragma: no cover - Windows specific
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        # Use -u flag for unbuffered output so tqdm updates appear in real-time
        self.process = subprocess.Popen([sys.executable, "-u", "-c", runner_code], **popen_kwargs)

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
                    # Process has exited - drain any remaining events
                    self._drain_remaining_events(timeout=1.0, max_attempts=5)
                    self._emit_fallback_error_if_needed()
                    if self.status == "running":
                        # Only update if no status was set via events
                        self.status = "completed" if self.process.returncode == 0 else "failed"
                    break

                has_new_events = self._process_event_file()
                if not has_new_events:
                    time.sleep(0.1)  # Poll every 100ms

            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"Event listener error: {e}")
                break

    def _process_event_file(self) -> bool:
        if not self.event_file:
            return False
        try:
            with open(self.event_file, "r") as f:
                all_events = json.load(f)
        except Exception as exc:
            if not self.stop_event.is_set():
                logger.debug(f"Event read error: {exc}")
            return False

        if len(all_events) <= len(self.events):
            return False

        new_events = all_events[len(self.events) :]
        with self.events_lock:
            self.events.extend(new_events)

        for event in new_events:
            self._handle_event(event)

        return True

    def _drain_remaining_events(self, *, timeout: float, max_attempts: int) -> None:
        attempts = max(max_attempts, 1)
        interval = timeout / attempts if timeout > 0 else 0.0

        while attempts > 0 and not self.stop_event.is_set():
            if self._process_event_file():
                attempts = max(max_attempts, 1)
                continue

            attempts -= 1
            if attempts <= 0 or interval <= 0:
                break

            time.sleep(interval)

    @staticmethod
    def _format_signal_message(signal_num: int, signal_name: Optional[str] = None) -> str:
        if signal_name is None:
            try:
                signal_name = signal.Signals(signal_num).name
            except ValueError:
                signal_name = None

        sigkill_value = getattr(signal, "SIGKILL", None)
        if signal_name == "SIGKILL" or (sigkill_value is not None and signal_num == sigkill_value):
            return "Training subprocess was killed by SIGKILL (likely out of system memory or critical system condition)."
        if signal_name:
            return f"Training subprocess was killed by {signal_name} (signal {signal_num})."
        return f"Training subprocess was killed by signal {signal_num}."

    def _format_signal_exit_message(self, exit_code: Optional[int]) -> Optional[str]:
        if exit_code is None or exit_code >= 0:
            return None
        return self._format_signal_message(-exit_code)

    def _emit_fallback_error_if_needed(self) -> None:
        if self._relayed_failure:
            return

        exit_code = self.process.returncode if self.process else None
        if exit_code in (None, 0):
            return

        synthesized = self._extract_error_from_logs(exit_code)
        if synthesized is None:
            message = self._format_signal_exit_message(exit_code) or f"Training failed with exit code {exit_code}."
            synthesized = {"message": message, "exit_code": exit_code, "source": "process"}
        else:
            synthesized.setdefault("exit_code", exit_code)

        synthetic_event = {
            "type": "error",
            "timestamp": time.time(),
            "data": synthesized,
        }

        with self.events_lock:
            self.events.append(synthetic_event)

        self._handle_event(synthetic_event)
        self._append_event("error", synthesized)

    def _extract_error_from_logs(self, exit_code: Optional[int]) -> Optional[Dict[str, Any]]:
        log_path = self.log_file
        if not log_path or not os.path.exists(log_path):
            return None

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
        except Exception as exc:
            logger.debug(f"Failed to read log file for {self.job_id}: {exc}")
            return None

        if not lines:
            return None

        tail_lines = [line.rstrip("\n") for line in lines[-200:]]
        meaningful_lines = [line.strip() for line in tail_lines if line.strip()]

        signal_message = self._format_signal_exit_message(exit_code)
        preferred_message: Optional[str] = None
        for line in reversed(tail_lines):
            lowered = line.lower().strip()
            if "cuda out of memory" in lowered:
                preferred_message = line.strip()
                break
            if "childfailederror" in lowered:
                preferred_message = line.strip()
                break
            if "died with <signals." in lowered:
                match = re.search(r"died with <Signals\.([A-Z0-9_]+):\s*(\d+)>", line)
                if match:
                    signal_name = match.group(1)
                    signal_num = int(match.group(2))
                    preferred_message = self._format_signal_message(signal_num, signal_name)
                    break
            # Also match accelerate's format: "Signal 9 (SIGKILL) received by PID 12345"
            if "signal" in lowered and "received by pid" in lowered:
                match = re.search(r"Signal\s+(\d+)\s+\(([A-Z_]+)\)\s+received by PID", line, re.IGNORECASE)
                if match:
                    signal_num = int(match.group(1))
                    signal_name = match.group(2)
                    preferred_message = self._format_signal_message(signal_num, signal_name)
                    break
            if "nccl" in lowered and "warning" in lowered:
                # Skip NCCL warnings as final result unless nothing better is found
                preferred_message = preferred_message or line.strip()

        if not preferred_message and signal_message:
            preferred_message = signal_message

        if not preferred_message and meaningful_lines:
            preferred_message = meaningful_lines[-1]

        if not preferred_message:
            preferred_message = f"Training failed with exit code {exit_code}."

        message = preferred_message[:512]

        traceback_marker = "Traceback (most recent call last):"
        traceback_start = None
        for idx, line in enumerate(tail_lines):
            if line.strip().startswith(traceback_marker):
                traceback_start = idx

        if traceback_start is not None:
            traceback_lines = tail_lines[traceback_start:]
        else:
            traceback_lines = tail_lines[-40:]

        traceback_text = "\n".join(traceback_lines).strip()
        if len(traceback_text) > 4000:
            traceback_text = traceback_text[-4000:]

        log_excerpt = "\n".join(tail_lines[-60:]).strip()
        if len(log_excerpt) > 4000:
            log_excerpt = log_excerpt[-4000:]

        payload: Dict[str, Any] = {
            "message": message,
            "source": "stdout",
        }

        if traceback_text:
            payload["traceback"] = traceback_text
        if log_excerpt:
            payload["log_excerpt"] = log_excerpt

        return payload

    def _stream_output(self):
        """Continuously read subprocess stdout to prevent pipe backpressure."""
        if not self.process or self.process.stdout is None:
            return

        log_handle = None
        if self.log_file:
            try:
                log_handle = open(self.log_file, "a", encoding="utf-8")
            except Exception as exc:
                logger.debug(f"Unable to open log file for {self.job_id}: {exc}")
                log_handle = None

        # Track last progress bar line to enable dynamic updates
        last_progress_line = None
        import re

        progress_pattern = re.compile(r".*\d+%\|[▏▎▍▌▋▊▉█\s]+\|.*\d+/\d+.*")

        try:
            for line in iter(self.process.stdout.readline, ""):
                if line == "":
                    break
                if log_handle is not None:
                    try:
                        log_handle.write(strip_ansi(line))
                        log_handle.flush()
                    except Exception as log_exc:
                        logger.debug(f"Failed to persist stdout for {self.job_id}: {log_exc}")
                        try:
                            log_handle.close()
                        except Exception:
                            pass
                        log_handle = None
                try:
                    # Check if this is a progress bar line
                    is_progress = progress_pattern.match(line.strip())

                    if is_progress:
                        # Use carriage return to update the same line
                        sys.stdout.write(f"\r{line.rstrip()}")
                        sys.stdout.flush()
                        last_progress_line = line
                    else:
                        # Regular line - print with newline
                        # If we were showing a progress bar, print a newline first
                        if last_progress_line:
                            sys.stdout.write("\n")
                            last_progress_line = None
                        sys.stdout.write(line)
                        sys.stdout.flush()
                except Exception:
                    # If stdout is unavailable, print to stderr with ANSI codes stripped
                    # to avoid double-formatting and escape codes in logs/webhooks
                    print(strip_ansi(line.rstrip()), file=sys.stderr)
        except Exception as exc:
            if not self.stop_event.is_set():
                logger.debug(f"stdout streaming error for {self.job_id}: {exc}")
        finally:
            # Print final newline if we ended on a progress bar
            if last_progress_line:
                try:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                except Exception:
                    # stdout may be unavailable; ignore errors during cleanup
                    pass
            if log_handle is not None:
                try:
                    log_handle.close()
                except Exception:
                    pass

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
                    state_data = event.get("data", {}) or {}
                    if "status" in state_data:
                        # Don't override terminated status with aborting
                        if not (self.status == "terminated" and state_data["status"] == "aborting"):
                            self.status = state_data["status"]
                            process_registry[self.job_id]["status"] = self.status

        event_type = str(event.get("type") or "").lower()
        event_data = event.get("data") or {}

        if event_type == "error":
            self._relay_failure_event(event_data)
        elif event_type == "state":
            status_text = str(event_data.get("status") or "").strip().lower()
            if status_text in {"failed", "error", "fatal"}:
                self._relay_failure_event(event_data)
            elif status_text in {"completed", "success"}:
                self._relay_completion_event(event_data)
            elif status_text in {"terminated", "cancelled", "canceled", "aborted", "stopped"}:
                self._relay_cancel_event(event_data)

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
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
        except Exception as exc:
            logger.debug(f"Failed to append event for {self.job_id}: {exc}")

    def terminate(self, timeout: int = 10) -> bool:
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
                    # Give the subprocess a moment to read the abort command
                    # before sending SIGTERM. The command checker polls every 100ms.
                    time.sleep(0.2)
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

    def _relay_failure_event(self, data: Dict[str, Any]) -> None:
        payload_data = dict(data or {})
        message = str(payload_data.get("message") or payload_data.get("error") or "").strip()
        if not message:
            message = "Training failed due to a fatal error."

        first_failure = not self._relayed_failure
        if first_failure:
            self._relayed_failure = True

        # If the message looks generic (like "Accelerate launch exited with status 1"),
        # try to extract more useful info from logs
        generic_patterns = [
            "accelerate launch exited with status",
            "training failed due to a fatal error",
            "exited with status",
            "exit code",
        ]
        message_lower = message.lower()
        is_generic = any(p in message_lower for p in generic_patterns)
        has_log_excerpt = bool(payload_data.get("log_excerpt"))

        if first_failure and is_generic and not has_log_excerpt:
            exit_code = payload_data.get("exit_code")
            if exit_code is None:
                # Try to extract exit code from message
                match = re.search(r"status\s+(-?\d+)", message)
                if match:
                    try:
                        exit_code = int(match.group(1))
                    except ValueError:
                        pass
            supplemental = self._extract_error_from_logs(exit_code)
            if supplemental:
                # Merge supplemental info into payload
                if supplemental.get("message") and supplemental["message"] != message:
                    # Use the more specific message from logs
                    better_message = supplemental["message"]
                    if len(better_message) > len(message) or "signal" in better_message.lower():
                        message = better_message
                if supplemental.get("log_excerpt"):
                    payload_data["log_excerpt"] = supplemental["log_excerpt"]
                if supplemental.get("traceback"):
                    payload_data.setdefault("traceback", supplemental["traceback"])

        payload_data.setdefault("status", "failed")
        normalized_status = str(payload_data.get("status") or "failed").strip().lower() or "failed"
        if not first_failure:
            payload_data["update"] = True

        public_status = normalized_status if normalized_status in {"failed", "error", "fatal"} else "failed"

        self.status = "failed"
        payload_data["status"] = public_status
        if first_failure and self.end_time is None:
            self.end_time = datetime.now().isoformat()

        with lock:
            if self.job_id in process_registry:
                process_registry[self.job_id]["status"] = self.status

        payload = {
            "type": "error",
            "severity": "error",
            "message": message,
            "job_id": self.job_id,
            "data": payload_data,
        }

        if first_failure:
            self._update_training_state_from_subprocess(status="error")
            # Release GPUs when job fails
            self._release_gpus()
        self._dispatch_callback_event(payload)
        self._dispatch_training_status_event(status="failed", data=payload_data, message=message)

    def _relay_completion_event(self, data: Dict[str, Any]) -> None:
        if self._relayed_completion:
            return
        self._relayed_completion = True

        payload_data = dict(data or {})
        payload_data.setdefault("status", "completed")
        payload = {
            "type": "training.summary",
            "status": "completed",
            "severity": "info",
            "message": data.get("message") or "Training completed successfully.",
            "job_id": self.job_id,
            "data": payload_data,
        }

        self._update_training_state_from_subprocess(status="completed")
        # Release GPUs when job completes
        self._release_gpus()
        self._dispatch_callback_event(payload)
        self._dispatch_training_status_event(status="completed", data=data, message=payload["message"])

    def _relay_cancel_event(self, data: Dict[str, Any]) -> None:
        if self._relayed_failure or self._relayed_completion:
            return

        payload_data = dict(data or {})
        payload_data.setdefault("status", "cancelled")
        payload = {
            "type": "training.status",
            "status": "cancelled",
            "severity": "warning",
            "message": data.get("message") or "Training was cancelled.",
            "job_id": self.job_id,
            "data": payload_data,
        }

        self._update_training_state_from_subprocess(status="cancelled")
        # Release GPUs when job is cancelled, but don't process pending jobs.
        # This prevents starting new jobs during bulk cancellation operations.
        self._release_gpus(process_pending=False)
        self._dispatch_callback_event(payload)

    def _release_gpus(self, *, process_pending: bool = True) -> None:
        """Release GPUs allocated to this job.

        Args:
            process_pending: If True, process pending jobs after release.
                Set to False during cancellation to avoid starting jobs
                that may also be about to be cancelled.
        """
        try:
            import asyncio

            from simpletuner.simpletuner_sdk.server.services.local_gpu_allocator import get_gpu_allocator

            def _do_release():
                async def _async_release():
                    allocator = get_gpu_allocator()
                    await allocator.release(self.job_id)
                    if process_pending:
                        # Process pending jobs to start the next one if GPUs are available
                        started = await allocator.process_pending_jobs()
                        if started:
                            logger.info(
                                "Started %d pending job(s) after GPU release from job %s",
                                len(started),
                                self.job_id,
                            )

                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        pool.submit(lambda: asyncio.run(_async_release()))
                except RuntimeError:
                    asyncio.run(_async_release())

            _do_release()
        except Exception:
            logger.debug(
                "Failed to release GPUs for job %s (may be normal for non-GPU jobs)",
                self.job_id,
                exc_info=True,
            )

    def _update_training_state_from_subprocess(self, *, status: str) -> None:
        normalized = status.strip().lower()
        try:
            from simpletuner.simpletuner_sdk.api_state import APIState

            if normalized == "completed":
                progress_state = APIState.get_state("training_progress") or {}
                if isinstance(progress_state, Mapping):
                    progress_state = dict(progress_state)
                    progress_state["percent"] = 100
                    APIState.set_state("training_progress", progress_state)
                APIState.set_state("training_status", "completed")
            elif normalized in {"cancelled", "canceled"}:
                APIState.set_state("training_status", "cancelled")
                APIState.set_state("training_progress", None)
            else:
                APIState.set_state("training_status", "error")
                APIState.set_state("training_progress", None)

            APIState.set_state("training_startup_stages", {})
            current_job = APIState.get_state("current_job_id")
            if current_job == self.job_id:
                logger.info(
                    "Clearing current_job_id=%s due to subprocess status event: %s",
                    self.job_id,
                    normalized,
                )
                APIState.set_state("current_job_id", None)
        except Exception:
            logger.debug("Failed to update API state for subprocess status '%s'", status, exc_info=True)

    def _dispatch_callback_event(self, payload: Dict[str, Any]) -> None:
        try:
            from simpletuner.simpletuner_sdk.server.services.callback_service import get_default_callback_service

            service = get_default_callback_service()
            service.handle_incoming(payload)
        except Exception:
            logger.debug("Failed to relay subprocess event to callback service", exc_info=True)

    def _dispatch_training_status_event(
        self, *, status: str, data: Optional[Dict[str, Any]] = None, message: str = ""
    ) -> None:
        normalized = status.strip().lower()
        status_payload = {
            "type": "training.status",
            "status": normalized,
            "severity": "error" if normalized in {"failed", "error", "fatal"} else "info",
            "message": message or None,
            "job_id": self.job_id,
            "data": (data or {}) | {"status": normalized},
        }
        self._dispatch_callback_event(status_payload)

    @staticmethod
    def _collect_child_pids_from_proc(parent_pid: int) -> List[int]:
        """Collect all descendant PIDs by traversing /proc (Linux fallback when psutil unavailable)."""
        children: List[int] = []
        to_visit = [parent_pid]
        visited = set()

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)

            # Find direct children by scanning /proc/*/stat for PPid
            try:
                for entry in os.listdir("/proc"):
                    if not entry.isdigit():
                        continue
                    try:
                        stat_path = f"/proc/{entry}/stat"
                        with open(stat_path, "r") as f:
                            stat_content = f.read()
                        # Format: pid (comm) state ppid ...
                        # The comm field may contain spaces/parens, so find the last ')'
                        last_paren = stat_content.rfind(")")
                        if last_paren == -1:
                            continue
                        fields = stat_content[last_paren + 2 :].split()
                        if len(fields) >= 2:
                            ppid = int(fields[1])
                            child_pid = int(entry)
                            if ppid == current and child_pid not in visited:
                                children.append(child_pid)
                                to_visit.append(child_pid)
                    except (FileNotFoundError, PermissionError, ValueError, IndexError):
                        continue
            except Exception:
                break

        return children

    def _force_kill_process_tree(self, timeout: int) -> None:
        """Terminate the subprocess tree with a short graceful window before force kill."""

        if not self.process or self.process.poll() is not None:
            return

        pid = self.process.pid
        # Allow more grace time for nested subprocesses (e.g., accelerate launch)
        # to terminate their children before force-killing
        grace_timeout = max(0.0, min(timeout, 5))
        remaining_timeout = max(0.0, timeout - grace_timeout)

        # Collect all child PIDs BEFORE sending any signals.
        # This is critical because if the parent process exits first (common with
        # accelerate launch), we won't be able to find its children later - they
        # get reparented to init and continue running, holding GPU memory.
        child_pids: List[int] = []
        if psutil:
            try:
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    try:
                        child_pids.append(child.pid)
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug(f"Failed to collect child PIDs for {self.job_id}: {exc}")
        elif os.path.isdir("/proc"):
            # Fallback: use /proc on Linux when psutil is unavailable
            child_pids = self._collect_child_pids_from_proc(pid)

        def _kill_collected_pids(sig: int) -> None:
            """Kill all collected child PIDs with the given signal."""
            for child_pid in child_pids:
                try:
                    os.kill(child_pid, sig)
                except ProcessLookupError:
                    pass  # Already dead
                except Exception as exc:
                    logger.debug(f"Failed to send signal {sig} to {child_pid}: {exc}")

        try:
            # First try to terminate so cleanup handlers can run.
            if os.name != "nt" and hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except Exception as exc:
                    logger.debug(f"SIGTERM killpg failed for {self.job_id}: {exc}")
            else:  # pragma: no cover - Windows specific
                try:
                    self.process.terminate()
                except Exception as exc:
                    logger.debug(f"terminate() failed for {self.job_id}: {exc}")

            try:
                if grace_timeout:
                    self.process.wait(timeout=grace_timeout)
            except Exception:
                pass

            still_running = self.process and self.process.poll() is None
            if still_running:
                if os.name != "nt" and hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except Exception as exc:
                        logger.debug(f"SIGKILL killpg failed for {self.job_id}: {exc}")
                else:  # pragma: no cover - Windows specific
                    try:
                        self.process.kill()
                    except Exception as exc:
                        logger.debug(f"kill() failed for {self.job_id}: {exc}")

            # Kill any children that may have survived using the PIDs we collected earlier.
            # This handles the case where the parent exits but children (e.g., accelerate
            # workers) continue running because they created their own process groups.
            _kill_collected_pids(signal.SIGKILL)

            try:
                self.process.wait(timeout=remaining_timeout or timeout)
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

            if self.ipc_dir and os.path.exists(self.ipc_dir):
                shutil.rmtree(self.ipc_dir)
        except Exception as e:
            logger.debug(f"Failed to clean up IPC dir: {e}")

        # Close subprocess pipes safely
        if self.process:
            self._safe_close_pipes()

        # Clear process reference
        self.process = None
        self.log_file = None

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
            return_code = process.process.returncode if process.process else None
            previous_status = process.status

            if process.status == "running":
                # Give the event thread a chance to flush remaining events
                if process.event_thread and process.event_thread.is_alive():
                    event_thread = process.event_thread
                    lock.release()
                    try:
                        deadline = time.time() + 1.0
                        while event_thread.is_alive() and time.time() < deadline:
                            remaining = deadline - time.time()
                            event_thread.join(timeout=min(0.2, max(0.0, remaining)))
                    finally:
                        lock.acquire()

                if process.status == "running":
                    if return_code == 0:
                        process.status = "completed"
                    elif return_code is None:
                        process.status = "crashed"
                    else:
                        process.status = "failed"
                    entry["status"] = process.status
                    logger.info(
                        "Process %s: status changed from %s to %s (return_code=%s, process not alive)",
                        job_id,
                        previous_status,
                        process.status,
                        return_code,
                    )
            elif process.status not in ["completed", "failed", "terminated", "crashed"]:
                entry["status"] = process.status

        return process.status


def get_process_pid(job_id: str) -> Optional[int]:
    """Get the PID of a running job's subprocess.

    Args:
        job_id: The job ID to look up

    Returns:
        The process ID if found, None otherwise
    """
    with lock:
        entry = process_registry.get(job_id)
        if entry and entry.get("process"):
            proc = entry["process"].process
            if proc:
                return proc.pid
    return None


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


def append_external_event(job_id: str, event: Dict[str, Any]) -> None:
    """
    Append an externally-generated event (e.g., Webhook notification) to a process event log.
    """
    if not job_id:
        return

    timestamp = event.get("timestamp") or time.time()
    event.setdefault("timestamp", timestamp)

    with lock:
        entry = process_registry.get(job_id)
        if not entry:
            return
        events = entry.setdefault("events", [])
        events.append(event)


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
