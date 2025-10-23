"""
Subprocess wrapper for running trainer in isolated process with IPC communication.
This runs inside the subprocess and handles command/event communication.
"""

import json
import logging
import signal
import threading
import time
import traceback
from multiprocessing import Queue
from queue import Empty
from typing import Any, Dict, Optional

logger = logging.getLogger("SubprocessWrapper")


class SubprocessTrainerWrapper:
    """Wrapper that runs inside subprocess to handle trainer execution and IPC."""

    def __init__(self, command_queue: Queue, event_pipe: Any):
        self.command_queue = command_queue
        self.event_pipe = event_pipe
        self.should_abort = False
        self.is_paused = False
        self.trainer = None
        self.command_thread = None

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, initiating shutdown")
        self.should_abort = True
        if self.trainer and hasattr(self.trainer, "abort"):
            self.trainer.abort()
        # Close event pipe to signal parent
        try:
            self.event_pipe.close()
        except:
            pass

    def _command_listener(self):
        """Listen for commands from the parent process."""
        while not self.should_abort:
            try:
                # Use get with timeout instead of checking empty()
                command = self.command_queue.get(timeout=0.1)
                self._handle_command(command)
            except Exception as e:
                # Timeout is normal, continue loop
                if not self.should_abort and not isinstance(e, Empty):
                    logger.error(f"Command listener error: {e}")
                    break
                continue

    def _handle_command(self, command: Dict[str, Any]):
        """Handle a command from the parent process."""
        cmd_type = command.get("command")
        data = command.get("data", {})

        logger.info(f"Received command: {cmd_type}")

        if cmd_type == "abort":
            self.should_abort = True
            if self.trainer and hasattr(self.trainer, "abort"):
                self.trainer.abort()
            self._send_event("state", {"status": "aborting"})

        elif cmd_type == "pause":
            self.is_paused = True
            if self.trainer and hasattr(self.trainer, "pause"):
                self.trainer.pause()
            self._send_event("state", {"status": "paused"})

        elif cmd_type == "resume":
            self.is_paused = False
            if self.trainer and hasattr(self.trainer, "resume"):
                self.trainer.resume()
            self._send_event("state", {"status": "resumed"})

        elif cmd_type == "update_config":
            # Allow dynamic config updates if trainer supports it
            if self.trainer and hasattr(self.trainer, "update_config"):
                self.trainer.update_config(data)
                self._send_event("state", {"status": "config_updated"})

    def _send_event(self, event_type: str, data: Dict[str, Any]):
        """Send an event to the parent process."""
        try:
            event = {"type": event_type, "timestamp": time.time(), "data": data}
            self.event_pipe.send(event)
        except Exception as e:
            logger.error(f"Failed to send event: {e}")

    def run_trainer(self, trainer_func, config: Dict[str, Any]):
        """Run the trainer with command handling and event reporting."""
        try:
            # Start command listener in background
            self.command_thread = threading.Thread(target=self._command_listener, daemon=True)
            self.command_thread.start()

            # Send startup event
            self._send_event("state", {"status": "starting"})

            # Check if trainer_func is callable (test function) or a class
            if callable(trainer_func) and not isinstance(trainer_func, type):
                # It's a test function, just call it
                self._send_event("state", {"status": "running", "config": config})
                result = trainer_func(config)
                if self.should_abort:
                    self._send_event("state", {"status": "aborted"})
                else:
                    self._send_event("state", {"status": "completed"})
            else:
                # Import and initialize trainer
                # trainer_func is expected to be the Trainer class or a factory function
                from simpletuner.helpers.configuration.json_file import normalize_args
                from simpletuner.helpers.training.trainer import Trainer

                # Create trainer instance
                normalized_config = normalize_args(config)
                self.trainer = Trainer(config=normalized_config)

                # Hook into trainer events if available
                self._setup_trainer_hooks()

                # Send running event
                self._send_event("state", {"status": "running", "config": config})

                # Run the trainer
                self.trainer.run()

                # Send completion event
                if self.should_abort:
                    self._send_event("state", {"status": "aborted"})
                else:
                    self._send_event("state", {"status": "completed"})

        except Exception as e:
            # Send error event
            self._send_event("error", {"message": str(e), "traceback": traceback.format_exc()})
            self._send_event("state", {"status": "failed"})
            raise
        finally:
            # Clean up resources
            self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        self.should_abort = True

        # Close event pipe
        try:
            self.event_pipe.close()
        except:
            pass

        # Clear command queue
        try:
            while not self.command_queue.empty():
                self.command_queue.get_nowait()
        except:
            pass

    def _setup_trainer_hooks(self):
        """Set up hooks to capture trainer events and progress."""
        if not self.trainer:
            return

        # Hook into the webhook handler if it exists
        if hasattr(self.trainer, "webhook_handler"):
            original_send = self.trainer.webhook_handler._send_request

            def wrapped_send(*args, **kwargs):
                """Intercept webhook sends and forward as events."""
                # Call original
                result = original_send(*args, **kwargs)

                # Forward to parent process
                message = args[0] if args else kwargs.get("message", "")
                self._send_event(
                    "webhook", {"message": message, "has_images": bool(args[1] if len(args) > 1 else kwargs.get("images"))}
                )

                return result

            self.trainer.webhook_handler._send_request = wrapped_send

        # Hook into state tracker if it exists
        if hasattr(self.trainer, "state") and hasattr(self.trainer.state, "get"):

            def send_progress_update():
                """Send periodic progress updates."""
                while not self.should_abort:
                    try:
                        state = {}
                        # Safely extract state information
                        if hasattr(self.trainer.state, "global_step"):
                            state["global_step"] = self.trainer.state.global_step
                        if hasattr(self.trainer.state, "current_epoch"):
                            state["current_epoch"] = self.trainer.state.current_epoch
                        if hasattr(self.trainer.state, "loss"):
                            state["loss"] = float(self.trainer.state.loss) if self.trainer.state.loss else None

                        self._send_event("progress", state)
                        time.sleep(10)  # Send updates every 10 seconds
                    except Exception as e:
                        logger.debug(f"Progress update error: {e}")

            progress_thread = threading.Thread(target=send_progress_update, daemon=True)
            progress_thread.start()


class SubprocessTrainerFactory:
    """Factory for creating trainer instances in subprocess context."""

    @staticmethod
    def create_trainer_runner(trainer_class_path: str):
        """Create a runner function for a specific trainer class."""

        def runner(config: Dict[str, Any], command_queue: Queue, event_pipe: Any):
            """Run a trainer from a class path."""
            # Import the trainer class dynamically
            module_path, class_name = trainer_class_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            trainer_class = getattr(module, class_name)

            # Create wrapper and run
            wrapper = SubprocessTrainerWrapper(command_queue, event_pipe)
            wrapper.run_trainer(trainer_class, config)

        return runner
