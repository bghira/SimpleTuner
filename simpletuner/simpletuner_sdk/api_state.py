"""
API State tracker to persist and resume tracker states and configurations.

If a server were to crash during a training job, we can immediately reload the system state
and continue training from the last checkpoint.
"""

import json
import logging
import os

logger = logging.getLogger("SimpleTunerSDK")
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class APIState:
    state = {}
    state_file = "api_state.json"
    trainer = None

    @classmethod
    def load_state(cls):
        if os.path.exists(cls.state_file):
            with open(cls.state_file, "r") as f:
                cls.state = json.load(f)
                logger.info(f"Loaded state from {cls.state_file}: {cls.state}")
        else:
            logger.info(f"No state file found at {cls.state_file}")

    @classmethod
    def save_state(cls):
        with open(cls.state_file, "w") as f:
            json.dump(cls.state, f)
            logger.info(f"Saved state to {cls.state_file}: {cls.state}")

    @classmethod
    def get_state(cls, key=None):
        if not key:
            return cls.state
        return cls.state.get(key)

    @classmethod
    def set_state(cls, key, value):
        cls.state[key] = value
        cls.save_state()

    @classmethod
    def delete_state(cls, key):
        if key in cls.state:
            del cls.state[key]
            cls.save_state()

    @classmethod
    def clear_state(cls):
        # Clean up trainer if it exists
        if hasattr(cls, "trainer") and cls.trainer is not None:
            try:
                # Abort any running processes
                if hasattr(cls.trainer, "abort"):
                    cls.trainer.abort()

                # Unload model components to free GPU memory
                if hasattr(cls.trainer, "model") and cls.trainer.model is not None:
                    if hasattr(cls.trainer.model, "unload"):
                        cls.trainer.model.unload()
            except Exception as e:
                # Log but don't fail on cleanup errors
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Error during trainer cleanup: {e}")
            finally:
                cls.trainer = None

        cls.state = {}
        cls.save_state()

    @classmethod
    def set_job(cls, job_id, job: dict):
        cls.set_state("current_job", job)
        cls.set_state("current_job_id", job_id)
        cls.set_state("status", "running")

    @classmethod
    def get_job(cls):
        return {
            "job_id": cls.get_state("current_job_id"),
            "job": cls.get_state("current_job"),
        }

    @classmethod
    def cancel_job(cls):
        cls.delete_state("current_job")
        cls.delete_state("current_job_id")
        cls.set_state("status", "cancelled")

    @classmethod
    def set_trainer(cls, trainer):
        cls.trainer = trainer

    @classmethod
    def get_trainer(cls):
        return cls.trainer

    @classmethod
    def get_active_jobs(cls):
        jobs = cls.state.get("jobs")
        if isinstance(jobs, dict):
            return jobs

        current_job_id = cls.state.get("current_job_id")
        current_job = cls.state.get("current_job")
        if current_job_id and current_job:
            return {current_job_id: current_job}
        return {}
