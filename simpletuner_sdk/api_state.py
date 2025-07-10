"""
API State tracker to persist and resume tracker states and configurations.

If a server were to crash during a training job, we can immediately reload the system state
and continue training from the last checkpoint.
"""

import os
import json
import logging

logger = logging.getLogger("SimpleTunerSDK")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING"))


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
