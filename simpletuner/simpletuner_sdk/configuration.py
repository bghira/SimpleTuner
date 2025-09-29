import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel

from simpletuner.helpers.configuration.cmd_args import get_default_config
from simpletuner.helpers.configuration.json_file import normalize_args
from simpletuner.simpletuner_sdk import thread_keeper
from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore


def _import_trainer_class():
    """Return the Trainer class, deferring import so tests can patch easily."""

    from simpletuner.helpers.training.trainer import Trainer as TrainerClass

    return TrainerClass


# Define a Pydantic model for input validation
class ConfigModel(BaseModel):
    job_id: str
    # the actual Trainer config
    trainer_config: dict
    # what we will write as config/multidatabackend.json
    dataloader_config: list
    # what we will write as config/webhooks.json
    webhook_config: dict
    # optional lycoris_config
    lycoris_config: dict = None
    # optional user_prompt_library
    user_prompt_library: dict = None


class Configuration:
    def __init__(self):
        self.router = APIRouter(prefix="/training/configuration")
        self.router.add_api_route("/preload", self.preload, methods=["POST"])
        self.router.add_api_route("/check", self.check, methods=["POST"])
        self.router.add_api_route("/default", self.default, methods=["GET"])
        self.router.add_api_route("/run", self.run, methods=["POST"])

    async def preload(self):
        """
        Download models for a given configuration
        """
        training_config = None
        TrainerClass = _import_trainer_class()
        trainer = TrainerClass(config=training_config)
        trainer.init_preprocessing_models(move_to_accelerator=False)
        trainer.init_unload_vae()
        trainer.init_unload_text_encoder()
        trainer.init_load_base_model(move_to_accelerator=False)

        return {"status": "successfully downloaded models"}

    def _config_clear(self):
        # clear prev configs from disk first
        for file in [
            "config/multidatabackend.json",
            "config/webhooks.json",
            "config/lycoris_config.json",
            "config/user_prompt_library.json",
        ]:
            if os.path.exists(file):
                os.remove(file)

    def _config_save(self, job_config: ConfigModel):
        job_config.trainer_config.setdefault("--report_to", "none")
        job_config.trainer_config.setdefault("report_to", "none")

        with open("config/multidatabackend.json", mode="w") as file_handler:
            json.dump(job_config.dataloader_config, file_handler, indent=4)
            job_config.trainer_config["data_backend_config"] = "config/multidatabackend.json"

        with open("config/webhooks.json", mode="w") as file_handler:
            json.dump(job_config.webhook_config, file_handler, indent=4)
            job_config.trainer_config["webhook_config"] = "config/webhooks.json"

        if hasattr(job_config, "lycoris_config"):
            logger.debug(f"LyCORIS config present: {job_config.lycoris_config}")
            with open("config/lycoris_config.json", "w") as f:
                f.write(json.dumps(job_config.lycoris_config, indent=4))
                job_config.trainer_config["lycoris_config"] = "config/lycoris_config.json"

        user_prompt_library_path = job_config.trainer_config.get("--user_prompt_library", None)
        logger.debug(f"User prompt library path: {user_prompt_library_path}")
        if user_prompt_library_path and hasattr(job_config, "user_prompt_library"):
            logger.debug(f"User prompt library present: {job_config.user_prompt_library}")
            with open(user_prompt_library_path, "w") as f:
                f.write(json.dumps(job_config.user_prompt_library, indent=4))
                job_config.trainer_config["user_prompt_library"] = "config/user_prompt_library.json"

    def _load_base_trainer_config(self) -> Dict[str, Any]:
        store = ConfigStore()
        candidates = []

        active_config = store.get_active_config()
        if active_config:
            candidates.append(active_config)

        # Prefer a folder-based default if the active config is missing
        candidates.extend(["default", "Default"])

        for name in candidates:
            try:
                config, _ = store.load_config(name)
                if isinstance(config, dict):
                    return config
            except (FileNotFoundError, ValueError):
                continue

        # Fallback to CLI parser defaults when no stored config exists
        defaults = get_default_config()
        return {f"--{key}": value for key, value in defaults.items() if value not in (None, "")}

    def _build_trainer_cli_args(self, trainer_config: Dict[str, Any]) -> List[str]:
        base_config = self._load_base_trainer_config()
        merged_config = dict(base_config)
        merged_config.update(trainer_config or {})
        return normalize_args(merged_config)

    async def check(self, job_config: ConfigModel):
        """
        Check for problems with a given configuration
        """
        try:
            trainer = APIState.get_trainer()
            if trainer:
                return {
                    "status": False,
                    "result": "Could not test configuration, a previous configuration was already loaded.",
                }
            self._config_clear()
            self._config_save(job_config)
            TrainerClass = _import_trainer_class()
            cli_args = self._build_trainer_cli_args(job_config.trainer_config)
            trainer = TrainerClass(config=cli_args)
            return {
                "status": "success",
                "result": "Configuration validated successfully",
            }
        except Exception as e:
            import traceback

            logger.error(traceback.format_exc())
            message = str(e) or "unknown error"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid config: {message}",
            )

    async def default(self) -> dict:
        """
        Get default configuration
        """
        return get_default_config()

    async def run(self, job_config: ConfigModel) -> dict:
        """
        Run the training job in a separate thread or subprocess.
        """
        logger.info("Received call")

        # Check execution mode from environment
        execution_mode = os.environ.get("SIMPLETUNER_EXECUTION_MODE", "thread")
        if execution_mode == "process":
            return await self._run_subprocess(job_config)
        current_job_id = APIState.get_state("current_job_id")
        job_id = job_config.job_id
        current_job_status = thread_keeper.get_thread_status(current_job_id)
        if current_job_id and isinstance(current_job_status, str) and current_job_status.lower() in {"running", "pending"}:
            return {
                "status": False,
                "result": f"Could not run job, '{current_job_id}' is already {current_job_status}.",
            }

        trainer = APIState.get_trainer()
        if trainer and isinstance(current_job_status, str) and current_job_status.lower() == "running":
            return {
                "status": False,
                "result": f"Could not run job, '{current_job_id}' is already running.",
            }
        self._config_clear()
        self._config_save(job_config)
        try:
            logger.info("Creating new Trainer instance..")
            TrainerClass = _import_trainer_class()
            cli_args = self._build_trainer_cli_args(job_config.trainer_config)
            trainer = TrainerClass(config=cli_args, job_id=job_id)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error loading configuration: {str(e)}",
            )

        APIState.set_trainer(trainer)
        APIState.set_job(job_config.job_id, job_config.__dict__)
        APIState.set_state("status", "pending")
        if not trainer:
            return {
                "status": "error",
                "result": "No training job has been configured yet. Trainer was unavailable.",
            }

        if isinstance(current_job_status, str) and current_job_status.lower() in ["running", "pending"]:
            return {
                "status": "error",
                "result": f"The training job '{job_config.job_id}' is already {current_job_status}.",
            }
        try:
            # Submit the job to the thread manager
            logger.info("Submitting job to thread..")
            thread_keeper.submit_job(job_id, trainer.run)
            APIState.set_state("status", "Running")
            return {
                "status": "success",
                "result": f"Started training run with job ID {job_id}.",
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error starting training run: {str(e)}",
            )

    async def _run_subprocess(self, job_config: ConfigModel) -> dict:
        """Run training in a subprocess."""
        from simpletuner.simpletuner_sdk.process_keeper import get_process_status
        from simpletuner.simpletuner_sdk.process_keeper import submit_job as submit_process_job

        current_job_id = APIState.get_state("current_job_id")
        job_id = job_config.job_id

        # Check if job is already running
        if current_job_id:
            current_status = get_process_status(current_job_id)
            if current_status in ["running", "pending"]:
                return {
                    "status": False,
                    "result": f"Could not run job, '{current_job_id}' is already {current_status}.",
                }

        self._config_clear()
        self._config_save(job_config)

        # Prepare config for subprocess
        config_dict = {
            "trainer_config": job_config.trainer_config,
            "dataloader_config": job_config.dataloader_config,
            "webhook_config": job_config.webhook_config,
            "job_id": job_id,
        }

        APIState.set_job(job_id, job_config.__dict__)
        APIState.set_state("status", "pending")

        try:
            # Submit to process manager
            logger.info(f"Submitting job {job_id} to subprocess..")

            # trainer_func will be handled by subprocess wrapper
            TrainerClass = _import_trainer_class()

            process = submit_process_job(job_id, TrainerClass, config_dict["trainer_config"])

            APIState.set_state("status", "running")
            APIState.set_state("execution_mode", "process")

            return {
                "status": "success",
                "result": f"Started training run with job ID {job_id} in subprocess.",
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error starting subprocess training: {str(e)}",
            )
