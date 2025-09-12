import json, logging, os

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from simpletuner.helpers.training.trainer import Trainer
from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.helpers.configuration.cmd_args import get_default_config
from simpletuner.helpers.configuration.json_file import normalize_args
from simpletuner.simpletuner_sdk.thread_keeper import submit_job, get_thread_status


# Define a Pydantic model for input validation
class ConfigModel(BaseModel):
    job_id: str
    # the actual Trainer config
    trainer_config: dict
    # what we will write as config/multidatabackend.json
    dataloader_config: list
    # what we will write as config/webhooks.json
    webhooks_config: dict
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
        trainer = Trainer(config=training_config)
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
        with open("config/multidatabackend.json", mode="w") as file_handler:
            json.dump(job_config.dataloader_config, file_handler, indent=4)
            job_config.trainer_config["data_backend_config"] = (
                "config/multidatabackend.json"
            )

        with open("config/webhooks.json", mode="w") as file_handler:
            json.dump(job_config.webhooks_config, file_handler, indent=4)
            job_config.trainer_config["webhook_config"] = "config/webhooks.json"

        if hasattr(job_config, "lycoris_config"):
            print(f"LyCORIS config present: {job_config.lycoris_config}")
            with open("config/lycoris_config.json", "w") as f:
                f.write(json.dumps(job_config.lycoris_config, indent=4))
                job_config.trainer_config["lycoris_config"] = (
                    "config/lycoris_config.json"
                )

        user_prompt_library_path = job_config.trainer_config.get(
            "--user_prompt_library", None
        )
        print(f"User prompt library path: {user_prompt_library_path}")
        if user_prompt_library_path and hasattr(job_config, "user_prompt_library"):
            print(f"User prompt library present: {job_config.user_prompt_library}")
            with open(user_prompt_library_path, "w") as f:
                f.write(json.dumps(job_config.user_prompt_library, indent=4))
                job_config.trainer_config["user_prompt_library"] = (
                    "config/user_prompt_library.json"
                )

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
            trainer = Trainer(config=normalize_args(job_config.trainer_config))
            return {
                "status": True,
                "result": f"Configuration validated successfully",
            }
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not validate configuration: {str(e)}",
            )

    async def default(self) -> dict:
        """
        Get default configuration
        """
        return get_default_config()

    async def run(self, job_config: ConfigModel) -> dict:
        """
        Run the training job in a separate thread.
        """
        logger.info("Received call")
        trainer = APIState.get_trainer()
        current_job_id = APIState.get_state("current_job_id")
        job_id = job_config.job_id
        current_job_status = get_thread_status(current_job_id)
        if trainer and current_job_status.lower() == "running":
            return {
                "status": False,
                "result": f"Could not run job, '{current_job_id}' is already running.",
            }
        self._config_clear()
        self._config_save(job_config)
        try:
            logger.info("Creating new Trainer instance..")
            trainer = Trainer(
                config=normalize_args(job_config.trainer_config), job_id=job_id
            )
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

        if current_job_status.lower() in ["running", "pending"]:
            return {
                "status": "error",
                "result": f"The training job '{job_config.job_id}' is already {current_job_status}.",
            }
        try:
            # Submit the job to the thread manager
            print("Submitting job to thread..")
            submit_job(job_id, trainer.run)
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
