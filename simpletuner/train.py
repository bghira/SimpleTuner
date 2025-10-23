import logging

# Quiet down, you.
ds_logger1 = logging.getLogger("DeepSpeed")
ds_logger2 = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
ds_logger1.setLevel("ERROR")
ds_logger2.setLevel("ERROR")
import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
from os import environ

environ["ACCELERATE_LOG_LEVEL"] = "WARNING"

from simpletuner.helpers import log_format
from simpletuner.helpers.logging import get_logger
from simpletuner.helpers.training.multi_process import _get_rank
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.trainer import Trainer

logger = get_logger("SimpleTuner")

if __name__ == "__main__":
    trainer = None
    try:
        import multiprocessing

        multiprocessing.set_start_method("fork")
    except Exception as e:
        logger.error(
            "Failed to set the multiprocessing start method to 'fork'. Unexpected behaviour such as high memory overhead or poor performance may result."
            f"\nError: {e}"
        )
    try:
        trainer = Trainer(
            exit_on_error=True,
        )
        trainer.configure_webhook()
        trainer.init_noise_schedule()
        trainer.init_seed()

        trainer.init_huggingface_hub()

        trainer.init_preprocessing_models()
        trainer.init_precision(preprocessing_models_only=True)
        trainer.init_data_backend()
        # trainer.init_validation_prompts()
        trainer.init_unload_text_encoder()
        trainer.init_unload_vae()

        trainer.init_load_base_model()
        trainer.init_controlnet_model()
        trainer.init_tread_model()
        trainer.init_precision()
        trainer.init_freeze_models()
        trainer.init_trainable_peft_adapter()
        trainer.init_ema_model()
        # EMA must be quantised if the base model is as well.
        trainer.init_precision(ema_only=True)

        trainer.move_models(destination="accelerator")
        trainer.init_distillation()
        trainer.init_validations()
        trainer.enable_sageattention_inference()
        trainer.init_benchmark_base_model()
        trainer.disable_sageattention_inference()

        trainer.resume_and_prepare()

        trainer.init_trackers()
        trainer.train()
    except KeyboardInterrupt:
        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message="Training has been interrupted by user action (lost terminal, or ctrl+C)."
            )
            StateTracker.get_webhook_handler().send_raw(
                structured_data={"status": "interrupted"},
                message_type="training.status",
                message_level="info",
                job_id=StateTracker.get_job_id(),
            )
    except Exception as e:
        import traceback

        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message=f"Training has failed. Please check the logs for more information: {e}"
            )
            StateTracker.get_webhook_handler().send_raw(
                structured_data={"status": "failed", "error": str(e), "traceback": traceback.format_exc()},
                message_type="training.status",
                message_level="error",
                job_id=StateTracker.get_job_id(),
            )
        print(e)
        print(traceback.format_exc())
    if trainer is not None and trainer.bf is not None:
        trainer.bf.stop_fetching()
