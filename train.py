from helpers.training.trainer import Trainer
from helpers.training.state_tracker import StateTracker
from helpers import log_format
import logging
from os import environ

logger = logging.getLogger("SimpleTuner")
logger.setLevel(environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

if __name__ == "__main__":
    global bf
    bf = None
    try:
        import multiprocessing

        multiprocessing.set_start_method("fork")
    except Exception as e:
        logger.error(
            "Failed to set the multiprocessing start method to 'fork'. Unexpected behaviour such as high memory overhead or poor performance may result."
            f"\nError: {e}"
        )
    try:
        trainer = Trainer()
        trainer.configure_webhook()
        trainer.init_noise_schedule()
        trainer.init_seed()

        trainer.init_huggingface_hub()

        trainer.init_preprocessing_models()
        trainer.init_data_backend()
        trainer.init_validation_prompts()
        trainer.init_unload_text_encoder()
        trainer.init_unload_vae()

        trainer.init_load_base_model()
        trainer.init_controlnet_model()
        trainer.init_freeze_models()
        trainer.init_trainable_peft_adapter()
        trainer.init_ema_model()
        trainer.init_precision()

        trainer.init_validations()
        trainer.init_benchmark_base_model()

        trainer.init_optimizer()
        trainer.init_lr_scheduler()

        trainer.init_hooks()
        trainer.init_resume_checkpoint()
        trainer.init_post_load_freeze()
        trainer.init_prepare_models()

        trainer.init_trackers()
        trainer.train()
    except KeyboardInterrupt:
        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message="Training has been interrupted by user action (lost terminal, or ctrl+C)."
            )
    except Exception as e:
        import traceback

        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message=f"Training has failed. Please check the logs for more information: {e}"
            )
        print(e)
        print(traceback.format_exc())
    if trainer.bf is not None:
        bf.stop_fetching()
