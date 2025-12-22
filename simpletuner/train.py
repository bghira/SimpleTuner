import atexit
import json
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
from pathlib import Path

environ["ACCELERATE_LOG_LEVEL"] = "WARNING"

from simpletuner.helpers import log_format
from simpletuner.helpers.logging import get_logger
from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase
from simpletuner.helpers.training.multi_process import _get_rank
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.trainer import Trainer

# Configure third-party loggers after imports
if hasattr(log_format, "configure_third_party_loggers"):
    log_format.configure_third_party_loggers()

logger = get_logger("SimpleTuner")


def _build_signal_consumer(signal_path_text: str | None, key: str):
    if not signal_path_text:
        return None

    signal_path = Path(signal_path_text)
    seen = 0
    pending = 0
    last_mtime = None
    read_error_logged = False

    def _consume():
        nonlocal seen, pending, last_mtime, read_error_logged

        try:
            stat = signal_path.stat()
        except FileNotFoundError:
            if not read_error_logged:
                logger.warning("Accelerate trigger file missing at %s", signal_path)
                read_error_logged = True
            return False
        except Exception as exc:
            if not read_error_logged:
                logger.warning("Failed to stat accelerate trigger file %s: %s", signal_path, exc)
                read_error_logged = True
            return False

        if last_mtime is None or stat.st_mtime > last_mtime:
            last_mtime = stat.st_mtime
            try:
                with signal_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                read_error_logged = False
            except Exception as exc:
                if not read_error_logged:
                    logger.warning("Failed to read accelerate trigger file %s: %s", signal_path, exc)
                    read_error_logged = True
                return False
            if not isinstance(payload, dict):
                if not read_error_logged:
                    logger.warning("Unexpected accelerate trigger payload in %s", signal_path)
                    read_error_logged = True
                return False
            try:
                count_value = int(payload.get(key, 0))
            except (TypeError, ValueError):
                count_value = 0
            if count_value > seen:
                pending += count_value - seen
                seen = count_value

        if pending > 0:
            pending -= 1
            return True
        return False

    return _consume


if __name__ == "__main__":
    trainer = None

    def _cleanup_trainer():
        if trainer is not None:
            try:
                trainer.cleanup()
            except Exception as exc:
                logger.error("Trainer cleanup failed during interpreter exit: %s", exc, exc_info=True)

    atexit.register(_cleanup_trainer)
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
        signal_file = environ.get("SIMPLETUNER_ACCELERATE_SIGNAL_FILE")
        validation_consumer = _build_signal_consumer(signal_file, "manual_validation")
        checkpoint_consumer = _build_signal_consumer(signal_file, "manual_checkpoint")
        if callable(validation_consumer):
            trainer.register_manual_validation_trigger(validation_consumer)
        if callable(checkpoint_consumer):
            trainer.register_manual_checkpoint_trigger(checkpoint_consumer)
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
        trainer.init_delete_model_caches()

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
        AttentionBackendController.apply(trainer.config, AttentionPhase.EVAL)
        trainer.init_benchmark_base_model()
        AttentionBackendController.apply(trainer.config, AttentionPhase.TRAIN)

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
    if trainer is not None:
        trainer.cleanup()
