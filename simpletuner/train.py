import atexit
import json
import logging
from os import environ
from pathlib import Path

# Import WebhookLogger setup FIRST before creating any loggers
from simpletuner.helpers.logging import get_logger

# Quiet down, you.
ds_logger1 = logging.getLogger("DeepSpeed")
ds_logger2 = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
ds_logger1.setLevel("ERROR")
ds_logger2.setLevel("ERROR")
import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,  # Changed from True to preserve WebhookLogger
    }
)

environ["ACCELERATE_LOG_LEVEL"] = "WARNING"

from simpletuner.helpers import log_format
from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase
from simpletuner.helpers.training.gpu_circuit_breaker import get_current_gpu_index, get_gpu_circuit_breaker, is_cuda_error
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


def _configure_last_ditch_webhook():
    """
    Attempt to configure a webhook handler when no Trainer instance is available.
    This is a last-ditch effort to send error notifications when training fails
    before the normal webhook configuration happens.
    """
    import json
    import sys

    # Try to extract webhook_config from CLI args
    webhook_config = None
    for i, arg in enumerate(sys.argv):
        if arg == "--webhook_config" and i + 1 < len(sys.argv):
            webhook_config = sys.argv[i + 1]
            break
        if arg.startswith("--webhook_config="):
            webhook_config = arg.split("=", 1)[1]
            break

    if not webhook_config:
        return

    # Parse the webhook config
    if webhook_config.startswith("{") or webhook_config.startswith("["):
        try:
            parsed_config = json.loads(webhook_config)
            if isinstance(parsed_config, dict):
                webhook_config = [parsed_config]
            elif isinstance(parsed_config, list):
                webhook_config = parsed_config
            else:
                return
        except json.JSONDecodeError:
            return
    elif Path(webhook_config).is_file():
        try:
            with open(webhook_config, "r") as f:
                parsed_config = json.load(f)
                if isinstance(parsed_config, dict):
                    webhook_config = [parsed_config]
                elif isinstance(parsed_config, list):
                    webhook_config = parsed_config
                else:
                    return
        except Exception:
            return
    else:
        return

    # Create a minimal webhook handler without accelerator
    from simpletuner.helpers.webhooks.handler import WebhookHandler

    handler = WebhookHandler(
        accelerator=None,
        project_name="SimpleTuner (emergency)",
        webhook_config=webhook_config,
    )
    StateTracker.set_webhook_handler(handler)
    logger.info("Configured last-ditch webhook handler for error reporting")


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

        # Check if this is a CUDA/GPU error and trigger circuit breaker
        if is_cuda_error(e):
            try:
                circuit_breaker = get_gpu_circuit_breaker(
                    webhook_handler=StateTracker.get_webhook_handler(),
                    job_id=StateTracker.get_job_id(),
                )
                gpu_idx = get_current_gpu_index()
                circuit_breaker.record_cuda_error(e, gpu_idx)
                logger.error(f"GPU circuit breaker triggered by CUDA error on GPU {gpu_idx}: {e}")
            except Exception as cb_err:
                logger.warning(f"Failed to trigger GPU circuit breaker: {cb_err}")

        # If webhook handler isn't configured yet (crash happened early), try to configure it now
        if StateTracker.get_webhook_handler() is None:
            try:
                _configure_last_ditch_webhook()
            except Exception as webhook_err:
                logger.warning(f"Failed to configure last-ditch webhook: {webhook_err}")

        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message=f"Training has failed. Please check the logs for more information: {e}"
            )
            # Send error event with proper message type so WebUI displays it correctly
            StateTracker.get_webhook_handler().send_raw(
                structured_data={
                    "title": f"Training failed: {type(e).__name__}",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
                message_type="error",
                message_level="error",
                job_id=StateTracker.get_job_id(),
            )
            # Also send status update to update job state
            StateTracker.get_webhook_handler().send_raw(
                structured_data={"status": "failed"},
                message_type="training.status",
                message_level="error",
                job_id=StateTracker.get_job_id(),
            )
        raise
    finally:
        if trainer is not None:
            trainer.cleanup()
