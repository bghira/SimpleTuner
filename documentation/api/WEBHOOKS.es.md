# Referencia de eventos de webhook

_Este archivo es generado automáticamente por `scripts/generate-webhook-documentation.py`. No lo edites manualmente._

## lifecycle.stage

| Stage Key | Status | Label | Message | Severity | Source |
| --- | --- | --- | --- | --- | --- |
| benchmark_base_model | running | Benchmarking base model | Base model benchmark begins | — | simpletuner/helpers/training/trainer.py:3730 |
| benchmark_base_model | completed | Benchmarking base model | Base model benchmark completed | — | simpletuner/helpers/training/trainer.py:3742 |
| checkpoint_save | running | Saving Checkpoint | Saving checkpoint to {save_path} | — | simpletuner/helpers/training/trainer.py:4764 |
| checkpoint_save | completed | Saving Checkpoint | Saved checkpoint to {save_path} | — | simpletuner/helpers/training/trainer.py:4796 |
| checkpoint_save_distiller | running | Saving Distillation States | Saving distillation states to {save_path_tmp} | — | simpletuner/helpers/training/trainer.py:4805 |
| checkpoint_save_distiller | completed | Saving Distillation States | Saved distillation states to {save_path_tmp} | — | simpletuner/helpers/training/trainer.py:4814 |
| final_validation | running | Running Final Validations | Generating final validation images... | — | simpletuner/helpers/training/trainer.py:5543 |
| final_validation | completed | Running Final Validations | Final validation images completed | — | simpletuner/helpers/training/trainer.py:5558 |
| init_data_backend | running | Configuring data backends | Configuring data backends... (this may take a while!) | — | simpletuner/helpers/training/trainer.py:2251 |
| init_data_backend | completed | Configuring data backends | Completed configuring data backends. | — | simpletuner/helpers/training/trainer.py:2270 |
| init_data_backend | failed | Configuring data backends | Failed to load data backends: {e} | — | simpletuner/helpers/training/trainer.py:2299 |
| init_load_base_model | running | Loading base model | webhook_msg | — | simpletuner/helpers/training/trainer.py:2222 |
| init_load_base_model | completed | Loading base model | Base model has loaded. | — | simpletuner/helpers/training/trainer.py:2233 |
| init_prepare_models | running | Preparing model components | Preparing model components | — | simpletuner/helpers/training/trainer.py:3490 |
| init_prepare_models | completed | Preparing model components | Completed preparing model components | — | simpletuner/helpers/training/trainer.py:3629 |
| init_resume_checkpoint | completed | Resume Checkpoint | No model to resume. Beginning fresh training run. | — | simpletuner/helpers/training/trainer.py:3772 |
| init_resume_checkpoint | running | Resume Checkpoint | Resuming model: {path} | — | simpletuner/helpers/training/trainer.py:3820 |
| init_resume_checkpoint | completed | Resume Checkpoint | Resumed from global_step {self.state['global_resume_step']} | — | simpletuner/helpers/training/trainer.py:3843 |
| init_resume_checkpoint | completed | Resuming checkpoint | — | info | simpletuner/helpers/training/trainer.py:3895 |
| init_vae_cache | completed | VAE Cache initialising | VAE cache initialization complete | — | simpletuner/helpers/caching/vae.py:1443 |
| model_save | running | Saving Final Model | Finalizing model and saving to {self.config.output_dir} | — | simpletuner/helpers/training/trainer.py:5530 |
| model_save | completed | Saving Final Model | Model saved to {self.config.output_dir} | — | simpletuner/helpers/training/trainer.py:5649 |
| training_abort | completed | Training Aborted | Aborting training run. | — | simpletuner/helpers/training/trainer.py:4527 |
| training_complete | completed | Training Complete | Training run complete. | — | simpletuner/helpers/training/trainer.py:5659 |
| type | — | readable_type | — | — | simpletuner/helpers/webhooks/mixin.py:23 |

## training.status

| Status | Message | Severity | Source |
| --- | --- | --- | --- |
| failed | Training failed: {e} | error | simpletuner/helpers/training/trainer.py:1767 |
| running | — | — | simpletuner/helpers/training/trainer.py:4254 |
| running | initial_msg | info | simpletuner/helpers/training/trainer.py:4450 |
| running | — | — | simpletuner/helpers/training/trainer.py:5342 |
| running | — | — | simpletuner/helpers/training/trainer.py:5389 |

## notification

| Message | Title | Severity | Source |
| --- | --- | --- | --- |
| Collected the following data backends: {collected_data_backend_keys} | — | info | simpletuner/helpers/training/trainer.py:2334 |
| Could not update learning rate scheduler LR value. | — | warning | simpletuner/helpers/training/trainer.py:3810 |
| Failed to pin EMA to CPU: {e} | — | warning | simpletuner/helpers/training/trainer.py:3596 |
| Prepared sample debug info | — | debug | simpletuner/helpers/image_manipulation/training_sample.py:390 |
| Training configuration initialized | Training Config | — | simpletuner/helpers/training/trainer.py:3990 |
| Training job has started, configuration has begun. | — | — | simpletuner/helpers/training/trainer.py:1903 |
| message | — | message_level | simpletuner/helpers/training/trainer.py:4307 |
| message | title or "Error" | error | simpletuner/helpers/webhooks/events.py:167 |

## error

| Message | Title | Severity | Source |
| --- | --- | --- | --- |
| Could not initialize trackers. Continuing without. {e} | — | error | simpletuner/helpers/training/trainer.py:3985 |
| Failed to run training: {e} | Fatal Error | error | simpletuner/helpers/training/trainer.py:1759 |

## training.checkpoint

| Label | Path | Final | Severity | Source |
| --- | --- | --- | --- | --- |
| Checkpoint saved to {save_path} | save_path | — | — | simpletuner/helpers/training/trainer.py:4834 |
