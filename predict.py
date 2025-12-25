"""Cog predictor entrypoint using the SimpleTuner trainer directly."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cog import BasePredictor, Input
from cog import Path as CogPath
from cog import Secret

from simpletuner.cog import CogWebhookReceiver, SimpleTunerCogRunner


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Initialise reusable runner state for Cog."""

        self.runner = SimpleTunerCogRunner()

    def predict(
        self,
        images: Optional[CogPath] = Input(
            description="Zip or tar archive of training images. Not required if dataloader_json points to external data.",
            default=None,
        ),
        hf_token: Optional[Secret] = Input(
            description="Hugging Face token for model downloads (set if the base model requires auth).",
            default=None,
        ),
        config_json: Optional[CogPath] = Input(
            description="Training config JSON (config.json). Defaults to config/config.json if present.",
            default=None,
        ),
        dataloader_json: Optional[CogPath] = Input(
            description="Multidatabackend config JSON. If not provided, a simple config is auto-generated from the images archive.",
            default=None,
        ),
        max_train_steps: Optional[int] = Input(
            description="Override --max_train_steps for quicker Cog runs.",
            default=None,
        ),
        return_logs: bool = Input(
            description="Print the tail of debug.log to Cog output.",
            default=True,
        ),
    ) -> CogPath:
        """Launch a SimpleTuner training job and return a zipped output directory."""

        token_value = hf_token.get_secret_value() if hf_token else None
        config_path = Path(config_json) if config_json else None
        dataloader_path = Path(dataloader_json) if dataloader_json else None
        dataset_archive = Path(images) if images else None

        # Start the webhook receiver to capture training events in Cog logs
        with CogWebhookReceiver() as webhook_receiver:
            webhook_config = [CogWebhookReceiver.build_webhook_config(webhook_receiver.url)]

            run_result = self.runner.run(
                dataset_archive=dataset_archive,
                hf_token=token_value,
                base_config_path=config_path,
                dataloader_config_path=dataloader_path,
                max_train_steps=max_train_steps,
                webhook_config=webhook_config,
            )

        archive_path = self.runner.package_output(Path(run_result["output_dir"]))

        if return_logs:
            log_tail = self.runner.read_debug_log()
            if log_tail:
                print("\n=== debug.log tail ===")
                print(log_tail[-5000:])

        return CogPath(archive_path)
