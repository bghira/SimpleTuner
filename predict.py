"""Cog predictor entrypoint using the SimpleTuner trainer directly."""

import json
import pathlib
from typing import Optional, Tuple

from cog import BasePredictor, Input, Path, Secret

from simpletuner.cog import CogWebhookReceiver, SimpleTunerCogRunner


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Initialise reusable runner state for Cog."""

        self.runner = SimpleTunerCogRunner()

    def _parse_json_or_path(self, value: str, param_name: str) -> Tuple[Optional[pathlib.Path], Optional[dict]]:
        """Parse a string as either inline JSON or a file path.

        Returns (path, None) if it's a file path, or (None, dict) if it's inline JSON.
        """
        value = value.strip()

        # Try parsing as JSON first
        if value.startswith("{") or value.startswith("["):
            try:
                parsed = json.loads(value)
                return None, parsed
            except json.JSONDecodeError as e:
                raise ValueError(f"{param_name} looks like JSON but failed to parse: {e}")

        # Treat as file path
        path = pathlib.Path(value)
        if not path.exists():
            raise FileNotFoundError(f"{param_name} file not found: {value}")
        return path, None

    def predict(
        self,
        images: Path = Input(
            description="Zip or tar archive of training images. Not required if dataloader_json points to external data.",
            default=None,
        ),
        hf_token: Secret = Input(
            description="Hugging Face token for model downloads (set if the base model requires auth).",
            default=None,
        ),
        config_json: str = Input(
            description="Training config: either a JSON string or path to config.json. Defaults to config/config.json if present.",
            default=None,
        ),
        dataloader_json: str = Input(
            description="Multidatabackend config: either a JSON string or path to file. If not provided, auto-generated from images.",
            default=None,
        ),
        max_train_steps: int = Input(
            description="Override --max_train_steps for quicker Cog runs.",
            default=None,
        ),
        return_logs: bool = Input(
            description="Print the tail of debug.log to Cog output.",
            default=True,
        ),
    ) -> Path:
        """Launch a SimpleTuner training job and return a zipped output directory."""

        token_value = hf_token.get_secret_value() if hf_token else None
        dataset_archive = pathlib.Path(images) if images else None

        # Parse config_json - can be JSON string or file path
        config_path = None
        config_dict = None
        if config_json:
            config_path, config_dict = self._parse_json_or_path(config_json, "config_json")

        # Parse dataloader_json - can be JSON string or file path
        dataloader_path = None
        dataloader_dict = None
        if dataloader_json:
            dataloader_path, dataloader_dict = self._parse_json_or_path(dataloader_json, "dataloader_json")

        # Start the webhook receiver to capture training events in Cog logs
        with CogWebhookReceiver() as webhook_receiver:
            webhook_config = [CogWebhookReceiver.build_webhook_config(webhook_receiver.url)]

            run_result = self.runner.run(
                dataset_archive=dataset_archive,
                hf_token=token_value,
                base_config_path=config_path,
                base_config_dict=config_dict,
                dataloader_config_path=dataloader_path,
                dataloader_config_dict=dataloader_dict,
                max_train_steps=max_train_steps,
                webhook_config=webhook_config,
            )

        archive_path = self.runner.package_output(pathlib.Path(run_result["output_dir"]))

        if return_logs:
            log_tail = self.runner.read_debug_log()
            if log_tail:
                print("\n=== debug.log tail ===")
                print(log_tail[-5000:])

        return Path(archive_path)
