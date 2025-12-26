"""Cog predictor entrypoint using the SimpleTuner trainer directly."""

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from cog import BasePredictor, Input, Path, Secret


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Initialise reusable runner state for Cog."""
        # Lazy import to avoid colored output during Cog introspection
        from simpletuner.cog import SimpleTunerCogRunner

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
        # S3 publishing options (override config)
        s3_bucket: Optional[str] = Input(
            description="S3-compatible bucket for publishing checkpoints (overrides config).",
            default=None,
        ),
        s3_region: Optional[str] = Input(
            description="S3 region (optional).",
            default=None,
        ),
        s3_endpoint_url: Optional[str] = Input(
            description="Custom S3 endpoint URL (for non-AWS providers like Backblaze B2, Cloudflare R2).",
            default=None,
        ),
        s3_base_path: Optional[str] = Input(
            description="Prefix inside the bucket (defaults to simpletuner/{job_id}).",
            default=None,
        ),
        s3_public_base_url: Optional[str] = Input(
            description="Public base URL to build shareable links (optional).",
            default=None,
        ),
        s3_access_key: Optional[Secret] = Input(
            description="S3 access key (leave blank to use IAM/instance roles).",
            default=None,
        ),
        s3_secret_key: Optional[Secret] = Input(
            description="S3 secret key (leave blank to use IAM/instance roles).",
            default=None,
        ),
        # HuggingFace Hub publishing options (override config)
        hub_model_id: Optional[str] = Input(
            description="HuggingFace Hub repo ID (e.g., 'username/my-lora') - overrides config.",
            default=None,
        ),
        hf_token: Secret = Input(
            description="Hugging Face token for model downloads and Hub publishing.",
            default=None,
        ),
        return_logs: bool = Input(
            description="Print the tail of debug.log to Cog output.",
            default=True,
        ),
    ) -> str:
        """Launch a SimpleTuner training job and return the output location."""

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

        # Build config overrides for publishing
        config_overrides: Dict[str, Any] = {}

        # S3 publishing config
        publishing_config: Optional[List[Dict[str, Any]]] = None
        if s3_bucket:
            s3_access_value = s3_access_key.get_secret_value() if s3_access_key else None
            s3_secret_value = s3_secret_key.get_secret_value() if s3_secret_key else None
            publishing_config = self._build_s3_publishing_config(
                bucket=s3_bucket,
                base_path=s3_base_path,
                region=s3_region,
                endpoint_url=s3_endpoint_url,
                access_key=s3_access_value,
                secret_key=s3_secret_value,
                public_base_url=s3_public_base_url,
            )
            config_overrides["publishing_config"] = publishing_config

        # HuggingFace Hub publishing config
        if hub_model_id:
            if not token_value:
                raise ValueError("hf_token is required when using hub_model_id for HuggingFace Hub publishing.")
            config_overrides["--push_to_hub"] = True
            config_overrides["--hub_model_id"] = hub_model_id
            config_overrides["--push_checkpoints_to_hub"] = True

        # Start the webhook receiver to capture training events in Cog logs
        from simpletuner.cog import CogWebhookReceiver

        with CogWebhookReceiver() as webhook_receiver:
            webhook_config = [CogWebhookReceiver.build_webhook_config(webhook_receiver.url)]

            run_result = self.runner.run(
                dataset_archive=dataset_archive,
                hf_token=token_value,
                base_config_path=config_path,
                base_config_dict=config_dict,
                dataloader_config_path=dataloader_path,
                dataloader_config_dict=dataloader_dict,
                config_overrides=config_overrides,
                max_train_steps=max_train_steps,
                webhook_config=webhook_config,
            )

        if return_logs:
            log_tail = self.runner.read_debug_log()
            if log_tail:
                print("\n=== debug.log tail ===")
                print(log_tail[-5000:])

        # Build output URL based on publishing destination
        if s3_bucket and publishing_config:
            path_prefix = publishing_config[0].get("base_path", "").lstrip("/")
            if s3_public_base_url:
                output_url = f"{s3_public_base_url.rstrip('/')}/{path_prefix}"
            elif s3_endpoint_url:
                output_url = f"{s3_endpoint_url.rstrip('/')}/{s3_bucket}/{path_prefix}"
            else:
                output_url = f"s3://{s3_bucket}/{path_prefix}"
            print(f"\nCheckpoints published to: {output_url}")
        elif hub_model_id:
            output_url = f"https://huggingface.co/{hub_model_id}"
            print(f"\nModel published to: {output_url}")
        else:
            # Publishing configured via config_json
            output_url = f"Training complete. Output: {run_result['output_dir']}"
            print(f"\n{output_url}")

        return output_url

    @staticmethod
    def _build_s3_publishing_config(
        *,
        bucket: str,
        base_path: Optional[str] = None,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        public_base_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build S3 publishing config for SimpleTuner."""
        path_prefix = base_path or "simpletuner"
        entry: Dict[str, Any] = {
            "provider": "s3",
            "bucket": bucket,
            "base_path": path_prefix,
        }
        if region:
            entry["region"] = region
        if endpoint_url:
            entry["endpoint_url"] = endpoint_url
        if access_key:
            entry["access_key"] = access_key
        if secret_key:
            entry["secret_key"] = secret_key
        if public_base_url:
            entry["public_base_url"] = public_base_url
        return [entry]
