"""Test publishing_config parsing and manager wiring."""

import json
import tempfile
import unittest
from pathlib import Path

from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args
from simpletuner.helpers.publishing.manager import PublishingManager
from simpletuner.helpers.publishing.providers.base import PublishingProvider


def _base_args():
    return [
        "--model_family=pixart_sigma",
        "--output_dir=/tmp/output",
        "--model_type=lora",
        "--optimizer=adamw_bf16",
        "--data_backend_config=/tmp/config.json",
    ]


class _StubProvider(PublishingProvider):
    def __init__(self, config):
        super().__init__("stub", config)
        self.calls = []

    def publish(self, artifact_path, *, artifact_name=None, metadata=None):
        metadata = metadata or {}
        self.calls.append({"artifact_path": Path(artifact_path), "artifact_name": artifact_name, "metadata": metadata})
        return self._record_result(
            artifact_path=Path(artifact_path),
            uri="stub://published",
            metadata=metadata,
        )


class TestPublishingConfigParsing(unittest.TestCase):
    def test_inline_publishing_config_json(self):
        config_json = json.dumps([{"provider": "s3", "bucket": "demo"}])
        args_list = _base_args() + [f"--publishing_config={config_json}"]

        args = parse_cmdline_args(input_args=args_list, exit_on_error=False)

        self.assertIsNotNone(args)
        self.assertIsInstance(args.publishing_config, list)
        self.assertEqual(args.publishing_config[0]["provider"], "s3")

    def test_file_based_publishing_config(self):
        config = [{"provider": "azure_blob", "container": "demo"}]
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            json.dump(config, handle)
            temp_path = handle.name
        try:
            args_list = _base_args() + [f"--publishing_config={temp_path}"]
            args = parse_cmdline_args(input_args=args_list, exit_on_error=True)
            self.assertEqual(args.publishing_config, config)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_publishing_manager_uses_registry(self):
        manager = PublishingManager([{"provider": "stub"}], provider_registry={"stub": _StubProvider})
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "artifact.txt"
            artifact.write_text("demo")
            metadata = {"custom": True}

            results = manager.publish(artifact, artifact_name="artifact", metadata=metadata)

        self.assertEqual(len(results), 1)
        stub = manager.providers[0]
        self.assertIsInstance(stub, _StubProvider)
        self.assertEqual(stub.calls[0]["artifact_name"], "artifact")
        self.assertEqual(stub.calls[0]["metadata"]["custom"], True)
        self.assertEqual(manager.latest_uri(), "stub://published")


if __name__ == "__main__":
    unittest.main()
