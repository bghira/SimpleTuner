from __future__ import annotations

import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simpletuner.helpers.utils.checkpoint_manager import CheckpointManager

try:
    from simpletuner.simpletuner_sdk.server.services.checkpoints_service import CheckpointsService
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    CheckpointsService = None  # type: ignore[assignment]
    _SKIP_REASON = str(exc)
else:
    _SKIP_REASON = ""


@unittest.skipIf(CheckpointsService is None, f"Dependencies unavailable: {_SKIP_REASON}")
class CheckpointCleanupTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        for step in range(5):
            os.makedirs(os.path.join(self.tempdir.name, f"checkpoint-{step+1}"))
        self.manager = CheckpointManager(self.tempdir.name)
        self.service = CheckpointsService()

    def _patch_manager(self):
        return patch.object(
            CheckpointsService,
            "_get_checkpoint_manager",
            return_value=self.manager,
        )

    def test_manager_cleanup_respects_limit(self) -> None:
        self.manager.cleanup_checkpoints(limit=1)
        remaining = sorted(d for d in os.listdir(self.tempdir.name) if d.startswith("checkpoint-"))
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0], "checkpoint-5")

    def test_preview_and_execute_cleanup_counts(self) -> None:
        with self._patch_manager():
            preview = self.service.preview_cleanup("env", limit=1)
            self.assertEqual(preview["count_to_remove"], 4)
            self.assertEqual(preview["checkpoints_to_keep"], 1)
            self.assertEqual(len(preview["checkpoints_to_remove"]), 4)

            result = self.service.execute_cleanup("env", limit=1)
            self.assertEqual(result["count_removed"], 4)
            self.assertEqual(len(result["removed_checkpoints"]), 4)

        remaining = sorted(d for d in os.listdir(self.tempdir.name) if d.startswith("checkpoint-"))
        self.assertEqual(remaining, ["checkpoint-5"])

    def test_build_checkpoint_preview_inlines_assets_in_readme(self) -> None:
        checkpoint_name = "checkpoint-1"
        checkpoint_path = Path(self.tempdir.name) / checkpoint_name
        assets_dir = checkpoint_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABAABDQottAAAAABJRU5ErkJggg=="
        )
        (assets_dir / "image.png").write_bytes(tiny_png)

        # Required files for validation
        (checkpoint_path / "training_state.json").write_text('{"global_step": 1}', encoding="utf-8")
        (checkpoint_path / "pytorch_lora_weights.safetensors").write_bytes(b"test")

        readme_content = (
            "---\n"
            "thumbnail: assets/image.png\n"
            "---\n"
            "# Sample\n\n"
            "![Val](assets/image.png)\n"
            "Inline ./assets/image.png usage\n"
        )
        (checkpoint_path / "README.md").write_text(readme_content, encoding="utf-8")

        checkpoint_info = {"name": checkpoint_name, "path": str(checkpoint_path)}
        preview = self.service._build_checkpoint_preview("env", self.manager, checkpoint_info)

        readme = preview.get("readme") or {}
        self.assertIsInstance(readme, dict)
        body = readme.get("body") or ""
        front_matter = readme.get("front_matter") or ""

        self.assertIn("data:image/png;base64", body)
        self.assertNotIn("assets/image.png", body)
        self.assertNotIn("./assets/image.png", body)

        # Front matter should remain untouched for downstream parsing
        self.assertIn("assets/image.png", front_matter)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
