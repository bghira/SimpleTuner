import tempfile
import unittest
from pathlib import Path

try:
    from simpletuner.simpletuner_sdk.server.utils.paths import resolve_config_path
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    resolve_config_path = None  # type: ignore[assignment]
    _SKIP_REASON = str(exc)
else:
    _SKIP_REASON = ""


@unittest.skipIf(resolve_config_path is None, f"Dependencies unavailable: {_SKIP_REASON}")
class ResolveConfigPathTestCase(unittest.TestCase):
    def test_trim_redundant_config_segment_when_config_dir_has_plural_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            configs_dir = Path(tmp_dir) / "configs"
            examples_dir = configs_dir / "examples"
            examples_dir.mkdir(parents=True, exist_ok=True)
            target_file = examples_dir / "lycoris_config.json"
            target_file.write_text("{}", encoding="utf-8")

            resolved = resolve_config_path(
                "config/examples/lycoris_config.json",
                config_dir=str(configs_dir),
                check_cwd_first=False,
            )

            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.resolve(), target_file.resolve())

    def test_trim_redundant_config_segment_when_names_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_dir = Path(tmp_dir) / "config"
            examples_dir = config_dir / "examples"
            examples_dir.mkdir(parents=True, exist_ok=True)
            target_file = examples_dir / "lycoris_config.json"
            target_file.write_text("{}", encoding="utf-8")

            resolved = resolve_config_path(
                "config/examples/lycoris_config.json",
                config_dir=str(config_dir),
                check_cwd_first=False,
            )

            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.resolve(), target_file.resolve())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
