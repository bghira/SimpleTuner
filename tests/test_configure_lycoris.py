import json
import tempfile
import unittest
from pathlib import Path

from simpletuner.configure import LycorisBuilderSession
from simpletuner.simpletuner_sdk.server.services.lycoris_builder_service import LYCORIS_BUILDER_SERVICE

if LYCORIS_BUILDER_SERVICE is None:  # pragma: no cover - guardrail for misconfigured envs
    raise RuntimeError("LyCORIS builder service is required for these tests")


class LycorisBuilderSessionTests(unittest.TestCase):
    """Exercise the interactive LyCORIS builder helpers without mocking."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def _session(self, name: str = "lycoris.json") -> LycorisBuilderSession:
        path = Path(self.temp_dir.name) / name
        return LycorisBuilderSession(LYCORIS_BUILDER_SERVICE, path)

    def test_initial_defaults_include_known_algorithm(self) -> None:
        session = self._session()
        algo = session.config.get("algo")
        self.assertIsInstance(algo, str)
        self.assertIn(algo, session.get_algorithm_names())
        self.assertGreaterEqual(session.config.get("linear_dim", 0), 1)

    def test_apply_preset_normalizes_target_lists(self) -> None:
        session = self._session()
        presets = session.get_preset_names()
        self.assertTrue(presets, "Expected at least one built-in LyCORIS preset")
        applied = session.apply_preset(presets[0])
        self.assertTrue(applied)
        payload = session.to_serializable().get("apply_preset")
        self.assertIsInstance(payload, dict)
        # All list-based entries should be stored as actual lists, not strings
        for field in [
            "target_module",
            "target_name",
            "unet_target_module",
            "unet_target_name",
            "text_encoder_target_module",
            "text_encoder_target_name",
            "exclude_name",
        ]:
            if field in payload:
                self.assertIsInstance(payload[field], list)

    def test_save_and_reload_roundtrip(self) -> None:
        session = self._session("roundtrip.json")
        session.set_algorithm("lora", reset=True)
        session.set_list("target_module", ["Attention", "FeedForward"])
        session.set_numeric("multiplier", 1.5)
        session.set_numeric("linear_dim", 96)
        saved_path = session.save_to_file()
        self.assertTrue(saved_path.exists())
        with saved_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        self.assertEqual(raw.get("algo"), "lora")

        clone = LycorisBuilderSession(LYCORIS_BUILDER_SERVICE, saved_path)
        clone.load_from_file()
        self.assertEqual(session.to_serializable(), clone.to_serializable())

    def test_override_management_flow(self) -> None:
        session = self._session("overrides.json")
        session.set_algorithm("lokr", reset=True)
        session.upsert_override("module", "Attention")
        session.set_override_algo("module", "Attention", "lokr")
        session.set_override_option("module", "Attention", "factor", 8)
        entries = session.get_override_entries("module")
        self.assertIn("Attention", entries)
        self.assertEqual(entries["Attention"].get("factor"), 8)

        session.rename_override("module", "Attention", "Attn")
        renamed = session.get_override_entries("module")
        self.assertIn("Attn", renamed)
        self.assertNotIn("Attention", renamed)
        session.delete_override("module", "Attn")
        self.assertFalse(session.get_override_entries("module"))


if __name__ == "__main__":
    unittest.main()
