import tempfile
import unittest
from pathlib import Path

from fastapi import status

from simpletuner.simpletuner_sdk.server.services.prompt_library_service import (
    PromptLibraryEntry,
    PromptLibraryError,
    PromptLibraryService,
)


class PromptLibraryServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self._tmpdir.name)
        self.service = PromptLibraryService(config_dir=self.base_dir)
        self.libs_dir = self.base_dir / "validation_prompt_libraries"
        self.libs_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_save_and_list_libraries(self) -> None:
        entries = {"hello": "Hello prompt", "bye": "Goodbye prompt"}
        result = self.service.save_library("user_prompt_library-test.json", entries)

        self.assertEqual(result["library"].filename, "user_prompt_library-test.json")
        records = self.service.list_libraries()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.prompt_count, len(entries))
        self.assertEqual(record.relative_path, "validation_prompt_libraries/user_prompt_library-test.json")

        payload = self.service.read_library("user_prompt_library-test.json")
        self.assertEqual(payload["entries"], entries)

    def test_rename_library_replaces_old_file(self) -> None:
        first_entries = {"a": "first"}
        self.service.save_library("user_prompt_library-old.json", first_entries)
        second_entries = {"b": "second"}
        self.service.save_library(
            "user_prompt_library-new.json",
            second_entries,
            previous_filename="user_prompt_library-old.json",
        )

        self.assertTrue((self.libs_dir / "user_prompt_library-new.json").exists())
        self.assertFalse((self.libs_dir / "user_prompt_library-old.json").exists())

    def test_invalid_filename_raises(self) -> None:
        with self.assertRaises(PromptLibraryError) as ctx:
            self.service.save_library("invalid name.json", {"test": "value"})
        self.assertEqual(ctx.exception.status_code, status.HTTP_400_BAD_REQUEST)

    def test_read_missing_library_raises(self) -> None:
        with self.assertRaises(PromptLibraryError) as ctx:
            self.service.read_library("user_prompt_library-missing.json")
        self.assertEqual(ctx.exception.status_code, status.HTTP_404_NOT_FOUND)

    def test_adapter_strength_entries_round_trip(self) -> None:
        entries = {
            "plain": "base prompt",
            "weighted": {"prompt": "slider prompt", "adapter_strength": 0.4},
            "entry_obj": PromptLibraryEntry(prompt="object prompt", adapter_strength=0.2),
        }
        result = self.service.save_library("user_prompt_library-slider.json", entries)
        saved_entries = result["entries"]
        self.assertEqual(
            saved_entries["weighted"],
            {"prompt": "slider prompt", "adapter_strength": 0.4},
        )
        self.assertEqual(
            saved_entries["entry_obj"],
            {"prompt": "object prompt", "adapter_strength": 0.2},
        )

        payload = self.service.read_library("user_prompt_library-slider.json")
        self.assertEqual(payload["entries"]["weighted"]["adapter_strength"], 0.4)
        self.assertEqual(payload["entries"]["entry_obj"]["prompt"], "object prompt")

    def test_invalid_adapter_strength_entry_rejected(self) -> None:
        with self.assertRaises(PromptLibraryError) as ctx:
            self.service.save_library(
                "user_prompt_library-invalid.json",
                {"broken": {"adapter_strength": 0.5}},
            )
        self.assertEqual(ctx.exception.status_code, status.HTTP_400_BAD_REQUEST)

    def test_bbox_entities_round_trip(self) -> None:
        entities = [
            {"label": "cat", "bbox": [0.2, 0.3, 0.6, 0.8]},
            {"label": "table", "bbox": [0.0, 0.5, 1.0, 1.0]},
        ]
        entries = {
            "cat_table": {
                "prompt": "a cat on a table",
                "bbox_entities": entities,
            },
        }
        result = self.service.save_library("user_prompt_library-bbox.json", entries)
        saved = result["entries"]["cat_table"]
        self.assertEqual(saved["prompt"], "a cat on a table")
        self.assertEqual(len(saved["bbox_entities"]), 2)
        self.assertEqual(saved["bbox_entities"][0]["label"], "cat")
        self.assertAlmostEqual(saved["bbox_entities"][0]["bbox"][0], 0.2)

        payload = self.service.read_library("user_prompt_library-bbox.json")
        reloaded = payload["entries"]["cat_table"]
        self.assertEqual(len(reloaded["bbox_entities"]), 2)
        self.assertEqual(reloaded["bbox_entities"][1]["label"], "table")


class PromptLibraryEntryBboxTestCase(unittest.TestCase):
    def test_from_payload_with_bbox_entities(self) -> None:
        payload = {
            "prompt": "test prompt",
            "bbox_entities": [
                {"label": "dog", "bbox": [0.1, 0.1, 0.5, 0.5]},
            ],
        }
        entry = PromptLibraryEntry.from_payload(payload)
        self.assertEqual(entry.prompt, "test prompt")
        self.assertIsNotNone(entry.bbox_entities)
        self.assertEqual(len(entry.bbox_entities), 1)
        self.assertEqual(entry.bbox_entities[0]["label"], "dog")

    def test_from_payload_string_has_no_bbox(self) -> None:
        entry = PromptLibraryEntry.from_payload("simple prompt")
        self.assertIsNone(entry.bbox_entities)

    def test_from_payload_without_bbox_has_none(self) -> None:
        entry = PromptLibraryEntry.from_payload({"prompt": "no bbox"})
        self.assertIsNone(entry.bbox_entities)

    def test_bbox_entities_not_list_raises(self) -> None:
        with self.assertRaises(PromptLibraryError):
            PromptLibraryEntry.from_payload({"prompt": "p", "bbox_entities": "bad"})

    def test_bbox_entities_item_not_dict_raises(self) -> None:
        with self.assertRaises(PromptLibraryError):
            PromptLibraryEntry.from_payload({"prompt": "p", "bbox_entities": ["bad"]})

    def test_bbox_entities_missing_label_raises(self) -> None:
        with self.assertRaises(PromptLibraryError):
            PromptLibraryEntry.from_payload(
                {
                    "prompt": "p",
                    "bbox_entities": [{"bbox": [0.1, 0.1, 0.5, 0.5]}],
                }
            )

    def test_bbox_entities_missing_bbox_raises(self) -> None:
        with self.assertRaises(PromptLibraryError):
            PromptLibraryEntry.from_payload(
                {
                    "prompt": "p",
                    "bbox_entities": [{"label": "a"}],
                }
            )

    def test_bbox_entities_wrong_length_raises(self) -> None:
        with self.assertRaises(PromptLibraryError):
            PromptLibraryEntry.from_payload(
                {
                    "prompt": "p",
                    "bbox_entities": [{"label": "a", "bbox": [0.1, 0.1, 0.5]}],
                }
            )

    def test_bbox_entities_degenerate_raises(self) -> None:
        with self.assertRaises(PromptLibraryError):
            PromptLibraryEntry.from_payload(
                {
                    "prompt": "p",
                    "bbox_entities": [{"label": "a", "bbox": [0.5, 0.5, 0.5, 0.5]}],
                }
            )

    def test_bbox_entities_clamping(self) -> None:
        entry = PromptLibraryEntry.from_payload(
            {
                "prompt": "p",
                "bbox_entities": [{"label": "a", "bbox": [-0.1, -0.2, 1.5, 1.3]}],
            }
        )
        bbox = entry.bbox_entities[0]["bbox"]
        self.assertAlmostEqual(bbox[0], 0.0)
        self.assertAlmostEqual(bbox[1], 0.0)
        self.assertAlmostEqual(bbox[2], 1.0)
        self.assertAlmostEqual(bbox[3], 1.0)

    def test_serialise_includes_bbox_entities(self) -> None:
        entry = PromptLibraryEntry(
            prompt="test",
            bbox_entities=[{"label": "cat", "bbox": [0.1, 0.2, 0.3, 0.4]}],
        )
        data = entry.serialise()
        self.assertIn("bbox_entities", data)
        self.assertEqual(data["bbox_entities"][0]["label"], "cat")

    def test_serialise_omits_bbox_when_none(self) -> None:
        entry = PromptLibraryEntry(prompt="test")
        data = entry.serialise()
        self.assertNotIn("bbox_entities", data)

    def test_serialise_entries_uses_dict_for_bbox(self) -> None:
        entries = {
            "plain": PromptLibraryEntry(prompt="simple"),
            "grounded": PromptLibraryEntry(
                prompt="with bbox",
                bbox_entities=[{"label": "a", "bbox": [0.1, 0.1, 0.5, 0.5]}],
            ),
        }
        result = PromptLibraryService.serialise_entries(entries)
        self.assertEqual(result["plain"], "simple")
        self.assertIsInstance(result["grounded"], dict)
        self.assertIn("bbox_entities", result["grounded"])

    def test_empty_bbox_entities_list_becomes_none(self) -> None:
        entry = PromptLibraryEntry.from_payload(
            {
                "prompt": "p",
                "bbox_entities": [],
            }
        )
        self.assertIsNone(entry.bbox_entities)


class PromptLibraryEntryBboxKeyframesTestCase(unittest.TestCase):
    def test_from_payload_with_bbox_keyframes(self) -> None:
        payload = {
            "prompt": "test prompt",
            "bbox_keyframes": [
                {"frame": 0, "entities": [{"label": "cat", "bbox": [0.1, 0.2, 0.5, 0.6]}]},
                {"frame": 10, "entities": [{"label": "cat", "bbox": [0.3, 0.4, 0.7, 0.8]}]},
            ],
        }
        entry = PromptLibraryEntry.from_payload(payload)
        self.assertEqual(entry.prompt, "test prompt")
        self.assertIsNotNone(entry.bbox_keyframes)
        self.assertEqual(len(entry.bbox_keyframes), 2)
        self.assertEqual(entry.bbox_keyframes[0]["frame"], 0)
        self.assertEqual(entry.bbox_keyframes[1]["frame"], 10)

    def test_from_payload_string_has_no_keyframes(self) -> None:
        entry = PromptLibraryEntry.from_payload("simple prompt")
        self.assertIsNone(entry.bbox_keyframes)

    def test_from_payload_without_keyframes_has_none(self) -> None:
        entry = PromptLibraryEntry.from_payload({"prompt": "no keyframes"})
        self.assertIsNone(entry.bbox_keyframes)

    def test_bbox_keyframes_not_list_raises(self) -> None:
        with self.assertRaises(PromptLibraryError):
            PromptLibraryEntry.from_payload({"prompt": "p", "bbox_keyframes": "bad"})

    def test_bbox_keyframes_item_not_dict_raises(self) -> None:
        with self.assertRaises(PromptLibraryError):
            PromptLibraryEntry.from_payload({"prompt": "p", "bbox_keyframes": ["bad"]})

    def test_bbox_keyframes_missing_frame_raises(self) -> None:
        with self.assertRaises(PromptLibraryError):
            PromptLibraryEntry.from_payload(
                {
                    "prompt": "p",
                    "bbox_keyframes": [{"entities": [{"label": "a", "bbox": [0.1, 0.1, 0.5, 0.5]}]}],
                }
            )

    def test_bbox_keyframes_negative_frame_raises(self) -> None:
        with self.assertRaises(PromptLibraryError):
            PromptLibraryEntry.from_payload(
                {
                    "prompt": "p",
                    "bbox_keyframes": [{"frame": -1, "entities": [{"label": "a", "bbox": [0.1, 0.1, 0.5, 0.5]}]}],
                }
            )

    def test_bbox_keyframes_empty_entities_raises(self) -> None:
        with self.assertRaises(PromptLibraryError):
            PromptLibraryEntry.from_payload(
                {
                    "prompt": "p",
                    "bbox_keyframes": [{"frame": 0, "entities": []}],
                }
            )

    def test_bbox_keyframes_invalid_entity_raises(self) -> None:
        with self.assertRaises(PromptLibraryError):
            PromptLibraryEntry.from_payload(
                {
                    "prompt": "p",
                    "bbox_keyframes": [{"frame": 0, "entities": [{"label": "", "bbox": [0, 0, 1, 1]}]}],
                }
            )

    def test_bbox_keyframes_sorted_by_frame(self) -> None:
        payload = {
            "prompt": "p",
            "bbox_keyframes": [
                {"frame": 10, "entities": [{"label": "a", "bbox": [0.1, 0.1, 0.5, 0.5]}]},
                {"frame": 0, "entities": [{"label": "a", "bbox": [0.2, 0.2, 0.6, 0.6]}]},
            ],
        }
        entry = PromptLibraryEntry.from_payload(payload)
        self.assertEqual(entry.bbox_keyframes[0]["frame"], 0)
        self.assertEqual(entry.bbox_keyframes[1]["frame"], 10)

    def test_empty_bbox_keyframes_list_becomes_none(self) -> None:
        entry = PromptLibraryEntry.from_payload({"prompt": "p", "bbox_keyframes": []})
        self.assertIsNone(entry.bbox_keyframes)

    def test_serialise_includes_bbox_keyframes(self) -> None:
        entry = PromptLibraryEntry(
            prompt="test",
            bbox_keyframes=[{"frame": 0, "entities": [{"label": "cat", "bbox": [0.1, 0.2, 0.3, 0.4]}]}],
        )
        data = entry.serialise()
        self.assertIn("bbox_keyframes", data)
        self.assertEqual(data["bbox_keyframes"][0]["frame"], 0)

    def test_serialise_omits_keyframes_when_none(self) -> None:
        entry = PromptLibraryEntry(prompt="test")
        data = entry.serialise()
        self.assertNotIn("bbox_keyframes", data)

    def test_serialise_entries_uses_dict_for_keyframes(self) -> None:
        entries = {
            "plain": PromptLibraryEntry(prompt="simple"),
            "keyed": PromptLibraryEntry(
                prompt="with keyframes",
                bbox_keyframes=[{"frame": 0, "entities": [{"label": "a", "bbox": [0.1, 0.1, 0.5, 0.5]}]}],
            ),
        }
        result = PromptLibraryService.serialise_entries(entries)
        self.assertEqual(result["plain"], "simple")
        self.assertIsInstance(result["keyed"], dict)
        self.assertIn("bbox_keyframes", result["keyed"])


class PromptLibraryKeyframeRoundTripTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self._tmpdir.name)
        self.service = PromptLibraryService(config_dir=self.base_dir)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_keyframes_round_trip(self) -> None:
        keyframes = [
            {"frame": 0, "entities": [{"label": "cat", "bbox": [0.1, 0.2, 0.5, 0.6]}]},
            {"frame": 20, "entities": [{"label": "cat", "bbox": [0.3, 0.4, 0.7, 0.8]}]},
        ]
        entries = {
            "moving_cat": {
                "prompt": "a cat walking across the room",
                "bbox_keyframes": keyframes,
            },
        }
        result = self.service.save_library("user_prompt_library-kf.json", entries)
        saved = result["entries"]["moving_cat"]
        self.assertEqual(saved["prompt"], "a cat walking across the room")
        self.assertEqual(len(saved["bbox_keyframes"]), 2)
        self.assertEqual(saved["bbox_keyframes"][0]["frame"], 0)
        self.assertEqual(saved["bbox_keyframes"][1]["frame"], 20)

        payload = self.service.read_library("user_prompt_library-kf.json")
        reloaded = payload["entries"]["moving_cat"]
        self.assertEqual(len(reloaded["bbox_keyframes"]), 2)
        self.assertEqual(reloaded["bbox_keyframes"][0]["entities"][0]["label"], "cat")


if __name__ == "__main__":
    unittest.main()
