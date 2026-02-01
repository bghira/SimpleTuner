"""Test data signals infrastructure and LoRA target extension."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.getcwd())

from simpletuner.helpers.models.common import ModelFoundation
from simpletuner.helpers.models.ltxvideo2.model import LTXVideo2


class MockModelFoundation(ModelFoundation):
    """Mock subclass for testing base class behavior."""

    NAME = "MockModel"
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    DEFAULT_LYCORIS_TARGET = ["Attention"]
    DEFAULT_CONTROLNET_LORA_TARGET = ["to_k", "to_q"]

    def __init__(self, config, accelerator):
        self.config = config
        self.accelerator = accelerator
        self._data_has_images = False
        self._data_has_video = False
        self._data_has_audio = False

    # Abstract method stubs
    def _encode_prompts(self, *args, **kwargs):
        pass

    def convert_negative_text_embed_for_pipeline(self, *args, **kwargs):
        pass

    def convert_text_embed_for_pipeline(self, *args, **kwargs):
        pass

    def model_predict(self, *args, **kwargs):
        pass


class MockLTXVideo2(LTXVideo2):
    """Mock LTXVideo2 that skips heavy initialization."""

    def __init__(self, config, accelerator):
        self.config = config
        self.accelerator = accelerator
        self._data_has_images = False
        self._data_has_video = False
        self._data_has_audio = False
        # Copy class attributes needed for LoRA target tests
        self.DEFAULT_LORA_TARGET = LTXVideo2.DEFAULT_LORA_TARGET
        self.DEFAULT_LYCORIS_TARGET = LTXVideo2.DEFAULT_LYCORIS_TARGET
        self.AUDIO_LORA_TARGETS = LTXVideo2.AUDIO_LORA_TARGETS


class TestDataSignalsBase(unittest.TestCase):
    """Test data signal infrastructure in base ModelFoundation class."""

    def setUp(self):
        self.config = MagicMock()
        self.config.lora_type = "standard"
        self.config.slider_lora_target = False
        self.config.controlnet = False
        self.config.peft_lora_target_modules = None
        self.accelerator = MagicMock()

    def test_default_data_signals_are_false(self):
        """Data signals should default to False."""
        model = MockModelFoundation(self.config, self.accelerator)
        self.assertFalse(model._data_has_images)
        self.assertFalse(model._data_has_video)
        self.assertFalse(model._data_has_audio)

    def test_configure_data_signals_sets_flags(self):
        """configure_data_signals should set the appropriate flags."""
        model = MockModelFoundation(self.config, self.accelerator)

        model.configure_data_signals(has_images=True, has_video=False, has_audio=True)

        self.assertTrue(model._data_has_images)
        self.assertFalse(model._data_has_video)
        self.assertTrue(model._data_has_audio)

    def test_configure_data_signals_all_true(self):
        """configure_data_signals should handle all flags being True."""
        model = MockModelFoundation(self.config, self.accelerator)

        model.configure_data_signals(has_images=True, has_video=True, has_audio=True)

        self.assertTrue(model._data_has_images)
        self.assertTrue(model._data_has_video)
        self.assertTrue(model._data_has_audio)

    def test_base_get_additional_lora_targets_returns_empty(self):
        """Base implementation should return empty list."""
        model = MockModelFoundation(self.config, self.accelerator)
        model.configure_data_signals(has_audio=True)

        additional = model._get_additional_lora_targets()

        self.assertEqual(additional, [])

    def test_get_lora_target_layers_without_additional(self):
        """Without additional targets, should return base targets."""
        model = MockModelFoundation(self.config, self.accelerator)

        targets = model.get_lora_target_layers()

        self.assertEqual(targets, ["to_k", "to_q", "to_v", "to_out.0"])

    def test_get_lora_target_layers_lycoris(self):
        """LyCORIS type should return lycoris targets."""
        self.config.lora_type = "lycoris"
        model = MockModelFoundation(self.config, self.accelerator)

        targets = model.get_lora_target_layers()

        self.assertEqual(targets, ["Attention"])

    def test_get_lora_target_layers_controlnet(self):
        """ControlNet should return controlnet targets."""
        self.config.controlnet = True
        model = MockModelFoundation(self.config, self.accelerator)

        targets = model.get_lora_target_layers()

        self.assertEqual(targets, ["to_k", "to_q"])

    def test_manual_peft_targets_override(self):
        """Manual peft_lora_target_modules should override everything."""
        self.config.peft_lora_target_modules = ["custom_module_1", "custom_module_2"]
        model = MockModelFoundation(self.config, self.accelerator)
        model.configure_data_signals(has_audio=True)

        targets = model.get_lora_target_layers()

        self.assertEqual(targets, ["custom_module_1", "custom_module_2"])


class TestModelWithAdditionalTargets(unittest.TestCase):
    """Test model that returns additional LoRA targets."""

    def setUp(self):
        self.config = MagicMock()
        self.config.lora_type = "standard"
        self.config.slider_lora_target = False
        self.config.controlnet = False
        self.config.peft_lora_target_modules = None
        self.accelerator = MagicMock()

    def test_additional_targets_combined_with_base(self):
        """Additional targets should be appended to base targets."""

        class ModelWithExtras(MockModelFoundation):
            def _get_additional_lora_targets(self):
                if self._data_has_audio:
                    return ["audio_proj_in", "audio_proj_out"]
                return []

        model = ModelWithExtras(self.config, self.accelerator)
        model.configure_data_signals(has_audio=True)

        targets = model.get_lora_target_layers()

        self.assertEqual(
            targets,
            ["to_k", "to_q", "to_v", "to_out.0", "audio_proj_in", "audio_proj_out"],
        )

    def test_no_duplicate_targets(self):
        """Should not add duplicates if additional target already in base."""

        class ModelWithDuplicates(MockModelFoundation):
            def _get_additional_lora_targets(self):
                return ["to_k", "new_module"]  # to_k is already in base

        model = ModelWithDuplicates(self.config, self.accelerator)
        model.configure_data_signals(has_audio=True)

        targets = model.get_lora_target_layers()

        # Should not have duplicate to_k
        self.assertEqual(targets.count("to_k"), 1)
        self.assertIn("new_module", targets)

    def test_additional_targets_not_added_when_flag_false(self):
        """Additional targets should not be added when data flag is False."""

        class ModelWithExtras(MockModelFoundation):
            def _get_additional_lora_targets(self):
                if self._data_has_audio:
                    return ["audio_proj_in", "audio_proj_out"]
                return []

        model = ModelWithExtras(self.config, self.accelerator)
        model.configure_data_signals(has_audio=False)

        targets = model.get_lora_target_layers()

        self.assertEqual(targets, ["to_k", "to_q", "to_v", "to_out.0"])
        self.assertNotIn("audio_proj_in", targets)


class TestLTXVideo2AudioLoraTargets(unittest.TestCase):
    """Test LTX-2 specific audio LoRA target behavior."""

    def setUp(self):
        self.config = MagicMock()
        self.config.lora_type = "standard"
        self.config.slider_lora_target = False
        self.config.controlnet = False
        self.config.peft_lora_target_modules = None
        self.accelerator = MagicMock()

    def test_ltx2_default_targets_without_audio(self):
        """LTX-2 without audio should use default targets."""
        model = MockLTXVideo2(self.config, self.accelerator)
        model.configure_data_signals(has_video=True, has_audio=False)

        targets = model.get_lora_target_layers()

        self.assertEqual(targets, ["to_k", "to_q", "to_v", "to_out.0"])

    def test_ltx2_adds_audio_targets_when_audio_present(self):
        """LTX-2 should add audio targets when audio data is present."""
        model = MockLTXVideo2(self.config, self.accelerator)
        model.configure_data_signals(has_video=True, has_audio=True)

        targets = model.get_lora_target_layers()

        # Should have base targets
        self.assertIn("to_k", targets)
        self.assertIn("to_q", targets)
        self.assertIn("to_v", targets)
        self.assertIn("to_out.0", targets)
        # Should have audio targets
        self.assertIn("audio_proj_in", targets)
        self.assertIn("audio_proj_out", targets)
        self.assertIn("audio_caption_projection.linear_1", targets)
        self.assertIn("audio_caption_projection.linear_2", targets)

    def test_ltx2_audio_only_mode(self):
        """LTX-2 with audio-only should still add audio targets."""
        model = MockLTXVideo2(self.config, self.accelerator)
        model.configure_data_signals(has_video=False, has_audio=True)

        targets = model.get_lora_target_layers()

        self.assertIn("audio_proj_in", targets)
        self.assertIn("audio_proj_out", targets)

    def test_ltx2_audio_targets_constant(self):
        """Verify LTX-2 AUDIO_LORA_TARGETS constant is correct."""
        expected = [
            "audio_proj_in",
            "audio_proj_out",
            "audio_caption_projection.linear_1",
            "audio_caption_projection.linear_2",
            "audio_ff.net.0.proj",
            "audio_ff.net.2",
        ]
        self.assertEqual(LTXVideo2.AUDIO_LORA_TARGETS, expected)

    def test_ltx2_get_additional_lora_targets_returns_copy(self):
        """_get_additional_lora_targets should return a copy, not the class constant."""
        model = MockLTXVideo2(self.config, self.accelerator)
        model.configure_data_signals(has_audio=True)

        targets1 = model._get_additional_lora_targets()
        targets2 = model._get_additional_lora_targets()

        # Should be equal but not the same object
        self.assertEqual(targets1, targets2)
        self.assertIsNot(targets1, targets2)


class TestFactoryDataSignals(unittest.TestCase):
    """Test factory's _configure_model_data_signals method."""

    def test_factory_detects_image_dataset(self):
        """Factory should detect image datasets and signal the model."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        model = MagicMock()
        args = MagicMock()
        accelerator = MagicMock()

        factory = FactoryRegistry(args, accelerator, None, None, model)

        config = [
            {"id": "images", "dataset_type": "image", "disabled": False},
        ]
        factory._configure_model_data_signals(config)

        model.configure_data_signals.assert_called_once_with(has_images=True, has_video=False, has_audio=False)

    def test_factory_detects_video_dataset(self):
        """Factory should detect video datasets and signal the model."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        model = MagicMock()
        args = MagicMock()
        accelerator = MagicMock()

        factory = FactoryRegistry(args, accelerator, None, None, model)

        config = [
            {"id": "videos", "dataset_type": "video", "disabled": False},
        ]
        factory._configure_model_data_signals(config)

        model.configure_data_signals.assert_called_once_with(has_images=False, has_video=True, has_audio=False)

    def test_factory_detects_audio_dataset(self):
        """Factory should detect audio datasets and signal the model."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        model = MagicMock()
        args = MagicMock()
        accelerator = MagicMock()

        factory = FactoryRegistry(args, accelerator, None, None, model)

        config = [
            {"id": "audio", "dataset_type": "audio", "disabled": False},
        ]
        factory._configure_model_data_signals(config)

        model.configure_data_signals.assert_called_once_with(has_images=False, has_video=False, has_audio=True)

    def test_factory_detects_mixed_datasets(self):
        """Factory should detect multiple dataset types."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        model = MagicMock()
        args = MagicMock()
        accelerator = MagicMock()

        factory = FactoryRegistry(args, accelerator, None, None, model)

        config = [
            {"id": "images", "dataset_type": "image", "disabled": False},
            {"id": "videos", "dataset_type": "video", "disabled": False},
            {"id": "audio", "dataset_type": "audio", "disabled": False},
        ]
        factory._configure_model_data_signals(config)

        model.configure_data_signals.assert_called_once_with(has_images=True, has_video=True, has_audio=True)

    def test_factory_ignores_disabled_datasets(self):
        """Factory should ignore disabled datasets."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        model = MagicMock()
        args = MagicMock()
        accelerator = MagicMock()

        factory = FactoryRegistry(args, accelerator, None, None, model)

        config = [
            {"id": "images", "dataset_type": "image", "disabled": True},
            {"id": "audio", "dataset_type": "audio", "disabled": False},
        ]
        factory._configure_model_data_signals(config)

        model.configure_data_signals.assert_called_once_with(has_images=False, has_video=False, has_audio=True)

    def test_factory_ignores_disable_flag(self):
        """Factory should also check 'disable' flag (alternative spelling)."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        model = MagicMock()
        args = MagicMock()
        accelerator = MagicMock()

        factory = FactoryRegistry(args, accelerator, None, None, model)

        config = [
            {"id": "images", "dataset_type": "image", "disable": True},
            {"id": "audio", "dataset_type": "audio", "disabled": False},
        ]
        factory._configure_model_data_signals(config)

        model.configure_data_signals.assert_called_once_with(has_images=False, has_video=False, has_audio=True)

    def test_factory_handles_no_model(self):
        """Factory should handle None model gracefully."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        args = MagicMock()
        accelerator = MagicMock()

        factory = FactoryRegistry(args, accelerator, None, None, None)

        config = [
            {"id": "images", "dataset_type": "image", "disabled": False},
        ]
        # Should not raise
        factory._configure_model_data_signals(config)

    def test_factory_ignores_non_primary_datasets(self):
        """Factory should ignore text_embeds, image_embeds, etc."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        model = MagicMock()
        args = MagicMock()
        accelerator = MagicMock()

        factory = FactoryRegistry(args, accelerator, None, None, model)

        config = [
            {"id": "text", "dataset_type": "text_embeds", "disabled": False},
            {"id": "img_embeds", "dataset_type": "image_embeds", "disabled": False},
        ]
        factory._configure_model_data_signals(config)

        model.configure_data_signals.assert_called_once_with(has_images=False, has_video=False, has_audio=False)


if __name__ == "__main__":
    unittest.main()
