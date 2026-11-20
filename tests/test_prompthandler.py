import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.prompts import PromptHandler


class TestPromptHandler(unittest.TestCase):
    def setUp(self):
        self.args = MagicMock()
        self.text_encoders = [MagicMock(), MagicMock()]
        self.tokenizers = [MagicMock(), MagicMock()]
        self.accelerator = MagicMock()
        self.model_type = "sdxl"
        self.data_backend = MagicMock()

    @patch("simpletuner.helpers.training.state_tracker.StateTracker.get_dataset")
    def test_prepare_instance_prompt_from_parquet(self, mock_get_dataset):
        # Setup
        image_path = "image_3.jpg"
        use_captions = True
        prepend_instance_prompt = True
        data_backend = MagicMock()
        instance_prompt = "Instance Prompt"
        sampler_backend_id = "sampler1"
        mock_metadata_backend = MagicMock()
        mock_metadata_backend.parquet_config = {"identifier_includes_extension": False}
        mock_metadata_backend.caption_cache_entry = MagicMock()
        mock_metadata_backend.caption_cache_entry.return_value = "a giant arcade game type claw..."
        mock_get_dataset.return_value = SimpleNamespace(metadata_backend=mock_metadata_backend)

        # Execute
        result_caption = PromptHandler.prepare_instance_prompt_from_parquet(
            image_path=image_path,
            use_captions=use_captions,
            prepend_instance_prompt=prepend_instance_prompt,
            data_backend=data_backend,
            instance_prompt=instance_prompt,
            sampler_backend_id=sampler_backend_id,
        )

        # Verify
        expected_caption = f"{instance_prompt} a giant arcade game type claw..."
        self.assertEqual(result_caption, expected_caption)
        mock_metadata_backend.caption_cache_entry.assert_called_once_with("image_3")

    def test_raises_value_error_on_missing_sampler_backend_id(self):
        with self.assertRaises(ValueError):
            PromptHandler.prepare_instance_prompt_from_parquet(
                image_path="path/to/image.jpg",
                use_captions=True,
                prepend_instance_prompt=True,
                data_backend=MagicMock(),
                instance_prompt="Instance Prompt",
                sampler_backend_id=None,  # This should cause a ValueError
            )

    @patch("builtins.open")
    @patch("simpletuner.helpers.prompts.BaseDataBackend")
    def test_instance_prompt_prepended_textfile(self, mock_backend, open_mock):
        # Setup
        open_mock.return_value.__enter__.return_value.read.return_value = "Caption from filename"
        instance_prompt = "Test Instance Prompt"
        caption_from_file = "Caption from file"
        mock_backend.exists.return_value = True
        mock_backend.read.return_value = caption_from_file

        # Instantiate PromptHandler with mocked backend and check the result
        handler = PromptHandler(
            args=self.args,
            text_encoders=["LameTest"],
            tokenizers=["LameTest"],
            accelerator=MagicMock(),
            model_type="sdxl",
        )
        result_caption = handler.magic_prompt(
            "path/to/image.png",
            use_captions=True,
            caption_strategy="textfile",
            prepend_instance_prompt=True,
            data_backend=mock_backend,
            instance_prompt=instance_prompt,
        )

        # Verify
        expected_caption = f"{instance_prompt} {caption_from_file}"
        self.assertEqual(result_caption, expected_caption)

    def test_instance_prompt_prepended_filename(self):
        # Setup
        instance_prompt = "Test Instance Prompt"
        image_filename = "image"

        # Execute
        handler = PromptHandler(
            args=self.args,
            text_encoders=["LameTest"],
            tokenizers=["LameTest"],
            accelerator=MagicMock(),
            model_type="sdxl",
        )
        result_caption = handler.magic_prompt(
            f"path/to/{image_filename}.png",
            use_captions=True,
            caption_strategy="filename",
            prepend_instance_prompt=True,
            instance_prompt=instance_prompt,
            data_backend=MagicMock(),
        )

        # Verify
        expected_caption = f"{instance_prompt} {image_filename}"
        self.assertEqual(result_caption, expected_caption)

    def test_filename_caption_matches_precache_normalisation(self):
        handler = PromptHandler(
            args=self.args,
            text_encoders=["LameTest"],
            tokenizers=["LameTest"],
            accelerator=MagicMock(),
            model_type="sdxl",
        )
        result_caption = handler.magic_prompt(
            "path/to/_a_photo_of_a_cat_.png",
            use_captions=True,
            caption_strategy="filename",
            prepend_instance_prompt=False,
            data_backend=MagicMock(),
        )

        self.assertEqual(result_caption, "a photo of a cat")
        self.assertEqual([result_caption], PromptHandler._normalize_caption_payload(result_caption))

    @patch("simpletuner.helpers.prompts.StateTracker.get_data_backend_config", return_value={})
    def test_textfile_training_captions_match_precache(self, *_):
        cases = (
            (" woven into a rug, the symbol 囍 appears. ", False, False, "woven into a rug, the symbol 囍 appears."),
            (b"  caption \r\n", False, False, ["caption"]),
            (" first caption \n \n second caption \n", False, False, ["first caption", "second caption"]),
            (" first line \n second line \n", True, False, "first line \n second line"),
            (" caption ", False, True, "style  caption"),
        )
        for content, disable_split, prepend, expected in cases:
            with self.subTest(content=content, disable_split=disable_split, prepend=prepend):
                self.data_backend.exists.return_value = True
                self.data_backend.read.return_value = content
                kwargs = dict(
                    use_captions=True,
                    prepend_instance_prompt=prepend,
                    data_backend=self.data_backend,
                    caption_strategy="textfile",
                    instance_prompt="style",
                    disable_multiline_split=disable_split,
                )
                training = PromptHandler.magic_prompt(image_path="image.png", **kwargs)
                config = {k: v for k, v in kwargs.items() if k != "data_backend"}
                dataset = DatasetRuntime("text", self.data_backend, None, config)
                precache = [caption for caption, _ in dataset.iter_caption_records(["image.png"])]
                self.assertEqual(training, expected)
                self.assertEqual([training] if isinstance(training, str) else training, precache)

    @patch("simpletuner.helpers.prompts.StateTracker.get_dataset")
    def test_webshart_caption_strategy_uses_metadata_caption_cache(self, mock_get_dataset):
        metadata_backend = MagicMock()
        metadata_backend.caption_cache_entry.return_value = [" caption one ", "caption two"]
        mock_get_dataset.return_value = SimpleNamespace(metadata_backend=metadata_backend)

        handler = PromptHandler(
            args=self.args,
            text_encoders=["LameTest"],
            tokenizers=["LameTest"],
            accelerator=MagicMock(),
            model_type="sdxl",
        )

        result_caption = handler.magic_prompt(
            "webshart://0/1/sample.webp",
            use_captions=True,
            caption_strategy="webshart",
            prepend_instance_prompt=True,
            data_backend=self.data_backend,
            instance_prompt="style",
            sampler_backend_id="webshart_backend",
        )

        self.assertEqual(result_caption, ["style caption one", "style caption two"])
        metadata_backend.caption_cache_entry.assert_called_once_with("webshart://0/1/sample.webp")
        mock_get_dataset.assert_called_once_with("webshart_backend")

    @patch("simpletuner.helpers.prompts.PromptHandler.prepare_instance_prompt_from_filename")
    def test_prepare_instance_prompt_from_filename_called(self, mock_prepare):
        """Ensure that prepare_instance_prompt_from_filename is called with correct arguments."""
        # Setup
        image_path = "path/to/image.png"
        use_captions = True
        prepend_instance_prompt = True
        instance_prompt = "test prompt"

        # Execute
        prompt_handler = PromptHandler(
            self.args,
            self.text_encoders,
            self.tokenizers,
            self.accelerator,
            self.model_type,
        )
        result = prompt_handler.magic_prompt(
            image_path=image_path,
            use_captions=use_captions,
            caption_strategy="filename",
            prepend_instance_prompt=prepend_instance_prompt,
            data_backend=self.data_backend,
            instance_prompt=instance_prompt,
        )

        # Verify
        mock_prepare.assert_called_once_with(
            image_path=image_path,
            use_captions=use_captions,
            prepend_instance_prompt=prepend_instance_prompt,
            instance_prompt=instance_prompt,
        )

    @patch("simpletuner.helpers.prompts.PromptHandler.prepare_instance_prompt_from_textfile")
    def test_prepare_instance_prompt_from_textfile_called(self, mock_prepare):
        """Ensure that prepare_instance_prompt_from_textfile is called when the caption_strategy is 'textfile'."""
        # Setup
        image_path = "path/to/image.png"
        use_captions = True
        prepend_instance_prompt = False
        instance_prompt = None
        caption_strategy = "textfile"

        # Execute
        prompt_handler = PromptHandler(
            self.args,
            self.text_encoders,
            self.tokenizers,
            self.accelerator,
            self.model_type,
        )
        result = prompt_handler.magic_prompt(
            image_path=image_path,
            use_captions=use_captions,
            caption_strategy=caption_strategy,
            prepend_instance_prompt=prepend_instance_prompt,
            instance_prompt=instance_prompt,
            data_backend=self.data_backend,
        )

        # Verify
        mock_prepare.assert_called_once_with(
            image_path,
            use_captions=use_captions,
            prepend_instance_prompt=prepend_instance_prompt,
            instance_prompt=instance_prompt,
            data_backend=self.data_backend,
            disable_multiline_split=False,
        )

    def test_magic_prompt_instanceprompt_trims_whitespace(self):
        """Ensure instanceprompt captions are normalized the same way as pre-compute."""
        handler = PromptHandler(
            self.args,
            self.text_encoders,
            self.tokenizers,
            self.accelerator,
            self.model_type,
        )

        result = handler.magic_prompt(
            image_path="path/to/image.png",
            use_captions=True,
            caption_strategy="instanceprompt",
            prepend_instance_prompt=False,
            data_backend=self.data_backend,
            instance_prompt="my-style, ",
        )

        self.assertEqual(result, "my-style,")

    def test_magic_prompt_instanceprompt_cleans_list_prompts(self):
        """Ensure instanceprompt lists are stripped and empties removed."""
        handler = PromptHandler(
            self.args,
            self.text_encoders,
            self.tokenizers,
            self.accelerator,
            self.model_type,
        )

        result = handler.magic_prompt(
            image_path="path/to/image.png",
            use_captions=True,
            caption_strategy="instanceprompt",
            prepend_instance_prompt=False,
            data_backend=self.data_backend,
            instance_prompt=["  keep-me  ", "  "],
        )

        self.assertEqual(result, ["keep-me"])

    def test_magic_prompt_raises_error_with_invalid_strategy(self):
        """Ensure magic_prompt raises ValueError with an unsupported caption strategy."""
        # Setup
        image_path = "path/to/image.png"
        use_captions = True
        prepend_instance_prompt = True
        caption_strategy = "invalid_strategy"
        prompt_handler = PromptHandler(
            self.args,
            self.text_encoders,
            self.tokenizers,
            self.accelerator,
            self.model_type,
        )

        # Verify
        with self.assertRaises(ValueError):
            prompt_handler.magic_prompt(
                image_path,
                use_captions,
                caption_strategy,
                prepend_instance_prompt,
                self.data_backend,
            )

    @patch("simpletuner.helpers.prompts.PromptHandler.filter_captions")
    def test_filter_captions_called(self, mock_filter):
        """Ensure that filter_captions is called with the correct arguments."""
        captions = ["caption 1", "caption 2"]
        prompt_handler = PromptHandler(
            self.args,
            self.text_encoders,
            self.tokenizers,
            self.accelerator,
            self.model_type,
        )
        prompt_handler.filter_captions(self.data_backend, captions)

        # Verify
        mock_filter.assert_called_once_with(self.data_backend, captions)


if __name__ == "__main__":
    unittest.main()
