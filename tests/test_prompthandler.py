import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from helpers.prompts import (
    PromptHandler,
)


class TestPromptHandler(unittest.TestCase):
    def setUp(self):
        self.args = MagicMock()
        self.text_encoders = [MagicMock(), MagicMock()]
        self.tokenizers = [MagicMock(), MagicMock()]
        self.accelerator = MagicMock()
        self.model_type = "sdxl"
        self.data_backend = MagicMock()

    @patch("helpers.training.state_tracker.StateTracker.get_parquet_database")
    @patch("helpers.training.state_tracker.StateTracker.get_data_backend")
    def test_prepare_instance_prompt_from_parquet(
        self, mock_get_data_backend, mock_get_parquet_database
    ):
        # Setup
        image_path = "image_3.jpg"
        use_captions = True
        prepend_instance_prompt = True
        data_backend = MagicMock()
        instance_prompt = "Instance Prompt"
        sampler_backend_id = "sampler1"
        filename_column = "filename"
        caption_column = "caption"
        mock_metadata_backend = MagicMock()
        mock_metadata_backend.caption_cache_entry = MagicMock()
        mock_metadata_backend.caption_cache_entry.return_value = (
            "a giant arcade game type claw..."
        )
        mock_get_data_backend.return_value = {
            "metadata_backend": mock_metadata_backend,
        }

        # Simulate the DataFrame structure and the expected row
        fallback_caption_column = "tags"
        mock_df = pd.DataFrame(
            [
                {
                    filename_column: "image_3",
                    caption_column: "a giant arcade game type claw...",
                    fallback_caption_column: "tags for image_3",
                }
            ]
        )

        # Configure the mock to return the simulated DataFrame
        mock_get_parquet_database.return_value = (
            mock_df,
            filename_column,
            caption_column,
            fallback_caption_column,
            False,
        )

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
        mock_get_parquet_database.assert_called_once_with(sampler_backend_id)

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
    @patch("helpers.prompts.BaseDataBackend")
    def test_instance_prompt_prepended_textfile(self, mock_backend, open_mock):
        # Setup
        open_mock.return_value.__enter__.return_value.read.return_value = (
            "Caption from filename"
        )
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

    @patch("helpers.prompts.PromptHandler.prepare_instance_prompt_from_filename")
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

    @patch("helpers.prompts.PromptHandler.prepare_instance_prompt_from_textfile")
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
        )

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

    @patch("helpers.prompts.PromptHandler.filter_captions")
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
