import unittest
from unittest.mock import patch, MagicMock
from helpers.prompts import (
    PromptHandler,
)


class TestPromptHandler(unittest.TestCase):
    def setUp(self):
        self.args = MagicMock()
        self.args.disable_compel = False
        self.text_encoders = [MagicMock(), MagicMock()]
        self.tokenizers = [MagicMock(), MagicMock()]
        self.accelerator = MagicMock()
        self.model_type = "sdxl"
        self.data_backend = MagicMock()

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
