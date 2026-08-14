import unittest

from simpletuner.helpers.configuration import cmd_args
from simpletuner.helpers.image_manipulation.nsfw_classifier import DEFAULT_NSFW_CHECK_MODELS_CSV


class TestParserTypeOverride(unittest.TestCase):
    def setUp(self):
        cmd_args._ARG_PARSER_CACHE = None

    def test_optimizer_beta1_uses_float_type(self):
        parser = cmd_args.get_argument_parser()
        action = next(action for action in parser._actions if "--optimizer_beta1" in action.option_strings)
        self.assertIs(action.type, float)

    def test_optimizer_beta2_uses_float_type(self):
        parser = cmd_args.get_argument_parser()
        action = next(action for action in parser._actions if "--optimizer_beta2" in action.option_strings)
        self.assertIs(action.type, float)

    def test_nsfw_integer_options_use_int_type(self):
        parser = cmd_args.get_argument_parser()
        for option in (
            "--nsfw_check_min_votes",
            "--nsfw_check_video_frame_count",
            "--nsfw_check_video_min_flagged_frames",
            "--validate_after_step",
            "--validate_after_epoch",
        ):
            with self.subTest(option=option):
                action = next(action for action in parser._actions if option in action.option_strings)
                self.assertIs(action.type, int)

    def test_nsfw_models_default_is_transformers_only(self):
        parser = cmd_args.get_argument_parser()
        action = next(action for action in parser._actions if "--nsfw_check_models" in action.option_strings)
        self.assertEqual(action.default, DEFAULT_NSFW_CHECK_MODELS_CSV)
        self.assertNotIn("Marqo/", action.default)

    def test_validate_after_options_reject_negative_values(self):
        base_args = [
            "--model_family=pixart_sigma",
            "--output_dir=output",
            "--model_type=lora",
            "--optimizer=adamw_bf16",
            "--data_backend_config=config/multidatabackend.json",
        ]
        for option in ("--validate_after_step", "--validate_after_epoch"):
            with self.subTest(option=option):
                with self.assertRaises(ValueError):
                    cmd_args.parse_cmdline_args(base_args + [f"{option}=-1"], exit_on_error=True)

    def test_validation_prompt_library_accepts_named_library(self):
        parser = cmd_args.get_argument_parser()

        args = parser.parse_args(
            [
                "--model_family=flux",
                "--output_dir=output/test",
                "--model_type=lora",
                "--optimizer=adamw_bf16",
                "--data_backend_config=config/multidatabackend.json",
                "--validation_prompt_library=audio",
            ]
        )

        self.assertEqual(args.validation_prompt_library, "audio")
