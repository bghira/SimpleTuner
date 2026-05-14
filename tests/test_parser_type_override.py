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
        ):
            with self.subTest(option=option):
                action = next(action for action in parser._actions if option in action.option_strings)
                self.assertIs(action.type, int)

    def test_nsfw_models_default_is_transformers_only(self):
        parser = cmd_args.get_argument_parser()
        action = next(action for action in parser._actions if "--nsfw_check_models" in action.option_strings)
        self.assertEqual(action.default, DEFAULT_NSFW_CHECK_MODELS_CSV)
        self.assertNotIn("Marqo/", action.default)
