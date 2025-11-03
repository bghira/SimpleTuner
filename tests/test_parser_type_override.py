import unittest

from simpletuner.helpers.configuration import cmd_args


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
