import unittest

from simpletuner.helpers.configuration.cmd_args import _normalize_sana_complex_instruction


class TestSanaComplexInstruction(unittest.TestCase):
    def test_multiline_string_is_normalized(self):
        raw_value = "Line one\nLine two\n\nLine three  "
        result = _normalize_sana_complex_instruction(raw_value)
        self.assertEqual(result, ["Line one", "Line two", "Line three"])

    def test_json_encoded_list_is_loaded(self):
        raw_value = '["first", "second"]'
        result = _normalize_sana_complex_instruction(raw_value)
        self.assertEqual(result, ["first", "second"])

    def test_none_like_values_return_none(self):
        self.assertIsNone(_normalize_sana_complex_instruction(None))
        self.assertIsNone(_normalize_sana_complex_instruction(""))
        self.assertIsNone(_normalize_sana_complex_instruction("None"))
