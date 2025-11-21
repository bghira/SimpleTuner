import unittest

from simpletuner.helpers.models.ace_step.lyrics_utils.lyric_tokenizer import (
    expand_abbreviations_multilingual,
    expand_numbers_multilingual,
    expand_symbols_multilingual,
)


class TestLyricTokenizer(unittest.TestCase):
    def test_expand_numbers_basic(self):
        # Simple test to verify num2words integration
        try:
            result = expand_numbers_multilingual("I have 5 apples.", "en")
            self.assertEqual(result, "I have five apples.")
        except NotImplementedError:
            self.skipTest("num2words not functioning as expected in this environment")

    def test_abbreviations_basic(self):
        result = expand_abbreviations_multilingual("Hello Mr. Smith.", "en")
        self.assertEqual(result, "Hello mister Smith.")

    def test_symbols_basic(self):
        result = expand_symbols_multilingual("100%", "en")
        self.assertEqual(result, "100 percent")


if __name__ == "__main__":
    unittest.main()
