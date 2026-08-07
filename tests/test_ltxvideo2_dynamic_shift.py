import ast
import inspect
import textwrap
import unittest

from simpletuner.helpers.models.ltxvideo2 import pipeline_ltx2, pipeline_ltx2_image2video


def _ltx2_pipeline_call_ast(pipeline_cls):
    source = inspect.getsource(pipeline_cls.__call__)
    return ast.parse(textwrap.dedent(source))


def _calculate_shift_calls(tree):
    def is_calculate_shift(node):
        return (isinstance(node, ast.Name) and node.id == "calculate_shift") or (
            isinstance(node, ast.Attribute) and node.attr == "calculate_shift"
        )

    return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and is_calculate_shift(node.func)]


class LTXVideo2DynamicShiftTests(unittest.TestCase):
    def assert_uses_actual_video_sequence_length_for_shift(self, pipeline_cls):
        tree = _ltx2_pipeline_call_ast(pipeline_cls)
        calls = _calculate_shift_calls(tree)

        self.assertGreaterEqual(len(calls), 1)
        self.assertTrue(
            any(
                any(isinstance(arg, ast.Name) and arg.id == "video_sequence_length" for arg in call.args)
                or any(
                    kw.arg == "video_sequence_length"
                    and isinstance(kw.value, ast.Name)
                    and kw.value.id == "video_sequence_length"
                    for kw in call.keywords
                )
                for call in calls
            )
        )

    def test_text_to_video_dynamic_shift_uses_actual_sequence_length(self):
        self.assert_uses_actual_video_sequence_length_for_shift(pipeline_ltx2.LTX2Pipeline)

    def test_image_to_video_dynamic_shift_uses_actual_sequence_length(self):
        self.assert_uses_actual_video_sequence_length_for_shift(pipeline_ltx2_image2video.LTX2ImageToVideoPipeline)


if __name__ == "__main__":
    unittest.main()
