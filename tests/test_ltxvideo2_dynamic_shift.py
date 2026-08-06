import ast
import inspect
import textwrap
import unittest

from simpletuner.helpers.models.ltxvideo2 import pipeline_ltx2, pipeline_ltx2_image2video


def _ltx2_pipeline_call_ast(pipeline_cls):
    source = inspect.getsource(pipeline_cls.__call__)
    return ast.parse(textwrap.dedent(source))


def _calculate_shift_calls(tree):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "calculate_shift"
    ]


class LTXVideo2DynamicShiftTests(unittest.TestCase):
    def assert_uses_actual_video_sequence_length_for_shift(self, pipeline_cls):
        tree = _ltx2_pipeline_call_ast(pipeline_cls)
        calls = _calculate_shift_calls(tree)

        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[0]), "video_sequence_length")

    def test_text_to_video_dynamic_shift_uses_actual_sequence_length(self):
        self.assert_uses_actual_video_sequence_length_for_shift(pipeline_ltx2.LTX2Pipeline)

    def test_image_to_video_dynamic_shift_uses_actual_sequence_length(self):
        self.assert_uses_actual_video_sequence_length_for_shift(pipeline_ltx2_image2video.LTX2ImageToVideoPipeline)


if __name__ == "__main__":
    unittest.main()
