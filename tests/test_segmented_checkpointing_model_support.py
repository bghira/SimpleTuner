"""Model-level segmented checkpointing capability coverage."""

import unittest


def assert_checkpointing_controls(
    test_case,
    model_cls,
    *,
    backend=False,
    interval=False,
    stride=False,
    offload=False,
    ffn=False,
    attention_offload=False,
):
    if backend:
        test_case.assertTrue(hasattr(model_cls, "set_gradient_checkpointing_backend"))
    if interval:
        test_case.assertTrue(hasattr(model_cls, "set_gradient_checkpointing_interval"))
    if stride:
        test_case.assertTrue(hasattr(model_cls, "set_gradient_checkpointing_segment_stride"))
    if offload:
        test_case.assertTrue(hasattr(model_cls, "set_gradient_checkpointing_offload_attention"))
    if ffn:
        test_case.assertTrue(getattr(model_cls, "_supports_ffn_gradient_checkpointing", False))
    if attention_offload:
        test_case.assertTrue(getattr(model_cls, "_supports_attention_activation_offload", False))


class AceStepSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.ace_step.transformer import ACEStepTransformer2DModel

        assert_checkpointing_controls(
            self,
            ACEStepTransformer2DModel,
            backend=True,
            interval=True,
            stride=True,
        )
