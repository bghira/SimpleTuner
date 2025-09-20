from tests.test_pipelines._common import (
    PipelineTestCase,
    CalculateShiftMixin,
    OptimizedScaleMixin,
    RetrieveLatentsMixin,
    RetrieveTimestepsMixin,
)


class TestSD3Pipeline(
    RetrieveTimestepsMixin,
    CalculateShiftMixin,
    RetrieveLatentsMixin,
    OptimizedScaleMixin,
    PipelineTestCase,
):
    module_name = "sd3"
