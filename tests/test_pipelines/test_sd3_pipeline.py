from tests.test_pipelines._common import (
    CalculateShiftMixin,
    OptimizedScaleMixin,
    PipelineTestCase,
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
