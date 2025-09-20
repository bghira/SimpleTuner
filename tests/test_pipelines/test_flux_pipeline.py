from tests.test_pipelines._common import (
    PipelineTestCase,
    CalculateShiftMixin,
    RetrieveLatentsMixin,
    RetrieveTimestepsMixin,
)


class TestFluxPipeline(
    RetrieveTimestepsMixin,
    CalculateShiftMixin,
    RetrieveLatentsMixin,
    PipelineTestCase,
):
    module_name = "flux"
