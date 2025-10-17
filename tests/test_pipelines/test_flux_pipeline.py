from tests.test_pipelines._common import CalculateShiftMixin, PipelineTestCase, RetrieveLatentsMixin, RetrieveTimestepsMixin


class TestFluxPipeline(
    RetrieveTimestepsMixin,
    CalculateShiftMixin,
    RetrieveLatentsMixin,
    PipelineTestCase,
):
    module_name = "flux"
