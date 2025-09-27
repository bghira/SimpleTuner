from tests.test_pipelines._common import PipelineTestCase, RescaleNoiseCfgMixin, RetrieveLatentsMixin, RetrieveTimestepsMixin


class TestSDXLPipeline(
    RetrieveTimestepsMixin,
    RetrieveLatentsMixin,
    RescaleNoiseCfgMixin,
    PipelineTestCase,
):
    module_name = "sdxl"
