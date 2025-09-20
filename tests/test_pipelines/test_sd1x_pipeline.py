from tests.test_pipelines._common import (
    PipelineTestCase,
    PreprocessMixin,
    RescaleNoiseCfgMixin,
    RetrieveLatentsMixin,
    RetrieveTimestepsMixin,
)


class TestSD1xPipeline(
    RetrieveTimestepsMixin,
    RetrieveLatentsMixin,
    RescaleNoiseCfgMixin,
    PreprocessMixin,
    PipelineTestCase,
):
    module_name = "sd1x"
