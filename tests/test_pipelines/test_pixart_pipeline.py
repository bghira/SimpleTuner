from tests.test_pipelines._common import PipelineTestCase, RetrieveLatentsMixin, RetrieveTimestepsMixin


class TestPixArtPipeline(RetrieveTimestepsMixin, RetrieveLatentsMixin, PipelineTestCase):
    module_name = "pixart"
