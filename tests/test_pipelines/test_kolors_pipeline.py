from tests.test_pipelines._common import PipelineTestCase, RetrieveLatentsMixin, RetrieveTimestepsMixin


class TestKolorsPipeline(RetrieveTimestepsMixin, RetrieveLatentsMixin, PipelineTestCase):
    module_name = "kolors"
