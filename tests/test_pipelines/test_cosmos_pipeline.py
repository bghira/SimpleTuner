from tests.test_pipelines._common import PipelineTestCase, RetrieveTimestepsMixin


class TestCosmosPipeline(RetrieveTimestepsMixin, PipelineTestCase):
    module_name = "cosmos"
