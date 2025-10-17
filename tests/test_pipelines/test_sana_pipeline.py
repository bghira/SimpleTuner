from tests.test_pipelines._common import ImportOnlyMixin, PipelineTestCase


class TestSanaPipeline(ImportOnlyMixin, PipelineTestCase):
    module_name = "sana"
