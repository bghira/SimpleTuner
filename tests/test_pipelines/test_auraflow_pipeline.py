from tests.test_pipelines._common import OptimizedScaleMixin, PipelineTestCase, RetrieveTimestepsMixin


class TestAuraflowPipeline(RetrieveTimestepsMixin, OptimizedScaleMixin, PipelineTestCase):
    module_name = "auraflow"
