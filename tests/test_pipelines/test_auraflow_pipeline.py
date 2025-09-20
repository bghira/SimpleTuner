from tests.test_pipelines._common import (
    PipelineTestCase,
    RetrieveTimestepsMixin,
    OptimizedScaleMixin,
)


class TestAuraflowPipeline(RetrieveTimestepsMixin, OptimizedScaleMixin, PipelineTestCase):
    module_name = "auraflow"
