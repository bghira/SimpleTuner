from tests.test_pipelines._common import PipelineTestCase, WanPromptCleaningMixin


class TestWanPipeline(WanPromptCleaningMixin, PipelineTestCase):
    module_name = "wan"
