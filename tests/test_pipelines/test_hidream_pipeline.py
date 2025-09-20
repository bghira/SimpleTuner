from tests.test_pipelines._common import PipelineTestCase, CalculateShiftMixin, RetrieveTimestepsMixin


class TestHiDreamPipeline(RetrieveTimestepsMixin, CalculateShiftMixin, PipelineTestCase):
    module_name = "hidream"
