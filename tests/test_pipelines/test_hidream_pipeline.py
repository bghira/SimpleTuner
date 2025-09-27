from tests.test_pipelines._common import CalculateShiftMixin, PipelineTestCase, RetrieveTimestepsMixin


class TestHiDreamPipeline(RetrieveTimestepsMixin, CalculateShiftMixin, PipelineTestCase):
    module_name = "hidream"
