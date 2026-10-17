import unittest

from simpletuner.helpers.models.prediction import PredictionTypes
from simpletuner.helpers.models.registry import LazyModelClass


class PredictionTypesTests(unittest.TestCase):
    def test_flow_prediction_enum_distinguishes_two_time_models(self):
        for alias in ("flow_map", "flow-map", "flowmap"):
            self.assertIs(PredictionTypes.from_str(alias), PredictionTypes.FLOW_MAP)
        self.assertTrue(PredictionTypes.FLOW_MAP.is_flow_prediction)
        self.assertTrue(PredictionTypes.FLOW_MATCHING.is_flow_prediction)
        self.assertFalse(PredictionTypes.EPSILON.is_flow_prediction)

    def test_lazy_metadata_returns_the_enum_without_importing_the_model(self):
        lazy = LazyModelClass({"prediction_type": "flow_map"})
        self.assertIs(lazy.PREDICTION_TYPE, PredictionTypes.FLOW_MAP)
        self.assertIsNone(lazy._real_class)

    def test_common_module_preserves_the_existing_enum_import(self):
        from simpletuner.helpers.models.common import PredictionTypes as CommonPredictionTypes

        self.assertIs(CommonPredictionTypes, PredictionTypes)
