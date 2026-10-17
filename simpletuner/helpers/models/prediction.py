"""Prediction conventions shared by model metadata and training."""

from enum import Enum


class PredictionTypes(Enum):
    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"
    FLOW_MATCHING = "flow_matching"
    FLOW_MAP = "flow_map"
    AUTOREGRESSIVE_NEXT_TOKEN = "autoregressive_next_token"

    @property
    def is_flow_prediction(self) -> bool:
        return self in (PredictionTypes.FLOW_MATCHING, PredictionTypes.FLOW_MAP)

    @staticmethod
    def from_str(label):
        if label in ("eps", "epsilon"):
            return PredictionTypes.EPSILON
        elif label in ("vpred", "v_prediction", "v-prediction"):
            return PredictionTypes.V_PREDICTION
        elif label in ("sample", "x_prediction", "x-prediction"):
            return PredictionTypes.SAMPLE
        elif label in ("flow", "flow_matching", "flow-matching"):
            return PredictionTypes.FLOW_MATCHING
        elif label in ("flow_map", "flow-map", "flowmap"):
            return PredictionTypes.FLOW_MAP
        elif label in ("ar", "autoregressive", "autoregressive_next_token"):
            return PredictionTypes.AUTOREGRESSIVE_NEXT_TOKEN
        else:
            raise NotImplementedError
