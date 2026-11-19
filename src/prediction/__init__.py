"""Reliability Forecasting Model Module.

Implements deep sequence forecasting architectures (LSTM / GRU) to predict
the future detection reliability of tracked objects over upcoming video frames
(e.g., t+1, t+3, t+5 frames ahead) to anticipate failure modes before they occur.
"""

from src.forecasting.baseline_forecaster import BaselineForecaster, PredictionRecord
from src.forecasting.lstm_forecaster import LSTMForecaster
from src.forecasting.model_factory import build_forecasting_model
from src.forecasting.multi_horizon_forecaster import (
    HorizonPredictionRecord,
    MultiHorizonForecaster,
    MultiHorizonForecastingSummary,
)

__all__ = [
    "BaselineForecaster",
    "PredictionRecord",
    "LSTMForecaster",
    "build_forecasting_model",
    "MultiHorizonForecaster",
    "HorizonPredictionRecord",
    "MultiHorizonForecastingSummary",
]


