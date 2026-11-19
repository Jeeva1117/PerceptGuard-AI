"""PerceptGuard AI Forecasting Package.

Predictive perception reliability forecasting module using baseline models
(Persistence, Historical Mean, Linear Trend) and future deep learning
temporal sequence models (LSTM/GRU).
"""

from src.forecasting.baseline_data_loader import (
    BaselineDataLoader,
    BaselineDataset,
    DataLoaderError,
    SequenceDatasetRecord,
)
from src.forecasting.baseline_experiment import BaselineExperiment
from src.forecasting.baseline_forecaster import (
    BaselineForecaster,
    BaselineForecastingError,
    PredictionRecord,
)
from src.forecasting.baseline_metrics import (
    MetricCalculationError,
    calculate_mae,
    calculate_metrics_summary,
    calculate_rmse,
    evaluate_predictions_by_horizon,
)
from src.forecasting.forecasting_config import (
    ArtifactsConfig,
    DataConfig,
    EvaluationConfig,
    ForecastingConfig,
    ForecastingConfigError,
    ModelConfig,
    ProjectConfig,
    RuntimeConfig,
    TrainingConfig,
)
from src.forecasting.forecasting_data_contract import (
    DataContractError,
    DataContractValidationResult,
    ForecastingDataContract,
    ModelInputSpec,
    ModelOutputSpec,
)
from src.forecasting.forecasting_environment import (
    EnvironmentVerificationResult,
    ForecastingEnvironmentVerifier,
)

__all__ = [
    # Configuration
    "ForecastingConfig",
    "ProjectConfig",
    "RuntimeConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ArtifactsConfig",
    "ForecastingConfigError",
    # Environment
    "ForecastingEnvironmentVerifier",
    "EnvironmentVerificationResult",
    # Data Contract
    "ForecastingDataContract",
    "DataContractValidationResult",
    "ModelInputSpec",
    "ModelOutputSpec",
    "DataContractError",
    # Baseline Forecaster
    "BaselineForecaster",
    "PredictionRecord",
    "BaselineForecastingError",
    # Baseline Data Loader
    "BaselineDataLoader",
    "BaselineDataset",
    "SequenceDatasetRecord",
    "DataLoaderError",
    # Baseline Experiment
    "BaselineExperiment",
    # Baseline Metrics
    "calculate_mae",
    "calculate_rmse",
    "calculate_metrics_summary",
    "evaluate_predictions_by_horizon",
    "MetricCalculationError",
]
