"""PerceptGuard AI — Predictive Early Warning & Perception Risk Assessment System.

Day 10 introduces the predictive early warning and risk assessment pipeline,
transforming Day 9 multi-horizon reliability forecasts into actionable real-time alerts.
"""

from src.early_warning.early_warning_config import (
    AlertGenerationConfig,
    DataIngestionConfig,
    EarlyWarningConfig,
    EarlyWarningConfigError,
    ForecastHorizonsConfig,
    LoggingConfig,
    ProjectConfig,
    RiskClassificationConfig,
    RuntimeConfig,
    TemporalMonitoringConfig,
    load_early_warning_config,
)
from src.early_warning.early_warning_data_contract import (
    EarlyWarningDataContract,
    EarlyWarningDataContractError,
    EarlyWarningDataContractValidationSummary,
    EarlyWarningInputRecord,
)
from src.early_warning.interfaces import (
    EarlyWarningAlert,
    IEarlyWarningDashboard,
    IEarlyWarningGenerator,
    IRiskAssessmentEngine,
    IRiskClassifier,
    ITemporalRiskMonitor,
    RiskLevel,
    RiskScore,
)

__all__ = [
    # Configuration
    "EarlyWarningConfig",
    "EarlyWarningConfigError",
    "load_early_warning_config",
    "ProjectConfig",
    "RuntimeConfig",
    "ForecastHorizonsConfig",
    "RiskClassificationConfig",
    "TemporalMonitoringConfig",
    "AlertGenerationConfig",
    "DataIngestionConfig",
    "LoggingConfig",
    # Data Contract
    "EarlyWarningInputRecord",
    "EarlyWarningDataContract",
    "EarlyWarningDataContractError",
    "EarlyWarningDataContractValidationSummary",
    # Interfaces & Domain Types
    "RiskLevel",
    "RiskScore",
    "EarlyWarningAlert",
    "IRiskAssessmentEngine",
    "IRiskClassifier",
    "ITemporalRiskMonitor",
    "IEarlyWarningGenerator",
    "IEarlyWarningDashboard",
]
