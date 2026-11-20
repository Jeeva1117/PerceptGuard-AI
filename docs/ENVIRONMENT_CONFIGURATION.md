# Environment Configuration & Profile Management

## 1. Executive Overview

PerceptGuard AI employs a centralized, strongly typed, environment-aware configuration architecture adhering to the Twelve-Factor App methodology. This foundation decouples code from runtime environments, guarantees type safety without introducing unvetted external dependencies, and enforces strict security and operational guardrails across development, testing, and production.

---

## 2. Environment Profiles

The system classifies deployment environments into three mutually exclusive profiles via `EnvironmentType` (`src/config/environment.py`):

| Profile | Key Purpose | `DEBUG` Mode | CORS Wildcard (`*`) | Host Wildcard (`*`) | Secret Key Requirements |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`development`** | Local engineering and exploration | `True` (default) | Allowed | Allowed | Insecure dev key allowed |
| **`testing`** | Automated pytest execution and CI/CD | `False` | Allowed | Allowed | Fast mock/dev key allowed |
| **`production`** | Live operational deployment | **Strictly `False`** | **Strictly Forbidden** | **Strictly Forbidden** | Cryptographically secure ($\ge 16$ chars) |

Aliases are supported case-insensitively (`dev`, `local`, `test`, `pytest`, `prod`, `live`). Unrecognized environment names (such as `staging` or `sandbox`) fail immediately at initialization with a descriptive `ConfigurationError`.

---

## 3. Configuration Hierarchy & Precedence

Settings are resolved in strict descending order of precedence:

```
[1. Explicit Runtime Overrides (e.g. reload_settings(env_override={...}))]
                             │
                             ▼
     [2. Operating System Environment Variables (os.environ)]
                             │
                             ▼
     [3. Local Environment File (.env via python-dotenv)]
                             │
                             ▼
           [4. Strongly Typed Dataclass Defaults]
```

---

## 4. Architecture & Component Subsystems

All configuration parameters reside under the unified `SystemSettings` class (`src/config/system_settings.py`), structured into focused dataclasses:

### 4.1. `AppConfig`
- `app_name: str` — Application identifier (`PerceptGuard AI`).
- `version: str` — Semantic application version (`1.0.0-preview`).
- `environment: EnvironmentType` — Active runtime environment profile.
- `debug: bool` — Diagnostic debug flag (must be `False` in production).
- `log_level: str` — Standardized logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

### 4.2. `APIConfig`
- `host: str` — Binding interface address (default `0.0.0.0`).
- `port: int` — Valid TCP port between 1 and 65535 (default `8000`).
- `prefix: str` — Base API route prefix (`/api/v1`).
- `reload: bool` — Hot reload trigger (for development).
- `workers: int` — Uvicorn process worker count (default `1` for CPU tier).
- `cors_origins: List[str]` — Explicitly allowed origin URLs (must include `http://` or `https://`).
- `timeout_seconds: float` — HTTP request execution deadline (default `30.0s`).

### 4.3. `SecurityConfig`
- `secret_key: SecretStr` — Masked cryptographic token container.
- `allowed_hosts: List[str]` — HTTP Host header whitelist.
- `rate_limit_per_minute: int` — Inbound request throttle threshold (default `120`).
- `enable_csrf: bool` — CSRF mitigation toggle.

### 4.4. `LoggingConfig`
- `log_level: str` — Root logger severity threshold.
- `json_format: bool` — Structured JSON logging switch.
- `log_file: Optional[Path]` — Target log file path.
- `log_dir: Path` — Directory storing operational logs (`BASE_DIR / logs`).

### 4.5. `ModelConfig`
- `device: str` — Hardware acceleration target (`cpu` or `cuda`).
- `yolo_model: str` — Detection model weights identifier (`yolov8n.pt` / `yolo11n.pt`).
- `lstm_model_path: Path` — Checkpoint path for multi-horizon LSTM reliability forecaster.
- `baseline_model_path: Path` — Checkpoint path for moving-average baseline forecaster.
- `confidence_threshold: float` — Object detection confidence threshold ($\in (0.0, 1.0]$).
- `iou_threshold: float` — Object detection NMS IoU threshold ($\in (0.0, 1.0]$).

### 4.6. `VideoProcessingConfig`
- `tracker_type: str` — Multi-object tracker identifier (`bytetrack`).
- `historical_sequence_length: int` — Feature sliding window depth (default `10` frames).
- `forecast_horizons: List[int]` — Prediction horizon offsets in frames (default `[1, 3, 5]`).
- `reliability_iou_correct_threshold: float` — Ground-truth IoU threshold for correct detection (`0.50`).
- `reliability_iou_high_threshold: float` — Ground-truth IoU threshold for high-quality detection (`0.70`).

### 4.7. `DashboardConfig`
- `host: str` — Streamlit web server host interface (`localhost`).
- `port: int` — Streamlit TCP listening port (`8501`).
- `api_base_url: str` — Backend service URL (`http://localhost:8000`).
- `auto_refresh_seconds: int` — Telemetry auto-refresh interval (`5s`).
- `telemetry_freshness_ttl_seconds: int` — Maximum latency before telemetry is marked stale (`30s`).
- `calibration_status_label: str` — Gate disclosure banner (`PROVISIONAL CALIBRATION`).

---

## 5. Backward Compatibility Guarantee

To preserve seamless compatibility with legacy scripts, model factories, and CLI entry points, `config/settings.py` and `config/__init__.py` act as zero-overhead adapter layers wrapping the new singleton `get_settings()`:

```python
# Legacy callers continue to import directly:
from config.settings import (
    BASE_DIR,
    DEVICE,
    DEFAULT_YOLO_MODEL,
    SEQUENCE_LENGTH,
    FORECAST_HORIZONS,
)
```

No existing caller needs modification, and 100% test suite compatibility is preserved.
