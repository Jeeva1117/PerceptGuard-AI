# Production Deployment & Configuration Guide

## 1. Production Overview

Deploying PerceptGuard AI to a production environment requires activating the `production` environment profile. This activates mandatory security guardrails, disables debug interfaces, enforces strict CORS policies, and validates credential strength at startup.

---

## 2. Mandatory Production Guardrails

If any of the following conditions are violated in production mode, the application will terminate immediately upon startup with a `ConfigurationError`:

| Parameter | Mandatory Production Rule | Failure Consequence |
| :--- | :--- | :--- |
| `APP_ENV` | Must be set to `production` (or alias `prod`, `live`). | Falls back to `development` if omitted. |
| `DEBUG` | **Must be `false`**. | Application halts with `ConfigurationError`. |
| `SECRET_KEY` | Must be non-default and $\ge 16$ characters in length. | Application halts with `ConfigurationError`. |
| `CORS_ORIGINS` | Cannot be empty or contain wildcard `*`. Must contain valid `http://` or `https://` URLs. | Application halts with `ConfigurationError`. |
| `ALLOWED_HOSTS` | Cannot contain wildcard `*`. Must be explicit domain/IP list. | Application halts with `ConfigurationError`. |
| `API_PORT` | Must be a valid TCP port ($1 \le \text{port} \le 65535$). | Application halts with `ConfigurationError`. |
| `STREAMLIT_PORT`| Must be a valid TCP port ($1 \le \text{port} \le 65535$). | Application halts with `ConfigurationError`. |
| `API_BASE_URL` | Must be a well-formed URL with scheme and host. | Application halts with `ConfigurationError`. |

---

## 3. Minimal Production Environment File (`.env.production`)

```ini
# Core Profile
APP_ENV=production
APP_NAME="PerceptGuard AI Production"
APP_VERSION=1.0.0-preview
DEBUG=false

# API Service
API_HOST=127.0.0.1
API_PORT=8000
API_PREFIX=/api/v1
API_WORKERS=2
API_TIMEOUT_SECONDS=30.0

# CORS & Allowed Hosts (Explicit domains only)
CORS_ORIGINS=https://perceptguard.internal.domain,https://dashboard.perceptguard.internal.domain
ALLOWED_HOSTS=api.perceptguard.internal.domain,127.0.0.1

# Cryptographic Security
SECRET_KEY=9f82d41a7c3e5b6081fa2c4d9e0b12738495a6b7c8d9e0f1a2b3c4d5e6f7a8b9
RATE_LIMIT_PER_MINUTE=240
ENABLE_CSRF=true

# Logging & Monitoring
LOG_LEVEL=INFO
LOG_JSON=true

# Model & Hardware
DEVICE=cpu
YOLO_MODEL=yolov8n.pt
CONFIDENCE_THRESHOLD=0.35
IOU_THRESHOLD=0.45

# Video Processing & Horizons
TRACKER_TYPE=bytetrack
HISTORICAL_SEQUENCE_LENGTH=10
FORECAST_HORIZONS=1,3,5

# Frontend Dashboard
STREAMLIT_HOST=127.0.0.1
STREAMLIT_PORT=8501
API_BASE_URL=https://api.perceptguard.internal.domain
TELEMETRY_FRESHNESS_TTL_SECONDS=30
CALIBRATION_STATUS_LABEL="PROVISIONAL CALIBRATION"
```

---

## 4. Reverse Proxy Architecture (Nginx)

In production, both FastAPI and Streamlit should be bound to `127.0.0.1` and served behind a hardened reverse proxy (e.g., Nginx, Traefik, or AWS ALB) with SSL termination:

```nginx
# PerceptGuard AI Backend API
server {
    listen 443 ssl http2;
    server_name api.perceptguard.internal.domain;

    ssl_certificate /etc/ssl/certs/perceptguard.crt;
    ssl_certificate_key /etc/ssl/private/perceptguard.key;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# PerceptGuard AI Streamlit Dashboard
server {
    listen 443 ssl http2;
    server_name dashboard.perceptguard.internal.domain;

    ssl_certificate /etc/ssl/certs/perceptguard.crt;
    ssl_certificate_key /etc/ssl/private/perceptguard.key;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## 5. Startup Verification Command

Before initiating systemd services or container tasks, verify configuration compliance via:

```bash
python -c "from src.config import get_settings; s = get_settings(); print(f'Successfully loaded {s.app.app_name} in {s.app.environment.value.upper()} mode.')"
```

If any variable fails the production validation matrix, the process will exit with code 1 and output the precise reason for rejection.
