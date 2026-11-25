# ==============================================================================
# PerceptGuard AI — Production Backend Dockerfile (FastAPI REST Service)
# Multi-stage CPU-optimized container with non-root security execution.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Dependency Builder
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install minimal build tools required for compiling C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated Python virtual environment for cleanly copying packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-optimized dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    APP__ENVIRONMENT=production \
    MODEL__DEVICE=cpu

# Install minimal runtime system libraries for OpenCV and health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user and group (UID/GID 1000)
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application configuration, model weights, and source code
COPY --chown=appuser:appgroup config/ config/
COPY --chown=appuser:appgroup models/ models/
COPY --chown=appuser:appgroup src/ src/
COPY --chown=appuser:appgroup dashboard/ dashboard/
COPY --chown=appuser:appgroup experiments/ experiments/
COPY --chown=appuser:appgroup main.py .

# Copy entrypoint script and ensure Unix line endings
COPY --chown=appuser:appgroup docker/entrypoint.sh /app/entrypoint.sh
RUN tr -d '\r' < /app/entrypoint.sh > /app/entrypoint_clean.sh && \
    mv /app/entrypoint_clean.sh /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Create required operational directories with non-root ownership
RUN mkdir -p /app/models /app/datasets /app/outputs /app/logs && \
    chown -R appuser:appgroup /app

# Switch to non-root execution
USER appuser

# Expose backend REST API default port
EXPOSE 8000

# Container liveness health check dynamically probing $PORT
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f "http://127.0.0.1:${PORT:-8000}/health/live" || exit 1

# Production entrypoint dynamically binding to Render's $PORT
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
