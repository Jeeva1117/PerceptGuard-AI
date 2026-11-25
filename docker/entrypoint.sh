#!/bin/sh
set -e

# ==============================================================================
# PerceptGuard AI — Production Container Startup Entrypoint
# Dynamically binds to Render's assigned $PORT (defaults to 8000)
# Initializes persistent storage directories for uploads and results
# ==============================================================================

# Dynamic PORT binding: Render injects $PORT (e.g. 10000)
export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"
export API_WORKERS="${API_WORKERS:-1}"

# Ensure outputs directory and uploads subdirectory exist on persistent storage
STORAGE_PATH="${PERCEPTGUARD_STORAGE_DIR:-${OUTPUTS_DIR:-/app/outputs}}"
mkdir -p "${STORAGE_PATH}/jobs/uploads" 2>/dev/null || true
mkdir -p /app/logs 2>/dev/null || true

# If container is invoked with custom arguments, execute them; otherwise start Uvicorn
if [ "$#" -gt 0 ]; then
    if [ "$1" = "uvicorn" ]; then
        exec uvicorn src.api.app:app --host "${HOST}" --port "${PORT}" --workers "${API_WORKERS}"
    else
        exec "$@"
    fi
else
    exec uvicorn src.api.app:app --host "${HOST}" --port "${PORT}" --workers "${API_WORKERS}"
fi
