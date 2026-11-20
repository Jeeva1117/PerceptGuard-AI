"""Static linting, schema validation, and security tests for Docker and Docker Compose."""

from __future__ import annotations

from pathlib import Path
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestDockerfilesStaticValidation:
    """Validate static configuration, multi-stage structure, and security controls in Dockerfiles."""

    def test_backend_dockerfile_structure_and_security(self):
        """Verify backend Dockerfile complies with multi-stage, non-root, and production requirements."""
        dockerfile_path = PROJECT_ROOT / "Dockerfile"
        assert dockerfile_path.exists(), "Root Dockerfile missing"

        content = dockerfile_path.read_text(encoding="utf-8")

        # Multi-stage build
        assert "FROM python:3.11-slim AS builder" in content
        assert "FROM python:3.11-slim AS runtime" in content

        # Non-root security user
        assert "useradd" in content
        assert "appuser" in content
        assert "USER appuser" in content

        # Working directory & port
        assert "WORKDIR /app" in content
        assert "EXPOSE 8000" in content

        # Healthcheck
        assert "HEALTHCHECK" in content
        assert "/health/live" in content

        # Production start command (Uvicorn)
        assert "uvicorn" in content
        assert "src.api.app:app" in content

        # CPU-only environment
        assert "MODEL__DEVICE=cpu" in content

    def test_frontend_dockerfile_structure_and_security(self):
        """Verify frontend Dockerfile complies with Streamlit requirements, multi-stage, and non-root."""
        frontend_dockerfile_path = PROJECT_ROOT / "Dockerfile.frontend"
        assert frontend_dockerfile_path.exists(), "Dockerfile.frontend missing"

        content = frontend_dockerfile_path.read_text(encoding="utf-8")

        # Multi-stage build
        assert "FROM python:3.11-slim AS builder" in content
        assert "FROM python:3.11-slim AS runtime" in content

        # Non-root security user
        assert "useradd" in content
        assert "USER appuser" in content

        # Port & healthcheck
        assert "EXPOSE 8501" in content
        assert "HEALTHCHECK" in content
        assert "/_stcore/health" in content

        # Streamlit production run command
        assert "streamlit" in content
        assert "dashboard/app.py" in content
        assert "--server.headless=true" in content

    def test_docker_dir_mirrors_exist(self):
        """Verify docker/ directory contains mirrored Dockerfile copies for modular builds."""
        docker_dir = PROJECT_ROOT / "docker"
        assert (docker_dir / "Dockerfile.backend").exists()
        assert (docker_dir / "Dockerfile.frontend").exists()


class TestDockerignoreSecurityExclusions:
    """Verify .dockerignore excludes sensitive files, secrets, raw datasets, and caches."""

    def test_dockerignore_rules(self):
        """Verify required security exclusions are present in .dockerignore."""
        dockerignore_path = PROJECT_ROOT / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore missing"

        content = dockerignore_path.read_text(encoding="utf-8")
        rules = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

        # Critical security exclusions
        assert ".git/" in rules
        assert ".env" in rules
        assert "venv/" in rules
        assert "__pycache__/" in rules
        assert "datasets/raw/" in rules
        assert "models/*.pt" in rules
        assert "outputs/" in rules


class TestDockerComposeOrchestration:
    """Validate docker-compose.yml syntax, service topologies, networks, and resource limits."""

    def test_docker_compose_schema_and_services(self):
        """Verify docker-compose.yml defines backend and frontend without superfluous services."""
        compose_path = PROJECT_ROOT / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml missing"

        content = compose_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        assert "services" in data
        services = data["services"]

        # Exactly 2 services: backend and frontend (no unnecessary database)
        assert "backend" in services
        assert "frontend" in services
        assert len(services) == 2, f"Expected 2 services, found: {list(services.keys())}"

        # Backend service configuration
        backend = services["backend"]
        assert "8000:8000" in backend["ports"]
        assert "perceptguard-net" in backend["networks"]
        assert backend.get("restart") == "unless-stopped"
        assert "healthcheck" in backend
        assert "test" in backend["healthcheck"]
        assert "deploy" in backend and "resources" in backend["deploy"]

        # Frontend service configuration
        frontend = services["frontend"]
        assert "8501:8501" in frontend["ports"]
        assert "perceptguard-net" in frontend["networks"]
        assert "backend" in frontend.get("depends_on", {})
        assert frontend["depends_on"]["backend"]["condition"] == "service_healthy"

        # Networks and Volumes
        assert "networks" in data and "perceptguard-net" in data["networks"]
        assert "volumes" in data
        assert "backend_outputs" in data["volumes"]
        assert "backend_logs" in data["volumes"]

    def test_referenced_source_directories_exist(self):
        """Verify all local host directories referenced in COPY or volumes exist."""
        required_dirs = [
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "dashboard",
            PROJECT_ROOT / "experiments",
            PROJECT_ROOT / "models",
            PROJECT_ROOT / "datasets" / "processed",
        ]
        for d in required_dirs:
            assert d.exists() and d.is_dir(), f"Referenced directory {d} does not exist"

        assert (PROJECT_ROOT / "requirements.txt").exists()
        assert (PROJECT_ROOT / "main.py").exists()
