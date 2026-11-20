"""Automated validation tests for GitHub Actions CI/CD workflows and deployment gates."""

from __future__ import annotations

from pathlib import Path
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
DEPLOY_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "deploy.yml"


class TestGitHubActionsCIWorkflow:
    """Validate structure, triggers, security policies, and jobs in ci.yml."""

    def test_ci_workflow_exists_and_is_valid_yaml(self):
        """Verify ci.yml exists and parses as valid YAML."""
        assert CI_WORKFLOW_PATH.exists(), f"CI workflow missing at {CI_WORKFLOW_PATH}"
        content = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert data is not None
        assert "name" in data
        assert "jobs" in data

    def test_ci_triggers_and_concurrency(self):
        """Verify trigger events, path filtering, and concurrency cancellation."""
        data = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))

        # Triggers
        triggers = data.get("on") or data.get(True)  # YAML parsers may interpret 'on' as boolean True
        assert triggers is not None
        assert "push" in triggers
        assert "pull_request" in triggers
        assert "workflow_dispatch" in triggers

        # Branches
        assert "main" in triggers["push"]["branches"]
        assert "main" in triggers["pull_request"]["branches"]

        # Concurrency
        assert "concurrency" in data
        assert data["concurrency"].get("cancel-in-progress") is True

    def test_ci_permissions_least_privilege(self):
        """Verify workflow adheres to least privilege permissions."""
        data = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
        assert "permissions" in data
        assert data["permissions"].get("contents") == "read"

    def test_ci_job_definitions(self):
        """Verify all 6 required verification jobs are configured."""
        data = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
        jobs = data["jobs"]

        expected_jobs = [
            "code-quality",
            "backend-ci",
            "frontend-ci",
            "docker-validation",
            "security-scan",
            "readiness-gate",
        ]
        for job in expected_jobs:
            assert job in jobs, f"Job '{job}' missing in CI workflow"

    def test_backend_ci_cpu_only_and_matrix(self):
        """Verify backend CI enforces CPU-only PyTorch and multi-version testing."""
        data = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
        backend_job = data["jobs"]["backend-ci"]

        # Matrix
        strategy = backend_job.get("strategy", {})
        matrix = strategy.get("matrix", {})
        assert "python-version" in matrix
        assert "3.11" in matrix["python-version"]

        # CPU torch install step
        steps = backend_job.get("steps", [])
        install_steps = [s for s in steps if "Install CPU-Optimized Dependencies" in s.get("name", "")]
        assert len(install_steps) == 1
        assert "--extra-index-url https://download.pytorch.org/whl/cpu" in install_steps[0]["run"]

    def test_docker_validation_non_root_check(self):
        """Verify Docker validation inspects non-root execution."""
        data = yaml.safe_load(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
        docker_job = data["jobs"]["docker-validation"]
        steps = docker_job.get("steps", [])
        step_names = [s.get("name", "") for s in steps]
        assert any("Non-Root User" in name for name in step_names)


class TestDeploymentApprovalGateWorkflow:
    """Validate deployment separation and manual authorization gates in deploy.yml."""

    def test_deploy_workflow_is_manual_only(self):
        """Verify deployment workflow is triggered solely by manual workflow_dispatch."""
        assert DEPLOY_WORKFLOW_PATH.exists(), f"Deploy workflow missing at {DEPLOY_WORKFLOW_PATH}"
        data = yaml.safe_load(DEPLOY_WORKFLOW_PATH.read_text(encoding="utf-8"))

        triggers = data.get("on") or data.get(True)
        # Must ONLY have workflow_dispatch
        assert "workflow_dispatch" in triggers
        assert "push" not in triggers
        assert "pull_request" not in triggers

    def test_deploy_workflow_requires_explicit_confirmation(self):
        """Verify deployment job enforces strict input confirmation token."""
        data = yaml.safe_load(DEPLOY_WORKFLOW_PATH.read_text(encoding="utf-8"))
        triggers = data.get("on") or data.get(True)
        inputs = triggers["workflow_dispatch"].get("inputs", {})
        assert "confirmation" in inputs

        deploy_job = data["jobs"]["deploy-authorized"]
        assert deploy_job.get("if") == "github.ref == 'refs/heads/main'"
