# Contributing to PerceptGuard AI

Thank you for your interest in contributing to **PerceptGuard AI**! We welcome contributions that enhance detection reliability forecasting, computer vision pipelines, dashboard capabilities, and edge performance.

---

## 1. Development Principles

1. **Safety & Scientific Rigor First:** Never fabricate machine learning metrics, ground truth annotations, or test results. Always distinguish between classification confidence and future reliability.
2. **CPU-First Optimization:** Code must remain fully executable on standard CPU architectures without mandatory GPU/CUDA hardware requirements.
3. **Evidence-Based Validation:** Every pull request must be supported by automated pytest unit tests and real-world execution evidence.
4. **Defensive Integration:** Maintain dual REST API and in-process fallback compatibility. Never introduce breaking changes to existing data contracts ($S = 1.0 - R$).

---

## 2. Setting Up Your Development Environment

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/Jeeva1117/PerceptGuard-AI.git
   cd PerceptGuard-AI
   ```

2. **Create and Activate Virtual Environment:**
   ```powershell
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   ```bash
   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Environment:**
   ```bash
   python main.py --info
   pytest tests/api/ -v
   ```

---

## 3. Coding Standards & Guidelines

- **Python Version:** Compatible with Python 3.10 through 3.13.
- **Type Annotations:** All new functions and methods must include static type annotations (`from __future__ import annotations`).
- **Docstrings:** Use Google-style or NumPy-style docstrings explaining parameters, returns, and exceptions.
- **Formatting:** Adhere to PEP 8 style standards with 100-character line length limits.
- **Null Safety:** Never coerce missing or null risk predictions to `0.0` or `LOW` risk; always use `None`, `"N/A"`, or `"UNKNOWN"`.

---

## 4. Testing Requirements

Before opening a pull request, run the entire test suite locally:

```bash
pytest -v
```

- **Zero Failing Tests:** All tests must pass before submission.
- **New Unit Tests:** Any new feature, route, or model capability must include corresponding unit and integration tests under `tests/`.
- **Regression Protection:** Ensure existing tests in `tests/api/`, `tests/detection/`, `tests/features/`, and `tests/frontend/` continue to pass.

---

## 5. Commit & Pull Request Guidelines

### Commit Messages
We follow the **Conventional Commits** specification:
- `feat:` A new feature or capability (e.g. `feat: add optical flow feature extractor`)
- `fix:` A bug fix (e.g. `fix: resolve NameError in video router`)
- `docs:` Documentation improvements (e.g. `docs: update user guide`)
- `test:` Adding or updating tests (e.g. `test: verify early warning suppression`)
- `chore:` Maintenance, dependency, or configuration updates

### Pull Request Process
1. Create a feature branch: `git checkout -b feat/my-improvement`.
2. Commit your changes with clear, concise commit messages.
3. Push to your fork and submit a Pull Request against `main`.
4. Fill out the PR description with:
   - Summary of changes
   - Motivation and context
   - Test execution evidence
   - Hardware profile tested on
