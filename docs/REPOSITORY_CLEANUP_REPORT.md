# PerceptGuard-AI — Repository Cleanup Execution Report

This report documents the repository cleanup, safety verification, dependency audit, and build validations performed on the **PerceptGuard-AI** project.

---

## 1. Summary of Removed Files

In strict compliance with the cleanup safety constraints, only confirmed temporary build artifacts, compiled bytecode, test caches, and transient test outputs were removed:

| Path / Category | Items Cleaned | Reason |
| :--- | :--- | :--- |
| `**/__pycache__/*.pyc` | All `.pyc` bytecode across `src/`, `tests/`, `config/`, `dashboard/`, `scratch/`, `scripts/` | Ephemeral compiled Python files automatically generated at runtime. |
| `.pytest_cache/` | Test runner cache directory | Stale local test invocation cache. |
| `.ultralytics/` | Ultralytics runtime cache | Temporary tracking runtime scratch directory. |
| `.yolo/` | YOLO temporary workspace | Ephemeral engine scratch folder. |
| `outputs/test_temp/` | Temporary scratch video directory | Transient artifacts from early pipeline validation. |
| `outputs/test_quick_preview/` | Transient test directory | Transient video test clips. |
| `outputs/test_low_light/` | Transient test directory | Transient test frames. |
| `outputs/test_verification/` | Transient test directory | Transient validation test frames. |
| `outputs/gemini_video_test/` | Obsolete test directory | Transient test folder. |

---

## 2. Retained Critical Assets

All critical components, model weights, configurations, and user data were preserved:

- **Model Weights**:
  - `models/yolo/yolo11n.pt` ($5.35\text{ MB}$, YOLO11n object detector).
  - `models/forecasting/lstm/model.pt` ($101.1\text{ KB}$, trained multi-horizon LSTM forecasting model).
- **Configurations**:
  - All YAML configs in `config/` (`yolo_config.yaml`, `tracker_config.yaml`, `video_config.yaml`, `feature_config.yaml`, `early_warning_config.yaml`, etc.).
- **Source Code**:
  - Entire backend architecture in `src/` (all 5 modules, video processing pipeline, API services).
  - Entire frontend SPA in `frontend/src/` (components, pages, context, styles).
- **User Videos & Historical Results**:
  - All historical jobs and raw video uploads in `outputs/jobs/` and `outputs/jobs/uploads/` preserved.
- **Reference Scenario Clips**:
  - Standard reference video clips in `outputs/day14_validation/clips/` (`scenario_a_normal_daytime.mp4`, etc.) preserved for testing.

---

## 3. Dependency Audit

### Backend Python Packages (`requirements.txt`)
All 15 packages in `requirements.txt` were audited and confirmed to be actively used:
- `torch`, `torchvision`: PyTorch CPU execution for LSTM forecasting.
- `opencv-python`, `ultralytics`: Frame ingestion and YOLO11n inference.
- `numpy`, `scipy`, `pandas`, `scikit-learn`: Feature extraction, kinematics, and statistics.
- `matplotlib`: Visualization generation.
- `fastapi`, `uvicorn`, `pydantic`, `httpx`: REST API server and test execution.
- `python-dotenv`, `psutil`, `pytest`, `pyyaml`: Configuration, diagnostics, and testing.

### Frontend Dependencies (`frontend/package.json`)
All frontend packages were verified:
- Framework: `react` (v19.3), `react-dom`, `react-router-dom` (v7.18).
- Icons & Charts: `lucide-react`, `recharts`.
- Styling: `tailwindcss`, `postcss`, `autoprefixer`, `tailwind-merge`, `clsx`.
- Build Tooling: `vite` (v8.3), `typescript` (~7.0), `@vitejs/plugin-react`.
- Unused import `Shield` was removed from `Sidebar.tsx` after brand icon integration.

---

## 4. Verification After Cleanup

1. **Frontend Production Build**:
   ```powershell
   npm --prefix frontend run build
   ```
   **Result**: Built in `3.72s` with zero errors. All chunks rendered cleanly.

2. **Backend System Verification**:
   ```powershell
   python main.py --info
   ```
   **Result**: 178/178 checks passed. Full Day 1–6 pipeline confirmed intact.
