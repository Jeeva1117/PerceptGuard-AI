# PerceptGuard-AI — Master Repository Cleanup Audit

This audit documents the comprehensive review of all files, directories, dependencies, and artifacts in the **PerceptGuard-AI** repository. It classifies files according to the 8 required operational categories and establishes explicit cleanup rules before any deletions are executed.

---

## 1. Classification of Repository Files

### Category 1: Required for Application Runtime
| File / Directory Path | Purpose / Description | Action |
| :--- | :--- | :--- |
| `src/camera_weather/` | Module 1: Optical degradation, haze estimation, visibility scoring | **RETAIN** |
| `src/detection/` | Module 2 (Part A): YOLO11n object detector implementation | **RETAIN** |
| `src/tracking/` | Module 2 (Part B): ByteTrack multi-object persistent tracking | **RETAIN** |
| `src/trajectory_prediction/` | Module 3: Metric ground-plane projection & Kalman trajectory forecasting | **RETAIN** |
| `src/collision_risk/` | Module 4: 1D/2D Time-To-Collision (TTC) & spatial conflict analyzer | **RETAIN** |
| `src/safety_recommendation/` | Module 5: Priority arbitration & driver safety advisory generation | **RETAIN** |
| `src/api/` | FastAPI REST services, job manager, routers (`video`, `health`, `telemetry`, `alerts`) | **RETAIN** |
| `src/video_processing/` | Complete pipeline orchestrator (`VideoPipelineRunner`, metadata generator) | **RETAIN** |
| `src/config/` | Centralized system settings, validators, and YAML config loaders | **RETAIN** |
| `config/*.yaml` | Detection, tracking, video, and forecasting configuration files | **RETAIN** |
| `main.py` | Command-line CLI entrypoint (`--info`, `--audit`, `--verify-env`) | **RETAIN** |
| `frontend/src/` | React 19 / TypeScript SPA source code, components, context, and pages | **RETAIN** |
| `frontend/package.json` | Frontend dependencies and build script definitions | **RETAIN** |
| `frontend/vite.config.ts` | Vite build & proxy configuration | **RETAIN** |
| `frontend/index.html` | SPA entry point and favicon anchor | **RETAIN** |

### Category 2: Required for Model Loading
| File / Directory Path | Purpose / Description | Action |
| :--- | :--- | :--- |
| `models/yolo/yolo11n.pt` | YOLO11n pretrained detection model weights (5.35 MB) | **RETAIN** |
| `models/forecasting/lstm/model.pt` | Trained multi-horizon LSTM checkpoint (101.1 KB) | **RETAIN** |
| `models/.gitkeep` | Git tracking anchor for models root | **RETAIN** |
| `models/yolo/.gitkeep` | Git tracking anchor for YOLO directory | **RETAIN** |
| `models/forecasting/.gitkeep`| Git tracking anchor for forecasting directory | **RETAIN** |

### Category 3: Required for Local Development
| File / Directory Path | Purpose / Description | Action |
| :--- | :--- | :--- |
| `venv/` | Python 3.13 virtual environment with installed packages | **RETAIN (Local, ignored by git)** |
| `frontend/node_modules/` | Node.js modules for Vite/React development | **RETAIN (Local, ignored by git)** |
| `.env` | Local development environment configuration | **RETAIN (Local, ignored by git)** |
| `.env.example` | Clean configuration template for other developers | **RETAIN** |
| `requirements.txt` | Python runtime dependency specifications | **RETAIN** |
| `pyproject.toml` / `setup.py` | Package build configuration | **RETAIN** |
| `docker-compose.yml` | Multi-container local orchestration | **RETAIN** |

### Category 4: Required for Testing
| File / Directory Path | Purpose / Description | Action |
| :--- | :--- | :--- |
| `tests/` | Comprehensive unit and integration test suites | **RETAIN** |
| `outputs/day14_validation/clips/` | Standard scenario reference driving clips used by tests | **RETAIN** |

### Category 5: Required for Future Deployment
| File / Directory Path | Purpose / Description | Action |
| :--- | :--- | :--- |
| `render.yaml` | Render Blueprint infrastructure-as-code definition | **RETAIN** |
| `Dockerfile` | Multi-stage production container build definition | **RETAIN** |
| `docker/Dockerfile.backend` | Backend service Dockerfile | **RETAIN** |
| `docker/entrypoint.sh` | Container entrypoint script resolving dynamic `$PORT` | **RETAIN** |
| `.dockerignore` | Build context exclusions ensuring fast, secure Docker builds | **RETAIN** |
| `docs/RENDER_DEPLOYMENT_GUIDE.md` | Deployment operations and persistent disk guide | **RETAIN** |

### Category 6: Unused or Duplicated Files (Cleanup Candidates)
| File / Directory Path | Reason | Proposed Action |
| :--- | :--- | :--- |
| `frontend/public/favicon.svg` | Generic Vite template lightning bolt; does not match PerceptGuard branding | **REPLACE with unique automotive logo** |
| `outputs/test_temp/` | Temporary scratch video test files from earlier validation | **DELETE** |
| `outputs/test_quick_preview/`| Transient pipeline test output directory | **DELETE** |
| `outputs/test_low_light/` | Transient test artifacts | **DELETE** |
| `outputs/test_verification/` | Transient sideways video test artifacts | **DELETE** |
| `outputs/gemini_video_test/` | Obsolete test directory | **DELETE** |

### Category 7: Temporary Generated Files (Cleanup Candidates)
| File / Directory Path | Reason | Proposed Action |
| :--- | :--- | :--- |
| `__pycache__/` & all sub-caches | Compiled Python bytecode (`.pyc`) generated at runtime | **DELETE (Clean safely)** |
| `.pytest_cache/` | Pytest execution state cache | **DELETE** |
| `.ultralytics/` | Ultralytics runtime tracking cache | **DELETE** |
| `.yolo/` | Temporary YOLO engine cache | **DELETE** |

### Category 8: Unnecessary Large Files
| File / Directory Path | Reason | Proposed Action |
| :--- | :--- | :--- |
| `datasets/processed/*.mp4` | Processed test clips generated during earlier experimentation | **RETAIN in local datasets/ (Ignored by git)** |
| `outputs/jobs/uploads/` | Uploaded user test videos (retained per explicit safety constraint) | **RETAIN (Preserve user data)** |

---

## 2. Safety Rule Verifications

1. **Required YOLO and LSTM Weights**: Confirmed retained at `models/yolo/yolo11n.pt` and `models/forecasting/lstm/model.pt`.
2. **ByteTrack and Configuration Files**: All files in `config/` (`yolo_config.yaml`, `tracker_config.yaml`, etc.) confirmed retained.
3. **Source Code**: No deletions in `src/` or `frontend/src/`.
4. **Working Five-Module Functionality**: All modules and endpoints confirmed preserved.
5. **User Uploads**: Preserved in `outputs/jobs/`.
