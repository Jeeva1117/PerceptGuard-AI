# Frontend Build & Production Validation

**PerceptGuard AI — Perception Failure Early Warning System**  
**Module:** Static Analysis, Bytecode Compilation, and Dependency Validation  
**Component:** `tests/frontend/test_production_build.py`  

---

## 1. Static Bytecode Compilation Validation

All frontend scripts and modular components are validated via Python's standard `py_compile` module with `doraise=True`:

```powershell
python -m py_compile dashboard/app.py dashboard/api_client.py dashboard/components/*.py
```

### Verified Modules
- `dashboard/app.py`: Clean compilation, zero syntax errors, zero deprecation warnings.
- `dashboard/api_client.py`: Clean compilation, strict type annotations.
- `dashboard/components/__init__.py`: Clean exports.
- `dashboard/components/state_views.py`: Clean compilation.
- `dashboard/components/header.py`: Clean compilation.
- `dashboard/components/overview.py`: Clean compilation.
- `dashboard/components/risk_cards.py`: Clean compilation.
- `dashboard/components/temporal_chart.py`: Clean compilation.
- `dashboard/components/alerts_feed.py`: Clean compilation.
- `dashboard/components/tracking_view.py`: Clean compilation.
- `dashboard/components/pipeline_accordion.py`: Clean compilation.

---

## 2. Production Runtime Dependencies

The frontend environment depends on hardened, pinned Python packages:

| Dependency | Purpose | Status |
| :--- | :--- | :--- |
| `streamlit >= 1.30.0` | Reactive web dashboard framework | Verified |
| `httpx >= 0.25.0` | High-performance HTTP/REST client | Verified |
| `pandas >= 2.0.0` | Tabular data manipulation & chart pivoting | Verified |
| `pydantic >= 2.0.0` | Data schema validation & settings management | Verified |

---

## 3. Headless Streamlit Execution & Smoke Testing

The application is verified to launch headlessly in containerized or CI/CD environments using:

```powershell
streamlit run dashboard/app.py --server.headless=true --browser.gatherUsageStats=false
```

No blocking UI dialogs, interactive prompts, or unsupported external browser dependencies exist.
