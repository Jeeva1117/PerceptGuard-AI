# Frontend Production Optimization

**PerceptGuard AI — Perception Failure Early Warning System**  
**Module:** Frontend Dashboard Optimization & Architecture Hardening  
**Target:** Production-Grade Reliability, Modular UI Architecture, and Performance  

---

## 1. Executive Summary

This document details the production optimization and hardening of the PerceptGuard AI frontend interactive dashboard (`dashboard/app.py` and `dashboard/components/`). Building on the Day 10 Streamlit dashboard and Day 11 backend REST API, this engineering upgrade modularizes the user interface into discrete, resilient component views, introduces dual deployment topology support (REST API and in-process fallback), enforces strict data presentation integrity, and provides resilient state views.

---

## 2. Architectural Modularization

The monolithic dashboard script was refactored into a layered, modular component architecture:

```
dashboard/
├── __init__.py
├── app.py                          # Application entry point, routing, sidebar, layout
├── api_client.py                   # Hardened HTTP REST API client with in-process fallback
└── components/
    ├── __init__.py                 # Clean public component exports
    ├── state_views.py              # Empty states, error boundaries, stale notices
    ├── header.py                   # Branding, versioning, environment & transport badges
    ├── overview.py                 # Section A: Pipeline status & operational health
    ├── risk_cards.py               # Section B: Multi-horizon risk cards (H1, H3, H5)
    ├── temporal_chart.py           # Section C: Trajectory line chart & transitions table
    ├── alerts_feed.py              # Section D: Alerts feed, filters, inspector, operator actions
    ├── tracking_view.py            # Section E: YOLO11n + ByteTrack authentic telemetry
    └── pipeline_accordion.py       # 8-Stage perception architecture pipeline status
```

### Key Architectural Benefits
- **Separation of Concerns:** Data retrieval, formatting, and UI rendering are decoupled.
- **Independent Testability:** Components can be isolated and verified through unit tests without running full browser instances.
- **Fail-Safe Isolation:** A failure in one section (e.g., telemetry latency spike) does not crash adjacent sections.

---

## 3. Strict Data Integrity Enforcements

1. **No Fake LOW Risk:** Missing or uncalculated multi-horizon predictions are strictly displayed as `"N/A"` with risk category `"UNKNOWN"` (grey badge). Missing forecasts are NEVER converted into `0.0` or `LOW` risk.
2. **Provisional Calibration Badge:** Because of zero-inflation in single-clip driving scenarios, multi-horizon failure risk thresholds ($S = 1.0 - R$) carry the mandatory operational gate label `PROVISIONAL CALIBRATION`.
3. **Authentic Telemetry Lineage:** Context metrics report authentic detection counts, unique tracking IDs, and processing latencies from YOLO11n and ByteTrack.

---

## 4. UI Performance & Resource Efficiency

- **Singleton Caching (`@st.cache_resource`):** The `DashboardApiClient` and `PipelineIntegrator` are instantiated once and cached across user interactions and re-runs.
- **Non-Blocking Network Probing:** Background HTTP health checks use tight 1.5s timeouts so unresponsive remote endpoints do not freeze UI thread execution.
- **Container-Width Responsive Layouts:** Dynamic Streamlit columns (`st.columns`) ensure responsiveness across desktop, workstation, and tablet viewport sizes.
