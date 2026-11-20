# Backend Production Hardening & Architecture Guide

## 1. Executive Summary

This document specifies the production-hardened REST API architecture for **PerceptGuard AI** (`src/api`). Built on top of FastAPI and Starlette, the service provides high-performance, asynchronous interfaces for querying multi-horizon perception failure risk assessments, temporal degradation trajectories, incident alerts, and video stream validation under strict CPU-first deployment constraints.

---

## 2. Application Architecture & Router Decomposition

The API follows a modular, decoupled structure:

```
src/api/
├── __init__.py           # Application export facade (create_app)
├── app.py                # FastAPI factory, lifespan manager, middleware assembly
├── dependencies.py       # Decoupled dependency injection providers
├── exceptions.py         # Centralized error classes & standardized exception handlers
├── middleware.py         # Request ID tracing, security headers, structured access logging
├── schemas.py            # Strongly typed Pydantic V2 request & response schemas
└── routers/
    ├── __init__.py       # Router bundle exports
    ├── health.py         # Liveness (/health/live), readiness (/health/ready), overall health
    ├── telemetry.py      # Risk overview (H1, H3, H5), pipeline status, temporal monitoring
    ├── alerts.py         # Alert history, filtering, detail, acknowledge, and resolve actions
    └── video.py          # Video format validation, file size checks, bounded processing
```

---

## 3. Application Lifecycle Management (Lifespan Context)

The application utilizes FastAPI's modern `lifespan` context manager (`@asynccontextmanager`) to manage application startup and shutdown deterministically:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    # 1. Resolve typed settings from singleton cache
    # 2. Instantiate singleton PipelineIntegrator
    # 3. Ingest pre-verified multi-horizon predictions artifact if available on disk
    # 4. Attach state to app.state.integrator and app.state.dashboard_service
    # 5. Record startup timestamp and log environment profile
    yield
    # SHUTDOWN
    # 1. Release references and clean up resources
    # 2. Log graceful termination
```

### Key Architectural Safeguards:
- **Zero Heavy Import Loading:** Models and heavy pipeline components are initialized during the lifecycle startup phase rather than during Python module import.
- **Graceful Zero-State Replay:** If ML checkpoints or prediction artifacts are absent on disk, the backend does not crash; instead, it enters an operational replay mode and reports `degraded` status through readiness probes.
- **Resource Determinism:** References are cleanly released on shutdown, preventing dangling file handles.

---

## 4. Defensive Middleware Pipeline

Incoming HTTP requests pass through three layered middleware components before reaching endpoint handlers:

1. **`RequestLoggingMiddleware`:**
   - Injects or propagates `X-Request-ID` (UUID4) across request and response headers.
   - Measures execution latency with microsecond precision (`time.perf_counter()`).
   - Generates structured operational access logs:
     `HTTP GET /api/v1/telemetry/risk-overview status=200 duration_ms=1.45 client=127.0.0.1 request_id=...`
2. **`SecurityHeadersMiddleware`:**
   - Injects `X-Content-Type-Options: nosniff`
   - Injects `X-Frame-Options: DENY`
   - Injects `X-XSS-Protection: 1; mode=block`
   - Injects `Referrer-Policy: strict-origin-when-cross-origin`
   - Injects `Strict-Transport-Security: max-age=31536000; includeSubDomains` (when `APP_ENV=production`).
3. **`CORSMiddleware` & `TrustedHostMiddleware`:**
   - Enforces origin domain whitelisting and host header validation from `SystemSettings`.

---

## 5. CPU Optimization & Memory Safety

Designed to execute on the reference hardware tier (Intel Core i3, 8 GB RAM, CPU-only):
- **Bounded Rolling Deques:** Downstream time series windows are bounded at $W=100$ records per horizon, restricting memory footprint to $< 5\text{ MB}$.
- **Synchronous CPU Offloading:** CPU-bound processing and file inspections run via Starlette's worker thread pool (`anyio.to_thread.run_sync`), preventing event-loop starvation.
- **Bounded Video Inspection:** Temporary uploads are capped at $100\text{ MB}$, and synchronous frame extraction caps processing to a maximum of $30$ frames per request.
