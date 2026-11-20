# Frontend Error Handling & State Resilience

**PerceptGuard AI — Perception Failure Early Warning System**  
**Module:** UI Error Handling, Empty States, and Resilient Degradation  
**Components:** `dashboard/components/state_views.py`, `dashboard/api_client.py`  

---

## 1. Exception Hierarchy

`DashboardApiClient` raises domain-specific exceptions inheriting from `FrontendAPIError`:

```
FrontendAPIError (Base Exception)
├── APIConnectionError       # Unreachable host, connection refused, DNS failure
├── APITimeoutError          # Request deadline exceeded
└── APIResponseError         # Non-2xx HTTP status returned by backend (e.g. 404, 422, 500)
```

Each exception captures:
- `message`: Human-readable summary suitable for display in UI error boundaries.
- `status_code`: Optional HTTP status code (for `APIResponseError`).
- `details`: Structured JSON error payload returned from the backend.

---

## 2. Component Error Boundaries

All major dashboard sections wrap client interactions within isolated error boundaries via `render_error_boundary(...)`:

```python
try:
  overview = client.get_risk_overview()
except FrontendAPIError as err:
  render_error_boundary(
      title="Failed to Load Risk Overview Telemetry",
      error_message=f"Backend communication error: {err.message}",
      retry_callback=st.rerun,
      key_suffix="risk_overview",
  )
  return
```

### Security & UX Principles
- **No Stack Traces Leaked:** Raw Python tracebacks and internal file paths are withheld from the user interface.
- **Section Isolation:** An API failure in one component (e.g., temporal monitoring) renders an inline error card with an actionable **"🔄 Retry Connection"** button, allowing other sections (e.g., detection telemetry) to remain operational.

---

## 3. Empty States (`render_empty_state`)

When queries return zero records (e.g., no active alerts matching a filter, or before video ingestion populates trajectories), `render_empty_state(...)` displays a structured card:

- Clear contextual title (e.g., *"No Matching Alerts"*).
- Informative message indicating why data is absent.
- Category icon (e.g., 🔔 for alerts, 📈 for charts, 🎯 for tracking).
- Prevents broken tables or empty line charts from rendering.

---

## 4. Telemetry Staleness Warnings (`render_stale_warning`)

In autonomous vehicle perception monitoring, stale data is dangerous if operators believe it is live. `render_stale_warning(...)` monitors data freshness:

- **Criteria:** Triggers when `freshness == "STALE"` or $(t_{\text{now}} - t_{\text{sync}}) > \text{TTL}$.
- **Presentation:** High-visibility amber banner indicating data age, last sync timestamp, and configured TTL.
- **Guarantee:** Informs operators that displayed metrics represent the last verified observation rather than real-time telemetry.
