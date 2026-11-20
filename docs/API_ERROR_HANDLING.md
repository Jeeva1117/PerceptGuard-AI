# API Error Handling & Exception Management Specification

## 1. Error Handling Philosophy

PerceptGuard AI adheres to a strict **Zero Information Leakage Policy** for API client responses while maintaining rich diagnostic traceability for server operations. In production:
1. Python stack traces, file paths, and memory addresses are **never** returned to API clients.
2. Sensitive keys, credentials, or authentication tokens are **never** exposed in error messages.
3. Every error response includes a correlation `request_id` matching server-side diagnostic logs.
4. HTTP status codes strictly adhere to RFC 9110 semantics.

---

## 2. Standardized Error Envelope Schema

All API errors return a uniform, predictable JSON payload:

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Alert with ID 'alert_12345' was not found.",
    "details": null
  },
  "request_id": "8f68d902-61f2-4cf4-9276-880c5e7b2498",
  "timestamp": "2026-11-20T17:45:00.123456+00:00"
}
```

### Schema Definitions (`src/api/schemas.py`):
- `error.code: str` — Machine-readable error categorization code.
- `error.message: str` — Human-readable operational explanation.
- `error.details: Optional[Any]` — Granular field validation errors (e.g. from Pydantic `RequestValidationError`).
- `request_id: str` — Unique UUID4 tracking identifier.
- `timestamp: str` — ISO 8601 UTC timestamp of error generation.

---

## 3. Exception Hierarchy & HTTP Status Mapping

The exception taxonomy is defined in `src/api/exceptions.py`:

| Exception Class | Error Code | HTTP Status | Typical Root Cause |
| :--- | :--- | :---: | :--- |
| `ResourceNotFoundError` | `RESOURCE_NOT_FOUND` | **404 Not Found** | Missing alert ID, sequence ID, or checkpoint file. |
| `InvalidInputError` | `INVALID_INPUT` | **400 Bad Request** | Unsupported horizon (e.g. 7), invalid severity, bad pagination. |
| `InvalidVideoError` | `INVALID_VIDEO_FILE` | **400 Bad Request** | Unsupported video extension, file size > 100MB, corrupt stream. |
| `RequestValidationError` | `VALIDATION_ERROR` | **422 Unprocessable Entity** | Pydantic schema validation failure (missing fields, wrong types). |
| `ModelInferenceError` | `MODEL_INFERENCE_ERROR` | **502 Bad Gateway** | Model outputting non-finite values (NaN / Inf) or compute failure. |
| `ServiceUnavailableError` | `SERVICE_UNAVAILABLE` | **503 Service Unavailable** | Core dashboard service or pipeline uninitialized. |
| Unhandled `Exception` | `INTERNAL_SERVER_ERROR` | **500 Internal Server Error** | Unexpected runtime crash. Logged securely with stack trace. |

---

## 4. Handler Registration & Logging Behavior

Exception handlers registered via `register_exception_handlers(app)` in `src/api/exceptions.py`:
- Log warnings for client errors ($4\text{xx}$):
  `WARNING perceptguard.api.exceptions: API Business Error: code=RESOURCE_NOT_FOUND status=404 message=... request_id=...`
- Log full tracebacks internally for server errors ($5\text{xx}$):
  `ERROR perceptguard.api.exceptions: Unhandled exception during request processing: ... request_id=...`
  Clients receive only the generic message: `"An unexpected internal server error occurred. Diagnostics logged under request_id."`
