# Backend Logging & Observability Specification

## 1. Observability Goals

PerceptGuard AI implements structured, low-overhead application logging designed for real-time diagnostics, performance auditing, and security incident investigations. The logging architecture adheres to the following principles:
1. **Correlation Tracing:** Every HTTP transaction carries a unique `X-Request-ID` propagated through log entries and client responses.
2. **Zero Secret Exposure:** Credentials, passwords, API tokens, and authorization headers are scrubbed before writing.
3. **Bounded Overhead:** Logging overhead is kept to $< 0.1\text{ ms}$ per request; high-frequency per-frame video loops suppress verbose logging.
4. **Structured Format:** Clean, parsable text format for development; structured JSON mode for production aggregation (Datadog, Loki, CloudWatch).

---

## 2. Request Tracing & Access Logging

The `RequestLoggingMiddleware` (`src/api/middleware.py`) records the complete lifecycle of every inbound HTTP request:

```
2026-11-20 17:45:00,123 [INFO] perceptguard.api.access: HTTP GET /api/v1/telemetry/risk-overview status=200 duration_ms=1.45 client=127.0.0.1 request_id=8f68d902-61f2-4cf4-9276-880c5e7b2498
```

### Logged Fields:
- `timestamp`: UTC wall-clock time.
- `log_level`: Severity level (`INFO`, `WARNING`, `ERROR`).
- `logger_name`: Originating subsystem logger (`perceptguard.api`, `perceptguard.api.access`, `perceptguard.api.exceptions`).
- `method`: HTTP verb (`GET`, `POST`, `OPTIONS`).
- `path`: Request path.
- `status`: HTTP response status code.
- `duration_ms`: Execution latency in milliseconds.
- `client`: Remote IP address or hostname.
- `request_id`: Tracing correlation identifier.

---

## 3. Lifecycle Events

The backend emits structured log entries during critical system transitions:
- **Startup:**
  `[INFO] perceptguard.api: Initializing PerceptGuard AI Backend: env=PRODUCTION, version=1.0.0-preview, device=cpu`
  `[INFO] perceptguard.api: Ingesting pre-verified multi-horizon predictions: predictions.json`
- **Shutdown:**
  `[INFO] perceptguard.api: Shutting down PerceptGuard AI Backend services...`
  `[INFO] perceptguard.api: Backend resources released cleanly.`
- **Operational Warnings:**
  `[WARNING] perceptguard.api: Multi-horizon predictions artifact not found at ... Operating in replay/zero-state telemetry mode.`
- **Business Exceptions:**
  `[WARNING] perceptguard.api.exceptions: API Business Error: code=RESOURCE_NOT_FOUND status=404 message=Alert 'alert_99' not found. request_id=...`
- **Internal Server Errors:**
  `[ERROR] perceptguard.api.exceptions: Unhandled exception during request processing: ... request_id=...` (accompanied by sanitized server traceback).

---

## 4. Security & Redaction Safeguards

- **Sensitive Headers Redacted:** `Authorization`, `Cookie`, `X-API-Key`, and `Set-Cookie` are never written to access logs.
- **Dictionary Sanitization:** All diagnostic configuration dumps pass through `redact_sensitive_dict()`.
- **Credential Masking:** Any `SecretStr` instances render as `**********` if logged accidentally.
