# Secrets Management & Redaction Specification

## 1. Zero-Exposure Philosophy

PerceptGuard AI operates under a strict **Zero-Exposure Policy** for application credentials, API keys, and sensitive tokens. Secrets must never appear in:
1. Git version control or repository commits.
2. Standard output / standard error streams (`stdout` / `stderr`).
3. Application log files or distributed trace payloads.
4. Serialized diagnostic dumps or API status responses.
5. Unhandled exception tracebacks or debugger variable inspection.

---

## 2. The `SecretStr` Type

To eliminate accidental leakage, all sensitive configuration fields are wrapped in `SecretStr` (`src/config/secrets.py`):

```python
from src.config.secrets import SecretStr

secret = SecretStr("actual-super-secure-production-key")

# String representation and debugging output are permanently masked:
str(secret)   # Returns: "**********"
repr(secret)  # Returns: "SecretStr('**********')"
f"{secret}"   # Returns: "**********"

# Unmasked retrieval is restricted to explicit calls:
raw_key = secret.get_secret_value()
```

### Key Technical Attributes:
- **Masked Representations:** `__str__` and `__repr__` strictly yield redaction masks.
- **Truthiness & Sizing:** `bool(secret)` returns boolean evaluation of the secret, while `len(secret)` provides the true string length for entropy checks without revealing characters.
- **Equality & Hashing:** Safe equality comparisons against other `SecretStr` instances without leaking contents.

---

## 3. Recursive Dictionary Redaction

Diagnostic reporting and telemetry aggregation frequently dump configuration snapshots. The utility `redact_sensitive_dict()` recursively cleans arbitrary data structures:

```python
from src.config.secrets import redact_sensitive_dict

clean_payload = redact_sensitive_dict(payload, mask="**********")
```

### Sensitive Substring Heuristics:
Any dictionary key containing any of the following substrings (case-insensitive) is automatically masked:
- `secret`
- `password`
- `token`
- `api_key` / `apikey`
- `credential`
- `private_key`
- `auth`

In addition, any value of type `SecretStr` encountered in nested dictionaries or lists is converted to the masking string `**********`.

---

## 4. Production Secret Key Requirements

When `APP_ENV=production`, PerceptGuard AI enforces stringent automated validation on `SECRET_KEY`:
1. **Non-Default Prohibition:** The default development token (`dev-insecure-secret-key-perceptguard-2026`) or any token containing `"dev-insecure"` or equal to `"secret"` is rejected with `ConfigurationError`.
2. **Entropy & Length Requirement:** Keys must have a minimum length of 16 characters (recommended: 32 bytes / 64 hex characters).

### Generating Production Secrets:
To generate an appropriate secret in Unix/Linux or Git Bash:
```bash
openssl rand -hex 32
```
Or via Python:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 5. Source Control Hygiene

Repository `.gitignore` guarantees that local environment overrides and cryptographic certificates remain on developer machines:

```gitignore
# Environment variables and secrets
.env
.env.local
*.key
*.pem
```

Only `.env.example` containing non-functional placeholder values is tracked in source control.
