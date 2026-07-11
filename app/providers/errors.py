"""Provider/policy error types shared by app/providers/router.py, the STT provider adapters, and
the STT routers.

- SttFallbackEligibleError / SttFailLoudError classify STT provider failures (setara-s94o.7):
  SttProviderRouter only falls back to a secondary provider on SttFallbackEligibleError; every
  other exception (including SttFailLoudError) propagates immediately with its message intact.
- SttPolicyRejectedError is raised by the policy layer (setara-s94o.9) before any provider is
  invoked. It carries the HTTP status the caller (routers/stt.py) should surface.
"""


class SttFallbackEligibleError(Exception):
    """Transient/infra STT provider failure - safe for SttProviderRouter to retry against a
    fallback provider (e.g. network timeout, provider 5xx, transient rate limiting)."""


class SttFailLoudError(Exception):
    """Non-transient STT provider failure - must surface to the caller immediately, never
    silently fall back (e.g. invalid API key, billing/quota exhausted, unsupported file format)."""


class SttPolicyRejectedError(Exception):
    """Raised by the policy layer before any provider is invoked. `status_code` is the HTTP
    status the caller should return; `detail` is a plain, user-safe message (no stack traces)."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
