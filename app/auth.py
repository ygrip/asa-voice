import secrets

from fastapi import Header, HTTPException

from app.config import settings

# Provider-override trust tiers (setara-s94o.8, plan §6.2). "production" is the safe default for
# any client not explicitly configured otherwise — override is opt-in per client, never opt-out.
DEFAULT_CLIENT_TIER = "production"
OVERRIDE_ALLOWED_TIERS = {"development", "admin", "test"}

_clients: dict[str, dict[str, str]] | None = None


def _get_clients() -> dict[str, dict[str, str]]:
    """Parse ALLOWED_CLIENTS ("client_id:secret" or "client_id:secret:tier", comma-separated).
    tier defaults to DEFAULT_CLIENT_TIER when omitted, so existing 2-field entries keep working
    unchanged and stay locked out of provider overrides unless explicitly upgraded."""
    global _clients
    if _clients is None:
        _clients = {}
        for entry in settings.allowed_clients.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) < 2:
                continue
            cid, sec = parts[0].strip(), parts[1].strip()
            tier = parts[2].strip().lower() if len(parts) > 2 and parts[2].strip() else DEFAULT_CLIENT_TIER
            if cid and sec:
                _clients[cid] = {"secret": sec, "tier": tier}
    return _clients


def validate_key(api_key: str | None) -> str:
    """Validate an API key string (client_id:secret). Returns client_id on success.
    Raises HTTPException on failure. If no clients configured, allows all (open dev mode)."""
    clients = _get_clients()
    if not clients:
        return "anonymous"
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    try:
        client_id, secret = api_key.split(":", 1)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid API key format")
    expected = clients.get(client_id)
    if expected is None or not secrets.compare_digest(expected["secret"].encode(), secret.encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return client_id


def get_client_tier(client_id: str) -> str:
    """Trust tier used for provider-override decisions. Open dev mode (no ALLOWED_CLIENTS
    configured, i.e. every request is already unauthenticated local dev) is treated as
    "development"; otherwise unknown/unconfigured clients get the safe "production" default."""
    clients = _get_clients()
    if not clients:
        return "development"
    record = clients.get(client_id)
    return record["tier"] if record else DEFAULT_CLIENT_TIER


def provider_override_allowed(client_id: str) -> bool:
    return get_client_tier(client_id) in OVERRIDE_ALLOWED_TIERS


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    return validate_key(x_api_key)
