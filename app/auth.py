import secrets

from fastapi import Header, HTTPException

from app.config import settings

_clients: dict[str, str] | None = None


def _get_clients() -> dict[str, str]:
    global _clients
    if _clients is None:
        _clients = {}
        for pair in settings.allowed_clients.split(","):
            pair = pair.strip()
            if ":" in pair:
                cid, sec = pair.split(":", 1)
                if cid.strip() and sec.strip():
                    _clients[cid.strip()] = sec.strip()
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
    if expected is None or not secrets.compare_digest(expected.encode(), secret.encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return client_id


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    return validate_key(x_api_key)
