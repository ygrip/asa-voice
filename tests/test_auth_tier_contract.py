"""Unit tests for client trust tiers used by provider-override decisions (setara-s94o.8)."""
from app import auth


def _reset_client_cache() -> None:
    auth._clients = None


def test_open_dev_mode_defaults_to_development_tier() -> None:
    from app.config import settings

    original = settings.allowed_clients
    settings.allowed_clients = ""
    _reset_client_cache()
    try:
        assert auth.get_client_tier("anyone") == "development"
        assert auth.provider_override_allowed("anyone") is True
    finally:
        settings.allowed_clients = original
        _reset_client_cache()


def test_configured_client_without_tier_defaults_to_production() -> None:
    from app.config import settings

    original = settings.allowed_clients
    settings.allowed_clients = "setara-ui-local:secret"
    _reset_client_cache()
    try:
        assert auth.get_client_tier("setara-ui-local") == "production"
        assert auth.provider_override_allowed("setara-ui-local") is False
    finally:
        settings.allowed_clients = original
        _reset_client_cache()


def test_client_with_explicit_dev_tier_allows_override() -> None:
    from app.config import settings

    original = settings.allowed_clients
    settings.allowed_clients = "setara-ui-local:secret:development,setara-core:secret2:production"
    _reset_client_cache()
    try:
        assert auth.get_client_tier("setara-ui-local") == "development"
        assert auth.provider_override_allowed("setara-ui-local") is True
        assert auth.get_client_tier("setara-core") == "production"
        assert auth.provider_override_allowed("setara-core") is False
    finally:
        settings.allowed_clients = original
        _reset_client_cache()


def test_unknown_client_defaults_to_production_when_clients_configured() -> None:
    from app.config import settings

    original = settings.allowed_clients
    settings.allowed_clients = "setara-ui-local:secret:admin"
    _reset_client_cache()
    try:
        assert auth.get_client_tier("someone-else") == "production"
        assert auth.provider_override_allowed("someone-else") is False
    finally:
        settings.allowed_clients = original
        _reset_client_cache()


def test_validate_key_still_works_after_tier_parsing() -> None:
    from app.config import settings

    original = settings.allowed_clients
    settings.allowed_clients = "setara-ui-local:s3cr3t:admin"
    _reset_client_cache()
    try:
        assert auth.validate_key("setara-ui-local:s3cr3t") == "setara-ui-local"
    finally:
        settings.allowed_clients = original
        _reset_client_cache()
