"""Unit tests for the policy layer v1 (setara-s94o.9): request validation + in-memory daily quota."""
import pytest

from app.providers.base import IN_MEMORY_AUDIO_MARKER, SttOptions
from app.providers.errors import SttPolicyRejectedError
from app.providers.policy import InMemoryDailyQuotaStore, RequestValidationPolicy


def test_rejects_missing_client_id() -> None:
    policy = RequestValidationPolicy()

    with pytest.raises(SttPolicyRejectedError) as exc_info:
        policy.validate_audio(IN_MEMORY_AUDIO_MARKER, SttOptions(client_id=None), duration_seconds=1.0)

    assert exc_info.value.status_code == 401


def test_rejects_unsupported_file_extension(tmp_path) -> None:
    path = tmp_path / "sample.xyz"
    path.write_bytes(b"data")
    policy = RequestValidationPolicy()

    with pytest.raises(SttPolicyRejectedError) as exc_info:
        policy.validate_audio(str(path), SttOptions(client_id="c1"), duration_seconds=1.0)

    assert exc_info.value.status_code == 415


def test_rejects_oversized_file(tmp_path) -> None:
    from app.config import settings

    path = tmp_path / "sample.wav"
    path.write_bytes(b"0" * 1024)
    original_limit = settings.max_upload_mb
    settings.max_upload_mb = 0  # anything is "too big"
    try:
        policy = RequestValidationPolicy()
        with pytest.raises(SttPolicyRejectedError) as exc_info:
            policy.validate_audio(str(path), SttOptions(client_id="c1"), duration_seconds=1.0)
        assert exc_info.value.status_code == 413
    finally:
        settings.max_upload_mb = original_limit


def test_rejects_audio_over_max_seconds_per_request(tmp_path) -> None:
    from app.config import settings

    path = tmp_path / "sample.wav"
    path.write_bytes(b"data")
    policy = RequestValidationPolicy()

    with pytest.raises(SttPolicyRejectedError) as exc_info:
        policy.validate_audio(
            str(path), SttOptions(client_id="c1"),
            duration_seconds=settings.max_stt_seconds_per_request + 1,
        )

    assert exc_info.value.status_code == 413


def test_quota_increments_and_rejects_when_exceeded() -> None:
    from app.config import settings

    original = settings.max_stt_seconds_per_client_per_day
    settings.max_stt_seconds_per_client_per_day = 10
    try:
        policy = RequestValidationPolicy()
        options = SttOptions(client_id="quota-client")

        policy.validate_audio(IN_MEMORY_AUDIO_MARKER, options, duration_seconds=1.0)  # 0 used, OK
        policy.record_usage(options, 6.0)
        policy.validate_audio(IN_MEMORY_AUDIO_MARKER, options, duration_seconds=1.0)  # 6 < 10, OK
        policy.record_usage(options, 6.0)  # now 12 >= 10

        with pytest.raises(SttPolicyRejectedError) as exc_info:
            policy.validate_audio(IN_MEMORY_AUDIO_MARKER, options, duration_seconds=1.0)
        assert exc_info.value.status_code == 429
    finally:
        settings.max_stt_seconds_per_client_per_day = original


def test_record_usage_ignores_missing_client_id() -> None:
    store = InMemoryDailyQuotaStore()
    policy = RequestValidationPolicy(quota_store=store)
    policy.record_usage(SttOptions(client_id=None), 5.0)
    assert store.used_seconds("") == 0.0


def test_quota_store_resets_by_day() -> None:
    store = InMemoryDailyQuotaStore()
    store.record("c1", 100.0)
    assert store.used_seconds("c1") == 100.0

    store._usage["c1"] = ("2000-01-01", 100.0)  # force a stale day
    assert store.used_seconds("c1") == 0.0
