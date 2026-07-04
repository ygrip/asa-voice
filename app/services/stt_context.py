from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import settings


class SttDecodeContext(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    language: str | None = Field(default=None, max_length=16)
    prompt: str | None = Field(default=None, max_length=1000)
    hotwords: list[str] = Field(default_factory=list, max_length=100)
    request_id: str | None = Field(default=None, alias="requestId", max_length=128)

    @field_validator("hotwords")
    @classmethod
    def validate_hotwords(cls, words: list[str]) -> list[str]:
        cleaned = [word.strip() for word in words if word.strip()]
        if any(len(word) > 100 for word in cleaned):
            raise ValueError("hotwords must not exceed 100 characters each")
        return cleaned


def build_prompt(context: SttDecodeContext | None) -> str | None:
    parts = [settings.stt_prompt.strip()]
    if context is not None and context.prompt:
        parts.append(context.prompt.strip())
    prompt = "\n".join(part for part in parts if part)
    return prompt or None


def build_hotwords(context: SttDecodeContext | None) -> str | None:
    words = settings.stt_hotwords.split()
    if context is not None:
        words.extend(context.hotwords)
    deduped = list(dict.fromkeys(word.strip() for word in words if word.strip()))
    return " ".join(deduped) or None


def resolve_language(context: SttDecodeContext | None, override: str | None = None) -> str | None:
    if override:
        return override
    if context is not None and context.language:
        return context.language
    return settings.stt_language or None
