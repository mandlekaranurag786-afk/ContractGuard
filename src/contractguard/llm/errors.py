"""Errors raised by language-model clients."""


class LLMError(Exception):
    """Base error for a language-model request that could not complete."""


class LLMQuotaError(LLMError):
    """A rate or quota limit remained after the client's retries."""


class LLMUnavailableError(LLMError):
    """An API failure other than an exhausted rate or quota limit."""


def exception_details(error: BaseException) -> str:
    """Retain exception types and underlying messages without traceback locals."""
    parts, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        parts.append(f"{type(error).__module__}.{type(error).__name__}: {error}")
        error = error.__cause__ or (
            error.__context__ if not error.__suppress_context__ else None
        )
    return "\nCaused by: ".join(parts)
