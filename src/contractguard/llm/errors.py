"""Errors raised by language-model clients."""


class LLMError(Exception):
    """Base error for a language-model request that could not complete."""


class LLMQuotaError(LLMError):
    """A rate or quota limit remained after the client's retries."""


class LLMUnavailableError(LLMError):
    """An API failure other than an exhausted rate or quota limit."""
