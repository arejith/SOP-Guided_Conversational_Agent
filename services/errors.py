"""Recoverable model output errors, separate from programming errors."""


class ModelResponseError(ValueError):
    """The model returned malformed, incomplete, or invalid output."""
