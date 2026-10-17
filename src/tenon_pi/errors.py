"""Stable client exceptions for Tenon REST and control transport failures."""

from __future__ import annotations

from typing import Any


class TenonError(Exception):
    """Base class for public SDK errors."""


class TenonTransportError(TenonError):
    """The request didn't produce an HTTP response.

    For a mutating request, retry with ``idempotency_key`` from this exception.
    """

    def __init__(self, message: str, *, idempotency_key: str | None = None) -> None:
        super().__init__(message)
        self.idempotency_key = idempotency_key


class TenonProtocolError(TenonError):
    """The peer returned a response that doesn't match the public contract."""


class TenonAPIError(TenonError):
    """A structured non-success response from Tenon."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        request_id: str,
        retryable: bool,
        details: dict[str, Any] | None = None,
        retry_after: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        super().__init__(f"{code}: {message} (request_id={request_id})")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        self.retryable = retryable
        self.details = details or {}
        self.retry_after = retry_after
        self.idempotency_key = idempotency_key


class AuthenticationError(TenonAPIError):
    """Authentication or credential rejection."""


class InsufficientCreditsError(TenonAPIError):
    """The billable resource couldn't reserve its quoted maximum."""


class VersionConflictError(TenonAPIError):
    """The If-Match resource version is stale or missing."""


class RateLimitError(TenonAPIError):
    """The request was rate limited; inspect ``retry_after``."""


class DatasetRejectedError(TenonError):
    """Dataset validation reached the rejected terminal state."""

    def __init__(self, version_id: str, findings: list[object]) -> None:
        super().__init__(f"dataset version {version_id} was rejected")
        self.version_id = version_id
        self.findings = findings


class PollTimeoutError(TenonError):
    """A resource didn't reach a requested terminal state before the local timeout."""


class BudgetExceededError(TenonError):
    """A quote exceeds the caller's explicit microcredit budget."""

    def __init__(self, *, quoted_microcredits: int, budget_microcredits: int) -> None:
        super().__init__(
            f"quote maximum {quoted_microcredits} microcredits exceeds budget {budget_microcredits}"
        )
        self.quoted_microcredits = quoted_microcredits
        self.budget_microcredits = budget_microcredits


_STATUS_ERROR_TYPES: dict[int, type[TenonAPIError]] = {
    401: AuthenticationError,
    402: InsufficientCreditsError,
    412: VersionConflictError,
    428: VersionConflictError,
    429: RateLimitError,
}


def error_type_for_status(status_code: int) -> type[TenonAPIError]:
    return _STATUS_ERROR_TYPES.get(status_code, TenonAPIError)
