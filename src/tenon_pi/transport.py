"""HTTP transport that enforces Tenon auth, idempotency, and error envelopes."""

from __future__ import annotations

import json
import secrets
import string
from collections.abc import Mapping
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ._version import __version__
from .errors import TenonProtocolError, TenonTransportError, error_type_for_status
from .models import ErrorEnvelope

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "https://tenon.dex-gem.ai/v1"
_VISIBLE_ASCII = set(string.printable) - set("\t\n\r\x0b\x0c")


def new_idempotency_key(prefix: str = "sdk") -> str:
    """Return a visible-ASCII key inside the contract's 8-128 character bound."""

    return f"{prefix}-{secrets.token_urlsafe(18)}"


def validate_idempotency_key(value: str) -> str:
    if not 8 <= len(value) <= 128:
        raise ValueError("Idempotency-Key must be 8-128 visible ASCII characters")
    if any(character not in _VISIBLE_ASCII for character in value):
        raise ValueError("Idempotency-Key must contain only visible ASCII characters")
    return value


def _wire_exclusions(value: Any) -> dict[Any, Any]:
    """Build recursive exclusions without collapsing absent and explicit null."""

    if isinstance(value, BaseModel):
        exclusions: dict[Any, Any] = {}
        required_defaults: frozenset[str] = getattr(
            type(value), "wire_required_defaults", frozenset()
        )
        for name, field in type(value).model_fields.items():
            if (
                name not in value.model_fields_set
                and not field.is_required()
                and name not in required_defaults
            ):
                exclusions[name] = True
                continue
            nested = _wire_exclusions(getattr(value, name))
            if nested:
                exclusions[name] = nested
        return exclusions
    if isinstance(value, Mapping):
        return {key: nested for key, item in value.items() if (nested := _wire_exclusions(item))}
    if isinstance(value, (list, tuple)):
        return {
            index: nested for index, item in enumerate(value) if (nested := _wire_exclusions(item))
        }
    return {}


def serialize_request_body(body: BaseModel) -> dict[str, Any]:
    """Serialize one request while preserving OpenAPI field-presence semantics.

    Required nullable fields remain present as JSON null. Optional fields are
    omitted when unset, while an explicitly supplied null remains present. A
    small amount of model metadata retains ergonomic SDK defaults that the
    frozen wire schema marks required.
    """

    return body.model_dump(mode="json", exclude=_wire_exclusions(body))


class RestTransport:
    """Synchronous REST transport.

    It intentionally doesn't retry mutating requests. If a timeout occurs, the
    raised :class:`TenonTransportError` carries the idempotency key for an exact
    caller-controlled retry.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float | httpx.Timeout = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=f"{self.base_url}/",
            timeout=timeout,
            follow_redirects=False,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": f"tenon-pi-client/{__version__}",
            },
        )
        self.last_request_id: str | None = None

    def __enter__(self) -> RestTransport:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        response_model: type[T],
        expected_status: int | set[int],
        body: BaseModel | Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        if_match: int | str | None = None,
        required_response_headers: Mapping[str, str] | None = None,
    ) -> T:
        expected = {expected_status} if isinstance(expected_status, int) else expected_status
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "User-Agent": f"tenon-pi-client/{__version__}",
        }
        if idempotency_key is not None:
            headers["Idempotency-Key"] = validate_idempotency_key(idempotency_key)
        if if_match is not None:
            version = str(if_match).strip('"')
            headers["If-Match"] = f'"{version}"'
        payload: Mapping[str, Any] | None
        payload = serialize_request_body(body) if isinstance(body, BaseModel) else body
        try:
            response = self._client.request(
                method,
                path.lstrip("/"),
                json=payload,
                params={key: value for key, value in (params or {}).items() if value is not None},
                headers=headers,
            )
        except httpx.RequestError as exc:
            raise TenonTransportError(
                f"request failed before an HTTP response: {exc}",
                idempotency_key=idempotency_key,
            ) from exc

        header_request_id = response.headers.get("X-Request-Id")
        if not header_request_id:
            raise TenonProtocolError("Tenon response is missing X-Request-Id")
        self.last_request_id = header_request_id
        for name, expected_value in (required_response_headers or {}).items():
            if response.headers.get(name) != expected_value:
                raise TenonProtocolError(
                    f"Tenon response requires {name}: {expected_value!r} "
                    f"(request_id={header_request_id})"
                )

        try:
            data = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TenonProtocolError(
                "Tenon returned non-JSON content "
                f"(status={response.status_code}, request_id={header_request_id})"
            ) from exc

        if response.status_code not in expected:
            try:
                envelope = ErrorEnvelope.model_validate(data)
            except ValidationError as exc:
                raise TenonProtocolError(
                    f"Tenon error response has an invalid envelope "
                    f"(status={response.status_code}, request_id={header_request_id})"
                ) from exc
            error = envelope.error
            if error.request_id != header_request_id:
                raise TenonProtocolError("error request_id doesn't match X-Request-Id")
            error_type = error_type_for_status(response.status_code)
            raise error_type(
                status_code=response.status_code,
                code=error.code,
                message=error.message,
                request_id=error.request_id,
                retryable=error.retryable,
                details=error.details,
                retry_after=response.headers.get("Retry-After"),
                idempotency_key=idempotency_key,
            )

        try:
            return response_model.model_validate(data)
        except ValidationError as exc:
            raise TenonProtocolError(
                f"Tenon success response violates {response_model.__name__} "
                f"(request_id={header_request_id}): {exc}"
            ) from exc
