from __future__ import annotations

import json

import httpx
import pytest

from tenon_pi.errors import (
    InsufficientCreditsError,
    TenonProtocolError,
    TenonTransportError,
)
from tenon_pi.models import Dataset, DatasetCreate
from tenon_pi.transport import RestTransport, validate_idempotency_key


def make_transport(handler) -> RestTransport:
    client = httpx.Client(
        base_url="https://api.fixture.invalid/v1/",
        transport=httpx.MockTransport(handler),
    )
    return RestTransport(
        "fixture-api-key",
        base_url="https://api.fixture.invalid/v1",
        client=client,
    )


def test_auth_idempotency_if_match_and_success_validation(contract_fixtures):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            headers={"X-Request-Id": "req_success"},
            json=contract_fixtures["dataset"],
        )

    transport = make_transport(handler)
    dataset = transport.request(
        "POST",
        "/datasets",
        response_model=Dataset,
        expected_status=201,
        body=DatasetCreate(name="fixture", description="test"),
        idempotency_key="dataset-fixture-key",
        if_match=3,
    )
    assert dataset.id == "dst_fixture"
    assert seen["method"] == "POST"
    assert seen["url"] == "https://api.fixture.invalid/v1/datasets"
    assert seen["headers"]["authorization"] == "Bearer fixture-api-key"
    assert seen["headers"]["idempotency-key"] == "dataset-fixture-key"
    assert seen["headers"]["if-match"] == '"3"'
    assert seen["body"] == {"name": "fixture", "description": "test"}
    assert transport.last_request_id == "req_success"


def test_structured_error_mapping(contract_fixtures):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            headers={"X-Request-Id": "req_fixture"},
            json=contract_fixtures["error"],
        )

    transport = make_transport(handler)
    with pytest.raises(InsufficientCreditsError) as raised:
        transport.request(
            "POST",
            "datasets",
            response_model=Dataset,
            expected_status=201,
            body=DatasetCreate(name="fixture"),
            idempotency_key="dataset-fixture-key",
        )
    error = raised.value
    assert error.status_code == 402
    assert error.code == "insufficient_credits"
    assert error.request_id == "req_fixture"
    assert error.idempotency_key == "dataset-fixture-key"


def test_transport_failure_preserves_idempotency_key():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("fixture network failure", request=request)

    transport = make_transport(handler)
    with pytest.raises(TenonTransportError) as raised:
        transport.request(
            "POST",
            "datasets",
            response_model=Dataset,
            expected_status=201,
            body=DatasetCreate(name="fixture"),
            idempotency_key="dataset-fixture-key",
        )
    assert raised.value.idempotency_key == "dataset-fixture-key"


def test_missing_request_id_is_protocol_error(contract_fixtures):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=contract_fixtures["dataset"])

    transport = make_transport(handler)
    with pytest.raises(TenonProtocolError, match="X-Request-Id"):
        transport.request(
            "GET", "datasets/dst_fixture", response_model=Dataset, expected_status=200
        )


def test_idempotency_key_validation():
    assert validate_idempotency_key("visible-123") == "visible-123"
    with pytest.raises(ValueError):
        validate_idempotency_key("short")
    with pytest.raises(ValueError):
        validate_idempotency_key("valid-but-has-newline\n")
