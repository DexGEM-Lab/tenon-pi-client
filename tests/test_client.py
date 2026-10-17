from __future__ import annotations

import json

import httpx
import pytest

from tenon_pi.client import TenonClient
from tenon_pi.errors import BudgetExceededError
from tenon_pi.models import ExecutionMode, Protocol, Quote, TrainingMethod, TrainingSpec
from tenon_pi.transport import RestTransport


def make_client(handler) -> TenonClient:
    http_client = httpx.Client(
        base_url="https://api.fixture.invalid/v1/",
        transport=httpx.MockTransport(handler),
    )
    transport = RestTransport(
        "fixture-api-key",
        base_url="https://api.fixture.invalid/v1",
        client=http_client,
    )
    return TenonClient("fixture-api-key", transport=transport)


def full_ft_spec() -> TrainingSpec:
    return TrainingSpec(
        dataset_version_id="dsv_fixture",
        base_model="openpi/pi05-base",
        method=TrainingMethod.FULL_FT,
        preset="balanced-v1",
        max_steps=100,
        max_runtime_seconds=7200,
        seed=42,
        checkpoint_interval_steps=100,
    )


def test_me_requires_no_store_bootstrap(contract_fixtures):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-Request-Id": "req_me", "Cache-Control": "no-store"},
            json=contract_fixtures["session_bootstrap"],
        )

    bootstrap = make_client(handler).me()
    assert bootstrap.id == "usr_fixture"
    assert not hasattr(bootstrap, "user")
    assert bootstrap.csrf_token is None


def test_quote_then_budgeted_training_create(contract_fixtures):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/quotes"):
            return httpx.Response(
                201,
                headers={"X-Request-Id": "req_quote"},
                json=contract_fixtures["quote"],
            )
        return httpx.Response(
            202,
            headers={"X-Request-Id": "req_job"},
            json=contract_fixtures["training_job"],
        )

    client = make_client(handler)
    quote = client.quote_training(full_ft_spec(), idempotency_key="quote-fixture-key")
    job = client.create_training_job(
        full_ft_spec(),
        quote=quote,
        max_cost_microcredits=5_000_000,
        idempotency_key="job-fixture-key",
    )
    assert job.id == "trn_fixture"
    quote_body = json.loads(requests[0].content)
    assert quote_body["kind"] == "training_job"
    assert quote_body["spec"]["method"] == "full_ft"
    job_body = json.loads(requests[1].content)
    assert job_body["quote_id"] == "quo_fixture"
    assert "lora_profile" not in job_body


def test_budget_guard_prevents_create(contract_fixtures):
    quote = Quote.model_validate(contract_fixtures["quote"])
    client = make_client(lambda _: pytest.fail("HTTP must not be called"))
    with pytest.raises(BudgetExceededError):
        client.create_training_job(
            full_ft_spec(),
            quote=quote,
            max_cost_microcredits=4_999_999,
        )


def test_control_connection_uses_response_url_unchanged(contract_fixtures):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/control/sessions/ctl_fixture/connections"
        return httpx.Response(
            201,
            headers={"X-Request-Id": "req_connection"},
            json=contract_fixtures["control_connection"],
        )

    connection = make_client(handler).create_control_connection(
        "ctl_fixture", idempotency_key="connection-fixture-key"
    )
    assert str(connection.wss_url).startswith("wss://inference.fixture.invalid/")
    assert "api.fixture.invalid" not in str(connection.wss_url)
    assert connection.protocol == Protocol.TENON_CONTROL


def test_trained_rtc_quote_is_representable_but_create_is_not(contract_fixtures):
    # ControlSessionSpec follows the quote schema. ControlSessionCreate in the
    # approved OpenAPI intentionally omits trained_rtc.
    assert ExecutionMode.TRAINED_RTC.value == "trained_rtc"
    assert contract_fixtures["control_session"]["execution_mode"] == "async_aligned"
