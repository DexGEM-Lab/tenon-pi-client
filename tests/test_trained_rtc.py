from __future__ import annotations

import copy
import json
from datetime import UTC, datetime

import httpx
import numpy as np
import pytest
from test_control import FakeSocket
from test_request_serialization import mock_client, validate_schema

from tenon_pi.control import ControlSocket, NativeWebSocketBackend, pack_message
from tenon_pi.data_fixture import example_profile
from tenon_pi.models import (
    Checkpoint,
    ControlConnection,
    ControlSessionSpec,
    DeploymentCreate,
    ExecutionMode,
    FallbackPolicy,
    LastAppliedAbsoluteTarget,
    Quote,
)
from tenon_pi.runtime import CommittedPrefix, InferenceRequest, Observation, RuntimeFault


def rtc_metadata(schema):
    return {
        "training_mode": "wan_rtc",
        "adapter_version": "pi05-prefix-indicator-v1",
        "conditioning": "known-prefix-indicator",
        "sampler": "held-prefix-euler",
        "loss": "per-example-tail-real-coordinate-mean",
        "max_prefix": schema.max_committed_prefix_steps,
        "action_horizon": schema.action_horizon_steps,
        "action_mask": [True] * 8,
        "pi05": True,
    }


def admitted_fixture(contract_fixtures):
    data = copy.deepcopy(contract_fixtures)
    schema = example_profile().robot_schema("sha256:" + "f" * 64)
    data["checkpoint"]["manifest"].update(
        training_mode="wan_rtc",
        effective_batch_size=8,
        rtc_config=rtc_metadata(schema),
        transform_digest=schema.transform_digest,
    )
    data["deployment"].update(
        execution_modes=["trained_rtc"], protocols=["tenon-control-v1"], state="ready"
    )
    data["control_session"].update(execution_mode="trained_rtc", protocol="tenon-control-v1")
    data["quote"]["kind"] = "control_session"
    data["capabilities"] = dict(
        contract_version="0.1.0",
        base_models=["openpi/pi05-base"],
        training_methods=["full_ft"],
        protocols=["tenon-control-v1"],
        execution_modes=["trained_rtc"],
        limits={},
    )
    return data, schema


def service(data, posts):
    def handler(request):
        path = request.url.path
        if request.method == "GET":
            key = {
                "/v1/checkpoints/ckp_fixture": "checkpoint",
                "/v1/deployments/dep_fixture": "deployment",
                "/v1/capabilities": "capabilities",
            }[path]
            return httpx.Response(200, headers={"X-Request-Id": "req_cap"}, json=data[key])
        body = json.loads(request.content)
        posts.append((path, body))
        schema, key, status = {
            "/v1/deployments": ("DeploymentCreate", "deployment", 202),
            "/v1/quotes": ("QuoteRequest", "quote", 201),
            "/v1/control/sessions": ("ControlSessionCreate", "control_session", 202),
        }[path]
        validate_schema(schema, body)
        return httpx.Response(status, headers={"X-Request-Id": "req_create"}, json=data[key])

    return mock_client(handler)


def session_spec():
    return ControlSessionSpec(
        deployment_id="dep_fixture",
        duration_seconds=300,
        protocol="tenon-control-v1",
        execution_mode="trained_rtc",
        fallback=FallbackPolicy(
            mode="drain_then_hold", hold_steps=2, hold_behavior=LastAppliedAbsoluteTarget()
        ),
    )


def test_checkpoint_and_server_capabilities_precede_trained_quote_create(contract_fixtures):
    data, schema = admitted_fixture(contract_fixtures)
    posts = []
    client = service(data, posts)
    request = DeploymentCreate(
        checkpoint_id="ckp_fixture",
        name="trained fixture",
        robot_schema=schema,
        default_prompt="task",
        protocols={"tenon-control-v1"},
        execution_modes={"trained_rtc"},
        fallback=session_spec().fallback,
        warm_idle_seconds=0,
    )
    client.create_deployment(request, idempotency_key="deployment-trained-fixture")
    quote = client.quote_control_session(session_spec(), idempotency_key="quote-trained-fixture")
    session = client.create_control_session(
        session_spec(),
        quote=quote,
        max_cost_microcredits=5_000_000,
        idempotency_key="session-trained-fixture",
    )
    assert session.execution_mode == ExecutionMode.TRAINED_RTC
    assert [p for p, b in posts] == ["/v1/deployments", "/v1/quotes", "/v1/control/sessions"]
    assert posts[-1][1]["execution_mode"] == "trained_rtc"
    assert posts[1][1]["spec"] == {k: v for k, v in posts[-1][1].items() if k != "quote_id"}


@pytest.mark.parametrize("operation", ["quote", "create"])
@pytest.mark.parametrize(
    "missing",
    [
        "standard",
        "metadata",
        "version",
        "capacity",
        "global_mode",
        "model",
        "deployment_mode",
        "schema",
        "artifacts",
    ],
)
def test_no_trained_mutation_without_checkpoint_and_endpoint_qualification(
    contract_fixtures, missing, operation
):
    data, _schema = admitted_fixture(contract_fixtures)
    posts = []
    if missing == "standard":
        data["checkpoint"]["manifest"]["training_mode"] = "standard"
    elif missing == "metadata":
        data["checkpoint"]["manifest"]["rtc_config"] = None
    elif missing == "version":
        data["checkpoint"]["manifest"]["rtc_config"]["adapter_version"] = ""
    elif missing == "capacity":
        data["checkpoint"]["manifest"]["rtc_config"]["max_prefix"] = True
    elif missing == "global_mode":
        data["capabilities"]["execution_modes"] = ["guided_rtc"]
    elif missing == "model":
        data["capabilities"]["base_models"] = ["another/model"]
    elif missing == "deployment_mode":
        data["deployment"]["execution_modes"] = ["async_aligned"]
    elif missing == "schema":
        data["deployment"]["schema_hash"] = "sha256:" + "0" * 64
    else:
        data["checkpoint"]["manifest"]["artifacts"] = []
    client = service(data, posts)
    with pytest.raises(ValueError):
        if operation == "quote":
            client.quote_control_session(session_spec())
        else:
            client.create_control_session(
                session_spec(),
                quote=Quote.model_validate(data["quote"]),
                max_cost_microcredits=5_000_000,
            )
    assert not posts


def test_trained_metadata_shape_must_match_deployment_schema(contract_fixtures):
    data, schema = admitted_fixture(contract_fixtures)
    data["checkpoint"]["manifest"]["rtc_config"]["action_horizon"] += 1
    with pytest.raises(ValueError, match="incompatible"):
        Checkpoint.model_validate(data["checkpoint"]).require_trained_rtc(schema)


@pytest.mark.parametrize("conditioning", ["trained", "guided"])
def test_trained_codec_keeps_prefix_and_rejects_relabeling(conditioning):
    prefix = CommittedPrefix.from_array(0, np.array([[0.25]], np.float32))
    request = InferenceRequest(
        0,
        0,
        Observation(
            0,
            -1,
            (0.0,),
            ({"camera_id": "base", "encoding": "png", "data": b"fixture"},),
            "task",
            0,
        ),
        0,
        1,
        3,
        4,
        500,
        prefix,
    )
    response = dict(
        type="actions",
        epoch=0,
        request_index=0,
        based_on_observation_index=0,
        start_action_index=0,
        valid_through_action_index_exclusive=2,
        actions=[[0.25], [0.5]],
        generation=dict(
            mode="trained_rtc",
            prefix_consumed_count=1,
            committed_prefix_sha256=prefix.sha256,
            sampler_conditioning=conditioning,
        ),
    )
    fake = FakeSocket([pack_message(response)])
    connection = ControlConnection(
        wss_url="wss://fixture.invalid/control",
        token="fixture-token",
        protocol="tenon-control-v1",
        expires_at=datetime.now(UTC),
    )
    socket = ControlSocket(connection, connector=lambda *a, **kw: fake)
    backend = NativeWebSocketBackend(
        socket,
        expected_mode=ExecutionMode.TRAINED_RTC,
        action_dim=1,
        max_committed_prefix_steps=1,
        authoritative_epoch=0,
    )
    if conditioning == "guided":
        with pytest.raises(RuntimeFault):
            backend.infer(request)
    else:
        result = backend.infer(request)
        assert result.sampler_conditioning == "trained"
    import msgpack

    sent = msgpack.unpackb(fake.sent[0], raw=False)
    assert sent["committed_prefix"] == prefix.to_wire(1)
