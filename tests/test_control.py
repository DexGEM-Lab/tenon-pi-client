from __future__ import annotations

import numpy as np
import pytest

from tenon_pi.control import (
    ControlSocket,
    NativeWebSocketBackend,
    pack_message,
    unpack_message,
)
from tenon_pi.errors import TenonProtocolError
from tenon_pi.models import ControlConnection, ExecutionMode, Protocol
from tenon_pi.runtime import CommittedPrefix, InferenceRequest, Observation


class FakeSocket:
    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent = []
        self.closed = None

    def send(self, message):
        self.sent.append(message)

    def recv(self, timeout=None):
        assert timeout is not None
        return self.incoming.pop(0)

    def close(self, code=1000, reason=""):
        self.closed = (code, reason)


def connection(protocol: Protocol) -> ControlConnection:
    return ControlConnection(
        wss_url="wss://negotiated.fixture.invalid/session/ctl_fixture",
        token="fixture-connection-token",
        expires_at="2026-10-14T12:01:00Z",
        protocol=protocol,
    )


def test_native_connection_uses_negotiated_url_and_bearer_header():
    hello = {
        "type": "hello",
        "protocol": "tenon-control-v1",
        "session_id": "ctl_fixture",
        "epoch": 0,
        "schema_hash": "sha256:schema",
        "execution_mode": "async_aligned",
        "rearm_required": False,
        "limits": {
            "max_request_bytes": 2_000_000,
            "max_committed_prefix_steps": 5,
            "max_in_flight": 1,
        },
    }
    fake = FakeSocket([pack_message(hello)])
    connect_args = {}

    def connector(url, **kwargs):
        connect_args["url"] = url
        connect_args.update(kwargs)
        return fake

    socket = ControlSocket(connection(Protocol.TENON_CONTROL), connector=connector)
    result = socket.native_handshake(
        schema_hash="sha256:schema", expected_mode=ExecutionMode.ASYNC_ALIGNED
    )
    assert result == hello
    assert connect_args["url"] == "wss://negotiated.fixture.invalid/session/ctl_fixture"
    assert connect_args["additional_headers"] == {
        "Authorization": "Bearer fixture-connection-token"
    }
    assert unpack_message(fake.sent[0]) == {
        "type": "ready",
        "epoch": 0,
        "schema_hash": "sha256:schema",
    }


def test_guided_handshake_is_explicit_and_mode_mismatch_closes():
    hello = {
        "type": "hello",
        "protocol": "tenon-control-v1",
        "session_id": "ctl_fixture",
        "epoch": 3,
        "schema_hash": "sha256:schema",
        "execution_mode": "guided_rtc",
        "rearm_required": False,
        "limits": {
            "max_request_bytes": 2_000_000,
            "max_committed_prefix_steps": 5,
            "max_in_flight": 1,
        },
    }
    accepted = FakeSocket([pack_message(hello)])
    socket = ControlSocket(
        connection(Protocol.TENON_CONTROL), connector=lambda *_args, **_kwargs: accepted
    )
    assert (
        socket.native_handshake(
            schema_hash="sha256:schema", expected_mode=ExecutionMode.GUIDED_RTC
        )["execution_mode"]
        == "guided_rtc"
    )

    mismatched = FakeSocket([pack_message(hello)])
    socket = ControlSocket(
        connection(Protocol.TENON_CONTROL), connector=lambda *_args, **_kwargs: mismatched
    )
    with pytest.raises(TenonProtocolError, match="mode"):
        socket.native_handshake(
            schema_hash="sha256:schema", expected_mode=ExecutionMode.ASYNC_ALIGNED
        )
    assert mismatched.closed == (4409, "execution mode mismatch")


def test_native_handshake_rejects_trained_rtc_before_consuming_hello():
    fake = FakeSocket([])
    socket = ControlSocket(
        connection(Protocol.TENON_CONTROL), connector=lambda *_args, **_kwargs: fake
    )
    with pytest.raises(ValueError, match="supports only"):
        socket.native_handshake(
            schema_hash="sha256:schema", expected_mode=ExecutionMode.TRAINED_RTC
        )


def test_schema_mismatch_closes_with_4409():
    hello = {
        "type": "hello",
        "protocol": "tenon-control-v1",
        "session_id": "ctl_fixture",
        "epoch": 0,
        "schema_hash": "sha256:other",
        "execution_mode": "async_aligned",
        "rearm_required": False,
        "limits": {},
    }
    fake = FakeSocket([pack_message(hello)])
    socket = ControlSocket(
        connection(Protocol.TENON_CONTROL), connector=lambda *_args, **_kwargs: fake
    )
    try:
        socket.native_handshake(
            schema_hash="sha256:expected", expected_mode=ExecutionMode.ASYNC_ALIGNED
        )
    except TenonProtocolError:
        pass
    else:
        raise AssertionError("expected TenonProtocolError")
    assert fake.closed == (4409, "schema mismatch")


def test_native_backend_emits_indexed_infer_and_accepts_epoch_bound_to_socket():
    response = {
        "type": "actions",
        "request_index": 4,
        "based_on_observation_index": 9,
        "start_action_index": 5,
        "valid_through_action_index_exclusive": 7,
        "actions": [[5.0, 1.0], [6.0, 1.0]],
        "generation": {"mode": "async_aligned"},
    }
    fake = FakeSocket([pack_message(response)])
    socket = ControlSocket(
        connection(Protocol.TENON_CONTROL), connector=lambda *_args, **_kwargs: fake
    )
    backend = NativeWebSocketBackend(
        socket,
        expected_mode=ExecutionMode.ASYNC_ALIGNED,
        action_dim=2,
        max_committed_prefix_steps=5,
        authoritative_epoch=2,
    )
    observation = Observation(
        9,
        4,
        (0.0,),
        ({"camera_id": "base", "encoding": "jpeg", "data": b"x"},),
        "task",
        123,
    )
    request = InferenceRequest(2, 4, observation, 5, 6, 7, 8, 100)
    result = backend.infer(request)
    assert result.epoch == 2
    sent = unpack_message(fake.sent[0])
    assert sent["epoch"] == 2
    assert sent["target_start_action_index"] == 5
    assert sent["committed_end_action_index_exclusive"] == 6
    assert "committed_prefix" not in sent


def test_guided_backend_emits_exact_existing_prefix_field_and_preserves_evidence():
    prefix = CommittedPrefix.from_array(5, np.asarray([[5.0, 1.0]], dtype=np.float32))
    response = {
        "type": "actions",
        "epoch": 2,
        "request_index": 4,
        "based_on_observation_index": 9,
        "start_action_index": 5,
        "valid_through_action_index_exclusive": 7,
        "actions": [[5.0, 1.0], [6.0, 1.0]],
        "generation": {
            "mode": "guided_rtc",
            "prefix_consumed_count": 1,
            "committed_prefix_sha256": prefix.sha256,
            "sampler_conditioning": "guided",
        },
    }
    fake = FakeSocket([pack_message(response)])
    socket = ControlSocket(
        connection(Protocol.TENON_CONTROL), connector=lambda *_args, **_kwargs: fake
    )
    backend = NativeWebSocketBackend(
        socket,
        expected_mode=ExecutionMode.GUIDED_RTC,
        action_dim=2,
        max_committed_prefix_steps=5,
        authoritative_epoch=2,
    )
    observation = Observation(
        9,
        4,
        (0.0,),
        ({"camera_id": "base", "encoding": "jpeg", "data": b"x"},),
        "task",
        123,
    )
    request = InferenceRequest(2, 4, observation, 5, 6, 7, 8, 100, prefix)
    result = backend.infer(request)
    sent = unpack_message(fake.sent[0])
    assert sent["committed_prefix"] == prefix.to_wire(2)
    assert result.generation_mode == "guided_rtc"
    assert result.prefix_consumed_count == 1
    assert result.committed_prefix_sha256 == prefix.sha256
    assert result.sampler_conditioning == "guided"


def test_openpi_connection_uses_api_key_header_and_metadata_first():
    fake = FakeSocket([pack_message({"model": "fixture"})])
    args = {}

    def connector(url, **kwargs):
        args.update(kwargs)
        return fake

    socket = ControlSocket(connection(Protocol.OPENPI_WS), connector=connector)
    assert socket.receive_openpi_metadata() == {"model": "fixture"}
    assert args["additional_headers"] == {"Authorization": "Api-Key fixture-connection-token"}
