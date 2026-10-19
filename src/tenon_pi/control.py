"""Negotiated control WebSocket connection primitives.

The local scheduling/runtime semantics are intentionally a later layer. This
module only opens the URL returned by REST and performs the native handshake.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast
from typing import Protocol as TypingProtocol

import msgpack  # type: ignore[import-untyped]
from websockets.sync.client import connect as websocket_connect

from .errors import TenonProtocolError
from .models import ControlConnection, ExecutionMode, Protocol
from .runtime import (
    PREFIX_CONDITIONED_MODES,
    ActionResponse,
    InferenceRequest,
    Observation,
    RuntimeFault,
    require_aligned_mode,
)


class WebSocketLike(TypingProtocol):
    def send(self, message: bytes | str) -> None: ...

    def recv(self, timeout: float | None = None) -> bytes | str: ...

    def close(self, code: int = 1000, reason: str = "") -> None: ...


Connector = Callable[..., WebSocketLike]


def pack_message(message: Mapping[str, Any]) -> bytes:
    return cast(bytes, msgpack.packb(dict(message), use_bin_type=True))


def unpack_message(payload: bytes | str) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TenonProtocolError("control protocol requires binary MessagePack frames")
    try:
        message = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise TenonProtocolError("invalid MessagePack control frame") from exc
    if not isinstance(message, dict) or not all(isinstance(key, str) for key in message):
        raise TenonProtocolError("control frame must be a string-keyed map")
    return message


class ControlSocket:
    """A connected socket using the opaque negotiated ``wss_url`` verbatim."""

    def __init__(
        self,
        connection: ControlConnection,
        *,
        connector: Connector = websocket_connect,
        open_timeout: float = 10,
    ) -> None:
        token = connection.token.get_secret_value()
        authorization = (
            f"Bearer {token}"
            if connection.protocol == Protocol.TENON_CONTROL
            else f"Api-Key {token}"
        )
        # Never inspect or combine this URL with the REST base URL.
        self.url = str(connection.wss_url)
        self.protocol = connection.protocol
        self._socket = connector(
            self.url,
            additional_headers={"Authorization": authorization},
            compression=None,
            max_size=None,
            open_timeout=open_timeout,
        )

    def __enter__(self) -> ControlSocket:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self, code: int = 1000, reason: str = "") -> None:
        self._socket.close(code=code, reason=reason)

    def receive_map(self, *, timeout: float | None = None) -> dict[str, Any]:
        return unpack_message(self._socket.recv(timeout=timeout))

    def send_map(self, message: Mapping[str, Any]) -> None:
        self._socket.send(pack_message(message))

    def native_handshake(
        self,
        *,
        schema_hash: str,
        expected_mode: ExecutionMode,
        timeout: float = 10,
    ) -> dict[str, Any]:
        if self.protocol != Protocol.TENON_CONTROL:
            raise ValueError("native_handshake is only valid for tenon-control-v1")
        mode = require_aligned_mode(expected_mode)
        hello = self.receive_map(timeout=timeout)
        required = {
            "type",
            "protocol",
            "session_id",
            "epoch",
            "schema_hash",
            "execution_mode",
            "rearm_required",
            "limits",
        }
        if not required.issubset(hello):
            raise TenonProtocolError(f"hello is missing fields: {sorted(required - set(hello))}")
        if hello["type"] != "hello" or hello["protocol"] != Protocol.TENON_CONTROL:
            raise TenonProtocolError("expected tenon-control-v1 hello")
        if hello["schema_hash"] != schema_hash:
            self.close(code=4409, reason="schema mismatch")
            raise TenonProtocolError("control hello schema_hash mismatch")
        if hello["execution_mode"] != mode.value:
            self.close(code=4409, reason="execution mode mismatch")
            raise TenonProtocolError(
                f"control hello mode {hello['execution_mode']!r} != expected {mode.value!r}"
            )
        limits = hello["limits"]
        if (
            not isinstance(limits, dict)
            or limits.get("max_in_flight") != 1
            or type(limits.get("max_request_bytes")) is not int
            or limits["max_request_bytes"] < 1
            or type(limits.get("max_committed_prefix_steps")) is not int
            or limits["max_committed_prefix_steps"] < 0
        ):
            raise TenonProtocolError("control hello limits violate the one-in-flight contract")
        epoch = hello["epoch"]
        if not isinstance(epoch, int) or epoch < 0:
            raise TenonProtocolError("control hello epoch must be a non-negative integer")
        self.send_map(
            {
                "type": "ready",
                "epoch": epoch,
                "schema_hash": schema_hash,
            }
        )
        return hello

    def receive_openpi_metadata(self, *, timeout: float = 10) -> dict[str, Any]:
        if self.protocol != Protocol.OPENPI_WS:
            raise ValueError("metadata-first handshake is only valid for openpi-ws-v1")
        return self.receive_map(timeout=timeout)


class NativeWebSocketBackend:
    """One-in-flight native backend for an explicitly negotiated aligned mode."""

    def __init__(
        self,
        socket: ControlSocket,
        *,
        expected_mode: ExecutionMode,
        action_dim: int,
        max_committed_prefix_steps: int,
        authoritative_epoch: int,
        receive_timeout: float = 10,
    ) -> None:
        if socket.protocol != Protocol.TENON_CONTROL:
            raise ValueError("native backend requires tenon-control-v1")
        if authoritative_epoch < 0:
            raise ValueError("authoritative_epoch must be non-negative")
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        if max_committed_prefix_steps < 0:
            raise ValueError("max_committed_prefix_steps must be non-negative")
        self.expected_mode = require_aligned_mode(expected_mode)
        self.action_dim = action_dim
        self.max_committed_prefix_steps = max_committed_prefix_steps
        self.socket = socket
        self.authoritative_epoch = authoritative_epoch
        self.receive_timeout = receive_timeout

    @staticmethod
    def _observation(observation: Observation) -> dict[str, Any]:
        return {
            "state": list(observation.state),
            "images": [dict(image) for image in observation.images],
            "prompt": observation.prompt,
        }

    def infer(self, request: InferenceRequest) -> ActionResponse:
        prefix = request.committed_prefix
        if self.expected_mode in PREFIX_CONDITIONED_MODES and prefix is None:
            raise RuntimeFault("conditioned request requires a committed prefix")
        if self.expected_mode == ExecutionMode.ASYNC_ALIGNED and prefix is not None:
            raise RuntimeFault("async_aligned request forbids committed_prefix")
        message: dict[str, Any] = {
            "type": "infer",
            "epoch": request.epoch,
            "request_index": request.request_index,
            "observation_index": request.observation.index,
            "observed_after_action_index": request.observation.observed_after_action_index,
            "target_start_action_index": request.target_start_action_index,
            "committed_end_action_index_exclusive": (request.committed_end_action_index_exclusive),
            "reply_needed_before_action_index": request.reply_needed_before_action_index,
            "max_server_processing_ms": request.max_server_processing_ms,
            "client_capture_monotonic_ns": request.observation.capture_monotonic_ns,
            "client_buffer_end_action_index_exclusive": (
                request.client_buffer_end_action_index_exclusive
            ),
            "observation": self._observation(request.observation),
        }
        if prefix is not None:
            message["committed_prefix"] = prefix.to_wire(self.action_dim)
        self.socket.send_map(message)
        message = self.socket.receive_map(timeout=self.receive_timeout)
        if message.get("type") == "error":
            raise RuntimeFault(f"policy error: {message.get('code', 'unknown')}")
        if message.get("type") != "actions":
            raise RuntimeFault("expected actions response")
        generation = message.get("generation") or {}
        try:
            if not isinstance(generation, dict):
                raise TypeError("generation must be a map")
            expected_generation_fields = (
                {
                    "mode",
                    "prefix_consumed_count",
                    "committed_prefix_sha256",
                    "sampler_conditioning",
                }
                if self.expected_mode in PREFIX_CONDITIONED_MODES
                else {"mode"}
            )
            if set(generation) != expected_generation_fields:
                raise TypeError("generation fields do not match the negotiated mode")
            prefix_count = generation.get("prefix_consumed_count")
            if prefix_count is not None and type(prefix_count) is not int:
                raise TypeError("prefix_consumed_count must be an integer")
            prefix_hash = generation.get("committed_prefix_sha256")
            if prefix_hash is not None and not isinstance(prefix_hash, str):
                raise TypeError("committed_prefix_sha256 must be a string")
            conditioning = generation.get("sampler_conditioning")
            if conditioning is not None and not isinstance(conditioning, str):
                raise TypeError("sampler_conditioning must be a string")
            if self.expected_mode in PREFIX_CONDITIONED_MODES:
                expected = (
                    "trained" if self.expected_mode == ExecutionMode.TRAINED_RTC else "guided"
                )
                if conditioning != expected:
                    raise TypeError("sampler conditioning does not match the negotiated mode")
            actions = tuple(tuple(float(value) for value in row) for row in message["actions"])
            return ActionResponse(
                # The approved actions example omits epoch; the authenticated
                # socket/request epoch is authoritative when it is absent.
                epoch=int(message.get("epoch", request.epoch)),
                request_index=int(message["request_index"]),
                based_on_observation_index=int(message["based_on_observation_index"]),
                start_action_index=int(message["start_action_index"]),
                valid_through_action_index_exclusive=int(
                    message["valid_through_action_index_exclusive"]
                ),
                actions=actions,
                generation_mode=str(generation.get("mode", "")),
                prefix_consumed_count=prefix_count,
                committed_prefix_sha256=prefix_hash,
                sampler_conditioning=conditioning,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeFault("malformed actions response") from exc

    def reset(self, *, epoch: int, reset_index: int, reason: str) -> int:
        self.socket.send_map(
            {"type": "reset", "epoch": epoch, "reset_index": reset_index, "reason": reason}
        )
        message = self.socket.receive_map(timeout=self.receive_timeout)
        if message.get("type") != "reset_ack" or message.get("reset_index") != reset_index:
            raise RuntimeFault("reset was not acknowledged")
        new_epoch = int(message["epoch"])
        self.authoritative_epoch = new_epoch
        return new_epoch

    def rearm(self, *, epoch: int, observation: Observation, rearm_index: int) -> bool:
        self.socket.send_map(
            {
                "type": "rearm",
                "epoch": epoch,
                "rearm_index": rearm_index,
                "observation_index": observation.index,
                "observed_after_action_index": observation.observed_after_action_index,
                "client_capture_monotonic_ns": observation.capture_monotonic_ns,
                "observation": self._observation(observation),
            }
        )
        message = self.socket.receive_map(timeout=self.receive_timeout)
        return (
            message.get("type") == "rearm_ack"
            and message.get("epoch") == epoch
            and message.get("rearm_index") == rearm_index
        )
