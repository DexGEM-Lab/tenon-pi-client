from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from websockets.sync.client import connect
from websockets.sync.server import serve

from tenon_pi.control import (
    ControlSocket,
    NativeWebSocketBackend,
    pack_message,
    unpack_message,
)
from tenon_pi.models import (
    ControlConnection,
    ExecutionMode,
    FallbackPolicy,
    LastAppliedAbsoluteTarget,
    Protocol,
)
from tenon_pi.runtime import (
    AsyncAlignedController,
    Observation,
    ResetPending,
    RuntimeFault,
)


def _observation(index: int, after: int) -> Observation:
    return Observation(
        index,
        after,
        (0.0,),
        ({"camera_id": "base", "encoding": "png", "data": b"fixture"},),
        "fixture",
        index,
    )


@pytest.mark.parametrize("mode", [ExecutionMode.ASYNC_ALIGNED, ExecutionMode.TRAINED_RTC])
@pytest.mark.parametrize(
    ("executor_workers", "disconnect_first"),
    [(None, False), (2, True)],
)
def test_native_socket_reset_waits_for_held_response_then_acks_once(
    executor_workers: int | None,
    disconnect_first: bool,
    mode: ExecutionMode,
):
    inference_started = threading.Event()
    release_inference = threading.Event()
    reset_received = threading.Event()
    reset_frames: list[dict] = []
    schema_hash = "sha256:" + "0" * 64

    def handler(websocket):
        websocket.send(
            pack_message(
                {
                    "type": "hello",
                    "protocol": "tenon-control-v1",
                    "session_id": "ctl_fixture",
                    "epoch": 0,
                    "schema_hash": schema_hash,
                    "execution_mode": mode.value,
                    "rearm_required": False,
                    "limits": {
                        "max_in_flight": 1,
                        "max_request_bytes": 4_000_000,
                        "max_committed_prefix_steps": 9,
                    },
                }
            )
        )
        infer_count = 0
        try:
            for raw in websocket:
                message = unpack_message(raw)
                if message["type"] == "ready":
                    continue
                if message["type"] == "infer":
                    infer_count += 1
                    if infer_count > 1:
                        inference_started.set()
                        assert release_inference.wait(3)
                    websocket.send(
                        pack_message(
                            {
                                "type": "actions",
                                "request_index": message["request_index"],
                                "based_on_observation_index": message["observation_index"],
                                "start_action_index": message["target_start_action_index"],
                                "valid_through_action_index_exclusive": (
                                    message["target_start_action_index"] + 10
                                ),
                                "actions": [[0.0]] * 10,
                                "generation": {
                                    "mode": mode.value,
                                    **(
                                        {
                                            "prefix_consumed_count": message["committed_prefix"][
                                                "count"
                                            ],
                                            "committed_prefix_sha256": message["committed_prefix"][
                                                "sha256"
                                            ],
                                            "sampler_conditioning": "trained",
                                        }
                                        if mode == ExecutionMode.TRAINED_RTC
                                        else {}
                                    ),
                                },
                            }
                        )
                    )
                elif message["type"] == "reset":
                    reset_frames.append(message)
                    reset_received.set()
                    websocket.send(
                        pack_message(
                            {
                                "type": "reset_ack",
                                "reset_index": message["reset_index"],
                                "epoch": 1,
                            }
                        )
                    )
        except Exception:
            pass

    with serve(handler, "127.0.0.1", 0) as server:
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        port = server.socket.getsockname()[1]
        connection = ControlConnection(
            wss_url="wss://fixture.invalid/control",
            token="synthetic-fixture-token",
            expires_at=datetime.now(UTC),
            protocol=Protocol.TENON_CONTROL,
        )
        with ControlSocket(
            connection,
            connector=lambda *_args, **_kwargs: connect(
                f"ws://127.0.0.1:{port}",
                compression=None,
                open_timeout=2,
                close_timeout=1,
            ),
        ) as socket:
            hello = socket.native_handshake(
                schema_hash=schema_hash,
                expected_mode=mode,
            )
            backend = NativeWebSocketBackend(
                socket,
                expected_mode=mode,
                action_dim=1,
                max_committed_prefix_steps=9,
                authoritative_epoch=hello["epoch"],
                receive_timeout=10,
            )
            executor = (
                None
                if executor_workers is None
                else ThreadPoolExecutor(max_workers=executor_workers)
            )
            runtime = AsyncAlignedController(
                backend,
                executor=executor,
                expected_mode=mode,
                controller_hz=20,
                state_dim=1,
                action_dim=1,
                action_horizon_steps=10,
                action_names=("joint",),
                action_semantics="joint_position",
                fallback=FallbackPolicy(
                    mode="drain_then_hold",
                    hold_steps=5,
                    hold_behavior=LastAppliedAbsoluteTarget(),
                ),
                committed_steps=3,
                clock=lambda: 0.0,
            )
            try:
                assert inspect.signature(runtime.reset).parameters["timeout_seconds"].default == 5
                assert backend.receive_timeout == 10
                runtime.warm_start([_observation(0, -1)])
                for action_index in range(7):
                    requested = runtime.next_action(
                        _observation(action_index + 1, action_index - 1)
                    )
                    runtime.record_applied(requested)
                assert inference_started.wait(1)
                if disconnect_first:
                    runtime.mark_disconnected("fixture disconnect during held receive")
                    assert runtime._future is None
                    assert len(runtime._orphaned_inference) == 1
                    with pytest.raises(RuntimeFault, match="receive is still owned"):
                        runtime.warm_start([_observation(100, 6)])

                with pytest.raises(ResetPending) as first:
                    runtime.reset("held WSS reset", timeout_seconds=0.01)
                assert first.value.reset_index == 1
                assert runtime.reset_pending and runtime.epoch == 0
                assert runtime.state == "disarmed"
                assert not reset_received.is_set()
                with pytest.raises(ResetPending):
                    runtime.reset("must reuse pending reset", timeout_seconds=0.01)
                assert not reset_received.is_set()

                release_inference.set()
                runtime.reset("complete pending reset", timeout_seconds=2)
                assert reset_received.wait(1)
                assert len(reset_frames) == 1
                assert reset_frames[0]["reset_index"] == 1
                assert runtime.epoch == 1
                assert backend.authoritative_epoch == 1
                assert not runtime.reset_pending
                assert runtime.buffer.remaining == 0
                assert runtime.metrics.accepted_responses == 0
            finally:
                release_inference.set()
                runtime.close()
                if executor is not None:
                    executor.shutdown(wait=True)
        server.shutdown()
        serving.join(2)
