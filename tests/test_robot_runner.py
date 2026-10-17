from __future__ import annotations

import io
import threading
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from robot_runner_fixtures import GantryAdapter, PlanarAdapter, runner_profile, settings
from websockets.sync.client import connect
from websockets.sync.server import serve

from tenon_pi.control import ControlSocket, pack_message, unpack_message
from tenon_pi.models import (
    ControlConnection,
    ExecutionMode,
    FallbackPolicy,
    LastAppliedAbsoluteTarget,
)
from tenon_pi.robot import CommandEncodingReceipt, ControllerFault, NativeCommandTrace
from tenon_pi.robot_runtime import CaptureValidator, run_connected_robot, run_robot_loop
from tenon_pi.runtime import RuntimeStopped


@pytest.mark.parametrize(
    "layout,adapter_class", [("planar", PlanarAdapter), ("gantry", GantryAdapter)]
)
def test_full_capture_control_loop_over_public_socket(layout, adapter_class):
    profile = runner_profile(layout)
    adapter = adapter_class(profile)
    schema = profile.robot_schema("sha256:" + "a" * 64)
    schema_hash = "sha256:" + "b" * 64
    observations = []
    action = profile.action.to_canonical(list(adapter.last_applied_action().values()))
    # Canonical action is chosen in profile order, independent of state width.
    action[0] += np.float32(0.01)

    def handler(ws):
        ws.send(
            pack_message(
                dict(
                    type="hello",
                    protocol="tenon-control-v1",
                    session_id="ctl_test",
                    epoch=0,
                    schema_hash=schema_hash,
                    execution_mode="async_aligned",
                    rearm_required=False,
                    limits=dict(
                        max_in_flight=1,
                        max_request_bytes=4_000_000,
                        max_committed_prefix_steps=profile.max_committed_prefix_steps,
                    ),
                )
            )
        )
        for encoded in ws:
            message = unpack_message(encoded)
            if message["type"] == "ready":
                continue
            assert message["type"] == "infer"
            observation = message["observation"]
            observations.append(observation)
            assert len(observation["state"]) == len(profile.state.canonical)
            for expected, image in zip(profile.cameras, observation["images"], strict=True):
                assert image["camera_id"] == expected.id
                decoded = Image.open(io.BytesIO(image["data"]))
                assert decoded.size == (expected.width, expected.height)
            start = message["target_start_action_index"]
            ws.send(
                pack_message(
                    dict(
                        type="actions",
                        epoch=0,
                        request_index=message["request_index"],
                        based_on_observation_index=message["observation_index"],
                        start_action_index=start,
                        valid_through_action_index_exclusive=start + profile.action_horizon_steps,
                        actions=[action.tolist()] * profile.action_horizon_steps,
                        generation=dict(mode="async_aligned"),
                    )
                )
            )

    with serve(handler, "127.0.0.1", 0) as server:
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        port = server.socket.getsockname()[1]
        connection = ControlConnection(
            wss_url="wss://fixture.invalid/control",
            token="fixture-token",
            protocol="tenon-control-v1",
            expires_at=datetime.now(UTC),
        )
        result = run_connected_robot(
            connection,
            adapter=adapter,
            profile=profile,
            schema=schema,
            schema_hash=schema_hash,
            execution_mode=ExecutionMode.ASYNC_ALIGNED,
            fallback=FallbackPolicy(
                mode="drain_then_hold", hold_steps=2, hold_behavior=LastAppliedAbsoluteTarget()
            ),
            settings=settings(profile),
            prompt="fixture",
            socket_factory=lambda c: ControlSocket(
                c, connector=lambda *a, **kw: connect(f"ws://127.0.0.1:{port}")
            ),
        )
        server.shutdown()
        serving.join(2)
    assert result["steps"] == len(adapter.commands) == 5
    assert len(adapter.captures) == 8 and len(observations) >= 3
    assert not adapter.armed and not result["physical_tracking_verified"]
    if layout == "gantry":
        assert set(adapter.stopped_groups) == {"motor_bank", "jaw"}
        assert observations[0]["state"][-1] == 0.25
    else:
        assert adapter.stops >= 1
    assert result["applied_overrides"] > 0


def fake_runtime(profile, adapter):
    target = profile.action.to_canonical(
        [adapter.last_applied_action()[c.name] for c in profile.action.native]
    )
    target[0] += 0.01
    return SimpleNamespace(
        warm_start=lambda observations: None,
        next_action=lambda observation: target,
        record_applied=lambda *a, **kw: None,
        reset_pending=False,
        metrics=SimpleNamespace(applied_overrides=0, prefix_invalidations=0, fallback_actions=0),
    )


@pytest.mark.parametrize("failure", ["capture", "stale", "skew", "alignment", "apply"])
def test_capture_and_partial_command_failures_stop_owned_actuators(failure):
    profile = runner_profile("planar")
    adapter = PlanarAdapter(profile)
    adapter.failure = failure
    with pytest.raises((ValueError, OSError, ControllerFault)):
        run_robot_loop(
            adapter, profile, settings(profile), fake_runtime(profile, adapter), prompt="task"
        )
    assert adapter.stops >= 1 and not adapter.armed
    if failure != "apply":
        assert not adapter.commands


def test_distinct_multi_group_adapter_stops_all_after_partial_application():
    profile = runner_profile("gantry")
    adapter = GantryAdapter(profile)
    adapter.partial_failure = True
    with pytest.raises(ControllerFault):
        run_robot_loop(
            adapter, profile, settings(profile), fake_runtime(profile, adapter), prompt="task"
        )
    assert len(adapter.commands) == 1
    assert set(adapter.stopped_groups) == {"motor_bank", "jaw"}


@pytest.mark.parametrize("kind", ["exhausted", "cancelled", "window_stop"])
def test_every_terminal_path_stops_hardware(kind):
    profile = runner_profile("planar")
    adapter = PlanarAdapter(profile)
    runtime = fake_runtime(profile, adapter)

    def next_action(observation):
        if kind == "cancelled":
            raise KeyboardInterrupt()
        raise RuntimeStopped("fallback hold budget exhausted")

    runtime.next_action = next_action
    if kind == "window_stop":
        adapter.stop_requested = True
    if kind == "cancelled":
        with pytest.raises(KeyboardInterrupt):
            run_robot_loop(adapter, profile, settings(profile), runtime, prompt="task")
    else:
        result = run_robot_loop(adapter, profile, settings(profile), runtime, prompt="task")
        assert result["steps"] == 0
    assert adapter.stops and not adapter.armed


@pytest.mark.parametrize(
    "bad", ["clock", "provenance", "orientation", "repeated", "before_action", "state_order"]
)
def test_capture_contract_rejects_unproven_alignment(bad):
    profile = runner_profile("planar")
    adapter = PlanarAdapter(profile)
    capture = adapter.capture(after_action_index=-1)
    validator = CaptureValidator(profile, settings(profile))
    applied = None
    if bad == "clock":
        capture = replace(capture, timestamp_clock="device_ticks")
    elif bad == "provenance":
        camera = capture.cameras[0]
        capture = replace(
            capture,
            cameras=(
                replace(
                    camera,
                    provenance=camera.provenance.model_copy(update={"rotation_degrees": 180}),
                ),
            ),
        )
    elif bad == "orientation":
        capture = replace(capture, cameras=(replace(capture.cameras[0], orientation="unknown"),))
    elif bad == "repeated":
        validator.observation(capture, index=0, after=-1, applied_at_ns=None, prompt="task")
    elif bad == "before_action":
        applied = capture.state_timestamp_ns + 1
    else:
        capture = replace(capture, native_state={"unknown": 0})
    with pytest.raises(ValueError):
        validator.observation(capture, index=1, after=-1, applied_at_ns=applied, prompt="task")


def test_native_change_signal_survives_canonical_quantization_alias():
    mapping = runner_profile("planar").action
    # Find a deterministic native neighboring pair whose division aliases.
    for value in np.linspace(0, 99, 1000, dtype=np.float32):
        native = np.array([0, 0, value], dtype=np.float32)
        canonical = mapping.to_canonical(native)
        receipt = CommandEncodingReceipt.from_canonical(canonical, mapping)
        submitted = receipt.encoded_native()
        applied = submitted.copy()
        applied[-1] = np.nextafter(applied[-1], np.float32(100))
        if np.array_equal(mapping.to_canonical(applied), mapping.to_canonical(submitted)):
            result = receipt.resolve_application(NativeCommandTrace.from_arrays(submitted, applied))
            assert result.native_changed and result.driver_changed
            np.testing.assert_array_equal(result.canonical_array(), mapping.to_canonical(applied))
            return
    pytest.fail("fixture did not produce the expected float32 native/canonical alias")
