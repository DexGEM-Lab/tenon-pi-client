"""Executable profile-driven observation/actuation loop for a supplied trusted adapter.

Camera orientation is capture provenance, not an instruction to rotate pixels.
Adapters own device connections, clock conversion and idempotent stop-all behavior.
"""

from __future__ import annotations

import importlib
import io
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from .control import ControlSocket, NativeWebSocketBackend
from .models import ControlConnection, ExecutionMode, FallbackPolicy, RobotSchema
from .robot import CommandEncodingReceipt, CommandLimiter, RobotController, RobotDriver
from .robot_profile import CaptureCamera, RobotProfile
from .runtime import AsyncAlignedController, Observation, RuntimeStopped, require_aligned_mode


@dataclass(frozen=True)
class CameraCapture:
    provenance: CaptureCamera
    rgb: np.ndarray
    timestamp_ns: int
    orientation: Literal["profile"] = "profile"


@dataclass(frozen=True)
class RobotCapture:
    native_state: Mapping[str, float]
    state_timestamp_ns: int
    cameras: tuple[CameraCapture, ...]
    observed_after_action_index: int
    calibration_digest: str
    timestamp_clock: Literal["host_monotonic_ns"] = "host_monotonic_ns"


class RobotAdapter(RobotDriver, Protocol):
    """Trusted installation code, explicitly selected by the user, never profile JSON.

    capture must acquire after the supplied action index and translate all device
    timestamps into this process's time.monotonic_ns domain using known clock
    calibration. It must not stamp stale device samples with the current time.
    RGB pixels must already match the profile's recorded orientation/provenance.
    stop must attempt every owned actuator even when one stop fails, and be
    idempotent. connected/armed/stop_requested reflect actual installation state.
    last_applied_action reports native command values, not physical telemetry.
    """

    @property
    def stop_requested(self) -> bool: ...
    def capture(self, *, after_action_index: int) -> RobotCapture: ...
    def last_applied_action(self) -> Mapping[str, float]: ...


class RobotRunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    lower: tuple[FiniteFloat, ...]
    upper: tuple[FiniteFloat, ...]
    max_delta: tuple[FiniteFloat, ...]
    max_capture_age_ms: float = Field(gt=0)
    max_capture_skew_ms: float = Field(ge=0)
    max_steps: int = Field(ge=1)
    max_server_processing_ms: int = Field(default=500, ge=1)


class CaptureValidator:
    def __init__(
        self,
        profile: RobotProfile,
        settings: RobotRunSettings,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ):
        self.profile, self.settings, self.clock_ns = profile, settings, clock_ns
        self._timestamps: dict[str, int] = {}

    def observation(
        self,
        capture: RobotCapture,
        *,
        index: int,
        after: int,
        applied_at_ns: int | None,
        prompt: str,
    ) -> Observation:
        from PIL import Image

        if (
            capture.timestamp_clock != "host_monotonic_ns"
            or capture.observed_after_action_index != after
            or capture.calibration_digest != self.profile.calibration_digest
        ):
            raise ValueError("capture clock/action alignment/calibration does not match the run")
        names = tuple(c.name for c in self.profile.state.native)
        if set(capture.native_state) != set(names):
            raise ValueError(
                "capture must report exactly the profile's named native state channels"
            )
        native = np.asarray([capture.native_state[n] for n in names], dtype=np.float32)
        lower, upper = self.profile.state.bounds("native")
        if (
            native.shape != lower.shape
            or not np.isfinite(native).all()
            or np.any(native < lower)
            or np.any(native > upper)
        ):
            raise ValueError("captured native state exceeds declared shape/calibration")
        cameras = {c.provenance.id: c for c in capture.cameras}
        if len(cameras) != len(capture.cameras) or set(cameras) != {
            c.id for c in self.profile.cameras
        }:
            raise ValueError(
                "capture cameras must exactly match the profile IDs without duplicates"
            )
        timestamps = {"state": capture.state_timestamp_ns}
        images = []
        for expected in self.profile.cameras:
            camera = cameras[expected.id]
            if camera.provenance != expected or camera.orientation != "profile":
                raise ValueError("capture camera provenance/orientation does not match the profile")
            pixels = np.asarray(camera.rgb)
            if pixels.dtype != np.uint8 or pixels.shape != (expected.height, expected.width, 3):
                raise ValueError("capture requires original profile-shaped RGB uint8 pixels")
            # Snapshot before encoding; adapters may reuse their acquisition buffers.
            out = io.BytesIO()
            Image.fromarray(pixels.copy()).save(out, format="PNG")
            images.append({"camera_id": expected.id, "encoding": "png", "data": out.getvalue()})
            timestamps[f"camera:{expected.id}"] = camera.timestamp_ns
        now = self.clock_ns()
        for name, stamp in timestamps.items():
            if type(stamp) is not int or stamp < 0 or stamp > now:
                raise ValueError(
                    "capture timestamps must be finite host-monotonic nanoseconds, not future times"
                )
            if now - stamp > self.settings.max_capture_age_ms * 1_000_000:
                raise ValueError("stale capture exceeds the explicit age limit")
            if stamp <= self._timestamps.get(name, -1):
                raise ValueError("capture must contain fresh samples from every source")
            if applied_at_ns is not None and stamp < applied_at_ns:
                raise ValueError("capture predates the reported action application")
        if (
            max(timestamps.values()) - min(timestamps.values())
            > self.settings.max_capture_skew_ms * 1_000_000
        ):
            raise ValueError("capture acquisition skew exceeds the explicit limit")
        self._timestamps = timestamps
        canonical = self.profile.state.to_canonical(native)
        return Observation(
            index,
            after,
            tuple(float(v) for v in canonical),
            tuple(images),
            prompt,
            max(timestamps.values()),
        )


def load_robot_adapter(
    factory: str, *, profile: RobotProfile, config: dict[str, Any]
) -> RobotAdapter:
    """Explicit trusted module:factory import; this string never comes from a profile."""
    module, separator, name = factory.partition(":")
    if not separator or not module or not name.isidentifier():
        raise ValueError("trusted adapter factory must be an importable module:function")
    constructor = getattr(importlib.import_module(module), name)
    if not callable(constructor):
        raise ValueError("trusted adapter factory is not callable")
    return constructor(profile=profile, config=config)


def _robot_controller(
    adapter: RobotAdapter, profile: RobotProfile, settings: RobotRunSettings
) -> RobotController:
    initial = adapter.last_applied_action()
    names = tuple(c.name for c in profile.action.native)
    if set(initial) != set(names):
        raise ValueError("adapter must report every last applied native action channel")
    return RobotController(
        adapter,
        profile.action,
        CommandLimiter(settings.lower, settings.upper, settings.max_delta),
        [initial[n] for n in names],
    )


def preflight_robot_adapter(
    adapter: RobotAdapter, profile: RobotProfile, settings: RobotRunSettings, *, prompt: str
) -> dict[str, Any]:
    """Validate local command/capture contracts without inference or send_action calls.

    Requires the operator's explicit connected/armed state; always stops owned
    actuators afterward. This is not commissioning or a reusable arming receipt.
    """
    try:
        _robot_controller(adapter, profile, settings)
        validator = CaptureValidator(profile, settings)
        for index in range(3):
            if not adapter.connected or not adapter.armed or adapter.stop_requested:
                raise RuntimeStopped("adapter disconnected, disarmed or stop requested")
            validator.observation(
                adapter.capture(after_action_index=-1),
                index=index,
                after=-1,
                applied_at_ns=None,
                prompt=prompt,
            )
        return {
            "profile_digest": profile.digest(),
            "captures": 3,
            "commands_sent": 0,
            "physical_tracking_verified": False,
        }
    finally:
        adapter.stop()


def run_robot_loop(
    adapter: RobotAdapter,
    profile: RobotProfile,
    settings: RobotRunSettings,
    runtime: AsyncAlignedController,
    *,
    prompt: str,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run finite real capture/apply ticks; every exit requests stop-all before cleanup.

    Conditioned command overrides retain the runtime's serialized reset ownership.
    This runner stops for explicit operator rearm instead of inventing a new epoch.
    """
    steps, observation_index, stopped_reason = 0, 0, "max_steps"
    applied_at = None
    validator = CaptureValidator(profile, settings, clock_ns)
    try:
        hardware = _robot_controller(adapter, profile, settings)

        def capture() -> Observation:
            nonlocal observation_index
            if not adapter.connected or not adapter.armed or adapter.stop_requested:
                raise RuntimeStopped("adapter disconnected, disarmed or stop requested")
            raw = adapter.capture(after_action_index=steps - 1)
            result = validator.observation(
                raw,
                index=observation_index,
                after=steps - 1,
                applied_at_ns=applied_at,
                prompt=prompt,
            )
            observation_index += 1
            return result

        # Each warm observation is a genuinely fresh capture; no hardware action
        # occurs until measured admission and all capture checks succeed.
        runtime.warm_start([capture() for _ in range(3)])
        deadline = clock_ns()
        for _ in range(settings.max_steps):
            delay = (deadline - clock_ns()) / 1e9
            if delay > 0:
                sleeper(delay)
            observation = capture()
            requested = runtime.next_action(observation)
            receipt = CommandEncodingReceipt.from_canonical(requested, profile.action)
            trace = hardware.apply_with_trace(receipt.encoded_native())
            applied_at = clock_ns()
            result = receipt.resolve_application(trace)
            runtime.record_applied(result.canonical_array(), command_changed=result.native_changed)
            steps += 1
            deadline += round(1e9 / profile.controller_hz)
            if runtime.reset_pending:
                stopped_reason = "conditioned_override_requires_reset_and_rearm"
                break
    except RuntimeStopped as exc:
        stopped_reason = str(exc)
    finally:
        adapter.stop()  # Must stop all owned actuators, including after partial application.
    return {
        "steps": steps,
        "observations": observation_index,
        "stopped_reason": stopped_reason,
        "applied_overrides": runtime.metrics.applied_overrides,
        "prefix_invalidations": runtime.metrics.prefix_invalidations,
        "fallback_actions": runtime.metrics.fallback_actions,
        "physical_tracking_verified": False,
    }


def run_connected_robot(
    connection: ControlConnection,
    *,
    adapter: RobotAdapter,
    profile: RobotProfile,
    schema: RobotSchema,
    schema_hash: str,
    fallback: FallbackPolicy,
    execution_mode: ExecutionMode,
    settings: RobotRunSettings,
    prompt: str,
    socket_factory: Callable[..., ControlSocket] = ControlSocket,
) -> dict[str, Any]:
    runtime = None
    try:
        if profile.robot_schema(schema.normalization_digest) != schema:
            raise ValueError("profile does not match the verified deployment schema")
        fallback.validate_for_schema(schema)
        mode = require_aligned_mode(execution_mode)
        with socket_factory(connection) as socket:
            hello = socket.native_handshake(schema_hash=schema_hash, expected_mode=mode)
            if hello["rearm_required"]:
                raise ValueError(
                    "robot runner requires a fresh session; reconnect needs explicit rearm"
                )
            backend = NativeWebSocketBackend(
                socket,
                expected_mode=mode,
                action_dim=len(schema.action_channels),
                authoritative_epoch=int(hello["epoch"]),
                max_committed_prefix_steps=int(hello["limits"]["max_committed_prefix_steps"]),
            )
            runtime = AsyncAlignedController(
                backend,
                expected_mode=mode,
                controller_hz=schema.controller_hz,
                state_dim=len(schema.state_channels),
                action_dim=len(schema.action_channels),
                action_horizon_steps=schema.action_horizon_steps,
                action_names=tuple(c.name for c in schema.action_channels),
                action_semantics=schema.action_semantics,
                fallback=fallback,
                action_lower=tuple(c.lower for c in schema.action_channels),
                action_upper=tuple(c.upper for c in schema.action_channels),
                committed_steps=schema.max_committed_prefix_steps,
                max_server_processing_ms=settings.max_server_processing_ms,
            )
            try:
                return run_robot_loop(adapter, profile, settings, runtime, prompt=prompt)
            finally:
                # Stop actuators before waiting for any outstanding network receive.
                try:
                    adapter.stop()
                finally:
                    runtime.close()
    finally:
        adapter.stop()
