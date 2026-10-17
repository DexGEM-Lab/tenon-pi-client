"""LIBERO observation adapter and single-control-thread async rollout loop."""

from __future__ import annotations

import io
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .runtime import AsyncAlignedController, Observation, RuntimeStopped

LIBERO_ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")
LIBERO_DUMMY_ACTION = np.array([0.0] * 6 + [-1.0], dtype=np.float32)


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("quaternion must be finite xyzw shape [4]")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] ** 2))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / denominator).astype(np.float32)


def _resize_with_pad(image: np.ndarray, size: int = 224) -> np.ndarray:
    rgb = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    scale = min(size / rgb.width, size / rgb.height)
    resized = rgb.resize(
        (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (size, size))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def _jpeg(image: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(
        output,
        format="JPEG",
        quality=95,
        optimize=False,
        progressive=False,
    )
    return output.getvalue()


def make_libero_observation(
    raw: dict[str, Any],
    *,
    prompt: str,
    observation_index: int,
    observed_after_action_index: int,
    capture_monotonic_ns: int | None = None,
) -> Observation:
    base = np.ascontiguousarray(np.asarray(raw["agentview_image"])[::-1, ::-1])
    wrist = np.ascontiguousarray(np.asarray(raw["robot0_eye_in_hand_image"])[::-1, ::-1])
    base = _resize_with_pad(base)
    wrist = _resize_with_pad(wrist)
    state = np.concatenate(
        [
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
            quaternion_to_axis_angle(np.asarray(raw["robot0_eef_quat"])),
            np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("LIBERO observation state must be finite shape [8]")
    return Observation(
        index=observation_index,
        observed_after_action_index=observed_after_action_index,
        state=tuple(float(value) for value in state),
        images=(
            {"camera_id": "base", "encoding": "jpeg", "data": _jpeg(base)},
            {"camera_id": "left_wrist", "encoding": "jpeg", "data": _jpeg(wrist)},
        ),
        prompt=prompt,
        capture_monotonic_ns=(
            time.monotonic_ns() if capture_monotonic_ns is None else capture_monotonic_ns
        ),
    )


@dataclass(frozen=True)
class LiberoRolloutResult:
    steps: int
    success: bool
    stopped_reason: str | None
    tick_lateness_ms_max: float
    accepted_responses: int
    dropped_responses: int
    fallback_actions: int
    applied_overrides: int
    prefix_invalidations: int


class LiberoAsyncRollout:
    """Wall-clock-paced LIBERO loop; this object alone calls ``env.step``."""

    def __init__(
        self,
        env: Any,
        controller: AsyncAlignedController,
        *,
        prompt: str,
        controller_hz: float = 20.0,
        clock: Any = time.monotonic,
        sleeper: Any = time.sleep,
        on_frame: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self.env = env
        self.controller = controller
        self.prompt = prompt
        self.period = 1.0 / controller_hz
        self.clock = clock
        self.sleeper = sleeper
        self.on_frame = on_frame

    def stabilize(self, observation: dict[str, Any], steps: int = 10) -> dict[str, Any]:
        current = observation
        for _ in range(steps):
            if self.on_frame is not None and not self.on_frame(current):
                raise RuntimeStopped("viewer closed during stabilization")
            current, _, _, _ = self.env.step(LIBERO_DUMMY_ACTION.tolist())
        return current

    def run(self, initial_observation: dict[str, Any], *, max_steps: int) -> LiberoRolloutResult:
        raw = initial_observation
        observation_index = 0
        warm_observations = []
        for _ in range(3):
            warm_observations.append(
                make_libero_observation(
                    raw,
                    prompt=self.prompt,
                    observation_index=observation_index,
                    observed_after_action_index=-1,
                )
            )
            observation_index += 1
        self.controller.warm_start(warm_observations)

        next_deadline = self.clock()
        max_lateness = 0.0
        stopped_reason: str | None = None
        success = False
        executed = 0
        for action_index in range(max_steps):
            if self.on_frame is not None and not self.on_frame(raw):
                stopped_reason = "viewer_closed"
                break
            now = self.clock()
            if now < next_deadline:
                self.sleeper(next_deadline - now)
            else:
                max_lateness = max(max_lateness, (now - next_deadline) * 1000)
            observation = make_libero_observation(
                raw,
                prompt=self.prompt,
                observation_index=observation_index,
                observed_after_action_index=action_index - 1,
            )
            observation_index += 1
            try:
                action = self.controller.next_action(observation)
            except RuntimeStopped as exc:
                stopped_reason = str(exc)
                break
            if action.shape != (7,) or not np.isfinite(action).all():
                stopped_reason = "runtime returned invalid LIBERO action"
                break
            try:
                raw, _, done, _ = self.env.step(action.tolist())
            except Exception as exc:
                self.controller.mark_disconnected("environment step failed")
                stopped_reason = f"environment step failed: {type(exc).__name__}"
                break
            self.controller.record_applied(action)
            executed += 1
            next_deadline += self.period
            if done:
                success = True
            if self.on_frame is not None and not self.on_frame(raw):
                stopped_reason = "viewer_closed"
                break
            if done:
                break
        return LiberoRolloutResult(
            steps=executed,
            success=success,
            stopped_reason=stopped_reason,
            tick_lateness_ms_max=max_lateness,
            accepted_responses=self.controller.metrics.accepted_responses,
            dropped_responses=self.controller.metrics.dropped_responses,
            fallback_actions=self.controller.metrics.fallback_actions,
            applied_overrides=self.controller.metrics.applied_overrides,
            prefix_invalidations=self.controller.metrics.prefix_invalidations,
        )
