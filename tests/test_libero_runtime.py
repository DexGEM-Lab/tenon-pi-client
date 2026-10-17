from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tenon_pi.libero_runtime import LIBERO_ACTION_NAMES, LiberoAsyncRollout
from tenon_pi.models import ExecutionMode, FallbackPolicy, ZeroDeltaPreserve
from tenon_pi.runtime import ActionResponse, AsyncAlignedController, InferenceRequest


def raw_observation():
    return {
        "agentview_image": np.full((256, 256, 3), 20, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((256, 256, 3), 40, dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
    }


class FakeEnv:
    def __init__(self, done_after=12):
        self.done_after = done_after
        self.actions = []
        self.threads = []

    def step(self, action):
        self.actions.append(list(action))
        self.threads.append(threading.get_ident())
        done = len(self.actions) >= self.done_after
        return raw_observation(), 1.0 if done else 0.0, done, {}


class FakeBackend:
    authoritative_epoch = 0

    def __init__(self, mode: ExecutionMode):
        self.mode = mode

    def infer(self, request: InferenceRequest) -> ActionResponse:
        actions = tuple(
            (0.01 * index, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
            for index in range(
                request.target_start_action_index,
                request.target_start_action_index + 10,
            )
        )
        prefix = request.committed_prefix
        assert (prefix is not None) == (self.mode == ExecutionMode.GUIDED_RTC)
        return ActionResponse(
            request.epoch,
            request.request_index,
            request.observation.index,
            request.target_start_action_index,
            request.target_start_action_index + 10,
            actions,
            generation_mode=self.mode.value,
            prefix_consumed_count=(prefix.count if prefix else None),
            committed_prefix_sha256=(prefix.sha256 if prefix else None),
            sampler_conditioning=("guided" if prefix else None),
        )

    def reset(self, *, epoch, reset_index, reason):
        return epoch + 1

    def rearm(self, *, epoch, observation, rearm_index):
        return True


class WarmClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        value = self.calls * 0.01
        self.calls += 1
        return value


class RolloutClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds
        time.sleep(0.001)  # yield to the inference worker without changing simulated time


@pytest.mark.parametrize("mode", [ExecutionMode.ASYNC_ALIGNED, ExecutionMode.GUIDED_RTC])
def test_libero_wrapper_is_single_env_step_thread_and_runs_explicit_mode(
    mode: ExecutionMode,
):
    fallback = FallbackPolicy(
        mode="drain_then_hold",
        hold_steps=5,
        hold_behavior=ZeroDeltaPreserve(preserve_channels=["gripper"]),
    )
    controller = AsyncAlignedController(
        FakeBackend(mode),
        expected_mode=mode,
        controller_hz=20,
        state_dim=8,
        action_dim=7,
        action_horizon_steps=10,
        action_names=LIBERO_ACTION_NAMES,
        action_semantics="cartesian_delta",
        fallback=fallback,
        action_lower=(-1.0,) * 7,
        action_upper=(1.0,) * 7,
        committed_steps=2,
        clock=WarmClock(),
    )
    env = FakeEnv(done_after=12)
    pacing = RolloutClock()
    try:
        result = LiberoAsyncRollout(
            env,
            controller,
            prompt="fixture task",
            clock=pacing,
            sleeper=pacing.sleep,
        ).run(raw_observation(), max_steps=20)
    finally:
        controller.close()
    assert result.success
    assert result.steps == 12
    assert len(set(env.threads)) == 1
    assert env.threads[0] == threading.get_ident()
    assert result.accepted_responses >= 1
    assert all(len(action) == 7 and np.isfinite(action).all() for action in env.actions)
