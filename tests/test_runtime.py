from __future__ import annotations

import hashlib
import threading
from concurrent.futures import Future
from dataclasses import dataclass, replace

import numpy as np
import pytest

from tenon_pi.models import (
    ExecutionMode,
    FallbackPolicy,
    LastAppliedAbsoluteTarget,
    ZeroDeltaPreserve,
)
from tenon_pi.runtime import (
    ActionResponse,
    AdmissionRejected,
    AsyncAlignedController,
    CommittedPrefix,
    FallbackExecutor,
    IndexedActionBuffer,
    InferenceRequest,
    Observation,
    ResetPending,
    RuntimeFault,
    RuntimeStopped,
    WarmTurnaroundGate,
    action_prefix_sha256,
)


def obs(index: int, after: int) -> Observation:
    return Observation(
        index=index,
        observed_after_action_index=after,
        state=(0.0,),
        images=({"camera_id": "base", "encoding": "jpeg", "data": b"x"},),
        prompt="fixture",
        capture_monotonic_ns=index,
    )


@dataclass
class FakeBackend:
    offset: float = 0.0
    horizon: int = 10
    authoritative_epoch: int = 0
    mode: ExecutionMode = ExecutionMode.ASYNC_ALIGNED
    resets: list[tuple[int, int, str]] = None  # type: ignore[assignment]
    rearms: list[tuple[int, int]] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.resets = []
        self.rearms = []

    def infer(self, request: InferenceRequest) -> ActionResponse:
        actions = tuple(
            (float(index) + self.offset, 1.0)
            for index in range(
                request.target_start_action_index,
                request.target_start_action_index + self.horizon,
            )
        )
        prefix = request.committed_prefix
        if self.mode in {ExecutionMode.GUIDED_RTC, ExecutionMode.TRAINED_RTC}:
            assert prefix is not None
        else:
            assert prefix is None
        return ActionResponse(
            epoch=request.epoch,
            request_index=request.request_index,
            based_on_observation_index=request.observation.index,
            start_action_index=request.target_start_action_index,
            valid_through_action_index_exclusive=(request.target_start_action_index + self.horizon),
            actions=actions,
            generation_mode=self.mode.value,
            prefix_consumed_count=(prefix.count if prefix is not None else None),
            committed_prefix_sha256=(prefix.sha256 if prefix is not None else None),
            sampler_conditioning=(
                ("trained" if self.mode == ExecutionMode.TRAINED_RTC else "guided")
                if prefix is not None
                else None
            ),
        )

    def reset(self, *, epoch: int, reset_index: int, reason: str) -> int:
        self.resets.append((epoch, reset_index, reason))
        self.authoritative_epoch = epoch + 1
        return self.authoritative_epoch

    def rearm(self, *, epoch: int, observation: Observation, rearm_index: int) -> bool:
        self.rearms.append((epoch, rearm_index))
        return True


class ManualExecutor:
    def __init__(self):
        self.future: Future[ActionResponse] | None = None
        self.request: InferenceRequest | None = None

    def submit(self, fn, request):
        del fn
        self.request = request
        self.future = Future()
        return self.future


class PairClock:
    def __init__(self, elapsed: float):
        self.elapsed = elapsed
        self.calls = 0

    def __call__(self):
        value = (self.calls // 2) + (self.elapsed if self.calls % 2 else 0.0)
        self.calls += 1
        return value


def fallback(holds: int = 2) -> FallbackPolicy:
    return FallbackPolicy(
        mode="drain_then_hold",
        hold_steps=holds,
        hold_behavior=ZeroDeltaPreserve(preserve_channels=["gripper"]),
    )


def controller(
    backend,
    executor=None,
    *,
    mode: ExecutionMode = ExecutionMode.ASYNC_ALIGNED,
    committed_steps=3,
    elapsed=0.05,
):
    return AsyncAlignedController(
        backend,
        expected_mode=mode,
        controller_hz=20,
        state_dim=1,
        action_dim=2,
        action_horizon_steps=10,
        action_names=("motion", "gripper"),
        action_semantics="cartesian_delta",
        fallback=fallback(),
        committed_steps=committed_steps,
        executor=executor,
        clock=PairClock(elapsed),
    )


def warm(runtime: AsyncAlignedController):
    runtime.warm_start([obs(0, -1), obs(1, -1), obs(2, -1)])


def apply(runtime: AsyncAlignedController, observation: Observation) -> np.ndarray:
    action = runtime.next_action(observation)
    runtime.record_applied(action)
    return action


def test_strict_action_bounds_reject_raw_overshoot_without_clipping():
    class OvershootBackend(FakeBackend):
        def infer(self, request):
            response = super().infer(request)
            actions = tuple((0.0, 1.009) for _ in response.actions)
            return ActionResponse(
                response.epoch,
                response.request_index,
                response.based_on_observation_index,
                response.start_action_index,
                response.valid_through_action_index_exclusive,
                actions,
            )

    runtime = AsyncAlignedController(
        OvershootBackend(),
        expected_mode=ExecutionMode.ASYNC_ALIGNED,
        controller_hz=20,
        state_dim=1,
        action_dim=2,
        action_horizon_steps=10,
        action_names=("motion", "gripper"),
        action_semantics="cartesian_delta",
        fallback=fallback(),
        action_lower=(-1.0, -1.0),
        action_upper=(1.0, 1.0),
        clock=PairClock(0.05),
    )
    with pytest.raises(RuntimeFault, match="outside deployment physical bounds"):
        warm(runtime)


def test_warm_turnaround_rejects_motion_when_horizon_cannot_cover_delay():
    gate = WarmTurnaroundGate(controller_hz=20, action_horizon_steps=10)
    gate.add(0.30)
    with pytest.raises(AdmissionRejected):
        gate.require_admissible()


def test_async_aligned_preserves_committed_actions_then_splices_future():
    initial = FakeBackend(offset=0)
    async_backend = initial
    executor = ManualExecutor()
    runtime = controller(async_backend, executor)
    warm(runtime)
    sequence = []
    observation_index = 3
    for action_index in range(8):
        action = apply(runtime, obs(observation_index, action_index - 1))
        observation_index += 1
        sequence.append(action.copy())
    assert executor.request is not None
    assert executor.request.target_start_action_index == 5
    assert executor.request.committed_end_action_index_exclusive == 8
    # A deliberately different plan arrives after action 7. The committed
    # interval [5, 8) remained from the original chunk.
    replacement = FakeBackend(offset=100).infer(executor.request)
    assert executor.future is not None
    executor.future.set_result(replacement)
    for action_index in range(8, 13):
        sequence.append(apply(runtime, obs(observation_index, action_index - 1)).copy())
        observation_index += 1
    assert [float(action[0]) for action in sequence] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        108,
        109,
        110,
        111,
        112,
    ]
    assert runtime.metrics.accepted_responses == 1


def test_each_requested_action_requires_one_actuator_truth_record():
    runtime = controller(FakeBackend())
    warm(runtime)
    requested = runtime.next_action(obs(3, -1))
    with pytest.raises(RuntimeFault, match="record_applied"):
        runtime.next_action(obs(4, 0))
    runtime.record_applied(requested)
    with pytest.raises(RuntimeFault, match="no pending"):
        runtime.record_applied(requested)


def test_async_actuator_override_updates_truth_without_claiming_a_guided_prefix():
    runtime = controller(FakeBackend())
    warm(runtime)
    observation_index = 3
    for action_index in range(5):
        apply(runtime, obs(observation_index, action_index - 1))
        observation_index += 1
    requested = runtime.next_action(obs(observation_index, 4))
    np.testing.assert_array_equal(requested, [5.0, 1.0])
    runtime.record_applied(np.array([4.0, 0.5], dtype=np.float32))
    assert runtime.state == "armed"
    assert runtime.metrics.applied_overrides == 1
    assert runtime.metrics.prefix_invalidations == 0
    np.testing.assert_array_equal(runtime.fallback.last_applied, [4.0, 0.5])


@pytest.mark.parametrize("mode", [ExecutionMode.GUIDED_RTC, ExecutionMode.TRAINED_RTC])
def test_guided_rtc_sends_exact_float32_prefix_and_preserves_it_before_splice(mode):
    backend = FakeBackend(mode=mode)
    executor = ManualExecutor()
    runtime = controller(backend, executor, mode=mode)
    warm(runtime)
    sequence = []
    observation_index = 3
    for action_index in range(6):
        sequence.append(apply(runtime, obs(observation_index, action_index - 1)).copy())
        observation_index += 1
    assert executor.request is not None
    prefix = executor.request.committed_prefix
    assert prefix is not None
    assert (prefix.start_action_index, prefix.end_action_index_exclusive, prefix.count) == (
        5,
        8,
        3,
    )
    prefix_array = prefix.as_array(2)
    np.testing.assert_array_equal(prefix_array, [[5.0, 1.0], [6.0, 1.0], [7.0, 1.0]])
    assert (
        prefix.sha256
        == hashlib.sha256(np.asarray(prefix_array, dtype="<f4").tobytes(order="C")).hexdigest()
    )
    assert prefix.sha256 == action_prefix_sha256(prefix_array)

    replacement = FakeBackend(
        offset=100,
        mode=mode,
    ).infer(executor.request)
    assert executor.future is not None
    executor.future.set_result(replacement)
    for action_index in range(6, 11):
        sequence.append(apply(runtime, obs(observation_index, action_index - 1)).copy())
        observation_index += 1
    assert [float(action[0]) for action in sequence] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        108,
        109,
        110,
    ]
    assert runtime.metrics.accepted_responses == 1


@pytest.mark.parametrize("mode", [ExecutionMode.GUIDED_RTC, ExecutionMode.TRAINED_RTC])
def test_guided_mode_mismatch_is_dropped_instead_of_relabeled(mode):
    backend = FakeBackend(mode=mode)
    executor = ManualExecutor()
    runtime = controller(backend, executor, mode=mode)
    warm(runtime)
    observation_index = 3
    for action_index in range(6):
        apply(runtime, obs(observation_index, action_index - 1))
        observation_index += 1
    assert executor.request is not None and executor.future is not None
    mismatched = FakeBackend(mode=ExecutionMode.ASYNC_ALIGNED).infer(
        InferenceRequest(
            **{
                **executor.request.__dict__,
                "committed_prefix": None,
            }
        )
    )
    executor.future.set_result(mismatched)
    action = apply(runtime, obs(observation_index, 5))
    assert action[0] == 6
    assert runtime.state == "rearm_required"
    assert runtime.metrics.dropped_responses == 1


@pytest.mark.parametrize("mode", [ExecutionMode.GUIDED_RTC, ExecutionMode.TRAINED_RTC])
def test_guided_prefix_hash_mismatch_drops_entire_response(mode):
    backend = FakeBackend(mode=mode)
    executor = ManualExecutor()
    runtime = controller(backend, executor, mode=mode)
    warm(runtime)
    observation_index = 3
    for action_index in range(6):
        apply(runtime, obs(observation_index, action_index - 1))
        observation_index += 1
    assert executor.request is not None and executor.future is not None
    response = backend.infer(executor.request)
    executor.future.set_result(replace(response, committed_prefix_sha256="0" * 64))
    action = apply(runtime, obs(observation_index, 5))
    assert action[0] == 6
    assert runtime.state == "rearm_required"
    assert runtime.metrics.dropped_responses == 1
    assert runtime.metrics.accepted_responses == 0


@pytest.mark.parametrize("mode", [ExecutionMode.GUIDED_RTC, ExecutionMode.TRAINED_RTC])
def test_exact_native_change_signal_invalidates_even_when_canonical_bytes_match(mode):
    backend = FakeBackend(mode=mode)
    runtime = controller(backend, mode=mode)
    warm(runtime)
    observation_index = 3
    for action_index in range(5):
        apply(runtime, obs(observation_index, action_index - 1))
        observation_index += 1
    requested = runtime.next_action(obs(observation_index, 4))
    runtime.record_applied(requested.copy(), command_changed=True)
    try:
        assert runtime.state == "rearm_required"
        assert runtime.metrics.applied_overrides == 1
        assert runtime.metrics.prefix_invalidations == 1
        assert runtime.buffer.remaining == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", [ExecutionMode.GUIDED_RTC, ExecutionMode.TRAINED_RTC])
def test_committed_guided_actuator_override_invalidates_result_and_resets_before_rearm(mode):
    backend = FakeBackend(mode=mode)
    runtime = controller(backend, mode=mode)
    warm(runtime)
    observation_index = 3
    for action_index in range(5):
        apply(runtime, obs(observation_index, action_index - 1))
        observation_index += 1

    requested = runtime.next_action(obs(observation_index, 4))
    observation_index += 1
    np.testing.assert_array_equal(requested, [5.0, 1.0])
    # A fake local actuator applies stricter limits than the requested/cloud
    # command. This reported value, not the prediction, becomes hold truth.
    applied = np.array([4.0, 0.5], dtype=np.float32)
    runtime.record_applied(applied)
    assert runtime.state == "rearm_required"
    assert runtime.metrics.applied_overrides == 1
    assert runtime.metrics.prefix_invalidations == 1
    assert runtime.buffer.remaining == 0

    hold = runtime.next_action(obs(observation_index, 5))
    observation_index += 1
    np.testing.assert_array_equal(hold, [0.0, 0.5])
    runtime.record_applied(hold)

    runtime.reconnect_and_rearm(
        backend,
        rearm_observation=obs(observation_index, 6),
        warm_observations=[
            obs(observation_index + 1, 6),
            obs(observation_index + 2, 6),
            obs(observation_index + 3, 6),
        ],
    )
    assert backend.resets and backend.resets[-1][0] == 0
    assert "differed from committed prefix" in backend.resets[-1][2]
    assert backend.rearms == []  # reset ACK permits fresh warm inference without reconnect rearm
    resumed = apply(runtime, obs(observation_index + 4, 6))
    assert resumed[0] == 7
    assert runtime.epoch == 1


def test_controller_rejects_backend_mode_or_dimension_mismatch():
    backend = FakeBackend(mode=ExecutionMode.ASYNC_ALIGNED)
    backend.expected_mode = ExecutionMode.ASYNC_ALIGNED  # type: ignore[attr-defined]
    backend.action_dim = 2  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="modes differ"):
        controller(backend, mode=ExecutionMode.GUIDED_RTC)
    backend.expected_mode = ExecutionMode.GUIDED_RTC  # type: ignore[attr-defined]
    backend.action_dim = 3  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="dimensions differ"):
        controller(backend, mode=ExecutionMode.GUIDED_RTC)
    backend.action_dim = 2  # type: ignore[attr-defined]
    backend.max_committed_prefix_steps = 1  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="prefix limit"):
        controller(backend, mode=ExecutionMode.GUIDED_RTC, committed_steps=2)


def test_controller_rejects_synchronous_mode():
    for mode in (ExecutionMode.SYNCHRONOUS_CHUNKED,):
        with pytest.raises(ValueError, match="supports only"):
            controller(FakeBackend(), mode=mode)


def test_absolute_fallback_repeats_last_actuator_returned_target():
    policy = FallbackPolicy(
        mode="drain_then_hold",
        hold_steps=1,
        hold_behavior=LastAppliedAbsoluteTarget(),
    )
    fallback_executor = FallbackExecutor(
        policy,
        action_dim=2,
        action_semantics="joint_position",
        action_names=("joint", "gripper"),
    )
    applied = np.array([0.25, 0.75], dtype=np.float32)
    fallback_executor.record_applied(applied)
    np.testing.assert_array_equal(fallback_executor.next_hold(), applied)


def test_committed_prefix_empty_hash_is_canonical():
    prefix = CommittedPrefix.from_array(4, np.empty((0, 2), dtype=np.float32))
    assert prefix.to_wire(2) == {
        "start_action_index": 4,
        "end_action_index_exclusive": 4,
        "count": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "actions": [],
    }


def test_late_reply_is_dropped_and_delta_hold_preserves_gripper():
    backend = FakeBackend()
    executor = ManualExecutor()
    runtime = controller(backend, executor, committed_steps=1)
    warm(runtime)
    observation_index = 3
    applied = []
    for action_index in range(10):
        applied.append(apply(runtime, obs(observation_index, action_index - 1)))
        observation_index += 1
    assert executor.request is not None and executor.future is not None
    executor.future.set_result(FakeBackend(offset=100).infer(executor.request))
    hold = apply(runtime, obs(observation_index, 9))
    assert np.array_equal(hold, np.array([0.0, 1.0], dtype=np.float32))
    assert runtime.metrics.late_responses == 1
    assert runtime.metrics.accepted_responses == 0


def test_disconnect_drains_accepted_buffer_then_bounded_hold_and_stop():
    runtime = controller(FakeBackend())
    warm(runtime)
    first = apply(runtime, obs(3, -1))
    assert first[0] == 0
    runtime.mark_disconnected("fixture disconnect")
    # Already accepted cloud actions may drain.
    for index in range(1, 10):
        assert apply(runtime, obs(3 + index, index - 1))[0] == index
    assert np.array_equal(apply(runtime, obs(13, 9)), [0.0, 1.0])
    assert np.array_equal(apply(runtime, obs(14, 10)), [0.0, 1.0])
    with pytest.raises(RuntimeStopped):
        runtime.next_action(obs(15, 11))


def test_nonfinite_async_response_faults_without_execution():
    backend = FakeBackend()
    executor = ManualExecutor()
    runtime = controller(backend, executor)
    warm(runtime)
    observation_index = 3
    for action_index in range(6):
        apply(runtime, obs(observation_index, action_index - 1))
        observation_index += 1
    assert executor.request is not None and executor.future is not None
    bad = backend.infer(executor.request)
    bad = ActionResponse(
        epoch=bad.epoch,
        request_index=bad.request_index,
        based_on_observation_index=bad.based_on_observation_index,
        start_action_index=bad.start_action_index,
        valid_through_action_index_exclusive=bad.valid_through_action_index_exclusive,
        actions=((float("nan"), 1.0), *bad.actions[1:]),
    )
    executor.future.set_result(bad)
    # The invalid cloud result is dropped. The already accepted local action is
    # still executed; the controller enters rearm-required fallback state.
    action = apply(runtime, obs(observation_index, 5))
    assert action[0] == 6
    assert runtime.metrics.dropped_responses == 1
    assert runtime.state == "rearm_required"


def test_buffer_drops_stale_epoch_and_reordered_request():
    buffer = IndexedActionBuffer(action_dim=1)
    buffer.seed(epoch=0, start_index=0, actions=np.arange(4, dtype=np.float32)[:, None])
    request = InferenceRequest(
        epoch=0,
        request_index=2,
        observation=obs(1, -1),
        target_start_action_index=0,
        committed_end_action_index_exclusive=1,
        reply_needed_before_action_index=4,
        client_buffer_end_action_index_exclusive=4,
        max_server_processing_ms=100,
    )
    stale_request = ActionResponse(0, 1, 1, 0, 4, ((10,), (11,), (12,), (13,)))
    assert (
        buffer.splice(
            request=request,
            response=stale_request,
            expected_mode=ExecutionMode.ASYNC_ALIGNED,
        ).reason
        == "stale_request"
    )
    buffer.reset(1)
    stale_epoch = ActionResponse(0, 2, 1, 0, 4, ((10,), (11,), (12,), (13,)))
    assert (
        buffer.splice(
            request=request,
            response=stale_epoch,
            expected_mode=ExecutionMode.ASYNC_ALIGNED,
        ).reason
        == "stale_epoch"
    )


def test_runtime_rejects_second_control_thread():
    runtime = controller(FakeBackend())
    warm(runtime)
    errors = []

    def drive():
        try:
            runtime.next_action(obs(3, -1))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=drive)
    thread.start()
    thread.join()
    assert errors and "control thread" in str(errors[0])


def test_reset_timeout_retains_inflight_work_and_serializes_exactly_one_reset():
    class HeldBackend(FakeBackend):
        def __post_init__(self):
            super().__post_init__()
            self.calls = 0
            self.started = threading.Event()
            self.release = threading.Event()
            self.active = False
            self.reset_overlapped = False

        def infer(self, request):
            self.calls += 1
            if self.calls > 1:
                self.active = True
                self.started.set()
                assert self.release.wait(2)
                self.active = False
            return super().infer(request)

        def reset(self, *, epoch, reset_index, reason):
            self.reset_overlapped = self.active
            return super().reset(
                epoch=epoch,
                reset_index=reset_index,
                reason=reason,
            )

    backend = HeldBackend()
    runtime = controller(backend)
    runtime.warm_start([obs(0, -1)])
    for action_index in range(7):
        apply(runtime, obs(action_index + 1, action_index - 1))
    assert backend.started.wait(1)

    with pytest.raises(ResetPending) as first:
        runtime.reset("explicit local reset", timeout_seconds=0.01)
    assert first.value.epoch == 0 and first.value.reset_index == 1
    assert runtime.reset_pending
    assert runtime.state == "disarmed"
    assert runtime.epoch == 0
    assert backend.resets == []
    with pytest.raises(ResetPending) as repeated:
        runtime.reset("must not create a second reset", timeout_seconds=0)
    assert repeated.value.reset_index == 1
    assert backend.resets == []

    backend.release.set()
    runtime.reset("same pending reset", timeout_seconds=1)
    assert not runtime.reset_pending
    assert runtime.state == "disarmed"
    assert runtime.epoch == 1
    assert backend.resets == [(0, 1, "explicit local reset")]
    assert not backend.reset_overlapped
    assert runtime.buffer.remaining == 0
    assert runtime.metrics.accepted_responses == 0


def test_invalid_reset_ack_never_forges_epoch_progress():
    class BadAckBackend(FakeBackend):
        def reset(self, *, epoch, reset_index, reason):
            self.resets.append((epoch, reset_index, reason))
            return epoch + 2

    backend = BadAckBackend()
    runtime = controller(backend)
    warm(runtime)
    with pytest.raises(RuntimeFault, match="unexpected epoch"):
        runtime.reset("bad ack")
    assert runtime.epoch == 0
    assert runtime.state == "rearm_required"
    assert runtime.reset_pending
    assert backend.resets == [(0, 1, "bad ack")]


def test_reset_increments_epoch_and_reconnect_requires_fresh_rearm_observation():
    backend = FakeBackend()
    runtime = controller(backend)
    warm(runtime)
    runtime.reset("new_episode")
    assert runtime.epoch == 1
    assert backend.resets == [(0, 1, "new_episode")]
    warm(runtime)
    apply(runtime, obs(3, -1))
    runtime.mark_disconnected("lost")
    new_backend = FakeBackend(offset=200, authoritative_epoch=42)
    runtime.reconnect_and_rearm(
        new_backend,
        rearm_observation=obs(4, 0),
        warm_observations=[obs(5, 0), obs(6, 0), obs(7, 0)],
    )
    action = apply(runtime, obs(8, 0))
    assert action[0] == 201
    assert runtime.epoch == 42
    assert new_backend.rearms == [(42, 1)]


def test_process_restart_starts_from_authoritative_hello_epoch():
    backend = FakeBackend(authoritative_epoch=17)
    runtime = controller(backend)
    warm(runtime)
    assert runtime.epoch == 17
    action = apply(runtime, obs(3, -1))
    assert action[0] == 0
