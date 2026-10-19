"""Local index-aligned execution for native async and guided server modes.

The client owns buffering, exact committed-prefix bytes, actuator truth, and
reset/rearm. Guided and trained sampling remain server functionality; their
negotiated conditioning evidence is checked distinctly here.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from .models import (
    ExecutionMode,
    FallbackPolicy,
    LastAppliedAbsoluteTarget,
    ZeroDeltaPreserve,
)

Action = np.ndarray


class AdmissionRejected(RuntimeError):
    """Warm measured turnaround cannot fit the declared action horizon."""


class RuntimeFault(RuntimeError):
    """The runtime detected invalid control data or protocol state."""


class ResetPending(RuntimeFault):
    """The reset is serialized behind unfinished inference and remains pending."""

    def __init__(self, *, epoch: int, reset_index: int) -> None:
        self.epoch = epoch
        self.reset_index = reset_index
        super().__init__(f"reset {reset_index} is pending behind inference in epoch {epoch}")


class RuntimeStopped(RuntimeError):
    """Fallback exhausted and motion must remain stopped."""


class RuntimeState(StrEnum):
    DISARMED = "disarmed"
    WARMING = "warming"
    ARMED = "armed"
    DISCONNECTED = "disconnected"
    REARM_REQUIRED = "rearm_required"
    STOPPED = "stopped"


@dataclass(frozen=True)
class Observation:
    index: int
    observed_after_action_index: int
    state: tuple[float, ...]
    images: tuple[dict[str, Any], ...]
    prompt: str
    capture_monotonic_ns: int

    def validate(self, state_dim: int) -> None:
        if self.index < 0:
            raise RuntimeFault("observation index must be non-negative")
        if len(self.state) != state_dim or not np.isfinite(np.asarray(self.state)).all():
            raise RuntimeFault("observation state has wrong dimension or non-finite values")
        if self.observed_after_action_index < -1:
            raise RuntimeFault("observed_after_action_index must be at least -1")
        if not self.images:
            raise RuntimeFault("at least one image is required")


PREFIX_CONDITIONED_MODES = frozenset({ExecutionMode.GUIDED_RTC, ExecutionMode.TRAINED_RTC})
SUPPORTED_ALIGNED_MODES = frozenset({ExecutionMode.ASYNC_ALIGNED, *PREFIX_CONDITIONED_MODES})


def require_aligned_mode(value: ExecutionMode | str) -> ExecutionMode:
    try:
        mode = ExecutionMode(value)
    except ValueError as exc:
        raise ValueError(f"unsupported local control execution mode: {value!r}") from exc
    if mode not in SUPPORTED_ALIGNED_MODES:
        raise ValueError(
            "local aligned controller supports only async_aligned, guided_rtc or trained_rtc; "
            f"got {mode.value}"
        )
    return mode


def action_prefix_sha256(actions: np.ndarray) -> str:
    array = np.array(actions, dtype="<f4", order="C", copy=True)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise RuntimeFault("committed prefix must be finite row-major actions")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class CommittedPrefix:
    start_action_index: int
    end_action_index_exclusive: int
    actions: tuple[tuple[float, ...], ...]
    sha256: str

    @classmethod
    def from_array(cls, start_action_index: int, actions: np.ndarray) -> CommittedPrefix:
        array = np.array(actions, dtype="<f4", order="C", copy=True)
        if start_action_index < 0 or array.ndim != 2 or not np.isfinite(array).all():
            raise RuntimeFault("invalid committed prefix range or actions")
        array.setflags(write=False)
        rows = tuple(tuple(float(value) for value in row) for row in array)
        return cls(
            start_action_index=start_action_index,
            end_action_index_exclusive=start_action_index + len(array),
            actions=rows,
            sha256=action_prefix_sha256(array),
        )

    @property
    def count(self) -> int:
        return self.end_action_index_exclusive - self.start_action_index

    def as_array(self, action_dim: int) -> np.ndarray:
        array = (
            np.empty((0, action_dim), dtype="<f4")
            if not self.actions
            else np.asarray(self.actions, dtype="<f4")
        )
        if array.shape != (self.count, action_dim):
            raise RuntimeFault("committed prefix range/actions disagree")
        if action_prefix_sha256(array) != self.sha256:
            raise RuntimeFault("committed prefix hash changed")
        return array

    def to_wire(self, action_dim: int) -> dict[str, Any]:
        actions = self.as_array(action_dim)
        return {
            "start_action_index": self.start_action_index,
            "end_action_index_exclusive": self.end_action_index_exclusive,
            "count": self.count,
            "sha256": self.sha256,
            "actions": actions.tolist(),
        }


@dataclass(frozen=True)
class InferenceRequest:
    epoch: int
    request_index: int
    observation: Observation
    target_start_action_index: int
    committed_end_action_index_exclusive: int
    reply_needed_before_action_index: int
    client_buffer_end_action_index_exclusive: int
    max_server_processing_ms: int
    committed_prefix: CommittedPrefix | None = None


@dataclass(frozen=True)
class ActionResponse:
    epoch: int
    request_index: int
    based_on_observation_index: int
    start_action_index: int
    valid_through_action_index_exclusive: int
    actions: tuple[tuple[float, ...], ...]
    generation_mode: str = "async_aligned"
    prefix_consumed_count: int | None = None
    committed_prefix_sha256: str | None = None
    sampler_conditioning: str | None = None

    def as_array(self, action_dim: int, expected_mode: ExecutionMode) -> np.ndarray:
        array = np.asarray(self.actions, dtype=np.float32)
        expected_rows = self.valid_through_action_index_exclusive - self.start_action_index
        if array.shape != (expected_rows, action_dim):
            raise RuntimeFault(
                f"action response shape {array.shape} != {(expected_rows, action_dim)}"
            )
        if expected_rows < 1 or not np.isfinite(array).all():
            raise RuntimeFault("action response is empty or non-finite")
        if self.generation_mode != expected_mode.value:
            raise RuntimeFault(
                f"{expected_mode.value} client received generation mode {self.generation_mode!r}"
            )
        return array


class PolicyBackend(Protocol):
    @property
    def authoritative_epoch(self) -> int: ...

    def infer(self, request: InferenceRequest) -> ActionResponse: ...

    def reset(self, *, epoch: int, reset_index: int, reason: str) -> int: ...

    def rearm(self, *, epoch: int, observation: Observation, rearm_index: int) -> bool: ...


@dataclass(frozen=True)
class SpliceResult:
    accepted: bool
    splice_index: int | None
    inserted_steps: int
    reason: str | None = None


class IndexedActionBuffer:
    """Thread-safe contiguous action buffer with an irrevocable future prefix."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        self.epoch = 0
        self.next_action_index = 0
        self.committed_end_action_index_exclusive = 0
        self._actions: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def end_action_index_exclusive(self) -> int:
        with self._lock:
            return max(self._actions, default=self.next_action_index - 1) + 1

    @property
    def remaining(self) -> int:
        with self._lock:
            return len([index for index in self._actions if index >= self.next_action_index])

    def reset(self, epoch: int, *, next_action_index: int = 0) -> None:
        if next_action_index < 0:
            raise ValueError("next_action_index must be non-negative")
        with self._lock:
            self.epoch = epoch
            self.next_action_index = next_action_index
            self.committed_end_action_index_exclusive = next_action_index
            self._actions.clear()

    def seed(self, *, epoch: int, start_index: int, actions: np.ndarray) -> None:
        self._validate_actions(actions)
        with self._lock:
            if epoch != self.epoch or start_index != self.next_action_index:
                raise RuntimeFault("seed doesn't match current epoch/index")
            self._actions = {
                start_index + offset: action.copy() for offset, action in enumerate(actions)
            }
            self.committed_end_action_index_exclusive = start_index

    def commit_steps(self, count: int) -> int:
        with self._lock:
            end = min(self.next_action_index + max(0, count), self._end_unlocked())
            self.committed_end_action_index_exclusive = max(
                self.committed_end_action_index_exclusive, end
            )
            return self.committed_end_action_index_exclusive

    def pop_indexed(self) -> tuple[int, np.ndarray] | None:
        with self._lock:
            index = self.next_action_index
            action = self._actions.pop(index, None)
            if action is None:
                return None
            self.next_action_index += 1
            if self.committed_end_action_index_exclusive < self.next_action_index:
                self.committed_end_action_index_exclusive = self.next_action_index
            return index, action.copy()

    def pop(self) -> np.ndarray | None:
        indexed = self.pop_indexed()
        return None if indexed is None else indexed[1]

    def snapshot(self, start: int, end: int) -> np.ndarray:
        with self._lock:
            if not self.next_action_index <= start <= end <= self._end_unlocked():
                raise RuntimeFault("committed prefix range is outside the local buffer")
            rows = [self._actions.get(index) for index in range(start, end)]
            if any(row is None for row in rows):
                raise RuntimeFault("committed prefix is not contiguous in the local buffer")
            if not rows:
                return np.empty((0, self.action_dim), dtype=np.float32)
            present = [row for row in rows if row is not None]
            return np.stack(present).astype(np.float32, copy=True)

    def advance_fallback(self) -> int:
        """Advance one local tick after emitting a fallback action."""
        with self._lock:
            if self._actions:
                raise RuntimeFault("cannot advance fallback while accepted actions remain")
            index = self.next_action_index
            self.next_action_index += 1
            self.committed_end_action_index_exclusive = self.next_action_index
            return index

    def splice(
        self,
        *,
        request: InferenceRequest,
        response: ActionResponse,
        expected_mode: ExecutionMode,
    ) -> SpliceResult:
        actions = response.as_array(self.action_dim, expected_mode)
        with self._lock:
            if response.epoch != self.epoch or request.epoch != self.epoch:
                return SpliceResult(False, None, 0, "stale_epoch")
            if response.request_index != request.request_index:
                return SpliceResult(False, None, 0, "stale_request")
            if response.based_on_observation_index != request.observation.index:
                return SpliceResult(False, None, 0, "observation_mismatch")
            if response.start_action_index != request.target_start_action_index:
                return SpliceResult(False, None, 0, "start_index_mismatch")
            if self.next_action_index >= request.reply_needed_before_action_index:
                return SpliceResult(False, None, 0, "late_reply")
            if self.next_action_index >= response.valid_through_action_index_exclusive:
                return SpliceResult(False, None, 0, "expired_reply")
            splice_index = max(
                self.next_action_index,
                request.committed_end_action_index_exclusive,
                response.start_action_index,
            )
            if splice_index >= response.valid_through_action_index_exclusive:
                return SpliceResult(False, None, 0, "empty_suffix")
            for index in [index for index in self._actions if index >= splice_index]:
                del self._actions[index]
            offset = splice_index - response.start_action_index
            suffix = actions[offset:]
            for delta, action in enumerate(suffix):
                self._actions[splice_index + delta] = action.copy()
            self._assert_contiguous_unlocked()
            return SpliceResult(True, splice_index, len(suffix))

    def _validate_actions(self, actions: np.ndarray) -> None:
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise RuntimeFault("action chunk has wrong dimension")
        if len(actions) < 1 or not np.isfinite(actions).all():
            raise RuntimeFault("action chunk is empty or non-finite")

    def _end_unlocked(self) -> int:
        return max(self._actions, default=self.next_action_index - 1) + 1

    def _assert_contiguous_unlocked(self) -> None:
        if not self._actions:
            return
        expected = set(range(self.next_action_index, self._end_unlocked()))
        if set(self._actions) != expected:
            raise RuntimeFault("action buffer became non-contiguous")


@dataclass
class WarmTurnaroundGate:
    controller_hz: float
    action_horizon_steps: int
    safety_steps: int = 1
    samples_seconds: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise RuntimeFault("turnaround measurement must be finite and non-negative")
        self.samples_seconds.append(seconds)

    @property
    def required_steps(self) -> int:
        if not self.samples_seconds:
            raise AdmissionRejected("no warm turnaround measurements")
        measured_steps = max(self.samples_seconds) * self.controller_hz
        return math.ceil(measured_steps - 1e-9) + self.safety_steps

    def require_admissible(self) -> int:
        required = self.required_steps
        if 2 * required + 1 > self.action_horizon_steps:
            raise AdmissionRejected(
                f"warm turnaround requires {required} steps but horizon "
                f"{self.action_horizon_steps} cannot sustain indexed native control"
            )
        return required


class FallbackExecutor:
    """Semantics-aware bounded hold based on the last actually applied target."""

    def __init__(
        self,
        policy: FallbackPolicy,
        *,
        action_dim: int,
        action_semantics: str,
        action_names: tuple[str, ...],
    ) -> None:
        self.policy = policy
        self.action_dim = action_dim
        self.action_semantics = action_semantics
        self.action_names = action_names
        self.last_applied: np.ndarray | None = None
        self.holds_emitted = 0

    def record_applied(self, action: np.ndarray) -> None:
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise RuntimeFault("applied action has wrong shape or non-finite values")
        self.last_applied = action.astype(np.float32, copy=True)

    def reset(self) -> None:
        self.last_applied = None
        self.holds_emitted = 0

    def next_hold(self) -> np.ndarray:
        if self.policy.mode == "immediate_stop" or self.policy.hold_steps == 0:
            raise RuntimeStopped("fallback requires immediate stop")
        if self.last_applied is None:
            raise RuntimeStopped("no applied target exists for fallback")
        if self.holds_emitted >= self.policy.hold_steps:
            raise RuntimeStopped("fallback hold budget exhausted")
        behavior = self.policy.hold_behavior
        if isinstance(behavior, ZeroDeltaPreserve):
            if self.action_semantics not in {"joint_delta", "cartesian_delta"}:
                raise RuntimeFault("zero-delta hold used with non-delta semantics")
            hold = np.zeros(self.action_dim, dtype=np.float32)
            for name in behavior.preserve_channels:
                try:
                    index = self.action_names.index(name)
                except ValueError as exc:
                    raise RuntimeFault(f"unknown preserved channel: {name}") from exc
                hold[index] = self.last_applied[index]
        elif isinstance(behavior, LastAppliedAbsoluteTarget):
            if self.action_semantics != "joint_position":
                raise RuntimeFault("absolute-target hold used with non-position semantics")
            hold = self.last_applied.copy()
        else:
            raise RuntimeFault("drain_then_hold lacks a valid hold behavior")
        self.holds_emitted += 1
        return hold


@dataclass
class RuntimeMetrics:
    accepted_responses: int = 0
    stale_responses: int = 0
    late_responses: int = 0
    dropped_responses: int = 0
    fallback_actions: int = 0
    requests_submitted: int = 0
    applied_overrides: int = 0
    prefix_invalidations: int = 0


@dataclass(frozen=True)
class _PendingAction:
    index: int
    requested: np.ndarray


@dataclass(frozen=True)
class _PendingReset:
    future: Future[int]
    from_epoch: int
    reset_index: int
    state_after_ack: RuntimeState
    requires_fresh_warm: bool


class AsyncAlignedController:
    """One-in-flight indexed scheduler for explicit async or guided server mode."""

    def __init__(
        self,
        backend: PolicyBackend,
        *,
        expected_mode: ExecutionMode,
        controller_hz: float,
        state_dim: int,
        action_dim: int,
        action_horizon_steps: int,
        action_names: tuple[str, ...],
        action_semantics: str,
        fallback: FallbackPolicy,
        action_lower: tuple[float, ...] | None = None,
        action_upper: tuple[float, ...] | None = None,
        committed_steps: int = 1,
        max_server_processing_ms: int = 500,
        executor: ThreadPoolExecutor | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        if len(action_names) != action_dim:
            raise ValueError("action_names length must match action_dim")
        if not 0 <= committed_steps < action_horizon_steps:
            raise ValueError("committed_steps must be within [0, action_horizon_steps)")
        self.expected_mode = require_aligned_mode(expected_mode)
        backend_mode = getattr(backend, "expected_mode", self.expected_mode)
        if backend_mode != self.expected_mode:
            raise ValueError("backend and controller expected execution modes differ")
        backend_action_dim = getattr(backend, "action_dim", action_dim)
        if backend_action_dim != action_dim:
            raise ValueError("backend and controller action dimensions differ")
        backend_max_prefix = getattr(
            backend, "max_committed_prefix_steps", action_horizon_steps - 1
        )
        if committed_steps > backend_max_prefix:
            raise ValueError("committed_steps exceeds the negotiated prefix limit")
        self.backend = backend
        self.controller_hz = controller_hz
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon_steps = action_horizon_steps
        self.committed_steps = committed_steps
        self.max_server_processing_ms = max_server_processing_ms
        self.action_lower = np.asarray(
            action_lower if action_lower is not None else [-np.inf] * action_dim,
            dtype=np.float32,
        )
        self.action_upper = np.asarray(
            action_upper if action_upper is not None else [np.inf] * action_dim,
            dtype=np.float32,
        )
        if (
            self.action_lower.shape != (action_dim,)
            or self.action_upper.shape != (action_dim,)
            or np.any(self.action_lower > self.action_upper)
        ):
            raise ValueError("action bounds must be ordered vectors matching action_dim")
        self.buffer = IndexedActionBuffer(action_dim)
        self.fallback = FallbackExecutor(
            fallback,
            action_dim=action_dim,
            action_semantics=action_semantics,
            action_names=action_names,
        )
        self.state = RuntimeState.DISARMED
        self.epoch = backend.authoritative_epoch
        if self.epoch < 0:
            raise ValueError("backend authoritative_epoch must be non-negative")
        self.buffer.reset(self.epoch)
        self.request_index = 0
        self.reset_index = 0
        self.rearm_index = 0
        self.metrics = RuntimeMetrics()
        self.gate = WarmTurnaroundGate(controller_hz, action_horizon_steps)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tenon-infer"
        )
        self._owns_executor = executor is None
        self._future: Future[ActionResponse] | None = None
        self._orphaned_inference: list[Future[ActionResponse]] = []
        self._request: InferenceRequest | None = None
        self._clock = clock
        self._control_thread: int | None = None
        self._last_observation_index = -1
        self._pending_action: _PendingAction | None = None
        self._guided_commitments: dict[int, np.ndarray] = {}
        self._pending_reset: _PendingReset | None = None
        self._locally_reset_rearm_epoch: int | None = None

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=True)

    @property
    def reset_pending(self) -> bool:
        return self._pending_reset is not None

    def warm_start(self, observations: list[Observation]) -> None:
        self._bind_control_thread()
        if self.state not in {RuntimeState.DISARMED, RuntimeState.REARM_REQUIRED}:
            raise RuntimeFault(f"cannot warm from {self.state}")
        if self._pending_reset is not None:
            raise RuntimeFault("serialized reset must complete before warming")
        self._require_no_orphaned_inference()
        if self._locally_reset_rearm_epoch is not None:
            raise RuntimeFault("committed-prefix override requires explicit rearm before warming")
        if len(observations) < 1:
            raise AdmissionRejected("at least one warm observation is required")
        self.state = RuntimeState.WARMING
        last_response: ActionResponse | None = None
        target_start = self.buffer.next_action_index
        for observation in observations:
            self._validate_observation(observation, expected_after=target_start - 1)
            request = self._make_request(
                observation,
                target_start=target_start,
                committed_end=target_start,
                reply_needed=target_start + self.action_horizon_steps,
                buffer_end=target_start,
            )
            start = self._clock()
            response = self.backend.infer(request)
            self.gate.add(self._clock() - start)
            actions = self._validate_response(request, response)
            if len(actions) != self.action_horizon_steps:
                raise AdmissionRejected("warm response doesn't cover the full action horizon")
            last_response = response
        refill_steps = self.gate.require_admissible()
        assert last_response is not None
        self.buffer.seed(
            epoch=self.epoch,
            start_index=target_start,
            actions=last_response.as_array(self.action_dim, self.expected_mode),
        )
        self._refill_threshold = min(
            self.action_horizon_steps - 1,
            refill_steps + max(0, self.committed_steps),
        )
        self.state = RuntimeState.ARMED

    def next_action(self, observation: Observation) -> np.ndarray:
        """Return one requested action; the caller must report actuator truth next."""
        self._bind_control_thread()
        if self._pending_action is not None:
            raise RuntimeFault("record_applied is required before requesting another action")
        expected_after = self.buffer.next_action_index - 1
        self._validate_observation(observation, expected_after=expected_after)
        self._poll_response()
        if self.state == RuntimeState.ARMED:
            if self._future is None and self.buffer.remaining <= self._refill_threshold:
                self._submit(observation)
            indexed = self.buffer.pop_indexed()
            if indexed is not None:
                return self._return_requested(*indexed)
            self.mark_disconnected("action buffer exhausted")
        if self.state in {RuntimeState.DISCONNECTED, RuntimeState.REARM_REQUIRED}:
            draining = self.buffer.pop_indexed()
            if draining is not None:
                return self._return_requested(*draining)
            hold = self.fallback.next_hold()
            index = self.buffer.advance_fallback()
            self.metrics.fallback_actions += 1
            return self._return_requested(index, hold)
        raise RuntimeStopped(f"runtime isn't armed: {self.state}")

    def record_applied(
        self,
        action: np.ndarray,
        *,
        command_changed: bool = False,
    ) -> None:
        """Record actuator-returned command plus optional exact representation change."""

        self._bind_control_thread()
        if type(command_changed) is not bool:
            raise TypeError("command_changed must be a boolean")
        pending = self._pending_action
        if pending is None:
            raise RuntimeFault("record_applied has no pending requested action")
        applied = np.asarray(action, dtype=np.float32)
        if applied.shape != (self.action_dim,):
            raise RuntimeFault("applied action must be one action vector")
        self._validate_action_bounds(applied)
        expected_prefix = self._guided_commitments.pop(pending.index, None)
        if expected_prefix is not None and not self._same_action_bytes(
            expected_prefix, pending.requested
        ):
            raise RuntimeFault("internal committed-prefix/requested action mismatch")
        overridden = command_changed or not self._same_action_bytes(applied, pending.requested)
        self._pending_action = None
        self.fallback.record_applied(applied)
        if overridden:
            self.metrics.applied_overrides += 1
        if expected_prefix is not None and overridden:
            self._invalidate_guided_prefix(
                f"applied command at index {pending.index} differed from committed prefix"
            )

    def mark_disconnected(self, reason: str) -> None:
        del reason
        if self.state == RuntimeState.STOPPED:
            return
        self.state = RuntimeState.REARM_REQUIRED
        self._request = None
        self._pending_action = None
        self._guided_commitments.clear()
        if self._future is not None:
            if not self._future.cancel():
                self._orphaned_inference.append(self._future)
            self._future = None

    def reset(self, reason: str, *, timeout_seconds: float = 5) -> None:
        """Serialize one reset behind inference; timeout retains explicit pending work."""

        self._bind_control_thread()
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        if self._pending_action is not None:
            raise RuntimeFault("record_applied is required before reset")
        if self._pending_reset is None:
            self._begin_serialized_reset(
                reason,
                state_after_ack=RuntimeState.DISARMED,
                next_action_index=0,
                clear_fallback=True,
                requires_fresh_warm=False,
            )
        self._finish_pending_reset(timeout_seconds=timeout_seconds)

    def reconnect_and_rearm(
        self,
        backend: PolicyBackend,
        *,
        rearm_observation: Observation,
        warm_observations: list[Observation],
    ) -> None:
        self._bind_control_thread()
        if self.state != RuntimeState.REARM_REQUIRED:
            raise RuntimeFault("reconnect requires rearm_required state")
        locally_reset = self._finish_pending_reset()
        self._drain_orphaned_inference(timeout_seconds=5)
        if getattr(backend, "expected_mode", self.expected_mode) != self.expected_mode:
            raise RuntimeFault("rearm backend execution mode differs from the controller")
        if getattr(backend, "action_dim", self.action_dim) != self.action_dim:
            raise RuntimeFault("rearm backend action dimension differs from the controller")
        if self.committed_steps > getattr(
            backend,
            "max_committed_prefix_steps",
            self.action_horizon_steps - 1,
        ):
            raise RuntimeFault("rearm backend prefix limit is smaller than committed_steps")
        self.backend = backend
        authoritative_epoch = backend.authoritative_epoch
        if locally_reset:
            if authoritative_epoch < self.epoch:
                raise RuntimeFault("rearm backend is older than the committed-prefix reset epoch")
        elif authoritative_epoch <= self.epoch:
            raise RuntimeFault("reconnect hello epoch must advance beyond the disconnected epoch")
        self.epoch = authoritative_epoch
        rearm_observation.validate(self.state_dim)
        if not locally_reset:
            self.rearm_index += 1
            if not backend.rearm(
                epoch=authoritative_epoch,
                observation=rearm_observation,
                rearm_index=self.rearm_index,
            ):
                raise RuntimeFault("rearm was not acknowledged")
        if not warm_observations or warm_observations[0].index <= rearm_observation.index:
            raise RuntimeFault("first inference after rearm requires a newer observation")
        self.buffer.reset(
            self.epoch,
            next_action_index=rearm_observation.observed_after_action_index + 1,
        )
        self._locally_reset_rearm_epoch = None
        self._guided_commitments.clear()
        self._pending_action = None
        self.fallback.reset()
        self.gate.samples_seconds.clear()
        self._last_observation_index = rearm_observation.index
        self.state = RuntimeState.REARM_REQUIRED
        self.warm_start(warm_observations)

    def _submit(self, observation: Observation) -> None:
        target_start = self.buffer.next_action_index
        committed_end = self.buffer.commit_steps(self.committed_steps)
        request = self._make_request(
            observation,
            target_start=target_start,
            committed_end=committed_end,
            reply_needed=self.buffer.end_action_index_exclusive,
            buffer_end=self.buffer.end_action_index_exclusive,
        )
        if request.committed_prefix is not None:
            prefix_actions = request.committed_prefix.as_array(self.action_dim)
            for offset, action in enumerate(prefix_actions):
                index = request.committed_prefix.start_action_index + offset
                existing = self._guided_commitments.get(index)
                if existing is not None and not self._same_action_bytes(existing, action):
                    raise RuntimeFault("guided committed action changed before application")
                self._guided_commitments[index] = action.copy()
        self._request = request
        self._future = self._executor.submit(self.backend.infer, request)
        self.metrics.requests_submitted += 1

    def _poll_response(self) -> None:
        if self._future is None or not self._future.done():
            return
        future, request = self._future, self._request
        self._future, self._request = None, None
        if request is None:
            self.metrics.dropped_responses += 1
            return
        try:
            response = future.result()
            self._validate_response(request, response)
            result = self.buffer.splice(
                request=request,
                response=response,
                expected_mode=self.expected_mode,
            )
        except Exception:
            self.metrics.dropped_responses += 1
            self.mark_disconnected("inference failure")
            return
        if result.accepted:
            self.metrics.accepted_responses += 1
        else:
            self.metrics.dropped_responses += 1
            if result.reason and result.reason.startswith("stale"):
                self.metrics.stale_responses += 1
            if result.reason in {"late_reply", "expired_reply", "empty_suffix"}:
                self.metrics.late_responses += 1

    def _make_request(
        self,
        observation: Observation,
        *,
        target_start: int,
        committed_end: int,
        reply_needed: int,
        buffer_end: int,
    ) -> InferenceRequest:
        prefix = None
        if self.expected_mode in PREFIX_CONDITIONED_MODES:
            prefix = CommittedPrefix.from_array(
                target_start,
                self.buffer.snapshot(target_start, committed_end),
            )
        request = InferenceRequest(
            epoch=self.epoch,
            request_index=self.request_index,
            observation=observation,
            target_start_action_index=target_start,
            committed_end_action_index_exclusive=committed_end,
            reply_needed_before_action_index=reply_needed,
            client_buffer_end_action_index_exclusive=buffer_end,
            max_server_processing_ms=self.max_server_processing_ms,
            committed_prefix=prefix,
        )
        self.request_index += 1
        return request

    def _validate_response(self, request: InferenceRequest, response: ActionResponse) -> np.ndarray:
        actions = response.as_array(self.action_dim, self.expected_mode)
        self._validate_action_bounds(actions)
        if response.epoch != request.epoch:
            raise RuntimeFault("response epoch mismatch")
        if response.request_index != request.request_index:
            raise RuntimeFault("response request index mismatch")
        if response.based_on_observation_index != request.observation.index:
            raise RuntimeFault("response observation index mismatch")
        if response.start_action_index != request.target_start_action_index:
            raise RuntimeFault("response start index mismatch")
        prefix = request.committed_prefix
        if self.expected_mode in PREFIX_CONDITIONED_MODES:
            if prefix is None:
                raise RuntimeFault("conditioned request omitted its committed prefix")
            if response.prefix_consumed_count != prefix.count:
                raise RuntimeFault("conditioned response consumed a different prefix count")
            if response.committed_prefix_sha256 != prefix.sha256:
                raise RuntimeFault("conditioned response echoed a different prefix hash")
            conditioning = (
                "trained" if self.expected_mode == ExecutionMode.TRAINED_RTC else "guided"
            )
            if response.sampler_conditioning != conditioning:
                raise RuntimeFault(
                    f"{self.expected_mode.value} response lacks "
                    f"{conditioning} conditioning evidence"
                )
        elif any(
            value is not None
            for value in (
                response.prefix_consumed_count,
                response.committed_prefix_sha256,
                response.sampler_conditioning,
            )
        ):
            raise RuntimeFault("async_aligned response carried conditioning evidence")
        return actions

    def _return_requested(self, index: int, action: np.ndarray) -> np.ndarray:
        requested = np.asarray(action, dtype=np.float32).copy()
        self._pending_action = _PendingAction(index=index, requested=requested)
        return requested.copy()

    @staticmethod
    def _same_action_bytes(left: np.ndarray, right: np.ndarray) -> bool:
        left_bytes = np.asarray(left, dtype="<f4").tobytes(order="C")
        right_bytes = np.asarray(right, dtype="<f4").tobytes(order="C")
        return left_bytes == right_bytes

    def _require_no_orphaned_inference(self) -> None:
        active: list[Future[ActionResponse]] = []
        for future in self._orphaned_inference:
            if future.done():
                with contextlib.suppress(Exception):
                    future.result()
            else:
                active.append(future)
        self._orphaned_inference = active
        if active:
            raise RuntimeFault(
                "backend inference receive is still owned; reset or wait before warming"
            )

    def _drain_orphaned_inference(self, *, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        owned = list(self._orphaned_inference)
        self._orphaned_inference.clear()
        for index, future in enumerate(owned):
            try:
                future.result(timeout=max(0.0, deadline - time.monotonic()))
            except FutureTimeoutError as exc:
                self._orphaned_inference = owned[index:]
                raise RuntimeFault(
                    "backend inference receive remains active; reconnect cannot warm yet"
                ) from exc
            except Exception:
                # The failed backend call still completed and released receive
                # ownership; its result remains intentionally unusable.
                pass

    def _invalidate_guided_prefix(self, reason: str) -> None:
        if self._pending_reset is not None:
            return
        self.metrics.prefix_invalidations += 1
        self._begin_serialized_reset(
            reason,
            state_after_ack=RuntimeState.REARM_REQUIRED,
            next_action_index=self.buffer.next_action_index,
            clear_fallback=False,
            requires_fresh_warm=True,
        )

    def _begin_serialized_reset(
        self,
        reason: str,
        *,
        state_after_ack: RuntimeState,
        next_action_index: int,
        clear_fallback: bool,
        requires_fresh_warm: bool,
    ) -> None:
        if self._pending_reset is not None:
            return
        owned_inference = list(self._orphaned_inference)
        self._orphaned_inference.clear()
        if self._future is not None:
            owned_inference.append(self._future)
        old_epoch = self.epoch
        self._future = None
        self._request = None
        self._pending_action = None
        self._guided_commitments.clear()
        self.buffer.reset(old_epoch, next_action_index=next_action_index)
        self.state = state_after_ack
        self.gate.samples_seconds.clear()
        self._locally_reset_rearm_epoch = None
        if clear_fallback:
            self.fallback.reset()
            self._last_observation_index = -1
        self.reset_index += 1
        reset_index = self.reset_index
        backend = self.backend

        def reset_after_inference() -> int:
            # The backend owns one socket receive lane. An inference timeout in
            # the caller is not completion; retain and drain this exact future
            # before issuing the single reset frame.
            for inference in owned_inference:
                with contextlib.suppress(Exception):
                    inference.result()
            return backend.reset(
                epoch=old_epoch,
                reset_index=reset_index,
                reason=reason,
            )

        future = self._executor.submit(reset_after_inference)
        self._pending_reset = _PendingReset(
            future=future,
            from_epoch=old_epoch,
            reset_index=reset_index,
            state_after_ack=state_after_ack,
            requires_fresh_warm=requires_fresh_warm,
        )

    def _finish_pending_reset(self, timeout_seconds: float = 5) -> bool:
        pending = self._pending_reset
        if pending is None:
            return self._locally_reset_rearm_epoch is not None
        try:
            new_epoch = pending.future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            raise ResetPending(
                epoch=pending.from_epoch,
                reset_index=pending.reset_index,
            ) from exc
        except Exception as exc:
            self.state = RuntimeState.REARM_REQUIRED
            raise RuntimeFault("serialized reset failed; server epoch is unknown") from exc
        if new_epoch != pending.from_epoch + 1:
            self.state = RuntimeState.REARM_REQUIRED
            raise RuntimeFault("reset ACK returned an unexpected epoch")
        next_action_index = self.buffer.next_action_index
        self.epoch = new_epoch
        self.buffer.reset(new_epoch, next_action_index=next_action_index)
        self.request_index = 0
        self.gate.samples_seconds.clear()
        self._pending_reset = None
        self.state = pending.state_after_ack
        self._locally_reset_rearm_epoch = new_epoch if pending.requires_fresh_warm else None
        return pending.requires_fresh_warm

    def _validate_action_bounds(self, actions: np.ndarray) -> None:
        array = np.asarray(actions)
        if array.shape[-1:] != (self.action_dim,) or not np.isfinite(array).all():
            raise RuntimeFault("action has wrong dimension or non-finite values")
        if np.any(array < self.action_lower) or np.any(array > self.action_upper):
            raise RuntimeFault("action is outside deployment physical bounds")

    def _validate_observation(self, observation: Observation, *, expected_after: int) -> None:
        observation.validate(self.state_dim)
        if observation.index <= self._last_observation_index:
            raise RuntimeFault("observation index must increase")
        if observation.observed_after_action_index != expected_after:
            raise RuntimeFault(
                f"observation captured after {observation.observed_after_action_index}, "
                f"expected {expected_after}"
            )
        self._last_observation_index = observation.index

    def _bind_control_thread(self) -> None:
        current = threading.get_ident()
        if self._control_thread is None:
            self._control_thread = current
        elif current != self._control_thread:
            raise RuntimeFault("only the control thread may drive the runtime")
