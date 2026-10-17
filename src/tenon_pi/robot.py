"""Profile-driven local actuation and exact canonical/native command receipts."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .robot_profile import VectorMapping


class RobotBoundaryError(RuntimeError):
    pass


class ControllerFault(RobotBoundaryError):
    pass


def _vector(values: object, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise RobotBoundaryError(f"command must be finite shape [{width}]")
    return array


@dataclass(frozen=True)
class NativeCommandTrace:
    """Immutable submitted and driver-reported native commands, not telemetry."""

    submitted: tuple[float, ...]
    applied: tuple[float, ...]

    @classmethod
    def from_arrays(cls, submitted: np.ndarray, applied: np.ndarray) -> NativeCommandTrace:
        a = _vector(submitted, len(submitted))
        b = _vector(applied, len(a))
        return cls(tuple(float(x) for x in a), tuple(float(x) for x in b))

    def submitted_array(self) -> np.ndarray:
        return np.asarray(self.submitted, dtype=np.float32).copy()

    def applied_array(self) -> np.ndarray:
        return np.asarray(self.applied, dtype=np.float32).copy()


@dataclass(frozen=True)
class AppliedCommandResult:
    canonical_float32: bytes
    native_changed: bool
    submitted_changed: bool
    driver_changed: bool

    def canonical_array(self) -> np.ndarray:
        return np.frombuffer(self.canonical_float32, dtype="<f4").copy()


@dataclass(frozen=True)
class CommandEncodingReceipt:
    """An exact canonical command token bound to its lossy native representation."""

    mapping: VectorMapping
    canonical_float32: bytes
    encoded_native_float32: bytes

    @classmethod
    def from_canonical(cls, values: object, mapping: VectorMapping) -> CommandEncodingReceipt:
        canonical = _vector(values, len(mapping.canonical)).copy()
        lower, upper = mapping.bounds("canonical")
        if np.any(canonical < lower) or np.any(canonical > upper):
            raise RobotBoundaryError("canonical command exceeds calibrated profile bounds")
        encoded = mapping.to_native(canonical)
        return cls(mapping, canonical.astype("<f4").tobytes(), encoded.astype("<f4").tobytes())

    def canonical_request(self) -> np.ndarray:
        return np.frombuffer(self.canonical_float32, dtype="<f4").copy()

    def encoded_native(self) -> np.ndarray:
        return np.frombuffer(self.encoded_native_float32, dtype="<f4").copy()

    def resolve_application(self, trace: NativeCommandTrace) -> AppliedCommandResult:
        width = len(self.mapping.native)
        submitted = _vector(trace.submitted, width).astype("<f4").tobytes()
        applied = _vector(trace.applied, width).astype("<f4").tobytes()
        submitted_changed = submitted != self.encoded_native_float32
        driver_changed = applied != submitted
        changed = submitted_changed or driver_changed
        # A genuine native change can alias in canonical float32. Keep an
        # independent signal so conditioned work is invalidated in that case.
        canonical = (
            self.mapping.to_canonical(trace.applied_array()).astype("<f4").tobytes()
            if changed
            else self.canonical_float32
        )
        return AppliedCommandResult(canonical, changed, submitted_changed, driver_changed)


def validate_capture_skew(
    timestamps_ns: dict[str, int], *, required: tuple[str, ...], max_skew_ms: float
) -> None:
    if not required or len(set(required)) != len(required):
        raise ValueError("declare distinct required capture sources")
    if not np.isfinite(max_skew_ms) or max_skew_ms < 0:
        raise ValueError("declare a finite nonnegative capture skew limit")
    if set(required) - set(timestamps_ns):
        raise RobotBoundaryError("missing synchronized captures")
    values = [timestamps_ns[name] for name in required]
    if any(type(v) is not int or v < 0 for v in values):
        raise RobotBoundaryError("capture times must be nonnegative monotonic nanoseconds")
    if max(values) - min(values) > max_skew_ms * 1_000_000:
        raise RobotBoundaryError("camera/state acquisition skew exceeds declared limit")


@dataclass(frozen=True)
class CommandLimiter:
    """Explicit native-unit bounds and per-tick slew; no guessed hardware limits."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    max_delta: tuple[float, ...]

    def __post_init__(self) -> None:
        width = len(self.lower)
        lower, upper, delta = (_vector(v, width) for v in (self.lower, self.upper, self.max_delta))
        if not width or np.any(lower > upper) or np.any(delta <= 0):
            raise ValueError("ordered calibration bounds and positive per-channel slew required")
        for name, value in (("lower", lower), ("upper", upper), ("max_delta", delta)):
            object.__setattr__(self, name, tuple(float(v) for v in value))

    def limit(self, requested: object, last_applied: object) -> np.ndarray:
        request = _vector(requested, len(self.lower))
        previous = _vector(last_applied, len(self.lower))
        if np.any(previous < self.lower) or np.any(previous > self.upper):
            raise RobotBoundaryError("last applied command exceeds local limits")
        bounded = np.clip(request, self.lower, self.upper)
        delta = np.asarray(self.max_delta, dtype=np.float32)
        return np.asarray(previous + np.clip(bounded - previous, -delta, delta), dtype=np.float32)


class RobotDriver(Protocol):
    """A supplied adapter must report actual connection/arming and applied values.

    Multi-device adapters own group ordering and stop *all* connected actuators
    on any partial failure. Do not implement these flags as unconditional true.
    """

    @property
    def connected(self) -> bool: ...
    @property
    def armed(self) -> bool: ...
    def send_action(self, action: dict[str, float]) -> dict[str, float]: ...
    def stop(self) -> None: ...


class RobotController:
    def __init__(
        self,
        driver: RobotDriver,
        mapping: VectorMapping,
        limiter: CommandLimiter,
        initial_applied: object,
    ) -> None:
        self.driver, self.mapping, self.limiter = driver, mapping, limiter
        self.names = tuple(c.name for c in mapping.native)
        if len(limiter.lower) != len(self.names):
            raise ValueError("limiter width must match the supplied action mapping")
        lower, upper = mapping.bounds("native")
        if np.any(np.asarray(limiter.lower) < lower) or np.any(np.asarray(limiter.upper) > upper):
            raise ValueError("local limits must be inside profile calibration")
        self.last_applied = _vector(initial_applied, len(self.names)).copy()
        if np.any(self.last_applied < limiter.lower) or np.any(self.last_applied > limiter.upper):
            raise ValueError("initial applied command exceeds local limits")
        self.faulted = False
        self._require_armed()

    def _require_armed(self) -> None:
        if self.faulted or not self.driver.connected or not self.driver.armed:
            raise ControllerFault("controller requires a connected, explicitly armed driver")

    def apply_with_trace(self, requested: object) -> NativeCommandTrace:
        try:
            self._require_armed()
            submitted = self.limiter.limit(requested, self.last_applied).copy()
            result = self.driver.send_action(dict(zip(self.names, submitted.tolist(), strict=True)))
            if set(result) != set(self.names):
                raise RobotBoundaryError("driver must return exactly all native action channels")
            applied = _vector([result[name] for name in self.names], len(self.names))
            if np.any(applied < self.limiter.lower) or np.any(applied > self.limiter.upper):
                raise RobotBoundaryError("driver returned command outside local calibration bounds")
        except Exception as exc:
            self.faulted = True
            with contextlib.suppress(Exception):
                self.driver.stop()
            raise ControllerFault(
                "actuation failed; adapter stop requested for all actuators"
            ) from exc
        self.last_applied = applied.copy()
        return NativeCommandTrace.from_arrays(submitted, applied)

    def apply(self, requested: object) -> np.ndarray:
        return self.apply_with_trace(requested).applied_array()

    def stop(self) -> None:
        self.faulted = True
        self.driver.stop()
