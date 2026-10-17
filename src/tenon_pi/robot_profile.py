"""Validated, data-only mappings from a supplied robot layout to canonical units.

Profiles are local preparation/actuation inputs, not new REST fields or claims
that a hosted model accepts a layout. No profile can execute Python expressions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .models import Channel, ChannelName, ImageTransform, RobotSchema

Unit = Literal[
    "radian", "radian_per_second", "meter", "meter_per_second", "normalized_0_1", "unitless"
]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class ProfileData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class NativeChannel(ProfileData):
    name: ChannelName
    unit: Literal[
        "degree",
        "millimeter",
        "open_percent",
        "closed_percent",
        "encoder_tick",
        "radian",
        "radian_per_second",
        "meter",
        "meter_per_second",
        "normalized_0_1",
        "unitless",
    ]
    lower: FiniteFloat
    upper: FiniteFloat

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.lower >= self.upper:
            raise ValueError("native calibration bounds must be strictly ordered")
        return self


class Coordinate(ProfileData):
    name: ChannelName
    unit: Unit
    native_index: Annotated[int, Field(ge=0)] | None
    conversion: Literal[
        "identity", "degrees_to_radians", "millimeters_to_meters", "gripper_closedness", "constant"
    ]
    constant: FiniteFloat | None = None
    native_closed: FiniteFloat | None = None
    native_open: FiniteFloat | None = None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.conversion == "constant":
            if self.native_index is not None or self.constant is None:
                raise ValueError("virtual coordinates require a constant and no native index")
        elif self.native_index is None or self.constant is not None:
            raise ValueError("physical coordinates require a native index and no constant")
        if self.conversion == "gripper_closedness":
            if (
                self.unit != "normalized_0_1"
                or self.native_closed is None
                or self.native_open is None
                or self.native_closed == self.native_open
            ):
                raise ValueError(
                    "gripper conversion requires distinct measured closed/open endpoints"
                )
        elif self.native_closed is not None or self.native_open is not None:
            raise ValueError("gripper endpoints only apply to gripper_closedness")
        return self


class VectorMapping(ProfileData):
    native: tuple[NativeChannel, ...] = Field(min_length=1)
    canonical: tuple[Coordinate, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def bijective(self) -> Self:
        for channels in (self.native, self.canonical):
            if len({c.name for c in channels}) != len(channels):
                raise ValueError("channel names must be unique within each vector")
        indices = [c.native_index for c in self.canonical if c.native_index is not None]
        if sorted(indices) != list(range(len(self.native))):
            raise ValueError("mapping must preserve each native coordinate exactly once")
        for c in self.canonical:
            if c.native_index is None:
                if c.unit == "normalized_0_1" and not 0 <= c.constant <= 1:
                    raise ValueError("normalized virtual coordinate outside [0,1]")
                continue
            native = self.native[c.native_index]
            pair = (native.unit, c.unit)
            if c.conversion == "identity" and pair[0] != pair[1]:
                raise ValueError("identity conversion requires identical declared units")
            if c.conversion == "degrees_to_radians" and pair != ("degree", "radian"):
                raise ValueError("degrees_to_radians requires degree -> radian")
            if c.conversion == "millimeters_to_meters" and pair != ("millimeter", "meter"):
                raise ValueError("millimeters_to_meters requires millimeter -> meter")
            if c.conversion == "gripper_closedness":
                if native.unit not in {
                    "open_percent",
                    "closed_percent",
                    "encoder_tick",
                    "meter",
                    "millimeter",
                    "normalized_0_1",
                    "unitless",
                }:
                    raise ValueError("unsupported native gripper unit")
                if sorted((c.native_closed, c.native_open)) != [native.lower, native.upper]:
                    raise ValueError("gripper endpoints must exactly span calibrated bounds")
            if (
                c.unit == "normalized_0_1"
                and c.conversion == "identity"
                and (native.lower < 0 or native.upper > 1)
            ):
                raise ValueError("normalized native bounds outside [0,1]")
        return self

    @staticmethod
    def _vector(values: object, width: int) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (width,) or not np.isfinite(array).all():
            raise ValueError(f"expected finite vector shape [{width}]")
        return array

    def to_canonical(self, values: object) -> np.ndarray:
        native = self._vector(values, len(self.native))
        result = np.empty(len(self.canonical), dtype=np.float32)
        for i, c in enumerate(self.canonical):
            if c.native_index is None:
                result[i] = c.constant
                continue
            value = native[c.native_index]
            if c.conversion == "degrees_to_radians":
                value = np.deg2rad(value)
            elif c.conversion == "millimeters_to_meters":
                value = value / np.float32(1000)
            elif c.conversion == "gripper_closedness":
                # Preserve float32 subtraction/division ordering, including
                # native open-percent's 1 - x/100 representation.
                value = np.float32(1) - (
                    (value - np.float32(c.native_closed))
                    / np.float32(c.native_open - c.native_closed)
                )
            result[i] = value
        return result

    def to_native(self, values: object) -> np.ndarray:
        canonical = self._vector(values, len(self.canonical))
        result = np.empty(len(self.native), dtype=np.float32)
        for i, c in enumerate(self.canonical):
            value = canonical[i]
            if c.native_index is None:
                if value != np.float32(c.constant):
                    raise ValueError("virtual coordinate differs from declared constant")
                continue
            if c.conversion == "degrees_to_radians":
                value = np.rad2deg(value)
            elif c.conversion == "millimeters_to_meters":
                value = value * np.float32(1000)
            elif c.conversion == "gripper_closedness":
                if not 0 <= value <= 1:
                    raise ValueError("gripper closedness outside [0,1]")
                value = np.float32(c.native_closed) + np.float32(
                    c.native_open - c.native_closed
                ) * (np.float32(1) - value)
            result[c.native_index] = value
        return result

    def bounds(self, space: Literal["native", "canonical"]) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray([c.lower for c in self.native], dtype=np.float32)
        upper = np.asarray([c.upper for c in self.native], dtype=np.float32)
        if space == "canonical":
            a, b = self.to_canonical(lower), self.to_canonical(upper)
            return np.minimum(a, b), np.maximum(a, b)
        return lower, upper

    def channels(self) -> list[Channel]:
        low, high = self.bounds("canonical")
        return [
            Channel(name=c.name, unit=c.unit, lower=float(low[i]), upper=float(high[i]))
            for i, c in enumerate(self.canonical)
        ]


class CaptureCamera(ProfileData):
    id: ChannelName
    width: Annotated[int, Field(ge=1)]
    height: Annotated[int, Field(ge=1)]
    serial: Annotated[str, Field(min_length=1)]
    mount: Annotated[str, Field(min_length=1)]
    capture_clock: Annotated[str, Field(min_length=1)]
    rotation_degrees: FiniteFloat
    calibration_digest: Digest


class ProfileImageTransform(ProfileData):
    target_width: Annotated[int, Field(ge=1)]
    target_height: Annotated[int, Field(ge=1)]
    resize_mode: Literal["pad", "center_crop", "stretch"]


class RobotProfile(ProfileData):
    profile_version: Literal["tenon-robot-profile-v1"]
    profile_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]*$")]
    calibration_digest: Digest
    state: VectorMapping
    action: VectorMapping
    action_semantics: Literal["joint_position", "joint_delta", "cartesian_delta"]
    controller_hz: Annotated[FiniteFloat, Field(gt=0, le=500)]
    cameras: tuple[CaptureCamera, ...] = Field(min_length=1)
    image_transform: ProfileImageTransform
    action_horizon_steps: Annotated[int, Field(ge=1, le=512)]
    max_committed_prefix_steps: Annotated[int, Field(ge=0, le=511)]
    model_state_width: Annotated[int, Field(ge=1)]
    model_action_width: Annotated[int, Field(ge=1)]
    output_projection: Literal["strict", "clip_bounds_v1"] = "strict"
    # One entry per canonical action: subtract this captured state coordinate
    # for model-space normalization, or keep absolute when null.
    model_delta_state_indices: tuple[Annotated[int, Field(ge=0)] | None, ...]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if len({c.id for c in self.cameras}) != len(self.cameras):
            raise ValueError("camera IDs must be unique")
        if self.max_committed_prefix_steps >= self.action_horizon_steps:
            raise ValueError("committed prefix must be shorter than action horizon")
        if self.model_state_width < len(self.state.canonical) or self.model_action_width < len(
            self.action.canonical
        ):
            raise ValueError("model widths cannot truncate canonical coordinates")
        if len(self.model_delta_state_indices) != len(self.action.canonical):
            raise ValueError("declare one model delta mapping per canonical action")
        for action, state in zip(
            self.action.canonical, self.model_delta_state_indices, strict=True
        ):
            if state is not None and (
                self.action_semantics != "joint_position"
                or action.native_index is None
                or state >= len(self.state.canonical)
                or self.state.canonical[state].unit != action.unit
            ):
                raise ValueError(
                    "model delta requires physical absolute action and matching state unit"
                )
        return self

    @classmethod
    def load(cls, path: str | Path) -> RobotProfile:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def descriptor(self) -> dict:
        return self.model_dump(mode="json")

    def digest(self) -> str:
        return (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    self.descriptor(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
        )

    def robot_schema(self, normalization_digest: str) -> RobotSchema:
        return RobotSchema(
            controller_hz=self.controller_hz,
            state_channels=self.state.channels(),
            action_channels=self.action.channels(),
            action_semantics=self.action_semantics,
            cameras=[
                dict(
                    id=c.id,
                    required=True,
                    accepted_encodings=["png"],
                    max_width=c.width,
                    max_height=c.height,
                )
                for c in self.cameras
            ],
            image_transform=ImageTransform(**self.image_transform.model_dump()),
            action_horizon_steps=self.action_horizon_steps,
            max_committed_prefix_steps=self.max_committed_prefix_steps,
            transform_digest=self.digest(),
            normalization_digest=normalization_digest,
        )

    def model_state(self, native_state: object) -> np.ndarray:
        canonical = self.state.to_canonical(native_state)
        padded = np.zeros(self.model_state_width, dtype=np.float32)
        padded[: len(canonical)] = canonical
        return padded

    def model_target(self, native_target: object, native_state: object) -> np.ndarray:
        state = self.state.to_canonical(native_state)
        target = self.action.to_canonical(native_target)
        padded = np.zeros(self.model_action_width, dtype=np.float32)
        padded[: len(target)] = self.model_actions(state, target[None])[0]
        return padded

    def canonical_target(
        self, model_action: object, native_state: object, *, project: bool = True
    ) -> np.ndarray:
        """Unnormalized model output → canonical target against captured native state.

        All incoming coordinates are checked before trim/virtual restoration.
        project=False preserves raw finite compatibility output for comparison;
        native serving applies the descriptor-selected projection before its
        schema-bound validation. Demonstrations and ordinary conversions never
        call this projection path.
        """
        action = self.action._vector(model_action, self.model_action_width)
        state = self.state.to_canonical(native_state)
        if not np.isfinite(state).all():
            raise ValueError("captured canonical state must be finite before output inverse")
        target = action[: len(self.action.canonical)].copy()
        with np.errstate(over="ignore", invalid="ignore"):
            for i, state_index in enumerate(self.model_delta_state_indices):
                if state_index is not None:
                    target[i] += state[state_index]
                coordinate = self.action.canonical[i]
                if coordinate.native_index is None:
                    target[i] = coordinate.constant
        if not np.isfinite(target).all():
            raise ValueError("canonical model output is nonfinite after inverse")
        if project and self.output_projection == "clip_bounds_v1":
            lower, upper = self.action.bounds("canonical")
            if not np.isfinite(lower).all() or not np.isfinite(upper).all():
                raise ValueError("projection bounds must be finite float32 coordinates")
            target = np.clip(target, lower, upper).astype(np.float32)
        return target

    def native_target(self, model_action: object, native_state: object) -> np.ndarray:
        target = self.canonical_target(model_action, native_state)
        lower, upper = self.action.bounds("canonical")
        if np.any(target < lower) or np.any(target > upper):
            raise ValueError("model target exceeds canonical bounds under strict projection")
        return self.action._vector(self.action.to_native(target), len(self.action.native))

    def model_actions(self, state: object, targets: object) -> np.ndarray:
        captured = self.state._vector(state, len(self.state.canonical))
        actions = np.asarray(targets, dtype=np.float32).copy()
        if actions.ndim != 2 or actions.shape[1] != len(self.action.canonical):
            raise ValueError("model targets must be canonical action rows")
        for i, state_index in enumerate(self.model_delta_state_indices):
            if state_index is not None:
                actions[:, i] -= captured[state_index]
        return actions
