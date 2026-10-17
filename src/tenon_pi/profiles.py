"""Versioned public robot profiles and reproducible schema digests."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, cast

from .models import Camera, Channel, ImageTransform, RobotSchema

LIBERO_PROFILE_VERSION = "openpi-pi05-libero-h10-v1"
LIBERO_OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
LIBERO_NORMALIZATION_SHA256 = "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
LIBERO_TRANSFORM_SHA256 = "81bd9293ace15b1a9209ddf514c79a88e1bf0c2a7b521a13b2449747644c20d6"
LIBERO_TRANSFORM_FILE_SHA256 = "4157e65cc7d6ba97b585ac8ae5c63a21d4624f7322c865f83883eb1ca0cae58b"
LIBERO_PROJECTION_REFERENCE_COMMIT = "933f91496d2adb15fd591a3f8a04c950e364eab9"


def _profile_file(name: str) -> Traversable:
    return files("tenon_pi").joinpath("profiles", "libero", name)


def _canonical_json_digest(data: dict[str, Any]) -> str:
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_libero_transform_descriptor() -> dict[str, Any]:
    value = json.loads(_profile_file("pi05-libero-transform-v1.json").read_text(encoding="utf-8"))
    return cast(dict[str, Any], value)


def load_libero_normalization_bytes() -> bytes:
    return _profile_file("pi05-libero-norm-stats.json").read_bytes()


def verify_libero_profile_assets() -> dict[str, str]:
    descriptor = load_libero_transform_descriptor()
    transform_sha256 = _canonical_json_digest(descriptor)
    normalization_sha256 = hashlib.sha256(load_libero_normalization_bytes()).hexdigest()
    if transform_sha256 != LIBERO_TRANSFORM_SHA256:
        raise ValueError("LIBERO transform descriptor digest mismatch")
    if normalization_sha256 != LIBERO_NORMALIZATION_SHA256:
        raise ValueError("LIBERO normalization asset digest mismatch")
    provenance = descriptor.get("provenance", {})
    if provenance.get("normalization_sha256") != normalization_sha256:
        raise ValueError("transform descriptor doesn't bind the normalization asset")
    if descriptor.get("openpi_revision") != LIBERO_OPENPI_REVISION:
        raise ValueError("transform descriptor OpenPI revision mismatch")
    if descriptor.get("protocol_output") != {
        "tenon-control-v1": "projected",
        "openpi-ws-v1": "raw",
    }:
        raise ValueError("LIBERO per-protocol output semantics mismatch")
    projection = descriptor.get("action", {}).get("output_projection", {})
    if projection != {
        "type": "componentwise_clip",
        "stage": "after_unnormalization_and_dimension_trim",
        "lower": [-1.0] * 7,
        "upper": [1.0] * 7,
        "applies_to": ["synchronous_chunked", "async_aligned", "guided_rtc"],
        "committed_prefix_space": "actually_applied_projected_commands",
        "raw_flow_sampler_unchanged": True,
        "controller_equivalence": projection.get("controller_equivalence"),
    }:
        raise ValueError("LIBERO output projection descriptor mismatch")
    return {
        "profile_version": LIBERO_PROFILE_VERSION,
        "openpi_revision": LIBERO_OPENPI_REVISION,
        "transform_digest": f"sha256:{transform_sha256}",
        "normalization_digest": f"sha256:{normalization_sha256}",
    }


def libero_robot_schema() -> RobotSchema:
    digests = verify_libero_profile_assets()
    return RobotSchema(
        controller_hz=20,
        state_channels=[
            Channel(name="eef_x", unit="meter", lower=-1.0, upper=1.0),
            Channel(name="eef_y", unit="meter", lower=-1.0, upper=1.0),
            Channel(name="eef_z", unit="meter", lower=0.0, upper=2.0),
            Channel(name="eef_rx", unit="radian", lower=-4.0, upper=4.0),
            Channel(name="eef_ry", unit="radian", lower=-4.0, upper=4.0),
            Channel(name="eef_rz", unit="radian", lower=-4.0, upper=4.0),
            Channel(name="gripper_qpos_0", unit="meter", lower=-0.05, upper=0.05),
            Channel(name="gripper_qpos_1", unit="meter", lower=-0.05, upper=0.05),
        ],
        action_channels=[
            Channel(name="delta_x", unit="unitless", lower=-1.0, upper=1.0),
            Channel(name="delta_y", unit="unitless", lower=-1.0, upper=1.0),
            Channel(name="delta_z", unit="unitless", lower=-1.0, upper=1.0),
            Channel(name="delta_rx", unit="unitless", lower=-1.0, upper=1.0),
            Channel(name="delta_ry", unit="unitless", lower=-1.0, upper=1.0),
            Channel(name="delta_rz", unit="unitless", lower=-1.0, upper=1.0),
            Channel(name="gripper", unit="unitless", lower=-1.0, upper=1.0),
        ],
        action_semantics="cartesian_delta",
        cameras=[
            Camera(
                id="base",
                required=True,
                accepted_encodings={"jpeg"},
                max_width=256,
                max_height=256,
            ),
            Camera(
                id="left_wrist",
                required=True,
                accepted_encodings={"jpeg"},
                max_width=256,
                max_height=256,
            ),
        ],
        image_transform=ImageTransform(
            target_width=224,
            target_height=224,
            resize_mode="pad",
        ),
        action_horizon_steps=10,
        max_committed_prefix_steps=5,
        transform_digest=digests["transform_digest"],
        normalization_digest=digests["normalization_digest"],
    )


def write_libero_robot_schema(path: str | Path) -> RobotSchema:
    schema = libero_robot_schema()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(schema.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return schema


def libero_profile_provenance() -> dict[str, Any]:
    return {
        **verify_libero_profile_assets(),
        "descriptor_resource": "tenon_pi/profiles/libero/pi05-libero-transform-v1.json",
        "normalization_resource": "tenon_pi/profiles/libero/pi05-libero-norm-stats.json",
        "transform_file_sha256": LIBERO_TRANSFORM_FILE_SHA256,
        "output_projection": load_libero_transform_descriptor()["action"]["output_projection"],
        "protocol_output": load_libero_transform_descriptor()["protocol_output"],
        "projection_reference_commit": LIBERO_PROJECTION_REFERENCE_COMMIT,
        "normalization_source": (
            "gs://openpi-assets/checkpoints/pi05_libero/assets/"
            "physical-intelligence/libero/norm_stats.json"
        ),
    }
