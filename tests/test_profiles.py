from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tenon_pi.libero_runtime import make_libero_observation
from tenon_pi.models import RobotSchema
from tenon_pi.profiles import (
    LIBERO_NORMALIZATION_SHA256,
    LIBERO_OPENPI_REVISION,
    LIBERO_PROJECTION_REFERENCE_COMMIT,
    LIBERO_TRANSFORM_FILE_SHA256,
    LIBERO_TRANSFORM_SHA256,
    libero_profile_provenance,
    libero_robot_schema,
    load_libero_normalization_bytes,
    load_libero_transform_descriptor,
    verify_libero_profile_assets,
    write_libero_robot_schema,
)


def _canonical_digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def test_released_assets_and_canonical_transform_digests_match_constants():
    descriptor = load_libero_transform_descriptor()
    normalization = load_libero_normalization_bytes()
    assert _canonical_digest(descriptor) == LIBERO_TRANSFORM_SHA256
    assert hashlib.sha256(normalization).hexdigest() == LIBERO_NORMALIZATION_SHA256
    assert descriptor["openpi_revision"] == LIBERO_OPENPI_REVISION
    descriptor_path = (
        Path(__file__).parents[1]
        / "src"
        / "tenon_pi"
        / "profiles"
        / "libero"
        / "pi05-libero-transform-v1.json"
    )
    assert hashlib.sha256(descriptor_path.read_bytes()).hexdigest() == LIBERO_TRANSFORM_FILE_SHA256
    assert descriptor["provenance"]["normalization_sha256"] == LIBERO_NORMALIZATION_SHA256
    assert verify_libero_profile_assets() == {
        "profile_version": "openpi-pi05-libero-h10-v1",
        "openpi_revision": LIBERO_OPENPI_REVISION,
        "transform_digest": f"sha256:{LIBERO_TRANSFORM_SHA256}",
        "normalization_digest": f"sha256:{LIBERO_NORMALIZATION_SHA256}",
    }


def test_generated_schema_matches_shipped_schema(tmp_path: Path):
    generated = tmp_path / "schema.json"
    write_libero_robot_schema(generated)
    shipped = Path(__file__).parents[1] / "schemas" / "libero-robot-schema.json"
    generated_schema = RobotSchema.model_validate_json(generated.read_text())
    shipped_schema = RobotSchema.model_validate_json(shipped.read_text())
    assert generated_schema == shipped_schema
    schema = libero_robot_schema()
    assert schema.controller_hz == 20
    assert len(schema.state_channels) == 8
    assert len(schema.action_channels) == 7
    assert schema.action_semantics == "cartesian_delta"
    assert [camera.id for camera in schema.cameras] == ["base", "left_wrist"]
    assert schema.image_transform.target_width == schema.image_transform.target_height == 224
    assert schema.action_horizon_steps == 10
    assert schema.transform_digest == f"sha256:{LIBERO_TRANSFORM_SHA256}"
    assert schema.normalization_digest == f"sha256:{LIBERO_NORMALIZATION_SHA256}"
    shipped_provenance = json.loads(
        (Path(__file__).parents[1] / "schemas" / "libero-profile-provenance.json").read_text()
    )
    assert shipped_provenance == libero_profile_provenance()


def test_runtime_camera_ids_match_shipped_profile():
    raw = {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    observation = make_libero_observation(
        raw,
        prompt="task",
        observation_index=0,
        observed_after_action_index=-1,
        capture_monotonic_ns=1,
    )
    assert [image["camera_id"] for image in observation.images] == ["base", "left_wrist"]
    assert [camera.id for camera in libero_robot_schema().cameras] == ["base", "left_wrist"]


def test_declared_output_projection_is_explicit_idempotent_and_bounded():
    descriptor = load_libero_transform_descriptor()
    projection = descriptor["action"]["output_projection"]
    assert descriptor["protocol_output"] == {
        "tenon-control-v1": "projected",
        "openpi-ws-v1": "raw",
    }
    assert projection["type"] == "componentwise_clip"
    assert projection["stage"] == "after_unnormalization_and_dimension_trim"
    assert projection["lower"] == [-1.0] * 7
    assert projection["upper"] == [1.0] * 7
    assert projection["applies_to"] == [
        "synchronous_chunked",
        "async_aligned",
        "guided_rtc",
    ]
    assert projection["committed_prefix_space"] == "actually_applied_projected_commands"
    assert projection["raw_flow_sampler_unchanged"] is True

    raw = np.array([[1.009, -1.009, 0.25, 0.0, -0.5, 0.9, 1.009]], dtype=np.float64)
    assert np.any(raw > 1.0) and np.any(raw < -1.0)  # strict native schema rejects raw
    openpi_compat_output = raw.copy()
    applied = np.clip(raw, projection["lower"], projection["upper"])
    assert np.array_equal(applied, [[1.0, -1.0, 0.25, 0.0, -0.5, 0.9, 1.0]])
    assert np.array_equal(np.clip(applied, projection["lower"], projection["upper"]), applied)
    assert np.array_equal(raw, [[1.009, -1.009, 0.25, 0.0, -0.5, 0.9, 1.009]])
    assert np.array_equal(openpi_compat_output, raw)  # raw openpi-ws-v1 behavior
    assert not np.array_equal(applied, openpi_compat_output)  # native physical behavior


def test_profile_provenance_points_to_real_released_asset():
    provenance = libero_profile_provenance()
    assert provenance["normalization_source"].endswith(
        "pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
    )
    assert provenance["openpi_revision"] == LIBERO_OPENPI_REVISION
    assert provenance["projection_reference_commit"] == LIBERO_PROJECTION_REFERENCE_COMMIT
