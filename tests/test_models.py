from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from tenon_pi.models import (
    Channel,
    ControlConnection,
    Dataset,
    DatasetFormat,
    DatasetVersionCreate,
    EvaluationSummary,
    ExecutionMode,
    FallbackPolicy,
    ImageTransform,
    LastAppliedAbsoluteTarget,
    Protocol,
    RobotSchema,
    SessionBootstrap,
    TrainingMethod,
    TrainingSpec,
    ZeroDeltaPreserve,
)


def test_session_bootstrap_is_flat(contract_fixtures):
    bootstrap = SessionBootstrap.model_validate(contract_fixtures["session_bootstrap"])
    assert bootstrap.id == "usr_fixture"
    assert bootstrap.csrf_token is None
    with pytest.raises(ValidationError):
        SessionBootstrap.model_validate(
            {"user": contract_fixtures["session_bootstrap"], "csrf_token": None}
        )


def test_response_ignores_unknown_fields(contract_fixtures):
    payload = {**contract_fixtures["dataset"], "future_field": "ignored"}
    dataset = Dataset.model_validate(payload)
    assert dataset.id == "dst_fixture"
    assert "future_field" not in dataset.model_dump()


def test_request_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainingSpec.model_validate(
            {
                "dataset_version_id": "dsv_fixture",
                "base_model": "openpi/pi05-base",
                "method": "full_ft",
                "preset": "balanced-v1",
                "max_steps": 100,
                "max_runtime_seconds": 3600,
                "seed": 42,
                "checkpoint_interval_steps": 100,
                "rank": 16,
            }
        )


def test_training_method_requires_named_lora_profile():
    common = {
        "dataset_version_id": "dsv_fixture",
        "base_model": "openpi/pi05-base",
        "preset": "balanced-v1",
        "max_steps": 100,
        "max_runtime_seconds": 3600,
        "seed": 42,
        "checkpoint_interval_steps": 100,
    }
    with pytest.raises(ValidationError, match="lora_profile is required"):
        TrainingSpec(**common, method=TrainingMethod.LORA)
    with pytest.raises(ValidationError, match="forbidden"):
        TrainingSpec(
            **common,
            method=TrainingMethod.FULL_FT,
            lora_profile="openpi-pi05-action-lora-v1",
        )


def test_non_finite_channel_bounds_rejected():
    with pytest.raises(ValidationError):
        Channel(name="joint", unit="radian", lower=-math.inf, upper=1.0)


def test_fallback_validates_against_action_semantics():
    schema = RobotSchema(
        controller_hz=20,
        state_channels=[Channel(name="state", unit="unitless", lower=-1, upper=1)],
        action_channels=[
            Channel(name="motion", unit="unitless", lower=-1, upper=1),
            Channel(name="gripper", unit="unitless", lower=-1, upper=1),
        ],
        action_semantics="cartesian_delta",
        cameras=[
            {
                "id": "base",
                "required": True,
                "accepted_encodings": ["jpeg"],
                "color_space": "rgb",
                "max_width": 640,
                "max_height": 480,
            }
        ],
        image_transform=ImageTransform(
            target_width=224,
            target_height=224,
            resize_mode="pad",
        ),
        action_horizon_steps=10,
        max_committed_prefix_steps=5,
        transform_digest="sha256:transform",
        normalization_digest="sha256:norm",
    )
    fallback = FallbackPolicy(
        mode="drain_then_hold",
        hold_steps=5,
        hold_behavior=ZeroDeltaPreserve(preserve_channels=["gripper"]),
    )
    fallback.validate_for_schema(schema)
    invalid = FallbackPolicy(
        mode="drain_then_hold",
        hold_steps=5,
        hold_behavior=LastAppliedAbsoluteTarget(),
    )
    with pytest.raises(ValueError, match="joint_position"):
        invalid.validate_for_schema(schema)


def test_native_v2_rejects_partner_profile():
    schema = RobotSchema(
        controller_hz=20,
        state_channels=[Channel(name="state", unit="unitless", lower=-1, upper=1)],
        action_channels=[Channel(name="action", unit="unitless", lower=-1, upper=1)],
        action_semantics="joint_delta",
        cameras=[
            {
                "id": "base",
                "required": True,
                "accepted_encodings": ["jpeg"],
                "color_space": "rgb",
                "max_width": 640,
                "max_height": 480,
            }
        ],
        image_transform={"target_width": 224, "target_height": 224, "resize_mode": "pad"},
        action_horizon_steps=10,
        max_committed_prefix_steps=5,
        transform_digest="sha256:transform",
        normalization_digest="sha256:norm",
    )
    file = {
        "relative_path": "meta/info.json",
        "size_bytes": 1,
        "sha256": "a" * 64,
        "content_type": "application/json",
    }
    with pytest.raises(ValidationError, match="tenon-native-v1"):
        DatasetVersionCreate(
            format="lerobot_v2",
            kind="demonstration",
            compatibility_profile="pi-partner-data-1caf4699",
            robot_schema=schema,
            files=[file],
        )
    request = DatasetVersionCreate(
        format="lerobot_v2",
        kind="demonstration",
        compatibility_profile="tenon-native-v1",
        robot_schema=schema,
        files=[file],
    )
    assert request.format == DatasetFormat.LEROBOT_V2


def test_evaluation_success_count_is_bounded():
    with pytest.raises(ValidationError, match="cannot exceed"):
        EvaluationSummary(episode_count=2, success_count=3)


def test_control_connection_requires_negotiated_wss():
    with pytest.raises(ValidationError, match="wss"):
        ControlConnection(
            wss_url="https://rest.example.test/v1/control",
            token="fixture-token",
            expires_at="2026-10-14T12:00:00Z",
            protocol=Protocol.TENON_CONTROL,
        )
    connection = ControlConnection(
        wss_url="wss://inference.example.test/session",
        token="fixture-token",
        expires_at="2026-10-14T12:00:00Z",
        protocol=Protocol.TENON_CONTROL,
    )
    assert connection.token.get_secret_value() == "fixture-token"
    assert ExecutionMode.TRAINED_RTC.value == "trained_rtc"
