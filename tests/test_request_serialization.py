from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from typer.testing import CliRunner

from tenon_pi import cli
from tenon_pi.client import TenonClient
from tenon_pi.models import (
    Ack,
    Camera,
    DatasetCreate,
    DatasetVersionCreate,
    DeploymentCreate,
    EvaluationCreate,
    EvaluationSummary,
    ExecutionMode,
    FallbackPolicy,
    FileManifest,
    ImageTransform,
    LastAppliedAbsoluteTarget,
    Protocol,
    RequestModel,
    RobotSchema,
    TrainingSpec,
    ZeroDeltaPreserve,
)
from tenon_pi.transport import RestTransport

ROOT = Path(__file__).parents[1]
OPENAPI = json.loads((ROOT / "spec" / "openapi.json").read_text())
runner = CliRunner()


def validate_schema(name: str, value: object) -> None:
    Draft202012Validator(
        {
            "$ref": f"#/components/schemas/{name}",
            "components": OPENAPI["components"],
        }
    ).validate(value)


def robot_schema(*, omit_defaults: bool = False) -> RobotSchema:
    value = json.loads((ROOT / "schemas" / "libero-robot-schema.json").read_text())
    if omit_defaults:
        value.pop("schema_version")
        value["image_transform"].pop("color_space")
        value["image_transform"].pop("dtype")
        for camera in value["cameras"]:
            camera.pop("color_space")
    return RobotSchema.model_validate(value)


def mock_client(handler: Any) -> TenonClient:
    http_client = httpx.Client(
        base_url="https://api.fixture.invalid/v1/",
        transport=httpx.MockTransport(handler),
    )
    transport = RestTransport(
        "fixture-api-key",
        base_url="https://api.fixture.invalid/v1",
        client=http_client,
    )
    return TenonClient("fixture-api-key", transport=transport)


def full_ft_spec() -> TrainingSpec:
    return TrainingSpec(
        dataset_version_id="dsv_fixture",
        base_model="openpi/pi05-base",
        method="full_ft",
        preset="balanced-v1",
        max_steps=100,
        max_runtime_seconds=7200,
        seed=42,
        checkpoint_interval_steps=100,
    )


def test_wire_required_default_metadata_matches_frozen_openapi() -> None:
    schemas = OPENAPI["components"]["schemas"]
    cases = [
        (Camera, "Camera"),
        (ImageTransform, "ImageTransform"),
        (RobotSchema, "RobotSchema"),
        (DatasetCreate, "DatasetCreate"),
        (DatasetVersionCreate, "DatasetVersionCreate"),
        (TrainingSpec, "TrainingSpec"),
    ]
    for model, schema_name in cases:
        defaulted = {name for name, field in model.model_fields.items() if not field.is_required()}
        required = set(schemas[schema_name]["required"])
        assert model.wire_required_defaults == defaulted & required

    hold_variants = schemas["HoldBehavior"]["oneOf"]
    assert all(
        set(variant["required"]) == {"type", "preserve_channels"} for variant in hold_variants
    )
    assert ZeroDeltaPreserve.wire_required_defaults == {"type"}
    assert LastAppliedAbsoluteTarget.wire_required_defaults == {"type", "preserve_channels"}


def test_sdk_emits_required_nullable_deployment_and_nested_fallback(
    contract_fixtures: dict[str, Any],
) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            202,
            headers={"X-Request-Id": "req_deployment_null"},
            json=contract_fixtures["deployment"],
        )

    request = DeploymentCreate(
        checkpoint_id="ckp_fixture",
        name="nullable defaults",
        robot_schema=robot_schema(omit_defaults=True),
        default_prompt=None,
        protocols={Protocol.TENON_CONTROL},
        execution_modes={ExecutionMode.ASYNC_ALIGNED},
        fallback=FallbackPolicy(mode="immediate_stop", hold_steps=0, hold_behavior=None),
        warm_idle_seconds=0,
    )
    deployment = mock_client(handler).create_deployment(
        request, idempotency_key="deployment-null-0001"
    )

    assert deployment.id == "dep_fixture"
    validate_schema("DeploymentCreate", seen)
    assert seen["default_prompt"] is None
    assert seen["fallback"] == {
        "mode": "immediate_stop",
        "hold_steps": 0,
        "hold_behavior": None,
    }
    assert seen["robot_schema"]["schema_version"] == "robot-schema-v1"
    assert seen["robot_schema"]["image_transform"]["color_space"] == "rgb"
    assert seen["robot_schema"]["image_transform"]["dtype"] == "uint8"
    assert all(camera["color_space"] == "rgb" for camera in seen["robot_schema"]["cameras"])


def test_sdk_required_defaults_emit_while_optional_defaults_omit(
    contract_fixtures: dict[str, Any],
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if request.url.path.endswith("/versions"):
            return httpx.Response(
                201,
                headers={"X-Request-Id": "req_version_defaults"},
                json={
                    "version": contract_fixtures["dataset_version"],
                    "upload": contract_fixtures["upload"],
                },
            )
        return httpx.Response(
            201,
            headers={"X-Request-Id": "req_quote_defaults"},
            json=contract_fixtures["quote"],
        )

    client = mock_client(handler)
    client.create_dataset_version(
        "dst_fixture",
        DatasetVersionCreate(
            kind="demonstration",
            compatibility_profile="tenon-native-v1",
            robot_schema=robot_schema(omit_defaults=True),
            files=[
                FileManifest(
                    relative_path="meta/info.json",
                    size_bytes=2,
                    sha256="0" * 64,
                    content_type="application/json",
                )
            ],
        ),
        idempotency_key="version-defaults-0001",
    )
    client.quote_training(full_ft_spec(), idempotency_key="quote-defaults-0001")

    version_body, quote_body = bodies
    validate_schema("DatasetVersionCreate", version_body)
    validate_schema("QuoteRequest", quote_body)
    assert version_body["format"] == "lerobot_v3"
    assert "source_evaluation_ids" not in version_body
    assert quote_body["kind"] == "training_job"
    assert "lora_profile" not in quote_body["spec"]


def test_optional_omitted_and_explicit_null_are_distinct_http_bodies(
    contract_fixtures: dict[str, Any],
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            202,
            headers={"X-Request-Id": f"req_evaluation_{len(bodies)}"},
            json=contract_fixtures["evaluation"],
        )

    client = mock_client(handler)
    common: dict[str, Any] = {
        "dataset_version_id": "dsv_eval_fixture",
        "checkpoint_id": "ckp_fixture",
        "deployment_id": "dep_fixture",
    }
    client.create_evaluation(
        EvaluationCreate(
            **common,
            summary=EvaluationSummary(episode_count=10, success_count=8),
        ),
        idempotency_key="evaluation-omitted-0001",
    )
    client.create_evaluation(
        EvaluationCreate(
            **common,
            control_session_id=None,
            summary=EvaluationSummary(
                episode_count=10,
                success_count=8,
                mean_episode_seconds=None,
            ),
        ),
        idempotency_key="evaluation-null-0001",
    )

    omitted, explicit_null = bodies
    validate_schema("EvaluationCreate", omitted)
    validate_schema("EvaluationCreate", explicit_null)
    assert "control_session_id" not in omitted
    assert "mean_episode_seconds" not in omitted["summary"]
    assert "labels" not in omitted
    assert explicit_null["control_session_id"] is None
    assert explicit_null["summary"]["mean_episode_seconds"] is None
    assert "labels" not in explicit_null


def test_optional_nonnullable_default_omits_but_explicit_value_emits(
    contract_fixtures: dict[str, Any],
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            201,
            headers={"X-Request-Id": f"req_dataset_{len(bodies)}"},
            json=contract_fixtures["dataset"],
        )

    client = mock_client(handler)
    client.create_dataset(DatasetCreate(name="omitted"), idempotency_key="dataset-omitted-0001")
    client.create_dataset(
        DatasetCreate(name="explicit", description=""),
        idempotency_key="dataset-explicit-0001",
    )

    assert bodies == [{"name": "omitted"}, {"name": "explicit", "description": ""}]
    for body in bodies:
        validate_schema("DatasetCreate", body)
    with pytest.raises(ValidationError):
        DatasetCreate(name="invalid", description=None)  # type: ignore[arg-type]


def test_explicit_null_survives_patch_without_implicit_optional_defaults() -> None:
    class PatchBody(RequestModel):
        note: str | None = None
        label: str = "unchanged"

    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"X-Request-Id": f"req_patch_{len(bodies)}"},
            json={"ok": True},
        )

    transport = mock_client(handler)._transport
    transport.request("PATCH", "fixture", response_model=Ack, expected_status=200, body=PatchBody())
    transport.request(
        "PATCH",
        "fixture",
        response_model=Ack,
        expected_status=200,
        body=PatchBody(note=None),
    )
    transport.request(
        "PATCH",
        "fixture",
        response_model=Ack,
        expected_status=200,
        body=PatchBody(label="unchanged"),
    )

    assert bodies == [{}, {"note": None}, {"label": "unchanged"}]
    with pytest.raises(ValidationError):
        PatchBody(label=None)  # type: ignore[arg-type]


def test_cli_deployment_required_null_reaches_http(
    monkeypatch: pytest.MonkeyPatch,
    contract_fixtures: dict[str, Any],
    tmp_path: Path,
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            202,
            headers={"X-Request-Id": "req_cli_deployment_null"},
            json=contract_fixtures["deployment"],
        )

    request_file = tmp_path / "deployment.json"
    request_file.write_text(
        json.dumps(
            {
                "checkpoint_id": "ckp_fixture",
                "name": "CLI required null",
                "robot_schema": robot_schema(omit_defaults=True).model_dump(
                    mode="json",
                    exclude={
                        "schema_version": True,
                        "cameras": {"__all__": {"color_space"}},
                        "image_transform": {"color_space", "dtype"},
                    },
                ),
                "default_prompt": None,
                "protocols": ["tenon-control-v1"],
                "execution_modes": ["async_aligned"],
                "fallback": {
                    "mode": "immediate_stop",
                    "hold_steps": 0,
                    "hold_behavior": None,
                },
                "warm_idle_seconds": 0,
            }
        )
    )
    client = mock_client(handler)
    monkeypatch.setattr(cli, "_client", lambda _ctx: client)
    result = runner.invoke(
        cli.app,
        [
            "--api-key",
            "fixture",
            "deployment",
            "create",
            "--request",
            str(request_file),
            "--no-wait",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(bodies) == 1
    validate_schema("DeploymentCreate", bodies[0])
    assert bodies[0]["default_prompt"] is None
    assert bodies[0]["fallback"]["hold_behavior"] is None
    assert bodies[0]["robot_schema"]["schema_version"] == "robot-schema-v1"


def test_cli_training_defaults_reach_http_without_implicit_null(
    monkeypatch: pytest.MonkeyPatch,
    contract_fixtures: dict[str, Any],
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            201,
            headers={"X-Request-Id": "req_cli_quote_defaults"},
            json=contract_fixtures["quote"],
        )

    client = mock_client(handler)
    monkeypatch.setattr(cli, "_client", lambda _ctx: client)
    result = runner.invoke(
        cli.app,
        [
            "--api-key",
            "fixture",
            "training",
            "quote",
            "--dataset-version-id",
            "dsv_fixture",
            "--method",
            "full_ft",
            "--max-steps",
            "100",
            "--max-runtime-seconds",
            "7200",
            "--checkpoint-interval-steps",
            "100",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(bodies) == 1
    body = bodies[0]
    validate_schema("QuoteRequest", body)
    assert body["kind"] == "training_job"
    assert body["spec"]["seed"] == 42
    assert body["spec"]["base_model"] == "openpi/pi05-base"
    assert body["spec"]["preset"] == "balanced-v1"
    assert "lora_profile" not in body["spec"]
