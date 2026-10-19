from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tenon_pi import cli
from tenon_pi.errors import TenonAPIError, TenonTransportError
from tenon_pi.models import (
    Capabilities,
    Checkpoint,
    ControlConnection,
    CreditBalance,
    Dataset,
    DatasetVersion,
    DatasetVersionCreated,
    FileCompleteRequest,
    Page,
    Quote,
    SessionBootstrap,
    TrainingJob,
    Upload,
)
from tenon_pi.uploads import FileCompletionContinuation, UploadCompletionPending

runner = CliRunner()


class FakeClient:
    def __init__(self, fixtures):
        self.fixtures = fixtures
        self.created = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def quote_training(self, _spec, **kwargs):
        return Quote.model_validate(self.fixtures["quote"])

    def credits(self):
        return CreditBalance(
            balance_microcredits=10_000_000,
            reserved_microcredits=0,
            available_microcredits=10_000_000,
            as_of="2026-10-14T12:00:00Z",
        )

    def require_budget(self, quote, max_cost_microcredits):
        assert quote.max_cost_microcredits <= max_cost_microcredits

    def create_training_job(self, _spec, *, quote, max_cost_microcredits, **kwargs):
        assert quote.id == "quo_fixture"
        assert max_cost_microcredits == 5_000_000
        self.created = True
        return TrainingJob.model_validate(self.fixtures["training_job"])

    def create_control_connection(self, session_id):
        assert session_id == "ctl_fixture"
        return ControlConnection.model_validate(self.fixtures["control_connection"])


def test_upload_pending_cli_payload_preserves_exact_resume_context():
    continuation = FileCompletionContinuation.create(
        "upl_fixture",
        "fil_fixture",
        FileCompleteRequest(parts=[{"part_number": 1, "etag": "etag"}]),
        "completion-key-0001",
    )
    api_error = TenonAPIError(
        status_code=409,
        code="completion_in_progress",
        message="still running",
        request_id="req_pending",
        retryable=True,
        retry_after="1",
        idempotency_key=continuation.idempotency_key,
    )
    pending = UploadCompletionPending(
        continuation=continuation,
        last_error=api_error,
        attempts=6,
        elapsed_seconds=65.0,
    )
    payload = cli._upload_pending_error(pending)
    assert payload["error"]["continuation"] == continuation.json_value()
    assert payload["error"]["last_api_error"] == {
        "status_code": 409,
        "code": "completion_in_progress",
        "request_id": "req_pending",
        "retryable": True,
        "retry_after": "1",
    }


def test_transport_error_payload_keeps_file_completion_context():
    continuation = FileCompletionContinuation.create(
        "upl_fixture",
        "fil_fixture",
        FileCompleteRequest(parts=[{"part_number": 1, "etag": "etag"}]),
        "completion-key-0002",
    )
    error = TenonTransportError(
        "response lost",
        idempotency_key=continuation.idempotency_key,
    )
    error.completion_continuation = continuation  # type: ignore[attr-defined]
    payload = cli._transport_error(error)
    assert payload["error"]["code"] == "transport_error"
    assert payload["error"]["completion_continuation"] == continuation.json_value()


def test_version_without_subcommand():
    result = runner.invoke(app=cli.app, args=["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == "0.3.0a1"


def test_local_v3_manifest_command(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text('{"codebase_version":"v3.0"}')
    output = tmp_path / "manifest.json"
    result = runner.invoke(app=cli.app, args=["dataset", "manifest", str(root), "-o", str(output)])
    assert result.exit_code == 0, result.output
    manifest = json.loads(output.read_text())
    assert manifest["format"] == "lerobot_v3"
    assert manifest["files"][0]["relative_path"] == "meta/info.json"


def test_libero_schema_command_writes_real_profile(tmp_path: Path):
    output = tmp_path / "schema.json"
    result = runner.invoke(app=cli.app, args=["libero", "schema", "-o", str(output)])
    assert result.exit_code == 0, result.output
    schema = json.loads(output.read_text())
    assert [camera["id"] for camera in schema["cameras"]] == ["base", "left_wrist"]
    assert schema["transform_digest"].endswith(
        "81bd9293ace15b1a9209ddf514c79a88e1bf0c2a7b521a13b2449747644c20d6"
    )
    assert schema["normalization_digest"].endswith(
        "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
    )


def test_training_spec_command_pins_libero_recipe(tmp_path: Path):
    output = tmp_path / "training.json"
    result = runner.invoke(
        app=cli.app,
        args=[
            "training",
            "spec",
            "--dataset-version-id",
            "dsv_fixture",
            "--method",
            "full_ft",
            "--base-model",
            "openpi/pi05-libero",
            "--preset",
            "libero10-demo-full-v1",
            "--max-steps",
            "100",
            "--max-runtime-seconds",
            "7200",
            "--checkpoint-interval-steps",
            "100",
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    spec = json.loads(output.read_text())
    assert spec["base_model"] == "openpi/pi05-libero"
    assert spec["preset"] == "libero10-demo-full-v1"
    assert spec["method"] == "full_ft"


def test_libero_manifest_command_is_explicitly_not_upload_ready(tmp_path: Path):
    output = tmp_path / "libero.json"
    result = runner.invoke(app=cli.app, args=["libero", "manifest", "-o", str(output)])
    assert result.exit_code == 0, result.output
    manifest = json.loads(output.read_text())
    assert manifest["total_frames"] == 2_806
    assert manifest["upload_contract"]["source_is_upload_ready"] is False
    assert manifest["upload_contract"]["recommended_demo_format"] == "lerobot_v2"


def test_agent_train_runs_one_upload_quote_confirmation_and_training(
    monkeypatch, contract_fixtures, tmp_path: Path
):
    class AgentClient(FakeClient):
        base_url = "https://api.fixture.invalid/v1"

        def me(self):
            return SessionBootstrap.model_validate(self.fixtures["session_bootstrap"])

        def capabilities(self):
            return Capabilities(
                contract_version="0.1.0",
                base_models=["openpi/pi05-base"],
                training_methods=["full_ft"],
                protocols=["tenon-control-v1"],
                limits={},
            )

        def get_dataset_version(self, *args):
            return DatasetVersion.model_validate(self.fixtures["dataset_version"])

        def get_checkpoint(self, _id):
            checkpoint = dict(self.fixtures["checkpoint"])
            checkpoint["manifest"] = {
                **checkpoint["manifest"],
                "transform_digest": "sha256:transform",
                "normalization_digest": "sha256:norm",
            }
            return Checkpoint.model_validate(checkpoint)

        def create_dataset(self, _request, **kwargs):
            return Dataset.model_validate(self.fixtures["dataset"])

        def create_dataset_version(self, _dataset_id, request, **kwargs):
            assert request.format == "lerobot_v3"
            return DatasetVersionCreated(
                version=DatasetVersion.model_validate(self.fixtures["dataset_version"]),
                upload=Upload.model_validate(self.fixtures["upload"]),
            )

        def wait_for_dataset_version(self, _dataset_id, _version_id):
            return DatasetVersion.model_validate(self.fixtures["dataset_version"])

        def wait_for_training_job(self, _job_id, *, timeout_seconds):
            del timeout_seconds
            payload = dict(self.fixtures["training_job"])
            payload["state"] = "succeeded"
            payload["checkpoint_ids"] = ["ckp_fixture"]
            return TrainingJob.model_validate(payload)

        def list_checkpoints(self, *, training_job_id):
            assert training_job_id == "trn_fixture"
            return Page[Checkpoint](
                items=[Checkpoint.model_validate(self.fixtures["checkpoint"])], next_cursor=None
            )

    class NoopUploader:
        def __init__(self, _client):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def upload(self, upload, _files):
            return upload

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text('{"codebase_version":"v3.0"}')
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "schema_version": "robot-schema-v1",
                "controller_hz": 20,
                "state_channels": [{"name": "state", "unit": "unitless", "lower": -1, "upper": 1}],
                "action_channels": [
                    {"name": "action", "unit": "unitless", "lower": -1, "upper": 1}
                ],
                "action_semantics": "joint_delta",
                "cameras": [
                    {
                        "id": "base",
                        "required": True,
                        "accepted_encodings": ["jpeg"],
                        "color_space": "rgb",
                        "max_width": 640,
                        "max_height": 480,
                    }
                ],
                "image_transform": {
                    "target_width": 224,
                    "target_height": 224,
                    "resize_mode": "pad",
                    "color_space": "rgb",
                    "dtype": "uint8",
                },
                "action_horizon_steps": 10,
                "max_committed_prefix_steps": 5,
                "transform_digest": "sha256:transform",
                "normalization_digest": "sha256:norm",
            }
        )
    )
    fake = AgentClient(contract_fixtures)
    monkeypatch.setattr(cli, "_client", lambda _ctx: fake)
    monkeypatch.setattr(cli, "DatasetUploader", NoopUploader)
    result = runner.invoke(
        app=cli.app,
        args=[
            "--api-key",
            "fixture",
            "agent",
            "train",
            str(root),
            "--context",
            str(tmp_path / "run.json"),
            "--schema",
            str(schema),
            "--name",
            "fixture",
            "--method",
            "full_ft",
            "--max-steps",
            "100",
            "--max-runtime-seconds",
            "7200",
            "--checkpoint-interval-steps",
            "100",
            "--max-cost-microcredits",
            "5000000",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["dataset_version"]["state"] == "ready"
    assert payload["training_job"]["state"] == "succeeded"
    assert payload["checkpoints"][0]["id"] == "ckp_fixture"


def test_training_run_confirms_quote_and_budget(monkeypatch, contract_fixtures):
    fake = FakeClient(contract_fixtures)
    monkeypatch.setattr(cli, "_client", lambda _ctx: fake)
    result = runner.invoke(
        app=cli.app,
        args=[
            "--api-key",
            "fixture",
            "training",
            "run",
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
            "--max-cost-microcredits",
            "5000000",
            "--yes",
            "--no-wait",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "maximum 5000000 microcredits" in result.output
    assert '"id": "trn_fixture"' in result.output
    assert fake.created


def test_session_connection_emits_server_negotiated_url(monkeypatch, contract_fixtures):
    fake = FakeClient(contract_fixtures)
    monkeypatch.setattr(cli, "_client", lambda _ctx: fake)
    result = runner.invoke(
        app=cli.app,
        args=["--api-key", "fixture", "session", "connection", "ctl_fixture"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["wss_url"].startswith("wss://inference.fixture.invalid/")
    assert payload["token"] == "fixture-connection-token"
