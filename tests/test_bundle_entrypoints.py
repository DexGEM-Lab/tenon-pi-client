from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from tenon_pi import cli
from tenon_pi.client import TenonClient
from tenon_pi.data_fixture import write_data_fixture
from tenon_pi.preparation import prepare_dataset
from tenon_pi.robot_profile import RobotProfile
from tenon_pi.transport import RestTransport
from tenon_pi.uploads import DatasetUploader, scan_dataset_directory

runner = CliRunner()


def _prepared_bundle(tmp_path: Path, format="lerobot_v2", dataset_v3_source=None):
    if format == "lerobot_v3":
        source, profile = dataset_v3_source
        return prepare_dataset(
            source,
            tmp_path / "prepared",
            input_space="canonical",
            dataset_repo_id="fixture/robot",
            profile=profile,
        )
    paths = write_data_fixture(tmp_path / "fixture")
    profile = RobotProfile.load(paths["profile"])
    return prepare_dataset(
        paths["dataset"],
        tmp_path / "prepared",
        input_space="native",
        dataset_repo_id="fixture/robot",
        profile=profile,
    )


def _created_response(
    contract_fixtures: dict[str, Any], request: dict[str, Any], mode="single"
) -> dict[str, Any]:
    version = dict(contract_fixtures["dataset_version"])
    version.update(format=request["format"], state="ready")
    upload = dict(contract_fixtures["upload"])
    upload["files"] = [
        {
            "id": f"fil_{index}",
            "relative_path": item["relative_path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
            "mode": mode,
            "state": "pending",
        }
        for index, item in enumerate(request["files"])
    ]
    return {"version": version, "upload": upload}


def _client(handler) -> TenonClient:
    http = httpx.Client(
        base_url="https://api.fixture.invalid/v1/",
        transport=httpx.MockTransport(handler),
    )
    return TenonClient(
        "fixture-api-key",
        transport=RestTransport(
            "fixture-api-key",
            base_url="https://api.fixture.invalid/v1",
            client=http,
        ),
    )


OPENAPI = json.loads((Path(__file__).parents[1] / "spec/openapi.json").read_text())


def validate_schema(name, value):
    Draft202012Validator(
        {"$ref": f"#/components/schemas/{name}", "components": OPENAPI["components"]}
    ).validate(value)


def real_upload_transport(handler):
    """Actual SDK sign/PUT/file-complete/upload-complete over HTTP transports."""
    state = {}
    transferred = {}

    def api(request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        if path.endswith("/parts/sign"):
            validate_schema("UploadPartSignRequest", body)
            payload = {
                "parts": [
                    {
                        **part,
                        "method": "PUT",
                        "url": f"https://upload.fixture.invalid/{part['file_id']}/{part['part_number']}",
                        "required_headers": {},
                        "expires_at": "2026-10-14T13:00:00Z",
                    }
                    for part in body["parts"]
                ]
            }
            for part in payload["parts"]:
                part.pop("content_length")
            validate_schema("UploadPartSignResponse", payload)
            return httpx.Response(200, headers={"X-Request-Id": "req_sign"}, json=payload)
        if "/uploads/" in path and path.endswith("/complete"):
            if "/files/" not in path:
                state["finished"] = True
            if "/files/" in path:
                validate_schema("FileCompleteRequest", body)
            return httpx.Response(
                200 if "/files/" in path else 202,
                headers={"X-Request-Id": "req_complete"},
                json=state["upload"],
            )
        if request.method == "POST" and path in {"/v1/quotes", "/v1/training/jobs"}:
            validate_schema(
                "QuoteRequest" if path.endswith("quotes") else "TrainingJobCreate", body
            )
        if path.endswith("/versions") and request.method == "POST":
            validate_schema("DatasetVersionCreate", body)
        response = handler(request)
        payload = response.json()
        if path.endswith("/versions") and request.method == "POST":
            validate_schema("DatasetVersionCreated", payload)
            state["upload"] = payload["upload"]
        return response

    def direct(request):
        assert "authorization" not in request.headers
        file_id, part_number = request.url.path.strip("/").split("/")
        transferred[(file_id, int(part_number))] = request.read()
        return httpx.Response(200, headers={"ETag": f'"{file_id}-{part_number}"'})

    direct_client = httpx.Client(transport=httpx.MockTransport(direct))
    return api, direct_client, state, transferred


def assert_transferred(prepared, state, transferred):
    local = {item.manifest.relative_path: item for item in scan_dataset_directory(prepared.root)}
    assert len(state["upload"]["files"]) == len(local)
    validate_schema("Upload", state["upload"])
    for item in state["upload"]["files"]:
        data = b"".join(
            value for (file_id, _), value in sorted(transferred.items()) if file_id == item["id"]
        )
        assert data == local[item["relative_path"]].path.read_bytes()


class NoopUploader:
    seen_paths: ClassVar[list[str]] = []

    def __init__(self, _client):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def upload(self, upload, files):
        type(self).seen_paths = [item.manifest.relative_path for item in files]
        return upload


@pytest.mark.parametrize("format", ["lerobot_v2", "lerobot_v3"])
@pytest.mark.parametrize("transfer_mode", ["single", "multipart"])
def test_dataset_upload_verified_bundle_reaches_http_with_all_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_fixtures: dict[str, Any],
    format,
    transfer_mode,
    dataset_v3_source,
) -> None:
    prepared = _prepared_bundle(tmp_path, format, dataset_v3_source)
    bodies: dict[str, dict[str, Any]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/datasets":
            return httpx.Response(
                201,
                headers={"X-Request-Id": "req_dataset"},
                json=contract_fixtures["dataset"],
            )
        if request.url.path.endswith("/versions"):
            body = json.loads(request.content)
            bodies["version"] = body
            return httpx.Response(
                201,
                headers={"X-Request-Id": "req_version"},
                json=_created_response(contract_fixtures, body, transfer_mode),
            )
        raise AssertionError(request.url)

    api, direct, state, transferred = real_upload_transport(handler)
    client = _client(api)
    monkeypatch.setattr(cli, "_client", lambda _ctx: client)
    monkeypatch.setattr(
        cli,
        "DatasetUploader",
        lambda client: DatasetUploader(client, direct_client=direct, part_size=4096),
    )
    result = runner.invoke(
        cli.app,
        [
            "--api-key",
            "fixture",
            "dataset",
            "upload",
            str(prepared.root),
            "--schema",
            str(prepared.schema),
            "--name",
            "Robot prepared bundle",
            "--no-wait",
        ],
    )

    assert result.exit_code == 0, result.output
    body = bodies["version"]
    assert body["format"] == format
    assert body["compatibility_profile"] == "tenon-native-v1"
    assert body["robot_schema"] == json.loads(prepared.schema.read_text())
    paths = {item["relative_path"] for item in body["files"]}
    assert len(paths) == len(scan_dataset_directory(prepared.root))
    assert_transferred(prepared, state, transferred)
    if transfer_mode == "multipart":
        assert len(transferred) > len(paths)
    assert "dataset/meta/info.json" in paths
    assert "assets/robot-profile-v1/norm_stats.json" in paths
    assert {"robot-schema.json", "preparation-manifest.json"} <= paths
    assert "meta/info.json" not in paths


@pytest.mark.parametrize("format", ["lerobot_v2", "lerobot_v3"])
def test_agent_train_verified_bundle_preserves_format_and_robot_spec_over_real_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_fixtures: dict[str, Any],
    format,
    dataset_v3_source,
) -> None:
    prepared = _prepared_bundle(tmp_path, format, dataset_v3_source)
    bodies: dict[str, dict[str, Any]] = {}
    calls = []

    def training_job(body: dict[str, Any], state: str) -> dict[str, Any]:
        value = dict(contract_fixtures["training_job"])
        value.update(state=state, spec={key: body[key] for key in body if key != "quote_id"})
        value["progress"] = {
            **value["progress"],
            "phase": "publishing" if state == "succeeded" else "dataset_loading",
            "completed_steps": body["max_steps"] if state == "succeeded" else 0,
            "max_steps": body["max_steps"],
        }
        value["checkpoint_ids"] = ["ckp_fixture"] if state == "succeeded" else []
        return value

    def handler(request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        calls.append((method, path))
        headers = {"X-Request-Id": "req_" + path.rsplit("/", 1)[-1]}
        if method == "GET" and path == "/v1/me":
            return httpx.Response(
                200,
                headers={**headers, "Cache-Control": "no-store"},
                json=contract_fixtures["session_bootstrap"],
            )
        if method == "GET" and path == "/v1/capabilities":
            return httpx.Response(
                200,
                headers=headers,
                json=dict(
                    contract_version="0.1.0",
                    base_models=["openpi/pi05-base"],
                    training_methods=["full_ft"],
                    protocols=["tenon-control-v1"],
                    limits={},
                ),
            )
        if method == "GET" and path == "/v1/uploads/upl_fixture":
            return httpx.Response(200, headers=headers, json=state["upload"])
        if method == "POST" and path == "/v1/datasets":
            return httpx.Response(201, headers=headers, json=contract_fixtures["dataset"])
        if method == "POST" and path.endswith("/versions"):
            body = json.loads(request.content)
            bodies["version"] = body
            return httpx.Response(
                201, headers=headers, json=_created_response(contract_fixtures, body)
            )
        if method == "GET" and "/datasets/" in path and "/versions/" in path:
            version = dict(contract_fixtures["dataset_version"])
            version.update(format=format, state="ready" if state.get("finished") else "uploading")
            return httpx.Response(200, headers=headers, json=version)
        if method == "POST" and path == "/v1/quotes":
            bodies["quote"] = json.loads(request.content)
            return httpx.Response(201, headers=headers, json=contract_fixtures["quote"])
        if method == "GET" and path == "/v1/credits":
            return httpx.Response(
                200,
                headers=headers,
                json={
                    "balance_microcredits": 10_000_000,
                    "reserved_microcredits": 0,
                    "available_microcredits": 10_000_000,
                    "as_of": "2026-10-14T12:00:00Z",
                },
            )
        if method == "POST" and path == "/v1/training/jobs":
            body = json.loads(request.content)
            bodies["job"] = body
            return httpx.Response(202, headers=headers, json=training_job(body, "queued"))
        if method == "GET" and path == "/v1/training/jobs/trn_fixture":
            return httpx.Response(
                200, headers=headers, json=training_job(bodies["job"], "succeeded")
            )
        if method == "GET" and path == "/v1/checkpoints/ckp_fixture":
            checkpoint = dict(contract_fixtures["checkpoint"])
            schema = json.loads(prepared.schema.read_text())
            checkpoint["manifest"] = {
                **checkpoint["manifest"],
                "transform_digest": schema["transform_digest"],
                "normalization_digest": schema["normalization_digest"],
            }
            return httpx.Response(200, headers=headers, json=checkpoint)
        raise AssertionError((method, path))

    api, direct, state, transferred = real_upload_transport(handler)
    monkeypatch.setattr(cli, "_client", lambda _ctx: _client(api))
    monkeypatch.setattr(
        cli,
        "DatasetUploader",
        lambda client, **kw: DatasetUploader(client, direct_client=direct, **kw),
    )
    args = [
        "--api-key",
        "fixture",
        "agent",
        "train",
        str(prepared.root),
        "--context",
        str(tmp_path / "run.json"),
        "--schema",
        str(prepared.schema),
        "--name",
        "Robot prepared bundle",
        "--method",
        "full_ft",
        "--base-model",
        "openpi/pi05-base",
        "--preset",
        "example-service-full-v1",
        "--max-steps",
        "100",
        "--max-runtime-seconds",
        "7200",
        "--checkpoint-interval-steps",
        "100",
        "--max-cost-microcredits",
        "5000000",
        "--yes",
    ]
    result = runner.invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    first_posts = [call for call in calls if call[0] == "POST"]
    first_transferred = dict(transferred)
    repeated = runner.invoke(cli.app, args)
    assert repeated.exit_code == 0, repeated.exception
    assert [call for call in calls if call[0] == "POST"] == first_posts
    assert transferred == first_transferred
    assert len(first_posts) == 4  # dataset, version, quote, job; transfer handled separately

    assert bodies["version"]["format"] == format
    assert len(bodies["version"]["files"]) == len(scan_dataset_directory(prepared.root))
    assert_transferred(prepared, state, transferred)
    assert bodies["quote"] == {
        "kind": "training_job",
        "spec": {
            "dataset_version_id": "dsv_fixture",
            "base_model": "openpi/pi05-base",
            "method": "full_ft",
            "preset": "example-service-full-v1",
            "max_steps": 100,
            "max_runtime_seconds": 7200,
            "seed": 42,
            "checkpoint_interval_steps": 100,
        },
    }
    assert bodies["job"]["preset"] == "example-service-full-v1"
    assert bodies["job"]["base_model"] == "openpi/pi05-base"


@pytest.mark.parametrize("format", ["lerobot_v2", "lerobot_v3"])
def test_plain_v2_v3_dataset_upload_paths_remain_flat_and_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_fixtures: dict[str, Any],
    format: str,
) -> None:
    root = tmp_path / format
    (root / "meta").mkdir(parents=True)
    version = "v2.1" if format == "lerobot_v2" else "v3.0"
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": version}))
    if format == "lerobot_v2":
        (root / "meta" / "tasks.jsonl").write_text('{"task_index":0,"task":"x"}\n')
        (root / "meta" / "episodes.jsonl").write_text(
            '{"episode_index":0,"tasks":["x"],"length":1}\n'
        )
        (root / "meta" / "episodes_stats.jsonl").write_text("{}\n")
        (root / "data").mkdir()
        (root / "data" / "fixture.bin").write_bytes(b"x")
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/datasets":
            return httpx.Response(
                201,
                headers={"X-Request-Id": "req_plain_dataset"},
                json=contract_fixtures["dataset"],
            )
        body = json.loads(request.content)
        seen.update(body)
        return httpx.Response(
            201,
            headers={"X-Request-Id": "req_plain_version"},
            json=_created_response(contract_fixtures, body),
        )

    monkeypatch.setattr(cli, "_client", lambda _ctx: _client(handler))
    monkeypatch.setattr(cli, "DatasetUploader", NoopUploader)
    schema = Path(__file__).parents[1] / "schemas" / "libero-robot-schema.json"
    result = runner.invoke(
        cli.app,
        [
            "--api-key",
            "fixture",
            "dataset",
            "upload",
            str(root),
            "--format",
            format,
            "--schema",
            str(schema),
            "--name",
            "plain",
            "--no-wait",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["format"] == format
    paths = {item["relative_path"] for item in seen["files"]}
    assert "meta/info.json" in paths
    assert not any(path.startswith("dataset/") or path.startswith("assets/") for path in paths)


@pytest.mark.parametrize("command", ["dataset", "agent"])
@pytest.mark.parametrize("format", ["lerobot_v2", "lerobot_v3"])
@pytest.mark.parametrize("tamper", ["bytes", "rehashed"])
def test_tampered_bundle_fails_before_client_or_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    format,
    tamper,
    dataset_v3_source,
) -> None:
    prepared = _prepared_bundle(tmp_path, format, dataset_v3_source)
    norm = prepared.assets / "norm_stats.json"
    if tamper == "bytes":
        norm.write_bytes(norm.read_bytes() + b"\n")
    else:
        from test_dataset_v3_prepare import rehash_wrong_stats

        rehash_wrong_stats(prepared)
    called = False

    def forbidden_client(_ctx):
        nonlocal called
        called = True
        raise AssertionError("auth/client construction must follow local bundle verification")

    monkeypatch.setattr(cli, "_client", forbidden_client)
    common = [
        str(prepared.root),
        "--schema",
        str(prepared.schema),
        "--name",
        "tampered",
    ]
    args = (
        ["dataset", "upload", *common, "--no-wait"]
        if command == "dataset"
        else [
            "agent",
            "train",
            *common,
            "--method",
            "full_ft",
            "--max-steps",
            "100",
            "--max-runtime-seconds",
            "7200",
            "--checkpoint-interval-steps",
            "100",
            "--max-cost-microcredits",
            "1",
            "--yes",
        ]
    )
    result = runner.invoke(cli.app, args)
    assert result.exit_code != 0
    assert not called
    assert result.exception is not None
    assert "norm_stats.json bytes" in str(result.exception)


@pytest.mark.parametrize("command", ["dataset", "agent"])
@pytest.mark.parametrize("format", ["lerobot_v2", "lerobot_v3"])
@pytest.mark.parametrize("mismatch", ["format", "schema"])
def test_verified_bundle_mismatch_fails_before_auth(
    tmp_path,
    monkeypatch,
    command,
    format,
    mismatch,
    dataset_v3_source,
):
    prepared = _prepared_bundle(tmp_path, format, dataset_v3_source)
    schema = prepared.schema
    extra = []
    if mismatch == "format":
        extra = ["--format", "lerobot_v3" if format == "lerobot_v2" else "lerobot_v2"]
    else:
        schema = tmp_path / "wrong-schema.json"
        value = json.loads(prepared.schema.read_text())
        value["normalization_digest"] = "sha256:" + "a" * 64
        schema.write_text(json.dumps(value))

    def forbidden_client(_ctx):
        raise AssertionError("authentication must not run for a mismatched bundle")

    monkeypatch.setattr(cli, "_client", forbidden_client)
    common = [str(prepared.root), "--schema", str(schema), "--name", "invalid", *extra]
    args = (
        ["dataset", "upload", *common, "--no-wait"]
        if command == "dataset"
        else [
            "agent",
            "train",
            *common,
            "--method",
            "full_ft",
            "--max-steps",
            "100",
            "--max-runtime-seconds",
            "7200",
            "--checkpoint-interval-steps",
            "100",
            "--max-cost-microcredits",
            "1",
            "--yes",
        ]
    )
    result = runner.invoke(cli.app, args)
    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)
    assert (
        "not lerobot" in result.output
        if mismatch == "format"
        else "--schema must match" in result.output
    )
