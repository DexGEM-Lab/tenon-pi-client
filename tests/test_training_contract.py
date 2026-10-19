from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError
from test_request_serialization import full_ft_spec, mock_client, validate_schema
from typer.testing import CliRunner

from tenon_pi import cli
from tenon_pi.data_fixture import example_profile
from tenon_pi.models import (
    Checkpoint,
    DatasetFormat,
    DatasetVersion,
    TrainingJob,
    TrainingJobCreate,
    TrainingMethod,
    TrainingMode,
    TrainingSpec,
)
from tenon_pi.transport import serialize_request_body
from tenon_pi.workflow import RunContext, run_training, verify_checkpoint


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"effective_batch_size": None, "training_mode": None},
        {"effective_batch_size": 8, "training_mode": "standard"},
        {"effective_batch_size": 16, "training_mode": "wan_rtc"},
    ],
)
def test_real_quote_and_create_bodies_match_amended_schema(options, contract_fixtures):
    captured = []
    spec = TrainingSpec(**full_ft_spec().model_dump(exclude_unset=True), **options)

    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        if request.url.path.endswith("quotes"):
            validate_schema("QuoteRequest", body)
            return httpx.Response(
                201, headers={"X-Request-Id": "req_quote"}, json=contract_fixtures["quote"]
            )
        validate_schema("TrainingJobCreate", body)
        value = copy.deepcopy(contract_fixtures["training_job"])
        value["spec"] = {k: v for k, v in body.items() if k != "quote_id"}
        return httpx.Response(202, headers={"X-Request-Id": "req_create"}, json=value)

    client = mock_client(handler)
    quote = client.quote_training(spec, idempotency_key="quote-training-fixture")
    client.create_training_job(
        spec,
        quote=quote,
        max_cost_microcredits=5_000_000,
        idempotency_key="create-training-fixture",
    )
    assert captured[0]["spec"] == {k: v for k, v in captured[1].items() if k != "quote_id"}
    for field in ("effective_batch_size", "training_mode"):
        if field in options:
            assert captured[0]["spec"][field] == options[field]
        else:
            assert field not in captured[0]["spec"]
    if not options:
        assert captured[0]["spec"] == full_ft_spec().model_dump(exclude_unset=True)


@pytest.mark.parametrize("invalid", [True, 0, -1, 2.5, "8"])
def test_effective_batch_requires_positive_integer_without_coercion(invalid):
    with pytest.raises(ValidationError):
        TrainingSpec(**full_ft_spec().model_dump(exclude_unset=True), effective_batch_size=invalid)


def test_cli_constructs_explicit_user_batch_and_training_mode(tmp_path):
    result = CliRunner().invoke(
        cli.app,
        [
            "training",
            "spec",
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
            "--effective-batch-size",
            "8",
            "--training-mode",
            "wan_rtc",
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["effective_batch_size"] == 8 and body["training_mode"] == "wan_rtc"
    validate_schema("TrainingSpec", body)


def old_run(tmp_path, fixtures):
    schema = example_profile().robot_schema("sha256:" + "f" * 64)
    # This is the v0.2 journal shape: no pilot fields, even as null.
    original_input = dict(
        files=[],
        schema=schema.model_dump(mode="json"),
        format="lerobot_v3",
        name="example",
        method="full_ft",
        max_steps=100,
        max_runtime_seconds=7200,
        checkpoint_interval_steps=100,
        seed=42,
        base_model=None,
        preset=None,
        budget=5_000_000,
    )
    original_spec = copy.deepcopy(fixtures["training_job"]["spec"])
    assert not {"effective_batch_size", "training_mode"} & set(original_spec)
    context = tmp_path / "old-run.json"
    with RunContext(context) as journal:
        for k, v in dict(
            input=original_input,
            training_spec=original_spec,
            dataset_id="dst_fixture",
            version_id="dsv_fixture",
            recipe=dict(base_model="openpi/pi05-base", preset="balanced-v1"),
            training_quote=fixtures["quote"],
        ).items():
            journal.set(k, v)
        journal.data["keys"]["training-job"] = "original-approved-job-key"
        journal.data["key_created_at"] = {"training-job": time.time()}
        journal.save()
    return context, schema, original_input, original_spec


class ResumeClient:
    base_url = "https://fixture.invalid/v1"

    def __init__(self, fixtures, schema, state="succeeded"):
        self.fixtures = copy.deepcopy(fixtures)
        self.schema = schema
        self.state = state
        self.creates = []

    def me(self):
        return SimpleNamespace(id="user-example")

    def get_dataset_version(self, *args):
        return DatasetVersion.model_validate(self.fixtures["dataset_version"])

    wait_for_dataset_version = get_dataset_version

    def create_training_job(self, spec, *, quote, max_cost_microcredits, idempotency_key):
        body = serialize_request_body(
            TrainingJobCreate(**spec.model_dump(exclude_unset=True), quote_id=quote.id)
        )
        self.creates.append((body, idempotency_key))
        return self.wait_for_training_job("trn_fixture")

    def wait_for_training_job(self, *args, **kwargs):
        return TrainingJob.model_validate(
            {
                **self.fixtures["training_job"],
                "state": self.state,
                "checkpoint_ids": ["ckp_fixture"],
            }
        )

    def get_checkpoint(self, identifier):
        v = self.fixtures["checkpoint"]
        v["manifest"]["transform_digest"] = self.schema.transform_digest
        return Checkpoint.model_validate(v)


def arguments(schema):
    return dict(
        files=[],
        schema=schema,
        format=DatasetFormat.LEROBOT_V3,
        name="example",
        method=TrainingMethod.FULL_FT,
        max_steps=100,
        max_runtime_seconds=7200,
        checkpoint_interval_steps=100,
        seed=42,
        base_model=None,
        preset=None,
        max_cost_microcredits=5_000_000,
        approve=lambda quote: pytest.fail("original approved create must not be reapproved"),
    )


def test_old_pending_context_preserves_exact_body_key_and_bound_records(
    tmp_path, contract_fixtures
):
    path, schema, original_input, original_spec = old_run(tmp_path, contract_fixtures)
    client = ResumeClient(contract_fixtures, schema)
    with RunContext(path) as journal:
        result = run_training(client, journal, **arguments(schema))
        assert journal.data["input"] == original_input
        assert journal.data["training_spec"] == original_spec
    assert result["training_job"]["state"] == "succeeded"
    body, key = client.creates[0]
    assert key == "original-approved-job-key"
    assert body == {**full_ft_spec().model_dump(exclude_unset=True), "quote_id": "quo_fixture"}
    with RunContext(path) as journal:
        run_training(client, journal, **arguments(schema))
    assert len(client.creates) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"effective_batch_size": 8},
        {"training_mode": TrainingMode.STANDARD},
        {"max_steps": 101},
        {"max_cost_microcredits": 6_000_000},
    ],
)
def test_changed_training_intent_requires_new_context(tmp_path, contract_fixtures, change):
    path, schema, original_input, original_spec = old_run(tmp_path, contract_fixtures)
    client = ResumeClient(contract_fixtures, schema)
    with RunContext(path) as journal:
        with pytest.raises(ValueError, match="differs"):
            run_training(client, journal, **{**arguments(schema), **change})
        assert (
            journal.data["input"] == original_input
            and journal.data["training_spec"] == original_spec
        )
    assert not client.creates


def test_failed_legacy_job_is_not_resurrected(tmp_path, contract_fixtures):
    path, schema, _, _ = old_run(tmp_path, contract_fixtures)
    client = ResumeClient(contract_fixtures, schema, "failed")
    with RunContext(path) as journal:
        journal.set("training_job_id", "trn_fixture")
        with pytest.raises(ValueError, match="failed"):
            run_training(client, journal, **arguments(schema))
    assert not client.creates


@pytest.mark.parametrize("bad", ["batch", "mode", "missing_rtc"])
def test_checkpoint_provenance_must_match_explicit_request(contract_fixtures, bad):
    schema = example_profile().robot_schema("sha256:" + "f" * 64)
    value = copy.deepcopy(contract_fixtures["checkpoint"])
    value["manifest"].update(
        transform_digest=schema.transform_digest,
        effective_batch_size=8,
        training_mode="wan_rtc",
        rtc_config={"adapter_version": "fixture-v1"},
    )
    if bad == "batch":
        value["manifest"]["effective_batch_size"] = 4
    elif bad == "mode":
        value["manifest"]["training_mode"] = "standard"
    else:
        value["manifest"]["rtc_config"] = None
    spec = TrainingSpec(
        **full_ft_spec().model_dump(exclude_unset=True),
        effective_batch_size=8,
        training_mode="wan_rtc",
    )
    with pytest.raises(ValueError):
        verify_checkpoint(
            Checkpoint.model_validate(value),
            job_id="trn_fixture",
            spec=spec,
            schema=schema,
            schema_hash=contract_fixtures["dataset_version"]["schema_hash"],
        )


def test_cli_omitted_pilot_fields_preserve_legacy_spec_file(tmp_path, contract_fixtures):
    output = tmp_path / "spec.json"
    result = CliRunner().invoke(
        cli.app,
        [
            "training",
            "spec",
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
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == contract_fixtures["training_job"]["spec"]
    assert json.loads(output.read_text()) == contract_fixtures["training_job"]["spec"]
