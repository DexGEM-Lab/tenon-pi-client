from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

from tenon_pi.data_fixture import example_profile
from tenon_pi.errors import TenonTransportError
from tenon_pi.models import Checkpoint, ControlSession, Deployment, Quote, TrainingSpec
from tenon_pi.profiles import libero_robot_schema
from tenon_pi.uploads import DatasetUploader, scan_dataset_directory
from tenon_pi.workflow import RunContext, deploy_run, resolve_recipe, verify_checkpoint


def test_context_is_exclusive_private_and_keeps_exact_keys(tmp_path):
    path = tmp_path / "run.json"
    with RunContext(path) as first:
        key = first.key("dataset")
        first.bind("input", {"value": 1})
        assert json.loads(path.read_text())["keys"]["dataset"] == key
        with pytest.raises(BlockingIOError), RunContext(path):
            pass
    with RunContext(path) as resumed:
        assert resumed.key("dataset") == key
        with pytest.raises(ValueError, match="differs"):
            resumed.bind("input", {"value": 2})
    assert path.stat().st_mode & 0o777 == 0o600


def test_completion_is_durable_before_lost_response(tmp_path, contract_fixtures):
    from test_uploads import FakeLifecycleClient

    data = tmp_path / "data/meta"
    data.mkdir(parents=True)
    (data / "info.json").write_bytes(b"hello world")
    context = tmp_path / "run.json"

    class LostResponse(FakeLifecycleClient):
        def complete_upload_file(self, upload_id, file_id, request, *, idempotency_key=None):
            saved = json.loads(context.read_text())["completions"][file_id]
            assert saved["idempotency_key"] == idempotency_key
            assert saved["request_json"] == request.model_dump_json()
            raise TenonTransportError("response lost", idempotency_key=idempotency_key)

    client = LostResponse(contract_fixtures["upload"])
    direct = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"ETag": '"original-etag"'})
        )
    )
    with (
        RunContext(context) as journal,
        DatasetUploader(
            client, direct_client=direct, completion_observer=journal.completion
        ) as uploader,
        pytest.raises(TenonTransportError),
    ):
        uploader.upload(client.upload, scan_dataset_directory(tmp_path / "data"))
    assert "original-etag" in context.read_text()
    assert "https://" not in context.read_text()


def test_libero_recipe_is_resolved_without_user_preset():
    catalog = SimpleNamespace(training_methods=["full_ft"], base_models=["openpi/pi05-libero"])
    client = SimpleNamespace(capabilities=lambda: catalog)
    assert resolve_recipe(client, libero_robot_schema(), "full_ft", None, None) == (
        "openpi/pi05-libero",
        "libero10-demo-full-v1",
    )


@pytest.mark.parametrize("mode", ["async_aligned", "trained_rtc"])
@pytest.mark.parametrize("failure", [None, "deployment", "session", "checkpoint"])
def test_deploy_reuses_exact_lineage_and_never_upgrades_terminal_state(
    tmp_path, contract_fixtures, failure, mode
):
    fixtures = deepcopy(contract_fixtures)
    schema = example_profile().robot_schema("sha256:" + "f" * 64)
    checkpoint = fixtures["checkpoint"]
    checkpoint["manifest"]["transform_digest"] = schema.transform_digest
    if mode == "trained_rtc":
        from test_trained_rtc import rtc_metadata

        fixtures["training_job"]["spec"].update(effective_batch_size=8, training_mode="wan_rtc")
        checkpoint["manifest"].update(
            effective_batch_size=8, training_mode="wan_rtc", rtc_config=rtc_metadata(schema)
        )
    calls = []

    class Client:
        base_url = "https://service.example.invalid/v1"

        def me(self):
            return SimpleNamespace(id="user-example")

        def capabilities(self):
            return SimpleNamespace(
                execution_modes=[mode], base_models=[checkpoint["manifest"]["base_model"]]
            )

        def get_checkpoint(self, identifier):
            value = deepcopy(checkpoint)
            if failure == "checkpoint":
                value["manifest"]["dataset_version_id"] = "another-dataset"
            return Checkpoint.model_validate(value)

        def create_deployment(self, request, *, idempotency_key):
            calls.append(("deployment", idempotency_key))
            self.fallback = request.fallback
            return self.wait_for_deployment("dep_fixture")

        def wait_for_deployment(self, identifier):
            value = deepcopy(fixtures["deployment"])
            value.update(
                state="failed" if failure == "deployment" else "ready",
                fallback=self.fallback.model_dump(mode="json"),
                execution_modes=[mode],
            )
            return Deployment.model_validate(value)

        def quote_control_session(self, spec, *, idempotency_key):
            calls.append(("quote", idempotency_key))
            value = {**fixtures["quote"], "kind": "control_session"}
            return Quote.model_validate(value)

        def create_control_session(self, spec, *, quote, max_cost_microcredits, idempotency_key):
            calls.append(("session", idempotency_key))
            return self.wait_for_control_session("ctl_fixture")

        def wait_for_control_session(self, identifier):
            value = {
                **fixtures["control_session"],
                "state": "expired" if failure == "session" else "active",
                "execution_mode": mode,
            }
            return ControlSession.model_validate(value)

    client = Client()
    with RunContext(tmp_path / "run.json") as journal:
        journal.set("input", {"schema": schema.model_dump(mode="json")})
        journal.set("training_spec", fixtures["training_job"]["spec"])
        journal.set("training_job_id", "trn_fixture")
        journal.set("dataset_version", fixtures["dataset_version"])
        journal.set("checkpoints", [checkpoint])
        kwargs = dict(
            name="policy",
            prompt="place block",
            execution_mode=mode,
            duration_seconds=300,
            max_cost_microcredits=5_000_000,
            approve=lambda quote: None,
        )
        if failure:
            with pytest.raises(ValueError):
                deploy_run(client, journal, **kwargs)
            if failure in {"checkpoint", "deployment"}:
                assert not any(c[0] == "session" for c in calls)
        else:
            result = deploy_run(client, journal, **kwargs)
            assert result["session"]["state"] == "active"
            again = deploy_run(client, journal, **kwargs)
            assert again == result
            assert [c[0] for c in calls] == ["deployment", "quote", "session"]


def test_checkpoint_readiness_requires_exact_source_binding(contract_fixtures):
    schema = example_profile().robot_schema("sha256:" + "f" * 64)
    checkpoint = Checkpoint.model_validate(contract_fixtures["checkpoint"])
    with pytest.raises(ValueError, match="lineage"):
        verify_checkpoint(
            checkpoint,
            job_id="trn_fixture",
            spec=TrainingSpec.model_validate(contract_fixtures["training_job"]["spec"]),
            schema=schema,
            schema_hash=contract_fixtures["dataset_version"]["schema_hash"],
        )


@pytest.mark.parametrize("pilot", [False, True])
def test_lost_training_create_replays_without_second_credit_approval(
    tmp_path, contract_fixtures, pilot
):
    from tenon_pi.models import DatasetFormat, DatasetVersion, TrainingJob, TrainingMethod
    from tenon_pi.workflow import run_training

    schema = example_profile().robot_schema("sha256:" + "f" * 64)
    fixtures = deepcopy(contract_fixtures)
    fixtures["checkpoint"]["manifest"]["transform_digest"] = schema.transform_digest
    additions = {"effective_batch_size": 8, "training_mode": "wan_rtc"} if pilot else {}
    if pilot:
        from test_trained_rtc import rtc_metadata

        fixtures["training_job"]["spec"].update(additions)
        fixtures["checkpoint"]["manifest"].update(**additions, rtc_config=rtc_metadata(schema))
    approvals, keys = [], []

    class Client:
        base_url = "https://service.example.invalid/v1"

        def me(self):
            return SimpleNamespace(id="user-example")

        def get_dataset_version(self, *args):
            return DatasetVersion.model_validate(fixtures["dataset_version"])

        wait_for_dataset_version = get_dataset_version

        def quote_training(self, spec, *, idempotency_key):
            return Quote.model_validate(fixtures["quote"])

        def create_training_job(self, spec, *, quote, max_cost_microcredits, idempotency_key):
            keys.append(idempotency_key)
            if len(keys) == 1:
                raise TenonTransportError(
                    "created and reserved; response lost", idempotency_key=idempotency_key
                )
            return self.wait_for_training_job("trn_fixture")

        def wait_for_training_job(self, identifier, **kwargs):
            return TrainingJob.model_validate(
                {
                    **fixtures["training_job"],
                    "state": "succeeded",
                    "checkpoint_ids": ["ckp_fixture"],
                }
            )

        def get_checkpoint(self, identifier):
            return Checkpoint.model_validate(fixtures["checkpoint"])

    def approve(quote):
        approvals.append(quote.id)
        assert len(approvals) == 1, "existing reservation must not be checked as new spending"

    context = tmp_path / "run.json"
    kwargs = dict(
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
        approve=approve,
        **additions,
    )
    with RunContext(context) as journal:
        journal.set("dataset_id", "dst_fixture")
        journal.set("version_id", "dsv_fixture")
        journal.set("recipe", dict(base_model="openpi/pi05-base", preset="balanced-v1"))
        with pytest.raises(TenonTransportError):
            run_training(Client(), journal, **kwargs)
    with RunContext(context) as journal:
        result = run_training(Client(), journal, **kwargs)
    assert result["training_job"]["state"] == "succeeded"
    assert len(keys) == 2 and keys[0] == keys[1]


def test_ambiguous_mutation_is_not_replayed_after_retention(tmp_path):
    with RunContext(tmp_path / "run.json") as journal:
        journal.key("training-job")
        journal.data["key_created_at"]["training-job"] = 0
        journal.save()
        with pytest.raises(ValueError, match="idempotency window"):
            journal.key("training-job")


def test_journal_schema_encoding_order_is_stable():
    from tenon_pi.workflow import _schema_json

    schema = example_profile().robot_schema("sha256:" + "f" * 64)
    schema.cameras[0].accepted_encodings = {"png", "jpeg", "raw_rgb8"}
    assert _schema_json(schema)["cameras"][0]["accepted_encodings"] == ["jpeg", "png", "raw_rgb8"]
