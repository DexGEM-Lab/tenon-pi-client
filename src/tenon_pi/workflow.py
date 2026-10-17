"""Resumable public lifecycle operations with a private local run journal.

Only non-secret specs, IDs, quotes and exact completion receipts are persisted.
Every mutation key is durable before its first request. The service remains the
source of truth for readiness; a journal never upgrades a remote state.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from .client import TenonClient
from .models import (
    Checkpoint,
    ControlSessionSpec,
    DatasetCreate,
    DatasetFormat,
    DatasetKind,
    DatasetVersionCreate,
    DeploymentCreate,
    ExecutionMode,
    FallbackPolicy,
    LastAppliedAbsoluteTarget,
    Protocol,
    Quote,
    RobotSchema,
    TrainingMethod,
    TrainingSpec,
    ZeroDeltaPreserve,
)
from .preparation import canonical_json_digest
from .profiles import LIBERO_TRANSFORM_SHA256
from .transport import new_idempotency_key
from .uploads import DatasetUploader, FileCompletionContinuation, LocalDatasetFile


class RunContext(AbstractContextManager):
    """One writer per local journal, including across a long polling operation.

    Advisory POSIX locks are released by the OS on process death. The context
    belongs outside dataset roots and is ignored by Git, never shared as source.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.data: dict[str, Any] = {}
        self._lock = None

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
        os.chmod(self._lock.name, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                self.data = json.loads(self.path.read_text())
                if not isinstance(self.data, dict) or self.data.get("version") != "tenon-run-v1":
                    raise ValueError("invalid run context; do not overwrite existing lineage")
            else:
                self.data = {"version": "tenon-run-v1", "keys": {}, "completions": {}}
        except BaseException:
            self._lock.close()
            self._lock = None
            raise
        return self

    def __exit__(self, *args):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def save(self) -> None:
        if self._lock is None:
            raise RuntimeError("run journal requires exclusive ownership")
        fd, name = tempfile.mkstemp(prefix=f".{self.path.name}-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as out:
                json.dump(
                    self.data, out, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
                )
                out.write("\n")
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def bind(self, name: str, value: Any) -> None:
        if name in self.data and self.data[name] != value:
            raise ValueError(
                f"run context {name} differs; inspect existing lineage before starting a new run"
            )
        self.data[name] = value
        self.save()

    def key(self, operation: str) -> str:
        keys = self.data["keys"]
        created = self.data.setdefault("key_created_at", {})
        if operation not in keys:
            keys[operation] = new_idempotency_key(operation)
            created[operation] = time.time()
            self.save()
        elif operation not in created or not 0 <= time.time() - created[operation] < 86400:
            raise ValueError(
                "mutation replay is outside the guaranteed idempotency window; "
                "reconcile the existing resource before retrying"
            )
        return keys[operation]

    def set(self, name: str, value: Any) -> None:
        self.data[name] = value
        self.save()

    def completion(self, receipt: FileCompletionContinuation) -> None:
        self.data["completions"][receipt.file_id] = receipt.json_value()
        self.save()

    def identity(self, client: TenonClient) -> None:
        # Never persist an endpoint URL that might contain query credentials.
        self.bind("service_digest", canonical_json_digest(client.base_url))
        self.bind("account_id", client.me().id)


def resolve_recipe(
    client: TenonClient,
    schema: RobotSchema,
    method: TrainingMethod,
    base_model: str | None,
    preset: str | None,
) -> tuple[str, str]:
    catalog = client.capabilities()
    if method not in catalog.training_methods:
        raise ValueError("requested method is absent from current service capabilities")
    if schema.transform_digest == f"sha256:{LIBERO_TRANSFORM_SHA256}":
        base_model = base_model or "openpi/pi05-libero"
        preset = (
            preset or f"libero10-demo-{'full' if method == TrainingMethod.FULL_FT else 'lora'}-v1"
        )
    else:
        base_model = base_model or "openpi/pi05-base"
        # The service, not the local profile, resolves/accepts this public preset.
        preset = preset or "balanced-v1"
    if base_model not in catalog.base_models:
        raise ValueError("resolved model is not offered by this endpoint; inspect capabilities")
    if method == TrainingMethod.LORA and not any(
        p.id == "openpi-pi05-action-lora-v1" and p.base_model == base_model
        for p in catalog.lora_profiles
    ):
        raise ValueError("named LoRA profile is not offered for the resolved model")
    return base_model, preset


def verify_checkpoint(
    checkpoint: Checkpoint,
    *,
    job_id: str,
    spec: TrainingSpec,
    schema: RobotSchema,
    schema_hash: str,
) -> None:
    manifest = checkpoint.manifest
    if (
        checkpoint.state != "ready"
        or checkpoint.training_job_id != job_id
        or manifest.dataset_version_id != spec.dataset_version_id
        or manifest.base_model != spec.base_model
        or manifest.method != spec.method
        or manifest.robot_schema_hash != schema_hash
        or manifest.transform_digest != schema.transform_digest
        or manifest.normalization_digest != schema.normalization_digest
        or not manifest.runtime_revision
        or not manifest.artifacts
        or any(a.size_bytes <= 0 for a in manifest.artifacts)
    ):
        raise ValueError(
            "checkpoint is not ready with the exact requested dataset/schema/model lineage"
        )
    if spec.method == TrainingMethod.LORA and (
        manifest.lora_profile != spec.lora_profile
        or not manifest.trainable_manifest_digest
        or not manifest.trainable_parameter_count
        or not manifest.adapter_ranks
    ):
        raise ValueError("LoRA checkpoint lacks named-profile/trainable-manifest evidence")


def _schema_json(schema: RobotSchema) -> dict[str, Any]:
    value = schema.model_dump(mode="json")
    for camera in value["cameras"]:
        camera["accepted_encodings"].sort()
    return value


def run_training(
    client: TenonClient,
    journal: RunContext,
    *,
    files: Sequence[LocalDatasetFile],
    schema: RobotSchema,
    format: DatasetFormat,
    name: str,
    method: TrainingMethod,
    max_steps: int,
    max_runtime_seconds: int,
    checkpoint_interval_steps: int,
    seed: int,
    base_model: str | None,
    preset: str | None,
    max_cost_microcredits: int,
    approve: Callable[[Quote], None],
    uploader_factory: Callable[..., DatasetUploader] = DatasetUploader,
) -> dict[str, Any]:
    journal.identity(client)
    journal.bind(
        "input",
        dict(
            files=[f.manifest.model_dump(mode="json") for f in files],
            schema=_schema_json(schema),
            format=format.value,
            name=name,
            method=method.value,
            max_steps=max_steps,
            max_runtime_seconds=max_runtime_seconds,
            checkpoint_interval_steps=checkpoint_interval_steps,
            seed=seed,
            base_model=base_model,
            preset=preset,
            budget=max_cost_microcredits,
        ),
    )
    if "recipe" not in journal.data:
        model, selected = resolve_recipe(client, schema, method, base_model, preset)
        journal.set("recipe", dict(base_model=model, preset=selected))
    if "dataset_id" not in journal.data:
        dataset = client.create_dataset(
            DatasetCreate(name=name, description="agent-prepared"),
            idempotency_key=journal.key("dataset"),
        )
        journal.set("dataset_id", dataset.id)
    dataset_id = journal.data["dataset_id"]
    if "version_id" not in journal.data:
        created = client.create_dataset_version(
            dataset_id,
            DatasetVersionCreate(
                format=format,
                kind=DatasetKind.DEMONSTRATION,
                compatibility_profile="tenon-native-v1",
                robot_schema=schema,
                files=[f.manifest for f in files],
            ),
            idempotency_key=journal.key("dataset-version"),
        )
        journal.set("version_id", created.version.id)
        journal.set("upload_id", created.upload.id)
    # A crash between the two ID writes can recover the authoritative upload ID.
    version = client.get_dataset_version(dataset_id, journal.data["version_id"])
    journal.bind("upload_id", version.upload_id)
    if version.state == "uploading":
        upload = client.get_upload(version.upload_id)
        receipts = {
            k: FileCompletionContinuation.from_json_value(v)
            for k, v in journal.data["completions"].items()
        }
        with uploader_factory(client, completion_observer=journal.completion) as uploader:
            uploader.upload(
                upload,
                files,
                completion_continuations=receipts,
                complete_idempotency_key=journal.key("complete-upload"),
            )
    ready = client.wait_for_dataset_version(dataset_id, version.id)
    journal.set("dataset_version", ready.model_dump(mode="json"))
    spec = TrainingSpec(
        dataset_version_id=ready.id,
        method=method,
        max_steps=max_steps,
        max_runtime_seconds=max_runtime_seconds,
        checkpoint_interval_steps=checkpoint_interval_steps,
        seed=seed,
        **({"lora_profile": "openpi-pi05-action-lora-v1"} if method == TrainingMethod.LORA else {}),
        **journal.data["recipe"],
    )
    journal.bind("training_spec", spec.model_dump(mode="json"))
    if "training_job_id" not in journal.data:
        if "training_quote" not in journal.data:
            quote = client.quote_training(spec, idempotency_key=journal.key("training-quote"))
            journal.set("training_quote", quote.model_dump(mode="json"))
        quote = Quote.model_validate(journal.data["training_quote"])
        # A saved create key proves approval already completed. Its first
        # request may have reserved credits even when the response was lost.
        # Replaying that exact request must not require those credits twice.
        if "training-job" not in journal.data["keys"]:
            approve(quote)
        job = client.create_training_job(
            spec,
            quote=quote,
            max_cost_microcredits=max_cost_microcredits,
            idempotency_key=journal.key("training-job"),
        )
        journal.set("training_job_id", job.id)
    job = client.wait_for_training_job(
        journal.data["training_job_id"], timeout_seconds=max_runtime_seconds + 1800
    )
    journal.set("training_job", job.model_dump(mode="json"))
    if job.spec != spec or job.state != "succeeded":
        raise ValueError(
            f"training job {job.id} is {job.state}; "
            "inspect durable job/failure, never retrain implicitly"
        )
    checkpoints = [client.get_checkpoint(c) for c in job.checkpoint_ids]
    verified = []
    for checkpoint in checkpoints:
        if checkpoint.state == "ready":
            verify_checkpoint(
                checkpoint, job_id=job.id, spec=spec, schema=schema, schema_hash=ready.schema_hash
            )
            verified.append(checkpoint)
    if not verified:
        raise ValueError("succeeded job has no verified ready checkpoint; inspect publication")
    journal.set("checkpoints", [c.model_dump(mode="json") for c in verified])
    return {
        "dataset_version": ready.model_dump(mode="json"),
        "training_quote": journal.data["training_quote"],
        "training_job": job.model_dump(mode="json"),
        "checkpoints": journal.data["checkpoints"],
    }


def deploy_run(
    client: TenonClient,
    journal: RunContext,
    *,
    name: str,
    prompt: str,
    execution_mode: ExecutionMode,
    duration_seconds: int,
    max_cost_microcredits: int,
    approve: Callable[[Quote], None],
) -> dict[str, Any]:
    journal.identity(client)
    if execution_mode == ExecutionMode.TRAINED_RTC:
        raise ValueError("trained_rtc is not admitted by the frozen session-create contract")
    if not journal.data.get("checkpoints"):
        raise ValueError("run context has no verified training checkpoint")
    schema = RobotSchema.model_validate(journal.data["input"]["schema"])
    spec = TrainingSpec.model_validate(journal.data["training_spec"])
    checkpoint_id = max(
        journal.data["checkpoints"], key=lambda c: (c["kind"] == "final", c["step"])
    )["id"]
    checkpoint = client.get_checkpoint(checkpoint_id)
    verify_checkpoint(
        checkpoint,
        job_id=journal.data["training_job_id"],
        spec=spec,
        schema=schema,
        schema_hash=journal.data["dataset_version"]["schema_hash"],
    )
    if schema.action_semantics == "joint_position":
        fallback = FallbackPolicy(
            mode="drain_then_hold", hold_steps=5, hold_behavior=LastAppliedAbsoluteTarget()
        )
    elif schema.transform_digest == f"sha256:{LIBERO_TRANSFORM_SHA256}":
        fallback = FallbackPolicy(
            mode="drain_then_hold",
            hold_steps=5,
            hold_behavior=ZeroDeltaPreserve(preserve_channels=["gripper"]),
        )
    else:
        fallback = FallbackPolicy(mode="immediate_stop", hold_steps=0, hold_behavior=None)
    request = DeploymentCreate(
        checkpoint_id=checkpoint.id,
        name=name,
        robot_schema=schema,
        default_prompt=prompt,
        protocols={Protocol.TENON_CONTROL},
        execution_modes={execution_mode},
        fallback=fallback,
        warm_idle_seconds=0,
    )
    journal.bind("deployment_spec", request.model_dump(mode="json"))
    if "deployment_id" not in journal.data:
        deployment = client.create_deployment(request, idempotency_key=journal.key("deployment"))
        journal.set("deployment_id", deployment.id)
    deployment = client.wait_for_deployment(journal.data["deployment_id"])
    journal.set("deployment", deployment.model_dump(mode="json"))
    if (
        deployment.state != "ready"
        or deployment.checkpoint_id != checkpoint.id
        or deployment.schema_hash != journal.data["dataset_version"]["schema_hash"]
        or execution_mode not in deployment.execution_modes
        or deployment.fallback != fallback
    ):
        raise ValueError("deployment is not ready with the exact checkpoint/schema/mode/fallback")
    session_spec = ControlSessionSpec(
        deployment_id=deployment.id,
        duration_seconds=duration_seconds,
        protocol=Protocol.TENON_CONTROL,
        execution_mode=execution_mode,
        fallback=fallback,
    )
    journal.bind("session_spec", session_spec.model_dump(mode="json"))
    journal.bind("session_budget", max_cost_microcredits)
    if "session_id" not in journal.data:
        if "session_quote" not in journal.data:
            quote = client.quote_control_session(
                session_spec, idempotency_key=journal.key("session-quote")
            )
            journal.set("session_quote", quote.model_dump(mode="json"))
        quote = Quote.model_validate(journal.data["session_quote"])
        if "session" not in journal.data["keys"]:
            approve(quote)
        session = client.create_control_session(
            session_spec,
            quote=quote,
            max_cost_microcredits=max_cost_microcredits,
            idempotency_key=journal.key("session"),
        )
        journal.set("session_id", session.id)
    session = client.wait_for_control_session(journal.data["session_id"])
    journal.set("session", session.model_dump(mode="json"))
    if (
        session.state != "active"
        or session.deployment_id != deployment.id
        or session.execution_mode != execution_mode
        or session.protocol != Protocol.TENON_CONTROL
    ):
        raise ValueError(
            f"session {session.id} is {session.state}; no implicit replacement or motion"
        )
    return {
        "deployment": deployment.model_dump(mode="json"),
        "session": session.model_dump(mode="json"),
    }
