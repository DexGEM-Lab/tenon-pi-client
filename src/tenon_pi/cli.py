"""Command-line lifecycle client for Tenon Pi v0.1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer
from pydantic import BaseModel

from ._version import __version__
from .client import TenonClient
from .data_fixture import write_data_fixture
from .datasets import detect_lerobot_format, validate_lerobot_root, validate_lerobot_v3_root
from .errors import TenonAPIError, TenonError, TenonTransportError
from .libero import acquire_libero10_source, write_libero10_acquisition_manifest
from .libero_v2 import materialize_libero10_v2
from .libero_v3 import materialize_libero10_v3
from .models import (
    ControlSessionSpec,
    DatasetCreate,
    DatasetFormat,
    DatasetKind,
    DatasetVersionCreate,
    DeploymentCreate,
    EvaluationCreate,
    Quote,
    RobotSchema,
    TrainingMethod,
    TrainingSpec,
)
from .preparation import (
    prepare_dataset,
    verify_preparation,
    write_bundle_manifest,
)
from .profiles import libero_profile_provenance, write_libero_robot_schema
from .robot_profile import RobotProfile
from .transport import DEFAULT_BASE_URL
from .uploads import (
    DatasetUploader,
    LocalDatasetFile,
    UploadCompletionPending,
    completion_continuation_from_error,
    scan_dataset_directory,
)

app = typer.Typer(
    name="tenon-pi",
    help="Tenon Pi managed robot-learning lifecycle client.",
    add_completion=False,
    no_args_is_help=True,
    invoke_without_command=True,
    pretty_exceptions_enable=False,
)
dataset_app = typer.Typer(help="Prepare, upload, and validate datasets.", no_args_is_help=True)
training_app = typer.Typer(help="Quote and manage training jobs.", no_args_is_help=True)
checkpoint_app = typer.Typer(help="Inspect checkpoints.", no_args_is_help=True)
deployment_app = typer.Typer(help="Create and manage deployments.", no_args_is_help=True)
session_app = typer.Typer(help="Quote and manage control sessions.", no_args_is_help=True)
evaluation_app = typer.Typer(help="Create and inspect evaluations.", no_args_is_help=True)
account_app = typer.Typer(help="Inspect account metadata and credits.", no_args_is_help=True)
libero_app = typer.Typer(help="Prepare the pinned LIBERO demo acquisition.", no_args_is_help=True)
data_app = typer.Typer(
    help="Inspect and prepare supplied-profile data and assets.", no_args_is_help=True
)
agent_app = typer.Typer(
    help="Run an agent-oriented end-to-end training recipe.", no_args_is_help=True
)

app.add_typer(account_app, name="account")
app.add_typer(dataset_app, name="dataset")
app.add_typer(training_app, name="training")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(deployment_app, name="deployment")
app.add_typer(session_app, name="session")
app.add_typer(evaluation_app, name="evaluation")
app.add_typer(libero_app, name="libero")
app.add_typer(data_app, name="data")
app.add_typer(agent_app, name="agent")


class SimulatorDisplayMode(StrEnum):
    AUTO = "auto"
    WINDOW = "window"
    HEADLESS = "headless"


class InputSpace(StrEnum):
    NATIVE = "native"
    CANONICAL = "canonical"


@dataclass
class CLIState:
    api_key: str | None
    base_url: str


@dataclass(frozen=True)
class _ResolvedDatasetInput:
    root: Path
    format: DatasetFormat
    schema: RobotSchema
    files: list[LocalDatasetFile]
    robot_bundle: bool


@app.callback()
def root(
    ctx: typer.Context,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            envvar="TENON_API_KEY",
            help="Tenon API key. Prefer TENON_API_KEY to shell history.",
            show_default=False,
        ),
    ] = None,
    base_url: Annotated[
        str,
        typer.Option("--base-url", envvar="TENON_BASE_URL", help="REST API base URL."),
    ] = DEFAULT_BASE_URL,
    version: Annotated[
        bool, typer.Option("--version", help="Show client version and exit.")
    ] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    ctx.obj = CLIState(api_key=api_key, base_url=base_url)


def _client(ctx: typer.Context) -> TenonClient:
    state: CLIState = ctx.ensure_object(CLIState)
    if not state.api_key:
        raise typer.BadParameter("set TENON_API_KEY or pass --api-key")
    return TenonClient(state.api_key, base_url=state.base_url)


def _echo(value: BaseModel | dict[str, Any] | list[Any]) -> None:
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


def _transport_error(exc: TenonTransportError) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": "transport_error",
        "message": str(exc),
        "idempotency_key": exc.idempotency_key,
    }
    continuation = completion_continuation_from_error(exc)
    if continuation is not None:
        error["completion_continuation"] = continuation.json_value()
    return {"error": error}


def _upload_pending_error(exc: UploadCompletionPending) -> dict[str, Any]:
    return {
        "error": {
            "code": "upload_completion_pending",
            "message": str(exc),
            "attempts": exc.attempts,
            "elapsed_seconds": exc.elapsed_seconds,
            "last_api_error": {
                "status_code": exc.last_error.status_code,
                "code": exc.last_error.code,
                "request_id": exc.last_error.request_id,
                "retryable": exc.last_error.retryable,
                "retry_after": exc.last_error.retry_after,
            },
            "continuation": exc.continuation.json_value(),
        }
    }


M = TypeVar("M", bound=BaseModel)


def _load_model(path: Path, model: type[M]) -> M:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise typer.BadParameter(f"cannot read {path}: {exc}") from exc
    except ValueError as exc:
        raise typer.BadParameter(f"invalid {model.__name__} JSON in {path}: {exc}") from exc


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise typer.BadParameter(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid {label} JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise typer.BadParameter(f"{label} JSON in {path} must contain an object")
    return value


def _resolve_dataset_input(
    directory: Path,
    schema_file: Path,
    declared_format: DatasetFormat | None,
) -> _ResolvedDatasetInput:
    root = directory.expanduser().resolve(strict=True)
    looks_like_robot_bundle = (root / "preparation-manifest.json").exists() or (
        (root / "dataset").is_dir() and (root / "assets" / "robot-profile-v1").is_dir()
    )
    if looks_like_robot_bundle:
        prepared = verify_preparation(root)
        detected = DatasetFormat(prepared.format)
        if declared_format is not None and declared_format != detected:
            raise typer.BadParameter(
                f"prepared profile bundle is {detected.value}, not {declared_format.value}"
            )
        supplied_schema = _load_model(schema_file, RobotSchema)
        prepared_schema = _load_model(prepared.schema, RobotSchema)
        if supplied_schema != prepared_schema:
            raise typer.BadParameter(
                "--schema must match the verified prepared bundle robot-schema.json"
            )
        return _ResolvedDatasetInput(
            root=root,
            format=detected,
            schema=prepared_schema,
            files=scan_dataset_directory(root),
            robot_bundle=True,
        )

    detected = DatasetFormat(detect_lerobot_format(root))
    if declared_format is not None and declared_format != detected:
        raise typer.BadParameter(
            f"declared {declared_format.value} disagrees with meta/info.json ({detected.value})"
        )
    validate_lerobot_root(root, detected.value)
    return _ResolvedDatasetInput(
        root=root,
        format=detected,
        schema=_load_model(schema_file, RobotSchema),
        files=scan_dataset_directory(root),
        robot_bundle=False,
    )


def _training_spec(
    dataset_version_id: str,
    method: TrainingMethod,
    max_steps: int,
    max_runtime_seconds: int,
    seed: int,
    checkpoint_interval_steps: int,
    base_model: str,
    preset: str,
) -> TrainingSpec:
    values: dict[str, object] = {
        "dataset_version_id": dataset_version_id,
        "base_model": base_model,
        "method": method,
        "preset": preset,
        "max_steps": max_steps,
        "max_runtime_seconds": max_runtime_seconds,
        "seed": seed,
        "checkpoint_interval_steps": checkpoint_interval_steps,
    }
    if method == TrainingMethod.LORA:
        values["lora_profile"] = "openpi-pi05-action-lora-v1"
    return TrainingSpec.model_validate(values)


def _confirm_billable(
    client: TenonClient,
    quote: Quote,
    *,
    max_cost_microcredits: int,
    yes: bool,
) -> None:
    client.require_budget(quote, max_cost_microcredits)
    credits = client.credits()
    typer.echo(
        f"Quote {quote.id}: maximum {quote.max_cost_microcredits} microcredits; "
        f"available {credits.available_microcredits} microcredits.",
        err=True,
    )
    typer.echo(
        f"Expires {quote.expires_at.isoformat()}; "
        f"resources {quote.resolved_resources.accelerator_count} x "
        f"{quote.resolved_resources.accelerator_model} ({quote.kind}).",
        err=True,
    )
    if quote.max_cost_microcredits > credits.available_microcredits:
        raise typer.BadParameter("quote exceeds currently available credits")
    if not yes and not typer.confirm("Create the billable resource with this maximum?"):
        raise typer.Abort()


@account_app.command("me")
def account_me(ctx: typer.Context) -> None:
    with _client(ctx) as client:
        _echo(client.me())


@account_app.command("credits")
def account_credits(ctx: typer.Context) -> None:
    with _client(ctx) as client:
        _echo(client.credits())


@account_app.command("capabilities")
def account_capabilities(ctx: typer.Context) -> None:
    with _client(ctx) as client:
        _echo(client.capabilities())


@dataset_app.command("manifest")
def dataset_manifest(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    format: Annotated[DatasetFormat | None, typer.Option("--format")] = None,
) -> None:
    """Validate a prepared LeRobot v2/v3 directory and write its file manifest."""
    detected = DatasetFormat(detect_lerobot_format(directory))
    if format is not None and format != detected:
        raise typer.BadParameter("--format disagrees with source metadata")
    format = detected
    validate_lerobot_root(directory, format.value)
    files = scan_dataset_directory(directory)
    payload = {
        "format": format.value,
        "files": [item.manifest.model_dump(mode="json") for item in files],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _echo(payload)


@dataset_app.command("upload")
def dataset_upload(
    ctx: typer.Context,
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    schema_file: Annotated[Path, typer.Option("--schema", exists=True, dir_okay=False)],
    name: Annotated[str, typer.Option("--name")],
    description: Annotated[str, typer.Option("--description")] = "",
    kind: Annotated[DatasetKind, typer.Option("--kind")] = DatasetKind.DEMONSTRATION,
    format: Annotated[DatasetFormat | None, typer.Option("--format")] = None,
    compatibility_profile: Annotated[
        str, typer.Option("--compatibility-profile")
    ] = "tenon-native-v1",
    dataset_id: Annotated[str | None, typer.Option("--dataset-id")] = None,
    wait: Annotated[bool, typer.Option("--wait/--no-wait")] = True,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=1)] = 1800,
) -> None:
    """Create a version, upload declared files, and optionally wait for validation."""
    resolved = _resolve_dataset_input(directory, schema_file, format)
    if resolved.robot_bundle and compatibility_profile != "tenon-native-v1":
        raise typer.BadParameter("prepared profile bundles require tenon-native-v1")
    with _client(ctx) as client:
        if dataset_id is None:
            dataset_id = client.create_dataset(DatasetCreate(name=name, description=description)).id
        created = client.create_dataset_version(
            dataset_id,
            DatasetVersionCreate(
                format=resolved.format,
                kind=kind,
                compatibility_profile=compatibility_profile,  # type: ignore[arg-type]
                robot_schema=resolved.schema,
                files=[item.manifest for item in resolved.files],
            ),
        )
        with DatasetUploader(client) as uploader:
            uploader.upload(created.upload, resolved.files)
        if wait:
            result = client.wait_for_dataset_version(
                dataset_id,
                created.version.id,
                timeout_seconds=timeout_seconds,
                on_update=lambda version: typer.echo(
                    f"dataset {version.id}: {version.state}", err=True
                ),
            )
            _echo(result)
        else:
            _echo(created.version)


@agent_app.command("train")
def agent_train(
    ctx: typer.Context,
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    schema_file: Annotated[Path, typer.Option("--schema", exists=True, dir_okay=False)],
    name: Annotated[str, typer.Option("--name")],
    method: Annotated[TrainingMethod, typer.Option("--method")],
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)],
    max_runtime_seconds: Annotated[int, typer.Option("--max-runtime-seconds", min=60)],
    checkpoint_interval_steps: Annotated[int, typer.Option("--checkpoint-interval-steps", min=1)],
    max_cost_microcredits: Annotated[int, typer.Option("--max-cost-microcredits", min=0)],
    format: Annotated[DatasetFormat | None, typer.Option("--format")] = None,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    base_model: Annotated[str | None, typer.Option("--base-model")] = None,
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    context_file: Annotated[Path, typer.Option("--context")] = Path(".tenon/run.json"),
) -> None:
    """Resume upload, validation, quote and training from a durable local context."""
    from .workflow import RunContext, run_training

    resolved = _resolve_dataset_input(directory, schema_file, format)
    context_path = context_file.expanduser().resolve()
    if context_path == resolved.root or resolved.root in context_path.parents:
        raise typer.BadParameter("--context must be outside the uploaded data root")
    with RunContext(context_path) as journal, _client(ctx) as client:
        result = run_training(
            client,
            journal,
            files=resolved.files,
            schema=resolved.schema,
            format=resolved.format,
            name=name,
            method=method,
            max_steps=max_steps,
            max_runtime_seconds=max_runtime_seconds,
            checkpoint_interval_steps=checkpoint_interval_steps,
            seed=seed,
            base_model=base_model,
            preset=preset,
            max_cost_microcredits=max_cost_microcredits,
            approve=lambda quote: _confirm_billable(
                client, quote, max_cost_microcredits=max_cost_microcredits, yes=yes
            ),
            uploader_factory=DatasetUploader,
        )
        _echo(result)


@agent_app.command("deploy")
def agent_deploy(
    ctx: typer.Context,
    name: Annotated[str, typer.Option("--name")],
    prompt: Annotated[str, typer.Option("--prompt")],
    max_cost_microcredits: Annotated[int, typer.Option("--max-cost-microcredits", min=0)],
    context_file: Annotated[Path, typer.Option("--context", exists=True)] = Path(".tenon/run.json"),
    execution_mode: Annotated[str, typer.Option("--execution-mode")] = "async_aligned",
    duration_seconds: Annotated[int, typer.Option("--duration-seconds", min=60, max=14400)] = 300,
    yes: Annotated[bool, typer.Option("--yes")] = False,
):
    """Deploy a verified run checkpoint and resume one quoted control-session admission."""
    from .models import ExecutionMode
    from .workflow import RunContext, deploy_run

    with RunContext(context_file) as journal, _client(ctx) as client:
        _echo(
            deploy_run(
                client,
                journal,
                name=name,
                prompt=prompt,
                execution_mode=ExecutionMode(execution_mode),
                duration_seconds=duration_seconds,
                max_cost_microcredits=max_cost_microcredits,
                approve=lambda quote: _confirm_billable(
                    client, quote, max_cost_microcredits=max_cost_microcredits, yes=yes
                ),
            )
        )


@agent_app.command("setup-simulator")
def agent_setup_simulator(
    directory: Annotated[Path, typer.Option("--directory")] = Path(".tenon/simulator"),
    client_source: Annotated[Path, typer.Option("--client-source")] = Path("."),
    display: Annotated[
        SimulatorDisplayMode, typer.Option("--display")
    ] = SimulatorDisplayMode.WINDOW,
    renderer: Annotated[str | None, typer.Option("--renderer")] = None,
    python: Annotated[
        str | None,
        typer.Option(
            "--python",
            help="Optional existing Python3.11; otherwise provision the pinned interpreter.",
        ),
    ] = None,
):
    """Create an owned isolated simulator and run CPU renderer/viewer preflight before spending."""
    from .simulator_setup import bootstrap_simulator

    _echo(
        bootstrap_simulator(
            directory,
            client_source=client_source,
            display=display.value,
            renderer=renderer,
            python=python,
        )
    )


@agent_app.command("start-simulator")
def agent_start_simulator(
    ctx: typer.Context,
    context_file: Annotated[Path, typer.Option("--context", exists=True)] = Path(".tenon/run.json"),
    python: Annotated[
        str | None,
        typer.Option("--python", help="Advanced Python override; runs a fresh local preflight."),
    ] = None,
    setup_file: Annotated[Path, typer.Option("--setup")] = Path(".tenon/simulator/setup.json"),
    task_suite: Annotated[str, typer.Option("--task-suite")] = "libero_10",
    task_id: Annotated[int, typer.Option("--task-id", min=0)] = 0,
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)] = 520,
    display: Annotated[SimulatorDisplayMode, typer.Option("--display")] = SimulatorDisplayMode.AUTO,
):
    """Launch a real installed LIBERO environment; one-time tokens remain in memory."""
    import subprocess

    from .profiles import LIBERO_TRANSFORM_SHA256
    from .simulator_receipt import credential_free_environment, qualified_simulator
    from .workflow import RunContext

    with (
        qualified_simulator(setup_file, python=python, display=display.value) as selected,
        RunContext(context_file) as journal,
        _client(ctx) as client,
    ):
        journal.identity(client)
        schema = journal.data["input"]["schema"]
        if schema["transform_digest"] != f"sha256:{LIBERO_TRANSFORM_SHA256}":
            raise typer.BadParameter("this launcher requires the pinned LIBERO schema")
        session = client.get_control_session(journal.data["session_id"])
        if (
            session.state != "active"
            or session.connection_state != "never_connected"
            or session.deployment_id != journal.data["deployment_id"]
        ):
            raise typer.BadParameter(
                "simulator launch requires a fresh active session for this deployment"
            )
        env = credential_free_environment(selected["environment"])
        state: CLIState = ctx.ensure_object(CLIState)
        env["TENON_API_KEY"], env["TENON_BASE_URL"] = state.api_key, state.base_url
        command = [
            selected["python"],
            "-m",
            "tenon_pi.simulator",
            "--setup-receipt",
            str(selected["receipt_path"]),
            "--display",
            display.value,
            "--session-id",
            session.id,
            "--schema-hash",
            journal.data["dataset_version"]["schema_hash"],
            "--execution-mode",
            journal.data["session_spec"]["execution_mode"],
            "--task-suite",
            task_suite,
            "--task-id",
            str(task_id),
            "--max-steps",
            str(max_steps),
        ]
        journal.set(
            "simulator_launch",
            {
                "session_id": session.id,
                "task_suite": task_suite,
                "task_id": task_id,
                "state": "started",
                "display_requested": display.value,
            },
        )
        result = subprocess.run(command, env=env, check=False)
        journal.set(
            "simulator_launch",
            {
                "session_id": session.id,
                "task_suite": task_suite,
                "task_id": task_id,
                "exit_code": result.returncode,
                "display_requested": display.value,
            },
        )
        if result.returncode:
            raise typer.Exit(code=result.returncode)


@agent_app.command("start-robot")
def agent_start_robot(
    ctx: typer.Context,
    profile_file: Annotated[Path, typer.Option("--profile", exists=True, dir_okay=False)],
    settings_file: Annotated[Path, typer.Option("--settings", exists=True, dir_okay=False)],
    adapter_factory: Annotated[
        str, typer.Option("--adapter", help="Explicit trusted module:factory; executes local code.")
    ],
    adapter_config: Annotated[Path, typer.Option("--adapter-config", exists=True, dir_okay=False)],
    prompt: Annotated[str, typer.Option("--prompt")],
    context_file: Annotated[Path, typer.Option("--context", exists=True)] = Path(".tenon/run.json"),
):
    """Run a configured, connected/armed robot using a verified profile and active session."""
    import signal

    from .models import ExecutionMode, Protocol
    from .robot_runtime import RobotRunSettings, load_robot_adapter, run_connected_robot
    from .workflow import RunContext

    profile = RobotProfile.load(profile_file)
    settings = RobotRunSettings.model_validate_json(settings_file.read_text())
    config = _load_json_object(adapter_config, "trusted adapter configuration")
    with RunContext(context_file) as journal, _client(ctx) as client:
        journal.identity(client)
        schema = RobotSchema.model_validate(journal.data["input"]["schema"])
        if profile.robot_schema(schema.normalization_digest) != schema:
            raise typer.BadParameter(
                "supplied profile must exactly match the run's verified schema"
            )
        session = client.get_control_session(journal.data["session_id"])
        deployment = client.get_deployment(journal.data["deployment_id"])
        mode = ExecutionMode(journal.data["session_spec"]["execution_mode"])
        if (
            session.state != "active"
            or session.connection_state != "never_connected"
            or session.protocol != Protocol.TENON_CONTROL
            or session.execution_mode != mode
            or session.deployment_id != deployment.id
            or deployment.state != "ready"
            or deployment.schema_hash != journal.data["dataset_version"]["schema_hash"]
            or deployment.checkpoint_id != journal.data["deployment_spec"]["checkpoint_id"]
        ):
            raise typer.BadParameter(
                "robot execution requires a fresh active session with the exact run lineage"
            )
        adapter = load_robot_adapter(adapter_factory, profile=profile, config=config)
        previous = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            raise KeyboardInterrupt("robot run termination requested")

        try:
            signal.signal(signal.SIGTERM, terminate)
            if not adapter.connected or not adapter.armed:
                raise typer.BadParameter(
                    "supplied adapter must be genuinely connected and explicitly armed"
                )
            journal.set(
                "robot_run",
                {"state": "starting", "session_id": session.id, "profile_digest": profile.digest()},
            )
            connection = client.create_control_connection(session.id)
            result = run_connected_robot(
                connection,
                adapter=adapter,
                profile=profile,
                schema=schema,
                schema_hash=deployment.schema_hash,
                fallback=deployment.fallback,
                execution_mode=mode,
                settings=settings,
                prompt=prompt,
            )
            journal.set(
                "robot_run",
                {**result, "session_id": session.id, "profile_digest": profile.digest()},
            )
            _echo(result)
        finally:
            try:
                adapter.stop()
            finally:
                signal.signal(signal.SIGTERM, previous)


@training_app.command("spec")
def training_spec(
    dataset_version_id: Annotated[str, typer.Option("--dataset-version-id")],
    method: Annotated[TrainingMethod, typer.Option("--method")],
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)],
    max_runtime_seconds: Annotated[int, typer.Option("--max-runtime-seconds", min=60)],
    checkpoint_interval_steps: Annotated[int, typer.Option("--checkpoint-interval-steps", min=1)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    base_model: Annotated[str, typer.Option("--base-model")] = "openpi/pi05-base",
    preset: Annotated[str, typer.Option("--preset")] = "balanced-v1",
) -> None:
    """Construct the exact TrainingSpec used inside a quote request, without network I/O."""
    spec = _training_spec(
        dataset_version_id,
        method,
        max_steps,
        max_runtime_seconds,
        seed,
        checkpoint_interval_steps,
        base_model,
        preset,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _echo(spec)


@training_app.command("quote")
def training_quote(
    ctx: typer.Context,
    dataset_version_id: Annotated[str, typer.Option("--dataset-version-id")],
    method: Annotated[TrainingMethod, typer.Option("--method")],
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)],
    max_runtime_seconds: Annotated[int, typer.Option("--max-runtime-seconds", min=60)],
    checkpoint_interval_steps: Annotated[int, typer.Option("--checkpoint-interval-steps", min=1)],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    base_model: Annotated[str, typer.Option("--base-model")] = "openpi/pi05-base",
    preset: Annotated[str, typer.Option("--preset")] = "balanced-v1",
) -> None:
    spec = _training_spec(
        dataset_version_id,
        method,
        max_steps,
        max_runtime_seconds,
        seed,
        checkpoint_interval_steps,
        base_model,
        preset,
    )
    with _client(ctx) as client:
        _echo(client.quote_training(spec))


@training_app.command("run")
def training_run(
    ctx: typer.Context,
    dataset_version_id: Annotated[str, typer.Option("--dataset-version-id")],
    method: Annotated[TrainingMethod, typer.Option("--method")],
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)],
    max_runtime_seconds: Annotated[int, typer.Option("--max-runtime-seconds", min=60)],
    checkpoint_interval_steps: Annotated[int, typer.Option("--checkpoint-interval-steps", min=1)],
    max_cost_microcredits: Annotated[int, typer.Option("--max-cost-microcredits", min=0)],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    base_model: Annotated[str, typer.Option("--base-model")] = "openpi/pi05-base",
    preset: Annotated[str, typer.Option("--preset")] = "balanced-v1",
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Confirm the displayed quote non-interactively."),
    ] = False,
    wait: Annotated[bool, typer.Option("--wait/--no-wait")] = True,
) -> None:
    spec = _training_spec(
        dataset_version_id,
        method,
        max_steps,
        max_runtime_seconds,
        seed,
        checkpoint_interval_steps,
        base_model,
        preset,
    )
    with _client(ctx) as client:
        quote = client.quote_training(spec)
        _confirm_billable(
            client,
            quote,
            max_cost_microcredits=max_cost_microcredits,
            yes=yes,
        )
        job = client.create_training_job(
            spec,
            quote=quote,
            max_cost_microcredits=max_cost_microcredits,
        )
        if wait:
            job = client.wait_for_training_job(
                job.id,
                timeout_seconds=max_runtime_seconds + 1800,
                on_update=lambda item: typer.echo(
                    f"job {item.id}: {item.state} {item.progress.completed_steps}/"
                    f"{item.progress.max_steps} ({item.progress.phase})",
                    err=True,
                ),
            )
        _echo(job)


@training_app.command("status")
def training_status(ctx: typer.Context, job_id: str) -> None:
    with _client(ctx) as client:
        _echo(client.get_training_job(job_id))


@training_app.command("cancel")
def training_cancel(
    ctx: typer.Context,
    job_id: str,
    resource_version: Annotated[int, typer.Option("--resource-version", min=1)],
) -> None:
    with _client(ctx) as client:
        _echo(client.cancel_training_job(job_id, resource_version=resource_version))


@checkpoint_app.command("list")
def checkpoint_list(
    ctx: typer.Context,
    training_job_id: Annotated[str | None, typer.Option("--training-job-id")] = None,
) -> None:
    with _client(ctx) as client:
        _echo(client.list_checkpoints(training_job_id=training_job_id))


@checkpoint_app.command("get")
def checkpoint_get(ctx: typer.Context, checkpoint_id: str) -> None:
    with _client(ctx) as client:
        _echo(client.get_checkpoint(checkpoint_id))


@checkpoint_app.command("download-url")
def checkpoint_download_url(ctx: typer.Context, checkpoint_id: str) -> None:
    with _client(ctx) as client:
        _echo(client.create_checkpoint_download_url(checkpoint_id))


@deployment_app.command("create")
def deployment_create(
    ctx: typer.Context,
    request_file: Annotated[Path, typer.Option("--request", exists=True, dir_okay=False)],
    wait: Annotated[bool, typer.Option("--wait/--no-wait")] = True,
) -> None:
    request = _load_model(request_file, DeploymentCreate)
    with _client(ctx) as client:
        deployment = client.create_deployment(request)
        if wait:
            deployment = client.wait_for_deployment(
                deployment.id,
                on_update=lambda item: typer.echo(
                    f"deployment {item.id}: {item.state}/{item.runtime_state}", err=True
                ),
            )
        _echo(deployment)


@deployment_app.command("get")
def deployment_get(ctx: typer.Context, deployment_id: str) -> None:
    with _client(ctx) as client:
        _echo(client.get_deployment(deployment_id))


@deployment_app.command("stop")
def deployment_stop(
    ctx: typer.Context,
    deployment_id: str,
    resource_version: Annotated[int, typer.Option("--resource-version", min=1)],
) -> None:
    with _client(ctx) as client:
        _echo(client.stop_deployment(deployment_id, resource_version=resource_version))


@session_app.command("quote")
def session_quote(
    ctx: typer.Context,
    spec_file: Annotated[Path, typer.Option("--spec", exists=True, dir_okay=False)],
) -> None:
    spec = _load_model(spec_file, ControlSessionSpec)
    with _client(ctx) as client:
        _echo(client.quote_control_session(spec))


@session_app.command("run")
def session_run(
    ctx: typer.Context,
    spec_file: Annotated[Path, typer.Option("--spec", exists=True, dir_okay=False)],
    max_cost_microcredits: Annotated[int, typer.Option("--max-cost-microcredits", min=0)],
    yes: Annotated[bool, typer.Option("--yes")] = False,
    wait: Annotated[bool, typer.Option("--wait/--no-wait")] = True,
) -> None:
    spec = _load_model(spec_file, ControlSessionSpec)
    with _client(ctx) as client:
        quote = client.quote_control_session(spec)
        _confirm_billable(
            client,
            quote,
            max_cost_microcredits=max_cost_microcredits,
            yes=yes,
        )
        session = client.create_control_session(
            spec,
            quote=quote,
            max_cost_microcredits=max_cost_microcredits,
        )
        if wait:
            session = client.wait_for_control_session(
                session.id,
                on_update=lambda item: typer.echo(
                    f"session {item.id}: {item.state}/{item.connection_state}", err=True
                ),
            )
        _echo(session)


@session_app.command("get")
def session_get(ctx: typer.Context, session_id: str) -> None:
    with _client(ctx) as client:
        _echo(client.get_control_session(session_id))


@session_app.command("connection")
def session_connection(ctx: typer.Context, session_id: str) -> None:
    """Issue a one-time token and the exact negotiated WSS URL.

    Treat command output as a secret. The SDK never derives this URL from REST.
    """
    with _client(ctx) as client:
        connection = client.create_control_connection(session_id)
        data = connection.model_dump(mode="json")
        data["token"] = connection.token.get_secret_value()
        _echo(data)


@session_app.command("close")
def session_close(
    ctx: typer.Context,
    session_id: str,
    resource_version: Annotated[int, typer.Option("--resource-version", min=1)],
) -> None:
    with _client(ctx) as client:
        _echo(client.close_control_session(session_id, resource_version=resource_version))


@evaluation_app.command("create")
def evaluation_create(
    ctx: typer.Context,
    request_file: Annotated[Path, typer.Option("--request", exists=True, dir_okay=False)],
    wait: Annotated[bool, typer.Option("--wait/--no-wait")] = True,
) -> None:
    request = _load_model(request_file, EvaluationCreate)
    with _client(ctx) as client:
        evaluation = client.create_evaluation(request)
        if wait:
            evaluation = client.wait_for_evaluation(evaluation.id)
        _echo(evaluation)


@evaluation_app.command("get")
def evaluation_get(ctx: typer.Context, evaluation_id: str) -> None:
    with _client(ctx) as client:
        _echo(client.get_evaluation(evaluation_id))


@data_app.command("inspect")
def data_inspect(directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)]):
    """Inspect local metadata or verify a prepared bundle without credentials."""
    from .data_import import inspect_data

    _echo(inspect_data(directory))


@data_app.command("import-jsonl")
def data_import_jsonl(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    profile_file: Annotated[Path, typer.Option("--profile", exists=True, dir_okay=False)],
    input_space: Annotated[InputSpace, typer.Option("--input-space")],
):
    """Import explicit ordered JSONL records and original PNG bytes into v2.1."""
    from .data_import import import_jsonl

    path = import_jsonl(
        source, output, profile=RobotProfile.load(profile_file), input_space=input_space.value
    )
    _echo({"format": "lerobot_v2", "path": str(path)})


@data_app.command("fixture")
def data_fixture(
    output: Annotated[Path, typer.Option("--output", "-o")],
    episodes: Annotated[int, typer.Option("--episodes", min=1)] = 2,
    frames_per_episode: Annotated[int, typer.Option("--frames-per-episode", min=2)] = 4,
    layout: Annotated[
        str, typer.Option("--layout", help="Synthetic planar or gantry layout.")
    ] = "planar",
) -> None:
    """Write a deterministic no-hardware native-v2 preparation fixture."""

    _echo(
        write_data_fixture(
            output, layout=layout, episodes=episodes, frames_per_episode=frames_per_episode
        )
    )


@data_app.command("prepare")
def preparation(
    source: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    input_space: Annotated[InputSpace, typer.Option("--input-space")],
    repo_id: Annotated[
        str,
        typer.Option(
            "--repo-id",
            help="Logical namespace/name identity; never a cache or filesystem path.",
        ),
    ],
    profile_file: Annotated[Path, typer.Option("--profile", exists=True, dir_okay=False)],
) -> None:
    """Prepare profile-canonical v2/v3 and exact model normalization assets."""
    result = prepare_dataset(
        source,
        output,
        input_space=input_space.value,
        dataset_repo_id=repo_id,
        profile=RobotProfile.load(profile_file),
    )
    _echo(result.json_value())


@data_app.command("manifest")
def data_manifest(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Verify and manifest the whole dataset/assets/schema preparation bundle."""

    _echo(write_bundle_manifest(directory, output))


@data_app.command("verify")
def data_verify(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
) -> None:
    """Regenerate data manifests/statistics/schema and verify every binding."""

    _echo(verify_preparation(directory).json_value())


@libero_app.command("schema")
def libero_schema(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("libero-robot-schema.json"),
) -> None:
    """Write the pinned pi05_libero robot schema with verified real digests."""
    schema = write_libero_robot_schema(output)
    _echo(
        {
            "schema": schema.model_dump(mode="json"),
            "provenance": libero_profile_provenance(),
            "path": str(output),
        }
    )


@libero_app.command("manifest")
def libero_manifest(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "libero10-mini-acquisition.json"
    ),
) -> None:
    """Write the immutable ten-episode source acquisition manifest."""
    _echo(write_libero10_acquisition_manifest(output))


@libero_app.command("materialize-v2")
def libero_materialize_v2(
    source_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
) -> None:
    """Create the immediate native-v2 demo artifact without temporal resampling."""
    result = materialize_libero10_v2(source_dir, output_dir)
    validate_lerobot_root(result, "lerobot_v2")
    _echo({"format": "lerobot_v2", "path": str(result)})


@libero_app.command("materialize-v3")
def libero_materialize_v3(
    source_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
) -> None:
    """Create canonical LeRobot v3 without resampling the retained transitions."""
    result = materialize_libero10_v3(source_dir, output_dir)
    validate_lerobot_v3_root(result)
    _echo({"format": "lerobot_v3", "path": str(result)})


@libero_app.command("acquire")
def libero_acquire(
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    """Download and verify 332.3 MiB of pinned v2 source material.

    Run ``materialize-v2`` for the immediate demo or ``materialize-v3`` for canonical v3.
    """
    if not yes and not typer.confirm("Download the immutable 332.3 MiB LIBERO source selection?"):
        raise typer.Abort()
    _echo(acquire_libero10_source(output_dir))


def main() -> None:
    try:
        app()
    except TenonAPIError as exc:
        typer.echo(
            json.dumps(
                {
                    "error": {
                        "status_code": exc.status_code,
                        "code": exc.code,
                        "message": exc.message,
                        "request_id": exc.request_id,
                        "retryable": exc.retryable,
                        "details": exc.details,
                        "retry_after": exc.retry_after,
                        "idempotency_key": exc.idempotency_key,
                    }
                },
                indent=2,
                sort_keys=True,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except TenonTransportError as exc:
        typer.echo(json.dumps(_transport_error(exc), indent=2, sort_keys=True), err=True)
        raise typer.Exit(code=1) from exc
    except UploadCompletionPending as exc:
        typer.echo(json.dumps(_upload_pending_error(exc), indent=2, sort_keys=True), err=True)
        raise typer.Exit(code=1) from exc
    except TenonError as exc:
        typer.echo(json.dumps({"error": {"message": str(exc)}}, indent=2), err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    main()
