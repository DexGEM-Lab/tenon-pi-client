"""Typed request and response models for the approved Tenon Pi v0.1 contract."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Generic, Literal, Self, TypeVar

from pydantic import (
    AnyHttpUrl,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    SecretStr,
    StringConstraints,
    model_validator,
)


class RequestModel(BaseModel):
    """Requests reject unknown fields and retain frozen-wire presence metadata."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    wire_required_defaults: ClassVar[frozenset[str]] = frozenset()


class ResponseModel(BaseModel):
    """Responses ignore unknown fields for forward compatibility."""

    model_config = ConfigDict(extra="ignore")


class TrainingMethod(StrEnum):
    FULL_FT = "full_ft"
    LORA = "lora"


class TrainingMode(StrEnum):
    STANDARD = "standard"
    WAN_RTC = "wan_rtc"


class Protocol(StrEnum):
    TENON_CONTROL = "tenon-control-v1"
    OPENPI_WS = "openpi-ws-v1"


class ExecutionMode(StrEnum):
    SYNCHRONOUS_CHUNKED = "synchronous_chunked"
    ASYNC_ALIGNED = "async_aligned"
    GUIDED_RTC = "guided_rtc"
    TRAINED_RTC = "trained_rtc"


class DatasetKind(StrEnum):
    DEMONSTRATION = "demonstration"
    EVALUATION = "evaluation"


class DatasetFormat(StrEnum):
    LEROBOT_V2 = "lerobot_v2"
    LEROBOT_V3 = "lerobot_v3"


class ErrorDetail(ResponseModel):
    code: str
    message: str
    request_id: str
    retryable: bool
    details: dict[str, Any] | None = None


class ErrorEnvelope(ResponseModel):
    error: ErrorDetail


T = TypeVar("T")


class Page(ResponseModel, Generic[T]):
    items: list[T]
    next_cursor: str | None


class Ack(ResponseModel):
    ok: bool


class User(ResponseModel):
    id: str
    email: str
    display_name: str
    tenant_id: str
    status: Literal["pending_activation", "active", "disabled"]
    roles: list[Literal["customer", "admin"]] = Field(default_factory=list)
    created_at: datetime
    resource_version: Annotated[int, Field(ge=1)]


class SessionBootstrap(User):
    csrf_token: str | None


class CreditBalance(ResponseModel):
    balance_microcredits: int
    reserved_microcredits: Annotated[int, Field(ge=0)]
    available_microcredits: int
    as_of: datetime


class LoraProfile(ResponseModel):
    id: str
    base_model: str
    paligemma_rank: int
    action_expert_rank: int
    trainable_heads: list[str] = Field(default_factory=list)
    trainable_manifest_digest: str


class Capabilities(ResponseModel):
    contract_version: Literal["0.1.0"]
    base_models: list[str]
    training_methods: list[TrainingMethod]
    protocols: list[Protocol]
    execution_modes: list[ExecutionMode] = Field(default_factory=list)
    limits: dict[str, int | float | str | bool]
    lora_profiles: list[LoraProfile] = Field(default_factory=list)


class PriceLine(ResponseModel):
    operation: Literal["full_ft", "lora", "control_session"]
    accelerator_class: str
    unit: Literal["gpu_millisecond"]
    unit_price_microcredits: Annotated[int, Field(ge=0)]


class PricingCatalog(ResponseModel):
    price_version: str
    credit_scale: Literal[1_000_000]
    lines: list[PriceLine]
    effective_at: datetime
    storage_credit_billed: Literal[False] = False


ChannelName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_.-]*$")]
Digest = Annotated[str, StringConstraints(pattern=r"^sha256:")]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class Channel(RequestModel):
    name: ChannelName
    unit: Literal[
        "radian",
        "radian_per_second",
        "meter",
        "meter_per_second",
        "normalized_0_1",
        "unitless",
    ]
    lower: FiniteFloat
    upper: FiniteFloat

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("channel lower bound must not exceed upper bound")
        return self


class Camera(RequestModel):
    wire_required_defaults = frozenset({"color_space"})

    id: ChannelName
    required: bool
    accepted_encodings: set[Literal["jpeg", "png", "raw_rgb8"]] = Field(min_length=1)
    color_space: Literal["rgb"] = "rgb"
    max_width: Annotated[int, Field(ge=1)]
    max_height: Annotated[int, Field(ge=1)]


class ImageTransform(RequestModel):
    wire_required_defaults = frozenset({"color_space", "dtype"})

    target_width: Annotated[int, Field(ge=1)]
    target_height: Annotated[int, Field(ge=1)]
    resize_mode: Literal["pad", "center_crop", "stretch"]
    color_space: Literal["rgb"] = "rgb"
    dtype: Literal["uint8"] = "uint8"


class RobotSchema(RequestModel):
    wire_required_defaults = frozenset({"schema_version"})

    schema_version: Literal["robot-schema-v1"] = "robot-schema-v1"
    controller_hz: Annotated[FiniteFloat, Field(gt=0, le=500)]
    state_channels: list[Channel] = Field(min_length=1)
    action_channels: list[Channel] = Field(min_length=1)
    action_semantics: Literal["joint_position", "joint_delta", "cartesian_delta"]
    cameras: list[Camera] = Field(min_length=1)
    image_transform: ImageTransform
    action_horizon_steps: Annotated[int, Field(ge=1, le=512)]
    max_committed_prefix_steps: Annotated[int, Field(ge=0, le=511)]
    transform_digest: Digest
    normalization_digest: Digest

    @model_validator(mode="after")
    def schema_invariants(self) -> Self:
        for label, values in (
            ("state channel", [channel.name for channel in self.state_channels]),
            ("action channel", [channel.name for channel in self.action_channels]),
            ("camera", [camera.id for camera in self.cameras]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} name")
        if self.max_committed_prefix_steps >= self.action_horizon_steps:
            raise ValueError("max_committed_prefix_steps must be less than action_horizon_steps")
        return self


class ZeroDeltaPreserve(RequestModel):
    wire_required_defaults = frozenset({"type"})

    type: Literal["zero_delta_preserve"] = "zero_delta_preserve"
    preserve_channels: list[str]


class LastAppliedAbsoluteTarget(RequestModel):
    wire_required_defaults = frozenset({"type", "preserve_channels"})

    type: Literal["last_applied_absolute_target"] = "last_applied_absolute_target"
    preserve_channels: list[str] = Field(default_factory=list, max_length=0)


HoldBehavior = ZeroDeltaPreserve | LastAppliedAbsoluteTarget


class FallbackPolicy(RequestModel):
    mode: Literal["immediate_stop", "drain_then_hold"]
    hold_steps: Annotated[int, Field(ge=0, le=500)]
    hold_behavior: HoldBehavior | None

    @model_validator(mode="after")
    def fallback_is_coherent(self) -> Self:
        if self.mode == "immediate_stop" and (
            self.hold_steps != 0 or self.hold_behavior is not None
        ):
            raise ValueError("immediate_stop requires hold_steps=0 and hold_behavior=null")
        if self.mode == "drain_then_hold" and (self.hold_steps < 1 or self.hold_behavior is None):
            raise ValueError("drain_then_hold requires a positive hold_steps and hold_behavior")
        return self

    def validate_for_schema(self, schema: RobotSchema) -> None:
        if self.mode == "immediate_stop":
            return
        assert self.hold_behavior is not None
        action_names = {channel.name for channel in schema.action_channels}
        if isinstance(self.hold_behavior, ZeroDeltaPreserve):
            if schema.action_semantics not in {"joint_delta", "cartesian_delta"}:
                raise ValueError("zero_delta_preserve requires delta action semantics")
            unknown = set(self.hold_behavior.preserve_channels) - action_names
            if unknown:
                raise ValueError(f"fallback preserves unknown action channels: {sorted(unknown)}")
        elif schema.action_semantics != "joint_position":
            raise ValueError("last_applied_absolute_target requires joint_position semantics")


class DatasetCreate(RequestModel):
    name: Annotated[str, Field(min_length=1, max_length=200)]
    description: Annotated[str, Field(max_length=2000)] = ""


class Dataset(ResponseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    resource_version: Annotated[int, Field(ge=1)]


class FileManifest(RequestModel):
    relative_path: Annotated[str, Field(min_length=1, max_length=1024)]
    size_bytes: Annotated[int, Field(ge=1)]
    sha256: Sha256Hex
    content_type: str


class DatasetVersionCreate(RequestModel):
    wire_required_defaults = frozenset({"format"})

    format: DatasetFormat = DatasetFormat.LEROBOT_V3
    kind: DatasetKind
    compatibility_profile: Literal["tenon-native-v1", "pi-partner-data-1caf4699"]
    robot_schema: RobotSchema
    files: list[FileManifest] = Field(min_length=1, max_length=100_000)
    source_evaluation_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def version_invariants(self) -> Self:
        paths = [file.relative_path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate relative_path in file manifest")
        if len(self.source_evaluation_ids) != len(set(self.source_evaluation_ids)):
            raise ValueError("duplicate source evaluation id")
        if (
            self.format == DatasetFormat.LEROBOT_V2
            and self.compatibility_profile != "tenon-native-v1"
        ):
            raise ValueError("lerobot_v2 supports only compatibility_profile=tenon-native-v1")
        return self


class ValidationFinding(ResponseModel):
    severity: Literal["error", "warning"]
    code: str
    path: str | None
    message: str


class DatasetVersion(ResponseModel):
    id: str
    dataset_id: str
    format: DatasetFormat
    kind: DatasetKind
    state: Literal[
        "uploading",
        "verifying",
        "promoting",
        "validating",
        "ready",
        "rejected",
        "expired",
        "aborted",
        "deleting",
        "deleted",
    ]
    manifest_digest: Digest
    schema_hash: Digest
    upload_id: str
    findings: list[ValidationFinding]
    created_at: datetime
    updated_at: datetime
    resource_version: Annotated[int, Field(ge=1)]
    content_manifest_digest: Digest | None = None


class UploadFile(ResponseModel):
    id: str
    relative_path: str
    size_bytes: int
    sha256: str
    mode: Literal["single", "multipart"]
    state: Literal["pending", "uploading", "complete", "failed"]


class Upload(ResponseModel):
    id: str
    dataset_version_id: str
    state: Literal["open", "completing", "verified", "expired", "aborted", "failed"]
    expires_at: datetime
    files: list[UploadFile]
    created_at: datetime
    updated_at: datetime
    resource_version: Annotated[int, Field(ge=1)]


class DatasetVersionCreated(ResponseModel):
    version: DatasetVersion
    upload: Upload


class PartSignItem(RequestModel):
    file_id: str
    part_number: Annotated[int, Field(ge=1, le=10_000)]
    content_length: Annotated[int, Field(ge=1)]


class UploadPartSignRequest(RequestModel):
    parts: list[PartSignItem] = Field(min_length=1, max_length=1000)


class SignedPart(ResponseModel):
    file_id: str
    part_number: int
    method: Literal["PUT"]
    url: AnyHttpUrl
    required_headers: dict[str, str]
    expires_at: datetime


class UploadPartSignResponse(ResponseModel):
    parts: list[SignedPart]


class CompletedPart(RequestModel):
    part_number: Annotated[int, Field(ge=1)]
    etag: str


class FileCompleteRequest(RequestModel):
    parts: list[CompletedPart] = Field(min_length=1)


class TrainingSpec(RequestModel):
    dataset_version_id: str
    base_model: str
    method: TrainingMethod
    preset: str
    max_steps: Annotated[int, Field(ge=1)]
    max_runtime_seconds: Annotated[int, Field(ge=60)]
    seed: Annotated[int, Field(ge=0)]
    checkpoint_interval_steps: Annotated[int, Field(ge=1)]
    lora_profile: Literal["openpi-pi05-action-lora-v1"] | None = None
    effective_batch_size: Annotated[int, Field(ge=1, strict=True)] | None = None
    training_mode: TrainingMode | None = None

    @model_validator(mode="after")
    def method_matches_lora_profile(self) -> Self:
        if self.method == TrainingMethod.LORA and self.lora_profile is None:
            raise ValueError("lora_profile is required when method=lora")
        if self.method == TrainingMethod.FULL_FT and self.lora_profile is not None:
            raise ValueError("lora_profile is forbidden when method=full_ft")
        return self


class TrainingJobCreate(TrainingSpec):
    quote_id: str


class TrainingProgress(ResponseModel):
    phase: Literal["dataset_loading", "normalizing", "training", "checkpointing", "publishing"]
    attempt: Annotated[int, Field(ge=1)]
    completed_steps: Annotated[int, Field(ge=0)]
    max_steps: Annotated[int, Field(ge=1)]


class TrainingJob(ResponseModel):
    id: str
    state: Literal[
        "queued",
        "starting",
        "running",
        "recovering",
        "canceling",
        "canceled",
        "succeeded",
        "failed",
    ]
    spec: TrainingSpec
    progress: TrainingProgress
    checkpoint_ids: list[str]
    reservation_id: str
    failure: ErrorDetail | None = None
    created_at: datetime
    updated_at: datetime
    resource_version: Annotated[int, Field(ge=1)]
    billable_gpu_milliseconds: Annotated[int, Field(ge=0)]


class JobEvent(ResponseModel):
    id: str
    seq: Annotated[int, Field(ge=0)]
    type: Literal[
        "queued",
        "state_changed",
        "progress",
        "checkpoint_published",
        "usage_reconciled",
        "failed",
    ]
    created_at: datetime
    payload: dict[str, Any]


class ControlSessionSpec(RequestModel):
    deployment_id: str
    duration_seconds: Annotated[int, Field(ge=60, le=14_400)]
    protocol: Protocol
    execution_mode: ExecutionMode
    fallback: FallbackPolicy

    @model_validator(mode="after")
    def protocol_mode_match(self) -> Self:
        if (
            self.protocol == Protocol.OPENPI_WS
            and self.execution_mode != ExecutionMode.SYNCHRONOUS_CHUNKED
        ):
            raise ValueError("openpi-ws-v1 permits only synchronous_chunked")
        return self


class ControlSessionCreate(ControlSessionSpec):
    # Candidate amendment: service/checkpoint capability still gates trained use.
    execution_mode: Literal[
        ExecutionMode.SYNCHRONOUS_CHUNKED,
        ExecutionMode.ASYNC_ALIGNED,
        ExecutionMode.GUIDED_RTC,
        ExecutionMode.TRAINED_RTC,
    ]
    quote_id: str


class TrainingQuoteRequest(RequestModel):
    wire_required_defaults = frozenset({"kind"})

    kind: Literal["training_job"] = "training_job"
    spec: TrainingSpec


class ControlQuoteRequest(RequestModel):
    wire_required_defaults = frozenset({"kind"})

    kind: Literal["control_session"] = "control_session"
    spec: ControlSessionSpec


QuoteRequest = TrainingQuoteRequest | ControlQuoteRequest


class QuoteLine(ResponseModel):
    unit: Literal["gpu_millisecond"]
    quantity_max: Annotated[int, Field(ge=0)]
    unit_price_microcredits: Annotated[int, Field(ge=0)]


class ResolvedResources(ResponseModel):
    accelerator_model: str
    accelerator_count: Annotated[int, Field(ge=1)]


class Quote(ResponseModel):
    id: str
    kind: Literal["training_job", "control_session"]
    price_version: str
    expires_at: datetime
    max_cost_microcredits: Annotated[int, Field(ge=0)]
    line_items: list[QuoteLine]
    resolved_resources: ResolvedResources
    request_hash: Digest


class Artifact(ResponseModel):
    name: str
    size_bytes: int
    sha256: Sha256Hex


class CheckpointManifest(ResponseModel):
    model_config = ConfigDict(extra="allow")

    base_model: str
    runtime_revision: str
    method: TrainingMethod
    dataset_version_id: str
    robot_schema_hash: str
    transform_digest: str
    normalization_digest: str
    artifacts: list[Artifact]
    lora_profile: str | None = None
    trainable_manifest_digest: str | None = None
    trainable_parameter_count: int | None = None
    adapter_ranks: dict[str, int] | None = None
    effective_batch_size: Annotated[int, Field(ge=1, strict=True)] | None = None
    training_mode: TrainingMode | None = None
    rtc_config: dict[str, Any] | None = None


class Checkpoint(ResponseModel):
    id: str
    training_job_id: str
    step: Annotated[int, Field(ge=0)]
    kind: Literal["intermediate", "final"]
    state: Literal["publishing", "ready", "failed", "deleting", "deleted"]
    manifest: CheckpointManifest
    exportable: bool
    created_at: datetime
    updated_at: datetime
    resource_version: Annotated[int, Field(ge=1)]

    def require_trained_rtc(self, schema: RobotSchema | None = None) -> None:
        """Check public provenance/shape; the service verifies saved model/worker identity."""
        config = self.manifest.rtc_config
        if (
            self.state != "ready"
            or self.manifest.training_mode != TrainingMode.WAN_RTC
            or not self.training_job_id
            or not self.manifest.runtime_revision
            or not self.manifest.artifacts
            or any(item.size_bytes <= 0 for item in self.manifest.artifacts)
        ):
            raise ValueError(
                "trained_rtc requires a ready WAN-trained checkpoint with artifact lineage"
            )
        if not isinstance(config, dict) or config.get("training_mode") != "wan_rtc":
            raise ValueError("trained checkpoint lacks matching conditioning metadata")
        for field in ("adapter_version", "conditioning", "sampler", "loss"):
            if not isinstance(config.get(field), str) or not config[field]:
                raise ValueError("trained checkpoint lacks versioned conditioning metadata")
        horizon, capacity = config.get("action_horizon"), config.get("max_prefix")
        mask = config.get("action_mask")
        if (
            type(horizon) is not int
            or not 1 < horizon <= 512
            or type(capacity) is not int
            or not 0 < capacity < horizon
            or not isinstance(mask, list)
            or not mask
            or not any(mask)
            or any(type(value) is not bool for value in mask)
        ):
            raise ValueError("trained checkpoint conditioning shape/capacity is invalid")
        if schema is not None and (
            horizon != schema.action_horizon_steps
            or capacity < schema.max_committed_prefix_steps
            or len(mask) < len(schema.action_channels)
        ):
            raise ValueError("trained checkpoint conditioning is incompatible with the schema")


class DownloadURL(ResponseModel):
    url: AnyHttpUrl
    expires_at: datetime


class DeploymentCreate(RequestModel):
    checkpoint_id: str
    name: Annotated[str, Field(min_length=1, max_length=200)]
    robot_schema: RobotSchema
    default_prompt: Annotated[str, Field(max_length=2000)] | None
    protocols: set[Protocol] = Field(min_length=1)
    execution_modes: set[ExecutionMode] = Field(min_length=1)
    fallback: FallbackPolicy
    warm_idle_seconds: Annotated[int, Field(ge=0, le=3600)]

    @model_validator(mode="after")
    def deployment_invariants(self) -> Self:
        self.fallback.validate_for_schema(self.robot_schema)
        if (
            Protocol.OPENPI_WS in self.protocols
            and ExecutionMode.SYNCHRONOUS_CHUNKED not in self.execution_modes
        ):
            raise ValueError("openpi-ws-v1 deployment requires synchronous_chunked")
        return self


class Deployment(ResponseModel):
    id: str
    name: str
    checkpoint_id: str
    state: Literal["creating", "ready", "failed", "draining", "stopped"]
    runtime_state: Literal["cold", "warming", "warm", "cooling", "error"]
    schema_hash: Digest
    protocols: list[Protocol]
    execution_modes: list[ExecutionMode]
    fallback: FallbackPolicy
    created_at: datetime
    updated_at: datetime
    resource_version: Annotated[int, Field(ge=1)]


class ControlSession(ResponseModel):
    id: str
    deployment_id: str
    state: Literal[
        "queued",
        "admitting",
        "active",
        "closing",
        "closed",
        "canceled",
        "queue_expired",
        "failed",
        "expired",
    ]
    connection_state: Literal[
        "never_connected", "connected", "disconnected_grace", "rearm_required"
    ]
    protocol: Protocol
    execution_mode: ExecutionMode
    duration_seconds: int
    active_at: datetime | None = None
    expires_at: datetime | None = None
    reservation_id: str
    close_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    resource_version: Annotated[int, Field(ge=1)]
    queue_expires_at: datetime
    billable_gpu_milliseconds: Annotated[int, Field(ge=0)]


class ControlConnection(ResponseModel):
    wss_url: AnyUrl
    token: SecretStr
    expires_at: datetime
    protocol: Protocol

    @model_validator(mode="after")
    def secure_websocket_url(self) -> Self:
        if self.wss_url.scheme != "wss":
            raise ValueError("control connection requires a negotiated wss:// URL")
        return self


class EvaluationSummary(RequestModel):
    episode_count: Annotated[int, Field(ge=1)]
    success_count: Annotated[int, Field(ge=0)]
    mean_episode_seconds: Annotated[FiniteFloat, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def successes_do_not_exceed_episodes(self) -> Self:
        if self.success_count > self.episode_count:
            raise ValueError("success_count cannot exceed episode_count")
        return self


class EvaluationCreate(RequestModel):
    dataset_version_id: str
    checkpoint_id: str
    deployment_id: str
    control_session_id: str | None = None
    summary: EvaluationSummary
    labels: list[Annotated[str, Field(max_length=100)]] = Field(default_factory=list)

    @model_validator(mode="after")
    def labels_are_unique(self) -> Self:
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("evaluation labels must be unique")
        return self


class Evaluation(ResponseModel):
    id: str
    state: Literal["processing", "ready", "failed"]
    dataset_version_id: str
    checkpoint_id: str
    deployment_id: str
    control_session_id: str | None = None
    summary: EvaluationSummary
    labels: list[str]
    failure: ErrorDetail | None = None
    created_at: datetime
    updated_at: datetime
    resource_version: Annotated[int, Field(ge=1)]
