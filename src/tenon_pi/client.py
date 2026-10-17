"""High-level synchronous SDK for the Tenon Pi v0.1 lifecycle."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import TypeVar

from .errors import BudgetExceededError, DatasetRejectedError, PollTimeoutError
from .models import (
    Ack,
    Capabilities,
    Checkpoint,
    ControlConnection,
    ControlQuoteRequest,
    ControlSession,
    ControlSessionCreate,
    ControlSessionSpec,
    CreditBalance,
    Dataset,
    DatasetCreate,
    DatasetVersion,
    DatasetVersionCreate,
    DatasetVersionCreated,
    Deployment,
    DeploymentCreate,
    DownloadURL,
    Evaluation,
    EvaluationCreate,
    FileCompleteRequest,
    JobEvent,
    Page,
    PricingCatalog,
    Quote,
    SessionBootstrap,
    TrainingJob,
    TrainingJobCreate,
    TrainingQuoteRequest,
    TrainingSpec,
    Upload,
    UploadPartSignRequest,
    UploadPartSignResponse,
)
from .transport import DEFAULT_BASE_URL, RestTransport, new_idempotency_key

R = TypeVar("R")


class TenonClient:
    """Tenon Pi lifecycle client.

    Mutations accept an optional ``idempotency_key``. When omitted, the SDK
    creates one. If transport fails before a response, the exception carries
    that key so the caller can perform an exact retry.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        transport: RestTransport | None = None,
    ) -> None:
        self._transport = transport or RestTransport(
            api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def __enter__(self) -> TenonClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def base_url(self) -> str:
        return self._transport.base_url

    @property
    def last_request_id(self) -> str | None:
        return self._transport.last_request_id

    def close(self) -> None:
        self._transport.close()

    @staticmethod
    def _key(value: str | None, prefix: str) -> str:
        return value or new_idempotency_key(prefix)

    # Account and metadata
    def me(self) -> SessionBootstrap:
        return self._transport.request(
            "GET",
            "me",
            response_model=SessionBootstrap,
            expected_status=200,
            required_response_headers={"Cache-Control": "no-store"},
        )

    def credits(self) -> CreditBalance:
        return self._transport.request(
            "GET", "credits", response_model=CreditBalance, expected_status=200
        )

    def capabilities(self) -> Capabilities:
        return self._transport.request(
            "GET", "capabilities", response_model=Capabilities, expected_status=200
        )

    def pricing(self) -> PricingCatalog:
        return self._transport.request(
            "GET", "pricing", response_model=PricingCatalog, expected_status=200
        )

    # Datasets
    def create_dataset(
        self,
        request: DatasetCreate,
        *,
        idempotency_key: str | None = None,
    ) -> Dataset:
        return self._transport.request(
            "POST",
            "datasets",
            response_model=Dataset,
            expected_status=201,
            body=request,
            idempotency_key=self._key(idempotency_key, "dataset"),
        )

    def get_dataset(self, dataset_id: str) -> Dataset:
        return self._transport.request(
            "GET",
            f"datasets/{dataset_id}",
            response_model=Dataset,
            expected_status=200,
        )

    def list_datasets(self, *, limit: int = 25, cursor: str | None = None) -> Page[Dataset]:
        return self._transport.request(
            "GET",
            "datasets",
            response_model=Page[Dataset],
            expected_status=200,
            params={"limit": limit, "cursor": cursor},
        )

    def create_dataset_version(
        self,
        dataset_id: str,
        request: DatasetVersionCreate,
        *,
        idempotency_key: str | None = None,
    ) -> DatasetVersionCreated:
        return self._transport.request(
            "POST",
            f"datasets/{dataset_id}/versions",
            response_model=DatasetVersionCreated,
            expected_status=201,
            body=request,
            idempotency_key=self._key(idempotency_key, "dataset-version"),
        )

    def get_dataset_version(self, dataset_id: str, version_id: str) -> DatasetVersion:
        return self._transport.request(
            "GET",
            f"datasets/{dataset_id}/versions/{version_id}",
            response_model=DatasetVersion,
            expected_status=200,
        )

    def wait_for_dataset_version(
        self,
        dataset_id: str,
        version_id: str,
        *,
        timeout_seconds: float = 1800,
        poll_interval_seconds: float = 2,
        on_update: Callable[[DatasetVersion], None] | None = None,
    ) -> DatasetVersion:
        terminal = {"ready", "rejected", "expired", "aborted", "deleted"}
        version = self._poll(
            lambda: self.get_dataset_version(dataset_id, version_id),
            lambda item: item.state in terminal,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            on_update=on_update,
            description=f"dataset version {version_id}",
        )
        if version.state == "rejected":
            raise DatasetRejectedError(version.id, list(version.findings))
        return version

    # Uploads
    def get_upload(self, upload_id: str) -> Upload:
        return self._transport.request(
            "GET", f"uploads/{upload_id}", response_model=Upload, expected_status=200
        )

    def sign_upload_parts(
        self,
        upload_id: str,
        request: UploadPartSignRequest,
        *,
        idempotency_key: str | None = None,
    ) -> UploadPartSignResponse:
        return self._transport.request(
            "POST",
            f"uploads/{upload_id}/parts/sign",
            response_model=UploadPartSignResponse,
            expected_status=200,
            body=request,
            idempotency_key=self._key(idempotency_key, "sign-parts"),
        )

    def complete_upload_file(
        self,
        upload_id: str,
        file_id: str,
        request: FileCompleteRequest,
        *,
        idempotency_key: str | None = None,
    ) -> Upload:
        return self._transport.request(
            "POST",
            f"uploads/{upload_id}/files/{file_id}/complete",
            response_model=Upload,
            expected_status=200,
            body=request,
            idempotency_key=self._key(idempotency_key, "complete-file"),
        )

    def complete_upload(self, upload_id: str, *, idempotency_key: str | None = None) -> Upload:
        return self._transport.request(
            "POST",
            f"uploads/{upload_id}/complete",
            response_model=Upload,
            expected_status=202,
            idempotency_key=self._key(idempotency_key, "complete-upload"),
        )

    def abort_upload(
        self,
        upload_id: str,
        *,
        resource_version: int,
        idempotency_key: str | None = None,
    ) -> Ack:
        return self._transport.request(
            "POST",
            f"uploads/{upload_id}/abort",
            response_model=Ack,
            expected_status=200,
            if_match=resource_version,
            idempotency_key=self._key(idempotency_key, "abort-upload"),
        )

    # Quotes and jobs
    def quote_training(
        self,
        spec: TrainingSpec,
        *,
        idempotency_key: str | None = None,
    ) -> Quote:
        return self._transport.request(
            "POST",
            "quotes",
            response_model=Quote,
            expected_status=201,
            body=TrainingQuoteRequest(spec=spec),
            idempotency_key=self._key(idempotency_key, "quote-training"),
        )

    def quote_control_session(
        self,
        spec: ControlSessionSpec,
        *,
        idempotency_key: str | None = None,
    ) -> Quote:
        return self._transport.request(
            "POST",
            "quotes",
            response_model=Quote,
            expected_status=201,
            body=ControlQuoteRequest(spec=spec),
            idempotency_key=self._key(idempotency_key, "quote-control"),
        )

    @staticmethod
    def require_budget(quote: Quote, max_cost_microcredits: int) -> Quote:
        if max_cost_microcredits < 0:
            raise ValueError("max_cost_microcredits must be non-negative")
        if quote.max_cost_microcredits > max_cost_microcredits:
            raise BudgetExceededError(
                quoted_microcredits=quote.max_cost_microcredits,
                budget_microcredits=max_cost_microcredits,
            )
        return quote

    def create_training_job(
        self,
        spec: TrainingSpec,
        *,
        quote: Quote,
        max_cost_microcredits: int,
        idempotency_key: str | None = None,
    ) -> TrainingJob:
        if quote.kind != "training_job":
            raise ValueError("training job requires a training_job quote")
        self.require_budget(quote, max_cost_microcredits)
        request = TrainingJobCreate(**spec.model_dump(exclude_unset=True), quote_id=quote.id)
        return self._transport.request(
            "POST",
            "training/jobs",
            response_model=TrainingJob,
            expected_status=202,
            body=request,
            idempotency_key=self._key(idempotency_key, "training-job"),
        )

    def get_training_job(self, job_id: str) -> TrainingJob:
        return self._transport.request(
            "GET", f"training/jobs/{job_id}", response_model=TrainingJob, expected_status=200
        )

    def cancel_training_job(
        self,
        job_id: str,
        *,
        resource_version: int,
        idempotency_key: str | None = None,
    ) -> TrainingJob:
        return self._transport.request(
            "POST",
            f"training/jobs/{job_id}/cancel",
            response_model=TrainingJob,
            expected_status=202,
            if_match=resource_version,
            idempotency_key=self._key(idempotency_key, "cancel-job"),
        )

    def list_job_events(
        self, job_id: str, *, limit: int = 25, cursor: str | None = None
    ) -> Page[JobEvent]:
        return self._transport.request(
            "GET",
            f"training/jobs/{job_id}/events",
            response_model=Page[JobEvent],
            expected_status=200,
            params={"limit": limit, "cursor": cursor},
        )

    def iter_job_events(self, job_id: str, *, page_size: int = 100) -> Iterator[JobEvent]:
        cursor: str | None = None
        while True:
            page = self.list_job_events(job_id, limit=page_size, cursor=cursor)
            yield from page.items
            if page.next_cursor is None:
                return
            cursor = page.next_cursor

    def wait_for_training_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 5,
        on_update: Callable[[TrainingJob], None] | None = None,
    ) -> TrainingJob:
        terminal = {"succeeded", "failed", "canceled"}
        return self._poll(
            lambda: self.get_training_job(job_id),
            lambda item: item.state in terminal,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            on_update=on_update,
            description=f"training job {job_id}",
        )

    # Checkpoints
    def list_checkpoints(
        self,
        *,
        training_job_id: str | None = None,
        limit: int = 25,
        cursor: str | None = None,
    ) -> Page[Checkpoint]:
        return self._transport.request(
            "GET",
            "checkpoints",
            response_model=Page[Checkpoint],
            expected_status=200,
            params={
                "training_job_id": training_job_id,
                "limit": limit,
                "cursor": cursor,
            },
        )

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        return self._transport.request(
            "GET", f"checkpoints/{checkpoint_id}", response_model=Checkpoint, expected_status=200
        )

    def create_checkpoint_download_url(
        self, checkpoint_id: str, *, idempotency_key: str | None = None
    ) -> DownloadURL:
        return self._transport.request(
            "POST",
            f"checkpoints/{checkpoint_id}/download-url",
            response_model=DownloadURL,
            expected_status=200,
            idempotency_key=self._key(idempotency_key, "checkpoint-download"),
        )

    # Deployments
    def create_deployment(
        self,
        request: DeploymentCreate,
        *,
        idempotency_key: str | None = None,
    ) -> Deployment:
        return self._transport.request(
            "POST",
            "deployments",
            response_model=Deployment,
            expected_status=202,
            body=request,
            idempotency_key=self._key(idempotency_key, "deployment"),
        )

    def get_deployment(self, deployment_id: str) -> Deployment:
        return self._transport.request(
            "GET", f"deployments/{deployment_id}", response_model=Deployment, expected_status=200
        )

    def wait_for_deployment(
        self,
        deployment_id: str,
        *,
        timeout_seconds: float = 1800,
        poll_interval_seconds: float = 2,
        on_update: Callable[[Deployment], None] | None = None,
    ) -> Deployment:
        terminal = {"ready", "failed", "stopped"}
        return self._poll(
            lambda: self.get_deployment(deployment_id),
            lambda item: item.state in terminal,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            on_update=on_update,
            description=f"deployment {deployment_id}",
        )

    def stop_deployment(
        self,
        deployment_id: str,
        *,
        resource_version: int,
        idempotency_key: str | None = None,
    ) -> Deployment:
        return self._transport.request(
            "POST",
            f"deployments/{deployment_id}/stop",
            response_model=Deployment,
            expected_status=202,
            if_match=resource_version,
            idempotency_key=self._key(idempotency_key, "stop-deployment"),
        )

    # Control sessions
    def create_control_session(
        self,
        spec: ControlSessionSpec,
        *,
        quote: Quote,
        max_cost_microcredits: int,
        idempotency_key: str | None = None,
    ) -> ControlSession:
        if spec.execution_mode == "trained_rtc":
            raise ValueError("trained_rtc isn't admitted by ControlSessionCreate in OpenAPI v0.1")
        if quote.kind != "control_session":
            raise ValueError("control session requires a control_session quote")
        self.require_budget(quote, max_cost_microcredits)
        request = ControlSessionCreate(**spec.model_dump(exclude_unset=True), quote_id=quote.id)
        return self._transport.request(
            "POST",
            "control/sessions",
            response_model=ControlSession,
            expected_status=202,
            body=request,
            idempotency_key=self._key(idempotency_key, "control-session"),
        )

    def get_control_session(self, session_id: str) -> ControlSession:
        return self._transport.request(
            "GET",
            f"control/sessions/{session_id}",
            response_model=ControlSession,
            expected_status=200,
        )

    def wait_for_control_session(
        self,
        session_id: str,
        *,
        timeout_seconds: float = 900,
        poll_interval_seconds: float = 2,
        on_update: Callable[[ControlSession], None] | None = None,
    ) -> ControlSession:
        terminal = {"active", "closed", "canceled", "queue_expired", "failed", "expired"}
        return self._poll(
            lambda: self.get_control_session(session_id),
            lambda item: item.state in terminal,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            on_update=on_update,
            description=f"control session {session_id}",
        )

    def create_control_connection(
        self, session_id: str, *, idempotency_key: str | None = None
    ) -> ControlConnection:
        # wss_url is returned and validated as wss://; callers must use it verbatim.
        return self._transport.request(
            "POST",
            f"control/sessions/{session_id}/connections",
            response_model=ControlConnection,
            expected_status=201,
            idempotency_key=self._key(idempotency_key, "control-connection"),
        )

    def close_control_session(
        self,
        session_id: str,
        *,
        resource_version: int,
        idempotency_key: str | None = None,
    ) -> ControlSession:
        return self._transport.request(
            "POST",
            f"control/sessions/{session_id}/close",
            response_model=ControlSession,
            expected_status=202,
            if_match=resource_version,
            idempotency_key=self._key(idempotency_key, "close-session"),
        )

    # Evaluations
    def create_evaluation(
        self,
        request: EvaluationCreate,
        *,
        idempotency_key: str | None = None,
    ) -> Evaluation:
        return self._transport.request(
            "POST",
            "evaluations",
            response_model=Evaluation,
            expected_status=202,
            body=request,
            idempotency_key=self._key(idempotency_key, "evaluation"),
        )

    def get_evaluation(self, evaluation_id: str) -> Evaluation:
        return self._transport.request(
            "GET", f"evaluations/{evaluation_id}", response_model=Evaluation, expected_status=200
        )

    def wait_for_evaluation(
        self,
        evaluation_id: str,
        *,
        timeout_seconds: float = 1800,
        poll_interval_seconds: float = 2,
        on_update: Callable[[Evaluation], None] | None = None,
    ) -> Evaluation:
        return self._poll(
            lambda: self.get_evaluation(evaluation_id),
            lambda item: item.state in {"ready", "failed"},
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            on_update=on_update,
            description=f"evaluation {evaluation_id}",
        )

    @staticmethod
    def _poll(
        getter: Callable[[], R],
        done: Callable[[R], bool],
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
        on_update: Callable[[R], None] | None,
        description: str,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> R:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be non-negative")
        deadline = clock() + timeout_seconds
        while True:
            item = getter()
            if on_update is not None:
                on_update(item)
            if done(item):
                return item
            if clock() >= deadline:
                raise PollTimeoutError(f"timed out waiting for {description}")
            sleeper(poll_interval_seconds)
