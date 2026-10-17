"""Local manifest generation and direct single/multipart upload helpers."""

from __future__ import annotations

import hashlib
import math
import mimetypes
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath

import httpx

from .client import TenonClient
from .errors import TenonAPIError, TenonError, TenonProtocolError, TenonTransportError
from .models import (
    CompletedPart,
    FileCompleteRequest,
    FileManifest,
    PartSignItem,
    Upload,
    UploadFile,
    UploadPartSignRequest,
)
from .transport import new_idempotency_key

DEFAULT_PART_SIZE = 64 * 1024 * 1024
DEFAULT_COMPLETION_MAX_ATTEMPTS = 6
DEFAULT_COMPLETION_MAX_WAIT_SECONDS = 90.0
_MIN_COMPLETION_RETRY_DELAY_SECONDS = 0.05
_HASH_CHUNK_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class LocalDatasetFile:
    path: Path
    manifest: FileManifest


@dataclass(frozen=True)
class FileCompletionContinuation:
    """Exact accepted completion operation needed for safe caller-controlled resume."""

    upload_id: str
    file_id: str
    request_json: str
    idempotency_key: str

    @classmethod
    def create(
        cls,
        upload_id: str,
        file_id: str,
        request: FileCompleteRequest,
        idempotency_key: str,
    ) -> FileCompletionContinuation:
        return cls(
            upload_id=upload_id,
            file_id=file_id,
            request_json=request.model_dump_json(),
            idempotency_key=idempotency_key,
        )

    @property
    def request(self) -> FileCompleteRequest:
        return FileCompleteRequest.model_validate_json(self.request_json)

    def json_value(self) -> dict[str, str]:
        return {
            "upload_id": self.upload_id,
            "file_id": self.file_id,
            "request_json": self.request_json,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_json_value(cls, value: Mapping[str, str]) -> FileCompletionContinuation:
        required = {"upload_id", "file_id", "request_json", "idempotency_key"}
        if set(value) != required or not all(
            isinstance(value[name], str) and value[name] for name in required
        ):
            raise ValueError("invalid file completion continuation")
        continuation = cls(
            upload_id=value["upload_id"],
            file_id=value["file_id"],
            request_json=value["request_json"],
            idempotency_key=value["idempotency_key"],
        )
        _ = continuation.request  # Validate the exact stored body before use.
        return continuation


class UploadCompletionPending(TenonError):
    """Bounded continuation expired while the accepted provider operation runs."""

    def __init__(
        self,
        *,
        continuation: FileCompletionContinuation,
        last_error: TenonAPIError,
        attempts: int,
        elapsed_seconds: float,
    ) -> None:
        self.continuation = continuation
        self.last_error = last_error
        self.attempts = attempts
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            "file completion remains pending after "
            f"{attempts} attempts/{elapsed_seconds:.3f}s; resume the exact continuation "
            f"(request_id={last_error.request_id})"
        )


class UploadContinuationRequired(TenonError):
    """Public upload state cannot safely reconstruct an accepted completion operation."""


def completion_continuation_from_error(
    error: BaseException,
) -> FileCompletionContinuation | None:
    """Return SDK completion context attached to an original propagated failure."""

    value = getattr(error, "completion_continuation", None)
    return value if isinstance(value, FileCompletionContinuation) else None


def _attach_completion_continuation(
    error: BaseException,
    continuation: FileCompletionContinuation,
) -> None:
    error.completion_continuation = continuation  # type: ignore[attr-defined]


class FileSlice:
    """Re-iterable bounded file stream for one signed upload part."""

    def __init__(self, path: Path, offset: int, length: int, *, chunk_size: int = _HASH_CHUNK_SIZE):
        self.path = path
        self.offset = offset
        self.length = length
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[bytes]:
        remaining = self.length
        with self.path.open("rb") as source:
            source.seek(self.offset)
            while remaining:
                block = source.read(min(self.chunk_size, remaining))
                if not block:
                    raise OSError(f"file ended before signed part length: {self.path}")
                remaining -= len(block)
                yield block


def _content_type(path: Path) -> str:
    if path.suffix == ".parquet":
        return "application/vnd.apache.parquet"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_HASH_CHUNK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _retry_after_seconds(value: str | None) -> float:
    if value is None:
        raise TenonProtocolError("retryable completion_in_progress response is missing Retry-After")
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TenonProtocolError("completion Retry-After is invalid") from exc
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - datetime.now(UTC)).total_seconds()
    if not math.isfinite(seconds) or seconds < 0:
        raise TenonProtocolError("completion Retry-After must be finite and non-negative")
    return max(seconds, _MIN_COMPLETION_RETRY_DELAY_SECONDS)


def scan_dataset_directory(root: str | Path) -> list[LocalDatasetFile]:
    """Build an immutable upload manifest from regular files below ``root``.

    Symlinks, empty files, device files, and path aliases are rejected before a
    server resource is created.
    """

    root_path = Path(root).expanduser().resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError(f"dataset root is not a directory: {root_path}")
    files: list[LocalDatasetFile] = []
    for directory, directory_names, file_names in os.walk(root_path, followlinks=False):
        directory_path = Path(directory)
        for name in list(directory_names):
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ValueError(f"symlink directory is forbidden: {candidate}")
        for name in file_names:
            path = directory_path / name
            if path.is_symlink():
                raise ValueError(f"symlink file is forbidden: {path}")
            if not path.is_file():
                raise ValueError(f"non-regular dataset entry is forbidden: {path}")
            relative = path.relative_to(root_path).as_posix()
            normalized = str(PurePosixPath(relative))
            if not relative or relative.startswith("/") or ".." in PurePosixPath(relative).parts:
                raise ValueError(f"unsafe relative path: {relative}")
            if normalized != relative:
                raise ValueError(f"non-canonical relative path: {relative}")
            size = path.stat().st_size
            if size < 1:
                raise ValueError(f"empty dataset files are forbidden by v0.1: {relative}")
            files.append(
                LocalDatasetFile(
                    path=path,
                    manifest=FileManifest(
                        relative_path=relative,
                        size_bytes=size,
                        sha256=_sha256(path),
                        content_type=_content_type(path),
                    ),
                )
            )
    files.sort(key=lambda item: item.manifest.relative_path)
    if not files:
        raise ValueError("dataset directory contains no files")
    return files


class DatasetUploader:
    """Upload declared files through short-lived direct PUT URLs."""

    def __init__(
        self,
        client: TenonClient,
        *,
        part_size: int = DEFAULT_PART_SIZE,
        direct_client: httpx.Client | None = None,
        direct_timeout: float | httpx.Timeout = 120.0,
        completion_max_attempts: int = DEFAULT_COMPLETION_MAX_ATTEMPTS,
        completion_max_wait_seconds: float = DEFAULT_COMPLETION_MAX_WAIT_SECONDS,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        completion_observer: Callable[[FileCompletionContinuation], None] | None = None,
    ) -> None:
        if part_size < 1:
            raise ValueError("part_size must be positive")
        if completion_max_attempts < 1:
            raise ValueError("completion_max_attempts must be positive")
        if not math.isfinite(completion_max_wait_seconds) or completion_max_wait_seconds <= 0:
            raise ValueError("completion_max_wait_seconds must be finite and positive")
        self.client = client
        self.part_size = part_size
        self.completion_max_attempts = completion_max_attempts
        self.completion_max_wait_seconds = completion_max_wait_seconds
        self._sleeper = sleeper
        self._clock = clock
        self._completion_observer = completion_observer
        self._owns_direct_client = direct_client is None
        self._direct_client = direct_client or httpx.Client(
            timeout=direct_timeout,
            follow_redirects=False,
        )

    def close(self) -> None:
        if self._owns_direct_client:
            self._direct_client.close()

    def __enter__(self) -> DatasetUploader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def upload(
        self,
        upload: Upload,
        local_files: Sequence[LocalDatasetFile],
        *,
        completion_continuations: Mapping[str, FileCompletionContinuation] | None = None,
        complete_idempotency_key: str | None = None,
    ) -> Upload:
        local_by_path = {item.manifest.relative_path: item for item in local_files}
        remote_by_path = {item.relative_path: item for item in upload.files}
        remote_by_id = {item.id: item for item in upload.files}
        if set(local_by_path) != set(remote_by_path):
            missing = sorted(set(remote_by_path) - set(local_by_path))
            undeclared = sorted(set(local_by_path) - set(remote_by_path))
            raise ValueError(
                f"local/upload manifest mismatch; missing={missing}, undeclared={undeclared}"
            )
        for relative_path, remote in remote_by_path.items():
            local = local_by_path[relative_path]
            if (
                remote.size_bytes != local.manifest.size_bytes
                or remote.sha256 != local.manifest.sha256
            ):
                raise ValueError(f"server manifest doesn't match local file: {relative_path}")
        continuations = dict(completion_continuations or {})
        unknown_continuations = set(continuations) - set(remote_by_id)
        if unknown_continuations:
            raise ValueError(
                f"completion continuation references unknown files: {sorted(unknown_continuations)}"
            )
        if upload.state in {"completing", "verified"}:
            return upload
        if upload.state != "open":
            raise UploadContinuationRequired(
                f"upload {upload.id} is terminal ({upload.state}); completion cannot continue"
            )
        for relative_path in sorted(remote_by_path):
            remote = remote_by_path[relative_path]
            local = local_by_path[relative_path]
            if remote.state == "complete":
                continue
            continuation = continuations.get(remote.id)
            if continuation is not None:
                if continuation.upload_id != upload.id or continuation.file_id != remote.id:
                    raise ValueError("completion continuation upload/file identity mismatch")
                self.continue_file_completion(continuation)
                continue
            if remote.state == "uploading":
                raise UploadContinuationRequired(
                    f"file {remote.id} is already uploading; automatic re-sign/PUT is unsafe. "
                    "Resume with the retained FileCompletionContinuation or inspect upload state."
                )
            if remote.state != "pending":
                raise UploadContinuationRequired(
                    f"file {remote.id} is terminal ({remote.state}); completion cannot continue"
                )
            self._upload_file(upload.id, remote, local.path)
        return self.client.complete_upload(
            upload.id,
            idempotency_key=complete_idempotency_key or new_idempotency_key("complete-upload"),
        )

    def continue_file_completion(
        self,
        continuation: FileCompletionContinuation,
    ) -> Upload:
        """Continue one accepted completion without signing or uploading bytes again."""

        return self._complete_file_with_continuation(continuation)

    @staticmethod
    def _is_completion_pending(error: TenonAPIError, idempotency_key: str) -> bool:
        return (
            error.status_code == 409
            and error.code == "completion_in_progress"
            and error.retryable is True
            and error.idempotency_key == idempotency_key
        )

    def _complete_file_with_continuation(
        self,
        continuation: FileCompletionContinuation,
    ) -> Upload:
        request = continuation.request
        # Persist the accepted part receipts before a completion can be sent.
        # Callback failures propagate: a missing durable receipt is not success.
        if self._completion_observer is not None:
            self._completion_observer(continuation)
        started = self._clock()
        for attempt in range(1, self.completion_max_attempts + 1):
            try:
                return self.client.complete_upload_file(
                    continuation.upload_id,
                    continuation.file_id,
                    request,
                    idempotency_key=continuation.idempotency_key,
                )
            except TenonAPIError as error:
                if not self._is_completion_pending(error, continuation.idempotency_key):
                    raise
                elapsed = self._clock() - started
                delay = _retry_after_seconds(error.retry_after)
                remaining = self.completion_max_wait_seconds - elapsed
                if attempt >= self.completion_max_attempts or remaining <= 0 or delay > remaining:
                    raise UploadCompletionPending(
                        continuation=continuation,
                        last_error=error,
                        attempts=attempt,
                        elapsed_seconds=max(0.0, elapsed),
                    ) from error
                # Caller interruption propagates, but keep the accepted operation
                # resumable without reconstructing its body or key.
                try:
                    self._sleeper(delay)
                except KeyboardInterrupt as interrupted:
                    _attach_completion_continuation(interrupted, continuation)
                    raise
            except TenonTransportError as error:
                _attach_completion_continuation(error, continuation)
                raise
            except KeyboardInterrupt as interrupted:
                _attach_completion_continuation(interrupted, continuation)
                raise
        raise AssertionError("completion attempt loop exhausted without result")  # pragma: no cover

    def _upload_file(self, upload_id: str, remote: UploadFile, path: Path) -> Upload:
        part_size = remote.size_bytes if remote.mode == "single" else self.part_size
        parts: list[PartSignItem] = []
        offset = 0
        part_number = 1
        while offset < remote.size_bytes:
            length = min(part_size, remote.size_bytes - offset)
            parts.append(
                PartSignItem(
                    file_id=remote.id,
                    part_number=part_number,
                    content_length=length,
                )
            )
            offset += length
            part_number += 1
        if len(parts) > 10_000:
            raise ValueError(f"file requires more than 10,000 parts: {remote.relative_path}")

        completed: list[CompletedPart] = []
        offset_by_part = {part.part_number: (part.part_number - 1) * part_size for part in parts}
        for batch_start in range(0, len(parts), 1000):
            batch = parts[batch_start : batch_start + 1000]
            signed = self.client.sign_upload_parts(
                upload_id,
                UploadPartSignRequest(parts=batch),
                idempotency_key=new_idempotency_key(f"sign-{remote.id}"),
            )
            expected_numbers = {part.part_number for part in batch}
            actual_numbers = {part.part_number for part in signed.parts}
            if expected_numbers != actual_numbers or any(
                part.file_id != remote.id for part in signed.parts
            ):
                raise TenonProtocolError("signed part response doesn't match requested file/parts")
            signed_by_number = {part.part_number: part for part in signed.parts}
            for requested in batch:
                part = signed_by_number[requested.part_number]
                headers = dict(part.required_headers)
                headers.setdefault("Content-Length", str(requested.content_length))
                stream = FileSlice(
                    path,
                    offset_by_part[requested.part_number],
                    requested.content_length,
                )
                try:
                    response = self._direct_client.put(
                        str(part.url),
                        headers=headers,
                        content=stream,
                    )
                except httpx.RequestError as exc:
                    raise TenonTransportError(
                        f"direct upload failed for {remote.relative_path} "
                        f"part {requested.part_number}: {exc}"
                    ) from exc
                if not 200 <= response.status_code < 300:
                    raise TenonTransportError(
                        f"direct upload returned HTTP {response.status_code} for "
                        f"{remote.relative_path} part {requested.part_number}"
                    )
                etag = response.headers.get("ETag")
                if not etag:
                    raise TenonProtocolError("direct upload response is missing ETag")
                completed.append(CompletedPart(part_number=requested.part_number, etag=etag))
        completed.sort(key=lambda item: item.part_number)
        request = FileCompleteRequest(parts=completed)
        continuation = FileCompletionContinuation.create(
            upload_id,
            remote.id,
            request,
            new_idempotency_key(f"complete-{remote.id}"),
        )
        return self._complete_file_with_continuation(continuation)
