from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from tenon_pi.client import TenonClient
from tenon_pi.errors import TenonAPIError, TenonProtocolError, TenonTransportError
from tenon_pi.models import (
    SignedPart,
    Upload,
    UploadPartSignResponse,
)
from tenon_pi.transport import RestTransport
from tenon_pi.uploads import (
    DatasetUploader,
    FileCompletionContinuation,
    UploadCompletionPending,
    UploadContinuationRequired,
    completion_continuation_from_error,
    scan_dataset_directory,
)


class FakeLifecycleClient:
    def __init__(self, upload_payload):
        self.upload = Upload.model_validate(upload_payload)
        self.sign_requests = []
        self.completed_files = []
        self.completed_uploads = []

    def sign_upload_parts(self, upload_id, request, *, idempotency_key=None):
        assert upload_id == self.upload.id
        self.sign_requests.append((request, idempotency_key))
        return UploadPartSignResponse(
            parts=[
                SignedPart(
                    file_id=part.file_id,
                    part_number=part.part_number,
                    method="PUT",
                    url=f"https://upload.fixture.invalid/{part.file_id}/{part.part_number}",
                    required_headers={"x-checksum-mode": "fixture"},
                    expires_at="2026-10-14T13:00:00Z",
                )
                for part in request.parts
            ]
        )

    def complete_upload_file(self, upload_id, file_id, request, *, idempotency_key=None):
        self.completed_files.append((upload_id, file_id, request, idempotency_key))
        return self.upload

    def complete_upload(self, upload_id, *, idempotency_key=None):
        self.completed_uploads.append((upload_id, idempotency_key))
        payload = self.upload.model_dump(mode="json")
        payload["state"] = "completing"
        return Upload.model_validate(payload)


def test_scan_and_direct_single_upload(tmp_path: Path, contract_fixtures):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_bytes(b"hello world")
    local_files = scan_dataset_directory(tmp_path)
    assert [item.manifest.relative_path for item in local_files] == ["meta/info.json"]
    assert local_files[0].manifest.sha256 == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )

    seen = []

    def direct_handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        seen.append((request, body))
        return httpx.Response(200, headers={"ETag": '"etag-fixture"'})

    direct = httpx.Client(transport=httpx.MockTransport(direct_handler))
    lifecycle = FakeLifecycleClient(contract_fixtures["upload"])
    uploader = DatasetUploader(lifecycle, direct_client=direct)  # type: ignore[arg-type]
    result = uploader.upload(lifecycle.upload, local_files)

    assert result.state == "completing"
    assert len(seen) == 1
    request, body = seen[0]
    assert body == b"hello world"
    assert request.headers["x-checksum-mode"] == "fixture"
    assert request.headers["content-length"] == "11"
    assert "authorization" not in request.headers
    completed = lifecycle.completed_files[0][2]
    assert completed.parts[0].etag == '"etag-fixture"'
    assert lifecycle.completed_uploads[0][0] == "upl_fixture"


def test_multipart_upload_preserves_order(tmp_path: Path, contract_fixtures):
    payload = deepcopy(contract_fixtures["upload"])
    payload["files"][0]["mode"] = "multipart"
    path = tmp_path / "meta"
    path.mkdir()
    (path / "info.json").write_bytes(b"hello world")
    local = scan_dataset_directory(tmp_path)
    lifecycle = FakeLifecycleClient(payload)
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        part = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, headers={"ETag": f'"etag-{part}"'})

    direct = httpx.Client(transport=httpx.MockTransport(handler))
    DatasetUploader(lifecycle, part_size=4, direct_client=direct).upload(lifecycle.upload, local)  # type: ignore[arg-type]
    assert bodies == [b"hell", b"o wo", b"rld"]
    complete = lifecycle.completed_files[0][2]
    assert [part.part_number for part in complete.parts] == [1, 2, 3]


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class CompletionLifecycleClient(FakeLifecycleClient):
    def __init__(self, upload_payload, outcomes):
        super().__init__(upload_payload)
        self.outcomes = list(outcomes)

    def complete_upload_file(self, upload_id, file_id, request, *, idempotency_key=None):
        self.completed_files.append((upload_id, file_id, request, idempotency_key))
        outcome = self.outcomes.pop(0)
        if outcome == "success":
            return self.upload
        code, retryable, retry_after, status = outcome
        raise TenonAPIError(
            status_code=status,
            code=code,
            message="fixture completion response",
            request_id=f"req_{len(self.completed_files)}",
            retryable=retryable,
            retry_after=retry_after,
            idempotency_key=idempotency_key,
        )


def _one_local_file(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_bytes(b"hello world")
    return scan_dataset_directory(tmp_path)


def _direct_client(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read())
        return httpx.Response(200, headers={"ETag": '"etag-fixture"'})

    return httpx.Client(transport=httpx.MockTransport(handler))


def _pending(retry_after="1"):
    return ("completion_in_progress", True, retry_after, 409)


def test_retryable_completion_continues_same_body_and_key_without_second_put(
    tmp_path: Path, contract_fixtures
):
    local = _one_local_file(tmp_path)
    lifecycle = CompletionLifecycleClient(
        contract_fixtures["upload"], [_pending(), _pending(), "success"]
    )
    direct_bodies = []
    clock = FakeClock()
    result = DatasetUploader(
        lifecycle,
        direct_client=_direct_client(direct_bodies),
        sleeper=clock.sleep,
        clock=clock,
    ).upload(lifecycle.upload, local)

    assert result.state == "completing"
    assert direct_bodies == [b"hello world"]
    assert len(lifecycle.sign_requests) == 1
    assert len(lifecycle.completed_files) == 3
    keys = {call[3] for call in lifecycle.completed_files}
    bodies = {call[2].model_dump_json() for call in lifecycle.completed_files}
    assert len(keys) == len(bodies) == 1
    assert clock.sleeps == [1.0, 1.0]
    assert len(lifecycle.completed_uploads) == 1


def test_rest_transport_continuation_reuses_exact_http_body_and_key(
    tmp_path: Path, contract_fixtures
):
    local = _one_local_file(tmp_path)
    upload = deepcopy(contract_fixtures["upload"])
    completion_requests = []
    counts = {"sign": 0, "put": 0, "whole": 0}

    def api_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/parts/sign"):
            counts["sign"] += 1
            body = json.loads(request.content)
            return httpx.Response(
                200,
                headers={"X-Request-Id": "req_sign"},
                json={
                    "parts": [
                        {
                            "file_id": body["parts"][0]["file_id"],
                            "part_number": 1,
                            "method": "PUT",
                            "url": "https://upload.fixture.invalid/fil_fixture/1",
                            "required_headers": {},
                            "expires_at": "2026-10-14T13:00:00Z",
                        }
                    ]
                },
            )
        if "/files/" in path and path.endswith("/complete"):
            completion_requests.append((request.content, request.headers["Idempotency-Key"]))
            if len(completion_requests) == 1:
                return httpx.Response(
                    409,
                    headers={"X-Request-Id": "req_pending", "Retry-After": "1"},
                    json={
                        "error": {
                            "code": "completion_in_progress",
                            "message": "still running",
                            "request_id": "req_pending",
                            "retryable": True,
                        }
                    },
                )
            completed = deepcopy(upload)
            completed["files"][0]["state"] = "complete"
            return httpx.Response(
                200,
                headers={"X-Request-Id": "req_complete"},
                json=completed,
            )
        if path.endswith("/complete"):
            counts["whole"] += 1
            completing = deepcopy(upload)
            completing["state"] = "completing"
            completing["files"][0]["state"] = "complete"
            return httpx.Response(
                202,
                headers={"X-Request-Id": "req_whole"},
                json=completing,
            )
        raise AssertionError(path)

    def direct_handler(request: httpx.Request) -> httpx.Response:
        counts["put"] += 1
        assert request.read() == b"hello world"
        return httpx.Response(200, headers={"ETag": '"etag-fixture"'})

    http = httpx.Client(
        base_url="https://api.fixture.invalid/v1/",
        transport=httpx.MockTransport(api_handler),
    )
    client = TenonClient(
        "fixture-key",
        transport=RestTransport(
            "fixture-key",
            base_url="https://api.fixture.invalid/v1",
            client=http,
        ),
    )
    clock = FakeClock()
    result = DatasetUploader(
        client,
        direct_client=httpx.Client(transport=httpx.MockTransport(direct_handler)),
        sleeper=clock.sleep,
        clock=clock,
    ).upload(Upload.model_validate(upload), local)

    assert result.state == "completing"
    assert counts == {"sign": 1, "put": 1, "whole": 1}
    assert len(completion_requests) == 2
    assert completion_requests[0] == completion_requests[1]
    assert clock.sleeps == [1.0]


def test_transport_loss_preserves_context_for_public_get_resume_without_reput(
    tmp_path: Path, contract_fixtures
):
    local = _one_local_file(tmp_path)
    upload = deepcopy(contract_fixtures["upload"])
    completion_requests = []
    counts = {"sign": 0, "put": 0, "whole": 0}

    def api_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/parts/sign"):
            counts["sign"] += 1
            body = json.loads(request.content)
            return httpx.Response(
                200,
                headers={"X-Request-Id": "req_sign"},
                json={
                    "parts": [
                        {
                            "file_id": body["parts"][0]["file_id"],
                            "part_number": 1,
                            "method": "PUT",
                            "url": "https://upload.fixture.invalid/fil_fixture/1",
                            "required_headers": {},
                            "expires_at": "2026-10-14T13:00:00Z",
                        }
                    ]
                },
            )
        if "/files/" in path and path.endswith("/complete"):
            completion_requests.append((request.content, request.headers["Idempotency-Key"]))
            if len(completion_requests) == 1:
                raise httpx.ReadTimeout("accepted response was lost", request=request)
            completed = deepcopy(upload)
            completed["files"][0]["state"] = "complete"
            return httpx.Response(
                200,
                headers={"X-Request-Id": "req_complete"},
                json=completed,
            )
        if path.endswith("/complete"):
            counts["whole"] += 1
            completing = deepcopy(upload)
            completing["state"] = "completing"
            completing["files"][0]["state"] = "complete"
            return httpx.Response(
                202,
                headers={"X-Request-Id": "req_whole"},
                json=completing,
            )
        raise AssertionError(path)

    def direct_handler(request: httpx.Request) -> httpx.Response:
        counts["put"] += 1
        assert request.read() == b"hello world"
        return httpx.Response(200, headers={"ETag": '"etag-fixture"'})

    http = httpx.Client(
        base_url="https://api.fixture.invalid/v1/",
        transport=httpx.MockTransport(api_handler),
    )
    client = TenonClient(
        "fixture-key",
        transport=RestTransport(
            "fixture-key",
            base_url="https://api.fixture.invalid/v1",
            client=http,
        ),
    )
    uploader = DatasetUploader(
        client,
        direct_client=httpx.Client(transport=httpx.MockTransport(direct_handler)),
    )
    with pytest.raises(TenonTransportError) as lost:
        uploader.upload(Upload.model_validate(upload), local)
    continuation = completion_continuation_from_error(lost.value)
    assert continuation is not None
    assert continuation.idempotency_key == lost.value.idempotency_key

    fresh = deepcopy(upload)
    fresh["files"][0]["state"] = "uploading"
    result = uploader.upload(
        Upload.model_validate(fresh),
        local,
        completion_continuations={continuation.file_id: continuation},
    )
    assert result.state == "completing"
    assert counts == {"sign": 1, "put": 1, "whole": 1}
    assert len(completion_requests) == 2
    assert completion_requests[0] == completion_requests[1]


def test_bounded_exhaustion_preserves_exact_continuation_and_safe_resume(
    tmp_path: Path, contract_fixtures
):
    local = _one_local_file(tmp_path)
    lifecycle = CompletionLifecycleClient(contract_fixtures["upload"], [_pending(), _pending()])
    direct_bodies = []
    clock = FakeClock()
    with pytest.raises(UploadCompletionPending) as raised:
        DatasetUploader(
            lifecycle,
            direct_client=_direct_client(direct_bodies),
            completion_max_attempts=2,
            sleeper=clock.sleep,
            clock=clock,
        ).upload(lifecycle.upload, local)
    pending = raised.value
    assert pending.attempts == 2
    assert pending.elapsed_seconds == 1.0
    assert pending.last_error.code == "completion_in_progress"
    assert pending.last_error.retryable
    assert pending.last_error.idempotency_key == pending.continuation.idempotency_key
    assert (
        FileCompletionContinuation.from_json_value(pending.continuation.json_value())
        == pending.continuation
    )
    assert direct_bodies == [b"hello world"]
    assert len(lifecycle.sign_requests) == 1

    resumed_payload = deepcopy(contract_fixtures["upload"])
    resumed_payload["files"][0]["state"] = "uploading"
    resumed = CompletionLifecycleClient(resumed_payload, ["success"])
    result = DatasetUploader(
        resumed,
        direct_client=_direct_client([]),
    ).upload(
        resumed.upload,
        local,
        completion_continuations={
            pending.continuation.file_id: pending.continuation,
        },
    )
    assert result.state == "completing"
    assert resumed.sign_requests == []
    assert len(resumed.completed_files) == 1
    resumed_call = resumed.completed_files[0]
    assert resumed_call[2].model_dump_json() == pending.continuation.request_json
    assert resumed_call[3] == pending.continuation.idempotency_key


def test_uploading_file_without_continuation_never_resigns(tmp_path: Path, contract_fixtures):
    local = _one_local_file(tmp_path)
    payload = deepcopy(contract_fixtures["upload"])
    payload["files"][0]["state"] = "uploading"
    lifecycle = CompletionLifecycleClient(payload, ["success"])
    with pytest.raises(UploadContinuationRequired, match="re-sign/PUT is unsafe"):
        DatasetUploader(lifecycle, direct_client=_direct_client([])).upload(lifecycle.upload, local)
    assert lifecycle.sign_requests == []
    assert lifecycle.completed_files == []


@pytest.mark.parametrize(
    "outcome",
    [
        ("invalid_state", False, None, 409),
        ("idempotency_conflict", False, None, 409),
        ("upload_expired", False, None, 409),
        ("other_retryable", True, "1", 503),
    ],
)
def test_noneligible_completion_errors_are_immediate(tmp_path: Path, contract_fixtures, outcome):
    local = _one_local_file(tmp_path)
    lifecycle = CompletionLifecycleClient(contract_fixtures["upload"], [outcome])
    clock = FakeClock()
    with pytest.raises(TenonAPIError) as raised:
        DatasetUploader(
            lifecycle,
            direct_client=_direct_client([]),
            sleeper=clock.sleep,
            clock=clock,
        ).upload(lifecycle.upload, local)
    assert raised.value.code == outcome[0]
    assert len(lifecycle.completed_files) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("retry_after", [None, "invalid", "-1", "nan"])
def test_pending_completion_requires_valid_retry_after(
    tmp_path: Path, contract_fixtures, retry_after
):
    local = _one_local_file(tmp_path)
    lifecycle = CompletionLifecycleClient(contract_fixtures["upload"], [_pending(retry_after)])
    with pytest.raises(TenonProtocolError, match="Retry-After"):
        DatasetUploader(lifecycle, direct_client=_direct_client([])).upload(lifecycle.upload, local)
    assert len(lifecycle.completed_files) == 1


def test_zero_retry_after_uses_nonzero_delay_floor(tmp_path: Path, contract_fixtures):
    local = _one_local_file(tmp_path)
    lifecycle = CompletionLifecycleClient(contract_fixtures["upload"], [_pending("0"), "success"])
    clock = FakeClock()
    DatasetUploader(
        lifecycle,
        direct_client=_direct_client([]),
        sleeper=clock.sleep,
        clock=clock,
    ).upload(lifecycle.upload, local)
    assert clock.sleeps == [0.05]


def test_completion_wait_is_bounded_without_hot_loop(tmp_path: Path, contract_fixtures):
    local = _one_local_file(tmp_path)
    lifecycle = CompletionLifecycleClient(contract_fixtures["upload"], [_pending("10")])
    clock = FakeClock()
    with pytest.raises(UploadCompletionPending) as raised:
        DatasetUploader(
            lifecycle,
            direct_client=_direct_client([]),
            completion_max_wait_seconds=1,
            sleeper=clock.sleep,
            clock=clock,
        ).upload(lifecycle.upload, local)
    assert raised.value.attempts == 1
    assert clock.sleeps == []


def test_interrupt_during_completion_http_call_retains_context(tmp_path: Path, contract_fixtures):
    local = _one_local_file(tmp_path)

    class InterruptingClient(FakeLifecycleClient):
        def complete_upload_file(self, upload_id, file_id, request, *, idempotency_key=None):
            self.completed_files.append((upload_id, file_id, request, idempotency_key))
            raise KeyboardInterrupt

    lifecycle = InterruptingClient(contract_fixtures["upload"])
    with pytest.raises(KeyboardInterrupt) as raised:
        DatasetUploader(lifecycle, direct_client=_direct_client([])).upload(lifecycle.upload, local)
    continuation = completion_continuation_from_error(raised.value)
    assert continuation is not None
    assert continuation.idempotency_key == lifecycle.completed_files[0][3]
    assert (
        continuation.request.model_dump_json() == lifecycle.completed_files[0][2].model_dump_json()
    )
    assert len(lifecycle.completed_files) == 1


def test_caller_interrupt_is_not_swallowed_or_retried(tmp_path: Path, contract_fixtures):
    local = _one_local_file(tmp_path)
    lifecycle = CompletionLifecycleClient(contract_fixtures["upload"], [_pending()])

    def interrupt(_seconds):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt) as raised:
        DatasetUploader(
            lifecycle,
            direct_client=_direct_client([]),
            sleeper=interrupt,
        ).upload(lifecycle.upload, local)
    assert isinstance(
        completion_continuation_from_error(raised.value),
        FileCompletionContinuation,
    )
    assert len(lifecycle.completed_files) == 1


@pytest.mark.parametrize("state", ["completing", "verified"])
def test_public_upload_resume_returns_already_advancing_state(
    tmp_path: Path, contract_fixtures, state
):
    local = _one_local_file(tmp_path)
    payload = deepcopy(contract_fixtures["upload"])
    payload["state"] = state
    lifecycle = CompletionLifecycleClient(payload, [])
    assert (
        DatasetUploader(lifecycle, direct_client=_direct_client([]))
        .upload(lifecycle.upload, local)
        .state
        == state
    )
    assert lifecycle.sign_requests == [] and lifecycle.completed_files == []


@pytest.mark.parametrize("state", ["expired", "aborted", "failed"])
def test_terminal_upload_state_never_restarts_provider_work(
    tmp_path: Path, contract_fixtures, state
):
    local = _one_local_file(tmp_path)
    payload = deepcopy(contract_fixtures["upload"])
    payload["state"] = state
    lifecycle = CompletionLifecycleClient(payload, [])
    with pytest.raises(UploadContinuationRequired, match="terminal"):
        DatasetUploader(lifecycle, direct_client=_direct_client([])).upload(lifecycle.upload, local)
    assert lifecycle.sign_requests == [] and lifecycle.completed_files == []


def test_scan_rejects_symlink(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_text("{}")
    (tmp_path / "link.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        scan_dataset_directory(tmp_path)
