from __future__ import annotations

import json
from pathlib import Path

SPEC = json.loads((Path(__file__).parents[1] / "spec" / "openapi.json").read_text())


def test_openapi_snapshot_identity():
    assert SPEC["openapi"] == "3.1.0"
    assert SPEC["info"]["version"] == "0.1.0"
    assert SPEC["servers"] == [{"url": "https://tenon.dex-gem.ai/v1"}]
    assert "b644badcd8edc2a97170755a5bb10ead20879107" in SPEC["info"]["description"]


def test_narrow_csrf_bootstrap_addition_is_present():
    bootstrap = SPEC["components"]["schemas"]["SessionBootstrap"]
    assert bootstrap["required"] == [
        "id",
        "email",
        "display_name",
        "tenant_id",
        "status",
        "created_at",
        "resource_version",
        "csrf_token",
    ]
    assert "user" not in bootstrap["properties"]
    for path in ["/auth/activate", "/auth/login"]:
        schema_ref = SPEC["paths"][path]["post"]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]["$ref"]
        assert schema_ref.endswith("/User")
    assert (
        SPEC["paths"]["/me"]["get"]["responses"]["200"]["headers"]["Cache-Control"]["schema"][
            "const"
        ]
        == "no-store"
    )


def test_post_approval_wire_expansions_are_absent():
    schemas = SPEC["components"]["schemas"]
    assert "credit_account_id" not in schemas["CreditBalance"]["properties"]
    assert "model_capability_id" not in schemas["TrainingSpec"]["properties"]
    assert "max_cost_microcredits" in schemas["Quote"]["properties"]
    assert "max_budget_microcredits" not in schemas["Quote"]["properties"]
    assert "dataset_timing" not in schemas["DatasetVersionCreate"]["properties"]


def test_sdk_lifecycle_paths_are_in_approved_snapshot():
    paths = SPEC["paths"]
    required = {
        "/credits",
        "/capabilities",
        "/quotes",
        "/datasets",
        "/datasets/{dataset_id}/versions",
        "/uploads/{upload_id}/parts/sign",
        "/uploads/{upload_id}/files/{file_id}/complete",
        "/uploads/{upload_id}/complete",
        "/training/jobs",
        "/training/jobs/{job_id}/cancel",
        "/training/jobs/{job_id}/events",
        "/checkpoints/{checkpoint_id}",
        "/deployments",
        "/control/sessions",
        "/control/sessions/{session_id}/connections",
        "/evaluations",
    }
    assert required <= set(paths)


def test_dataset_upload_format_has_only_approved_v2_v3_extension():
    schemas = SPEC["components"]["schemas"]
    assert schemas["DatasetVersionCreate"]["properties"]["format"]["enum"] == [
        "lerobot_v2",
        "lerobot_v3",
    ]
    assert schemas["DatasetVersion"]["properties"]["format"]["enum"] == [
        "lerobot_v2",
        "lerobot_v3",
    ]
    assert "dataset_timing" not in schemas["DatasetVersionCreate"]["properties"]


def test_connection_url_is_explicit_and_trained_rtc_create_boundary_is_preserved():
    schemas = SPEC["components"]["schemas"]
    wss = schemas["ControlConnection"]["properties"]["wss_url"]
    assert "not derived from REST origin" in wss["description"]
    create_modes = schemas["ControlSessionCreate"]["properties"]["execution_mode"]["enum"]
    quote_modes = schemas["ControlSessionSpec"]["properties"]["execution_mode"]["enum"]
    assert "trained_rtc" not in create_modes
    assert "trained_rtc" in quote_modes


def test_training_method_and_named_lora_profile_are_fixed():
    schema = SPEC["components"]["schemas"]["TrainingSpec"]
    assert schema["properties"]["method"]["enum"] == ["full_ft", "lora"]
    assert schema["properties"]["lora_profile"]["enum"] == [
        "openpi-pi05-action-lora-v1",
        None,
    ]
    assert "rank" not in schema["properties"]
    assert "alpha" not in schema["properties"]
