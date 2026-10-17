from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dataset_v3_fixture import LENGTHS
from typer.testing import CliRunner

from tenon_pi import cli
from tenon_pi.normalization import RunningStats
from tenon_pi.preparation import (
    PreparationError,
    build_bundle_manifest,
    build_transform_descriptor,
    canonical_json_digest,
    compute_normalization,
    dataset_file_manifest,
    file_sha256,
    iter_horizon_samples,
    prepare_dataset,
    verify_preparation,
)


def prepare(source, tmp_path):
    root, profile = source
    return prepare_dataset(
        root,
        tmp_path / "prepared",
        input_space="canonical",
        dataset_repo_id="owner/original-recording",
        profile=profile,
    )


def test_v3_whole_bundle_preserves_original_bytes_rows_timing_and_h4(dataset_v3_source, tmp_path):
    source, profile = dataset_v3_source
    before = dataset_file_manifest(source)
    result = prepare(dataset_v3_source, tmp_path)
    assert result.format == "lerobot_v3"
    assert result.dataset_repo_id == "owner/original-recording"
    assert result == verify_preparation(result.root)
    assert dataset_file_manifest(result.dataset) == before == dataset_file_manifest(source)
    assert not list(result.root.rglob("*view-manifest.json"))
    descriptor = build_transform_descriptor(profile)
    samples = list(iter_horizon_samples(result.dataset, descriptor))
    tables = [pq.read_table(p) for p in sorted((source / "data").rglob("*.parquet"))]
    table = pa.concat_tables(tables)
    states = np.asarray(table["observation.state"].to_pylist(), np.float32)
    actions = np.asarray(table["action"].to_pylist(), np.float32)
    timestamps = table["timestamp"].to_numpy()
    cursor = 0
    for episode, length in enumerate(LENGTHS):
        for frame in range(length):
            i = cursor + frame
            sample = samples[i]
            indices = np.minimum(np.arange(i, i + 4), cursor + length - 1)
            np.testing.assert_array_equal(sample["observation.state"], states[i])
            np.testing.assert_array_equal(sample["action"], actions[indices])
            assert sample["observation.state"].dtype == sample["action"].dtype == np.float32
            assert sample["episode_index"] == episode and sample["frame_index"] == frame
            assert sample["timestamp"] == float(timestamps[i])
        cursor += length
    # Full 4 consecutive rows in ep1, not a repeated single action.
    np.testing.assert_array_equal(samples[7]["action"], actions[7:11])
    assert len(np.unique(samples[7]["action"][:, 0])) > 1
    manifest = json.loads((result.assets / "normalization-manifest.json").read_text())
    assert manifest["dataset_format"] == "lerobot_v3"
    assert manifest["episode_tail_padding"] == "repeat_last_action"
    assert manifest["statistics_schedule"]["per_row_updates"] == [
        ["state", [1, 4]],
        ["actions", [4, 4]],
    ]
    assert manifest["state_count"] == sum(LENGTHS)
    assert manifest["action_count"] == sum(LENGTHS) * 4
    bundle = build_bundle_manifest(result.root)
    assert bundle["format"] == "lerobot_v3"
    paths = {item["relative_path"] for item in bundle["files"]}
    assert {"dataset/" + item["path"] for item in before} <= paths
    assert "assets/robot-profile-v1/norm_stats.json" in paths


def test_generated_video_really_has_off_keyframe_shared_episode(dataset_v3_source):
    source, _ = dataset_v3_source
    path = source / "videos/observation.images.view_0/chunk-000/file-001.mp4"
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=key_frame,pts_time",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    frames = json.loads(output)["frames"]
    assert len(frames) == sum(LENGTHS[1:])
    assert frames[0]["key_frame"] == 1
    assert frames[LENGTHS[1]]["key_frame"] == 0
    assert float(frames[LENGTHS[1]]["pts_time"]) == LENGTHS[1] / 20


def test_v3_statistics_schedule_and_dtypes_are_exact(dataset_v3_source, monkeypatch):
    root, profile = dataset_v3_source
    descriptor = build_transform_descriptor(profile)
    updates = []
    original = RunningStats.update

    def record(self, batch):
        updates.append(batch.copy())
        original(self, batch)

    monkeypatch.setattr(RunningStats, "update", record)
    encoded, rows = compute_normalization(root, descriptor)
    again, _ = compute_normalization(root, descriptor)
    assert encoded == again
    assert rows == sum(LENGTHS)
    samples = list(iter_horizon_samples(root, descriptor))
    for i, sample in enumerate(samples):
        state, actions = updates[2 * i : 2 * i + 2]
        assert state.shape == (4,) and actions.shape == (4, 4)
        assert state.dtype == actions.dtype == np.float64
        np.testing.assert_array_equal(state, sample["observation.state"].astype(np.float64))
        expected = sample["action"].copy()
        expected[:, list((0, 1))] -= sample["observation.state"][list((0, 1))]
        np.testing.assert_array_equal(actions, expected.astype(np.float64))
        np.testing.assert_array_equal(actions[:, [3]], sample["action"][:, [3]])
        assert np.all(actions[:, [2]] == 0)


def test_v3_prepare_cli_works_without_credentials(dataset_v3_source, tmp_path):
    source, profile = dataset_v3_source
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(json.dumps(profile))
    result = CliRunner().invoke(
        cli.app,
        [
            "data",
            "prepare",
            str(source),
            "-o",
            str(tmp_path / "prepared"),
            "--input-space",
            "canonical",
            "--repo-id",
            "owner/original-recording",
            "--profile",
            str(profile_file),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["format"] == "lerobot_v3"


@pytest.mark.parametrize(
    "feature",
    ["timestamp", "index", "episode_index", "frame_index", "task_index"],
)
def test_v3_scalar_feature_shape_is_required_during_prepare(dataset_v3_source, tmp_path, feature):
    source, profile = dataset_v3_source
    root = tmp_path / "source"
    shutil.copytree(source, root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"][feature]["shape"] = [2]
    info_path.write_text(json.dumps(info))
    with pytest.raises(PreparationError, match=r"shape \[1\]"):
        prepare((root, profile), tmp_path)
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize(
    "feature",
    ["timestamp", "index", "episode_index", "frame_index", "task_index"],
)
def test_v3_scalar_feature_shape_is_rechecked_during_verify(dataset_v3_source, tmp_path, feature):
    result = prepare(dataset_v3_source, tmp_path)
    info_path = result.dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"][feature]["shape"] = [2]
    info_path.write_text(json.dumps(info))
    with pytest.raises(PreparationError, match=r"shape \[1\]"):
        verify_preparation(result.root)


@pytest.mark.parametrize(
    "variant",
    [
        "version",
        "data_template",
        "video_template",
        "image",
        "native",
        "view",
        "missing_video",
        "extra_data",
        "episode_range",
        "data_owner",
        "video_offset",
        "video_duration",
        "video_reversed",
        "video_truncated",
        "video_overlap",
        "video_overlong",
        "row_owner",
        "row_task",
        "timestamp",
        "dummy",
        "gripper",
        "bounds",
        "vector_dtype",
        "fractional_index",
        "null_timestamp",
        "tasks",
        "symlink",
    ],
)
def test_unsupported_or_invalid_v3_is_rejected_before_output(dataset_v3_source, tmp_path, variant):
    source, profile = dataset_v3_source
    root = tmp_path / "source"
    shutil.copytree(source, root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pq.read_table(metadata_path).to_pylist()
    data_path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data_path)
    if variant == "version":
        info["codebase_version"] = "v3.1"
    elif variant in {"data_template", "video_template"}:
        info[variant.split("_")[0] + "_path"] = "other/{file_index}.parquet"
    elif variant == "image":
        info["features"]["observation.images.view_0"]["dtype"] = "image"
    elif variant == "native":
        info["features"]["action"]["shape"] = [12]
    elif variant == "view":
        info["tenon_view"] = {}
    elif variant == "missing_video":
        next((root / "videos").rglob("*.mp4")).unlink()
    elif variant == "extra_data":
        shutil.copy2(data_path, data_path.with_name("file-999.parquet"))
    elif variant == "episode_range":
        metadata[1]["dataset_from_index"] += 1
    elif variant == "data_owner":
        metadata[0]["data/file_index"] = 1
    elif variant == "video_offset":
        metadata[2]["videos/observation.images.view_0/from_timestamp"] += 0.05
        metadata[2]["videos/observation.images.view_0/to_timestamp"] += 0.05
    elif variant == "video_duration":
        metadata[0]["videos/observation.images.view_0/to_timestamp"] += 0.05
    elif variant in {"video_reversed", "video_truncated", "video_overlap", "video_overlong"}:
        prefix = "videos/observation.images.view_0/"
        if variant == "video_reversed":
            metadata[2][prefix + "to_timestamp"] = 0.0
        elif variant == "video_truncated":
            metadata[1][prefix + "to_timestamp"] -= 0.05
        elif variant == "video_overlap":
            metadata[2][prefix + "from_timestamp"] -= 0.05
            metadata[2][prefix + "to_timestamp"] -= 0.05
        else:
            metadata[2][prefix + "to_timestamp"] += 100.0
    elif variant == "tasks":
        pq.write_table(
            pa.table({"task_index": [0, 0], "task": ["a", "b"]}), root / "meta/tasks.parquet"
        )
    elif variant == "symlink":
        (root / "link").symlink_to(info_path)
    else:
        key = {
            "row_owner": "episode_index",
            "row_task": "task_index",
            "timestamp": "timestamp",
            "null_timestamp": "timestamp",
            "fractional_index": "index",
            "vector_dtype": "action",
        }.get(variant, "action")
        values = table[key].to_pylist()
        dtype = table.schema.field(key).type
        if variant == "vector_dtype":
            dtype = pa.list_(pa.float64(), 4)
        elif variant == "fractional_index":
            values[0], dtype = 0.5, pa.float64()
        elif variant == "null_timestamp":
            values[0] = None
        elif variant == "dummy":
            values[0][2] = 0.1
        elif variant == "gripper":
            values[0][3] = 1.1
        elif variant == "bounds":
            values[0][0] = 10.0
        else:
            values[0] = {"row_owner": 1, "row_task": 1, "timestamp": 0.01}[variant]
        table = table.set_column(
            table.schema.get_field_index(key), key, pa.array(values, type=dtype)
        )
    info_path.write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist(metadata), metadata_path)
    pq.write_table(table, data_path)
    with pytest.raises((PreparationError, ValueError)):
        prepare((root, profile), tmp_path)
    assert not (tmp_path / "prepared").exists()


def rehash_wrong_stats(result):
    """Forge all digest bindings consistently; verification must still regenerate."""
    norm_path = result.assets / "norm_stats.json"
    payload = json.loads(norm_path.read_text())
    payload["norm_stats"]["actions"]["q99"][0] += 0.1
    norm_path.write_text(json.dumps(payload, indent=2))
    norm_digest = file_sha256(norm_path)
    manifest_path = result.assets / "normalization-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["normalization_digest"] = norm_digest
    manifest_path.write_text(json.dumps(manifest))
    schema = json.loads(result.schema.read_text())
    schema["normalization_digest"] = norm_digest
    result.schema.write_text(json.dumps(schema))
    prep_path = result.root / "preparation-manifest.json"
    prep = json.loads(prep_path.read_text())
    prep["normalization_digest"] = norm_digest
    prep["robot_schema_digest"] = canonical_json_digest(schema)
    for name, path in (
        ("norm_stats.json", norm_path),
        ("normalization-manifest.json", manifest_path),
        ("robot-schema.json", result.schema),
    ):
        prep["artifacts"][name] = file_sha256(path)
    prep_path.write_text(json.dumps(prep))


def test_self_consistently_rehashed_wrong_assets_still_fail(dataset_v3_source, tmp_path):
    result = prepare(dataset_v3_source, tmp_path)
    rehash_wrong_stats(result)
    with pytest.raises(PreparationError, match="regenerated horizon"):
        verify_preparation(result.root)


@pytest.mark.parametrize("source_kind", ["generated", "upstream_writer"])
def test_optional_same_input_external_statistics_comparator(
    dataset_v3_source, tmp_path, source_kind
):
    """External comparator protocol: SOURCE DESCRIPTOR OUTPUT (norm_stats.json).

    The external executable independently reads these exact source rows, builds
    same-episode H4 windows and runs the pinned production transform/statistics.
    It must not import this client's transforms, sampler, RunningStats or serializer.
    No private package is a public client dependency.
    """
    comparator = os.environ.get("TENON_STATS_COMPARATOR")
    if comparator is None:
        pytest.skip("set TENON_STATS_COMPARATOR to a test-only external reference executable")
    source = dataset_v3_source
    if source_kind == "upstream_writer":
        writer_source = os.environ.get("TENON_V3_WRITER_SOURCE")
        if writer_source is None:
            pytest.skip("set TENON_V3_WRITER_SOURCE to an actual upstream-written v3 source")
        source = (Path(writer_source), *dataset_v3_source[1:])
    result = prepare(source, tmp_path)
    expected = tmp_path / "reference-norm_stats.json"
    subprocess.run(
        [comparator, str(source[0]), str(result.assets / "transform.json"), str(expected)],
        check=True,
    )
    actual = (result.assets / "norm_stats.json").read_bytes()
    assert expected.read_bytes() == actual
    reference = json.loads(expected.read_bytes())["norm_stats"]
    public = json.loads(actual)["norm_stats"]
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            np.testing.assert_array_equal(public[key][field], reference[key][field])
