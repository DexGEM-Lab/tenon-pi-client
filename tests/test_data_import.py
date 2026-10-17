from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from tenon_pi.cli import app
from tenon_pi.data_fixture import example_profile, write_data_fixture
from tenon_pi.data_import import import_jsonl, inspect_data
from tenon_pi.preparation import PreparationError, prepare_dataset


def raw_source(tmp_path):
    paths = write_data_fixture(tmp_path / "fixture")
    root = tmp_path / "raw"
    root.mkdir()
    rows = []
    for path in sorted((Path(paths["dataset"]) / "data").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            name = f"{row['index']}.png"
            (root / name).write_bytes(row["observation.images.view_0"]["bytes"])
            rows.append(
                dict(
                    episode_index=row["episode_index"],
                    frame_index=row["frame_index"],
                    timestamp=row["timestamp"],
                    state=row["observation.state"],
                    action=row["action"],
                    task="place block",
                    images={"view_0": name},
                )
            )
    source = root / "rows.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return source, rows, paths


def test_import_preserves_raw_values_and_encoded_pixels(tmp_path):
    source, rows, _ = raw_source(tmp_path)
    profile = example_profile()
    root = import_jsonl(source, tmp_path / "imported", profile=profile, input_space="native")
    actual = [
        row
        for path in sorted((root / "data").rglob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    ]
    assert len(actual) == len(rows)
    for src, dst in zip(rows, actual, strict=True):
        assert dst["observation.state"] == src["state"]
        assert dst["action"] == src["action"]
        assert dst["timestamp"] == src["timestamp"]
        assert dst["episode_index"] == src["episode_index"]
        assert (
            dst["observation.images.view_0"]["bytes"]
            == (source.parent / src["images"]["view_0"]).read_bytes()
        )
    result = prepare_dataset(
        root,
        tmp_path / "prepared",
        input_space="native",
        dataset_repo_id="example/imported",
        profile=profile,
    )
    assert inspect_data(result.root)["verified"] is True
    assert inspect_data(root)["verified"] is False
    assert inspect_data(source.parent)["kind"] == "unrecognized"


@pytest.mark.parametrize("bad", ["precision", "order", "camera", "traversal", "time", "shape"])
def test_import_rejects_ambiguous_or_lossy_data(tmp_path, bad):
    source, rows, _ = raw_source(tmp_path)
    if bad == "precision":
        rows[0]["state"][0] = 0.1
    elif bad == "order":
        rows[1]["frame_index"] = 9
    elif bad == "camera":
        rows[0]["images"] = {}
    elif bad == "traversal":
        rows[0]["images"]["view_0"] = "../fixture/secret.png"
    elif bad == "time":
        rows[1]["timestamp"] += 0.001
    else:
        rows[0]["action"].append(0)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((PreparationError, ValueError)):
        import_jsonl(source, tmp_path / "out", profile=example_profile(), input_space="native")
    assert not (tmp_path / "out").exists()


def test_offline_cli_fixture_inspect_prepare_verify(tmp_path):
    runner = CliRunner()
    fixture = tmp_path / "fixture"
    out = runner.invoke(app, ["data", "fixture", "-o", str(fixture), "--layout", "gantry"])
    assert out.exit_code == 0, out.output
    paths = json.loads(out.output)
    out = runner.invoke(app, ["data", "inspect", paths["dataset"]])
    assert out.exit_code == 0 and json.loads(out.output)["verified"] is False
    prepared = tmp_path / "prepared"
    out = runner.invoke(
        app,
        [
            "data",
            "prepare",
            paths["dataset"],
            "--profile",
            paths["profile"],
            "--input-space",
            "native",
            "--repo-id",
            "example/gantry",
            "-o",
            str(prepared),
        ],
    )
    assert out.exit_code == 0, out.output
    out = runner.invoke(app, ["data", "verify", str(prepared)])
    assert out.exit_code == 0, out.output
    schema = json.loads((prepared / "robot-schema.json").read_text())
    assert len(schema["state_channels"]) == 5 and len(schema["action_channels"]) == 4
    assert np.isfinite([c["lower"] for c in schema["state_channels"]]).all()
