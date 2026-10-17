from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jsonschema import Draft202012Validator

from tenon_pi.data_fixture import example_profile, write_data_fixture
from tenon_pi.dataset_metadata import write_v2_metadata_statistics
from tenon_pi.preparation import (
    PreparationError,
    dataset_file_manifest,
    iter_horizon_samples,
    prepare_dataset,
    verify_preparation,
)
from tenon_pi.robot import (
    CommandEncodingReceipt,
    CommandLimiter,
    ControllerFault,
    NativeCommandTrace,
    RobotController,
)
from tenon_pi.robot_profile import RobotProfile


@pytest.mark.parametrize("layout,dimensions", [("planar", (3, 4, 3, 4)), ("gantry", (5, 5, 4, 4))])
@pytest.mark.parametrize("revision", ["v2.0", "v2.1"])
def test_distinct_layouts_preserve_rows_pixels_and_compute_real_statistics(
    tmp_path, layout, dimensions, revision
):
    profile = example_profile(layout)
    assert (
        tuple(
            len(x)
            for x in (
                profile.state.native,
                profile.state.canonical,
                profile.action.native,
                profile.action.canonical,
            )
        )
        == dimensions
    )
    fixture = write_data_fixture(tmp_path / "source", layout=layout)
    source = Path(fixture["dataset"])
    info = json.loads((source / "meta/info.json").read_text())
    if revision == "v2.0":
        info["codebase_version"] = revision
        (source / "meta/info.json").write_text(json.dumps(info))
        write_v2_metadata_statistics(source, info)
        (source / "meta/episodes_stats.jsonl").unlink()
    before = dataset_file_manifest(source)
    result = prepare_dataset(
        source,
        tmp_path / "prepared",
        input_space="native",
        dataset_repo_id="example/calibrated",
        profile=profile,
    )
    assert verify_preparation(result.root) == result
    assert dataset_file_manifest(source) == before
    for src in sorted((source / "data").rglob("*.parquet")):
        a, b = pq.read_table(src), pq.read_table(result.dataset / src.relative_to(source))
        for key in a.column_names:
            if key in ("action", "observation.state"):
                mapping = profile.action if key == "action" else profile.state
                expected = np.stack([mapping.to_canonical(row) for row in a[key].to_pylist()])
                np.testing.assert_array_equal(b[key].to_pylist(), expected)
            else:
                assert a[key].equals(b[key]), key
    samples = list(iter_horizon_samples(result.dataset, profile.descriptor()))
    norms = json.loads((result.assets / "norm_stats.json").read_text())["norm_stats"]
    states = np.stack([s["observation.state"] for s in samples]).astype(np.float64)
    targets = np.concatenate(
        [profile.model_actions(s["observation.state"], s["action"]) for s in samples]
    ).astype(np.float64)
    np.testing.assert_allclose(norms["state"]["mean"], states.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(norms["actions"]["mean"], targets.mean(axis=0), atol=1e-12)
    # Last horizon stays in its own episode, never borrows the next episode.
    np.testing.assert_array_equal(
        samples[3]["action"],
        np.repeat(samples[3]["action"][:1], profile.action_horizon_steps, axis=0),
    )
    schema = json.loads(result.schema.read_text())
    spec = json.loads((Path(__file__).parents[1] / "spec/openapi.json").read_text())
    Draft202012Validator(
        {"$ref": "#/components/schemas/RobotSchema", "components": spec["components"]}
    ).validate(schema)
    assert len(schema["state_channels"]) == dimensions[1]
    assert len(schema["action_channels"]) == dimensions[3]
    copied = prepare_dataset(
        result.dataset,
        tmp_path / "identity",
        input_space="canonical",
        dataset_repo_id="example/calibrated",
        profile=profile,
    )
    assert dataset_file_manifest(copied.dataset) == dataset_file_manifest(result.dataset)
    assert (copied.assets / "norm_stats.json").read_bytes() == (
        result.assets / "norm_stats.json"
    ).read_bytes()


@pytest.mark.parametrize(
    "bad",
    ["duplicate", "missing", "unit", "gripper", "virtual", "delta", "fps", "unknown", "truncate"],
)
def test_invalid_profile_never_guesses(bad):
    value = example_profile().descriptor()
    if bad == "duplicate":
        value["action"]["canonical"][0]["native_index"] = 0
    elif bad == "missing":
        del value["calibration_digest"]
    elif bad == "unit":
        value["action"]["canonical"][0]["conversion"] = "identity"
    elif bad == "gripper":
        value["state"]["canonical"][3]["native_open"] = 50
    elif bad == "virtual":
        value["action"]["canonical"][2]["constant"] = None
    elif bad == "delta":
        value["model_delta_state_indices"][0] = 99
    elif bad == "fps":
        value["controller_hz"] = float("nan")
    elif bad == "unknown":
        value["expression"] = "arbitrary code is not a profile field"
    else:
        value["model_action_width"] = 2
    with pytest.raises(ValueError):
        RobotProfile.model_validate(value)


@pytest.mark.parametrize(
    "bad",
    [
        "names",
        "fps",
        "virtual",
        "bounds",
        "timestamps",
        "episode",
        "image",
        "stats",
        "scalar",
        "symlink",
    ],
)
def test_data_contract_failures_are_explicit_and_atomic(tmp_path, bad):
    profile = example_profile()
    paths = write_data_fixture(tmp_path / "fixture")
    root = Path(paths["dataset"])
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(path)
    if bad == "names":
        info["features"]["action"]["names"].reverse()
    elif bad == "fps":
        info["fps"] = 30
    elif bad == "image":
        info["features"]["observation.images.view_0"]["shape"] = [100, 100, 3]
    elif bad == "scalar":
        info["features"]["frame_index"]["shape"] = [2]
    elif bad == "stats":
        (root / "meta/episodes_stats.jsonl").write_text("{}\n")
    elif bad == "symlink":
        (root / "link").symlink_to(info_path)
    elif bad == "virtual":
        # Inject a contradictory physical mapping into an otherwise valid profile.
        value = profile.descriptor()
        value["state"]["canonical"][2]["constant"] = 1
        profile = RobotProfile.model_validate(value)
        # Native input legitimately produces that value, so test canonical corruption below.
        first = prepare_dataset(
            root,
            tmp_path / "first",
            input_space="native",
            dataset_repo_id="example/test",
            profile=profile,
        )
        root = first.dataset
        path = next((root / "data").rglob("*.parquet"))
        table = pq.read_table(path)
        values = table["observation.state"].to_pylist()
        values[0][2] = 0
        table = table.set_column(
            table.schema.get_field_index("observation.state"),
            "observation.state",
            pa.array(values, type=pa.list_(pa.float32(), 4)),
        )
    else:
        key = {"bounds": "action", "timestamps": "timestamp", "episode": "episode_index"}[bad]
        values = table[key].to_pylist()
        if bad == "bounds":
            values[0][0] = 999
        elif bad == "timestamps":
            values[1] += 0.001
        else:
            values[0] = 99
        table = table.set_column(
            table.schema.get_field_index(key),
            key,
            pa.array(values, type=table.schema.field(key).type),
        )
    if bad != "virtual":
        info_path.write_text(json.dumps(info))
    pq.write_table(table, path)
    with pytest.raises((PreparationError, ValueError, KeyError)):
        prepare_dataset(
            root,
            tmp_path / "invalid",
            input_space="canonical" if bad == "virtual" else "native",
            dataset_repo_id="example/test",
            profile=profile,
        )
    assert not (tmp_path / "invalid").exists()


class EchoDriver:
    connected = True
    armed = True
    stopped = False

    def send_action(self, action):
        return dict(action)

    def stop(self):
        self.stopped = True


@pytest.mark.parametrize("layout", ["planar", "gantry"])
def test_exact_receipts_limits_and_driver_faults(layout):
    profile = example_profile(layout)
    mapping = profile.action
    low, high = mapping.bounds("native")
    native = np.asarray((low + high) / 2, np.float32)
    canonical = mapping.to_canonical(native)
    # A token often cannot survive a lossy unit roundtrip bit-for-bit.
    canonical[0] = np.float32(0.1234567 if layout == "planar" else 0.0234567)
    receipt = CommandEncodingReceipt.from_canonical(canonical, mapping)
    encoded = receipt.encoded_native()
    exact = receipt.resolve_application(NativeCommandTrace.from_arrays(encoded, encoded))
    assert not exact.native_changed
    assert exact.canonical_float32 == canonical.astype("<f4").tobytes()
    applied = encoded.copy()
    applied[-1] = np.nextafter(applied[-1], np.float32(0))
    changed = receipt.resolve_application(NativeCommandTrace.from_arrays(encoded, applied))
    assert changed.native_changed and changed.driver_changed
    driver = EchoDriver()
    controller = RobotController(
        driver, mapping, CommandLimiter(tuple(low), tuple(high), tuple([0.01] * len(low))), native
    )
    trace = controller.apply_with_trace(encoded)
    assert np.all(np.abs(trace.submitted_array() - native) <= 0.010001)
    driver.armed = False
    with pytest.raises(ControllerFault):
        controller.apply(encoded)
    assert driver.stopped and controller.faulted


def test_profile_digest_is_canonical_and_immutable():
    profile = example_profile()
    value = copy.deepcopy(profile.descriptor())
    assert (
        RobotProfile.model_validate(dict(reversed(list(value.items())))).digest()
        == profile.digest()
    )
    value["action"]["native"][0]["upper"] = 80
    assert profile.action.native[0].upper == 90
    with pytest.raises(ValueError):
        profile.action.native[0].upper = 80


def test_single_coordinate_vector_is_not_scalar_metadata(tmp_path):
    value = example_profile().descriptor()
    for key in ("state", "action"):
        value[key] = {
            "native": [{"name": "axis", "unit": "meter", "lower": -1, "upper": 1}],
            "canonical": [
                {"name": "axis", "unit": "meter", "native_index": 0, "conversion": "identity"}
            ],
        }
    value["model_delta_state_indices"] = [0]
    profile = RobotProfile.model_validate(value)
    fixture = write_data_fixture(tmp_path / "fixture", profile=profile)
    result = prepare_dataset(
        fixture["dataset"],
        tmp_path / "prepared",
        input_space="native",
        dataset_repo_id="example/axis",
        profile=profile,
    )
    assert verify_preparation(result.root) == result
    table = pq.read_table(next((result.dataset / "data").rglob("*.parquet")))
    metadata = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    assert metadata["observation.state"]["_type"] == "Sequence"
    assert metadata["frame_index"]["_type"] == "Value"
