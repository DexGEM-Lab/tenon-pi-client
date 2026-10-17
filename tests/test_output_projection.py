from __future__ import annotations

import json

import numpy as np
import pytest

from tenon_pi.data_fixture import example_profile, write_data_fixture
from tenon_pi.preparation import prepare_dataset, verify_preparation
from tenon_pi.robot import CommandEncodingReceipt
from tenon_pi.robot_profile import RobotProfile


def projection_profile(mode="strict"):
    value = example_profile().descriptor()
    value["state"] = dict(
        native=[
            dict(name="position", unit="meter", lower=-1, upper=1),
            dict(name="sensor", unit="unitless", lower=-10, upper=10),
        ],
        canonical=[
            dict(name="sensor", unit="unitless", native_index=1, conversion="identity"),
            dict(name="position", unit="meter", native_index=0, conversion="identity"),
        ],
    )
    value["action"] = dict(
        native=[dict(name="position", unit="meter", lower=-1, upper=1)],
        canonical=[
            dict(name="position", unit="meter", native_index=0, conversion="identity"),
            dict(
                name="virtual",
                unit="unitless",
                native_index=None,
                conversion="constant",
                constant=0.25,
            ),
        ],
    )
    value["model_delta_state_indices"] = [1, None]
    value["model_state_width"] = value["model_action_width"] = 4
    value["output_projection"] = mode
    return RobotProfile.model_validate(value)


def test_projection_is_explicit_digest_bound_and_raw_compatibility_is_unchanged():
    strict, clipped = projection_profile(), projection_profile("clip_bounds_v1")
    action = [0.3, 99, 100, 100]
    state = [0.9, 0]
    np.testing.assert_allclose(strict.canonical_target(action, state), [1.2, 0.25], atol=1e-7)
    with pytest.raises(ValueError, match="strict"):
        strict.native_target(action, state)
    np.testing.assert_array_equal(clipped.native_target(action, state), [1.0])
    np.testing.assert_allclose(
        clipped.canonical_target(action, state, project=False), [1.2, 0.25], atol=1e-7
    )
    assert strict.digest() != clipped.digest()
    assert strict.descriptor()["output_projection"] == "strict"
    omitted = strict.descriptor()
    omitted.pop("output_projection")
    assert RobotProfile.model_validate(omitted).descriptor() == strict.descriptor()
    # Neither ordinary conversion nor receipt encoding silently clips input commands.
    np.testing.assert_allclose(clipped.action.to_native([1.2, 0.25]), [1.2])
    with pytest.raises(RuntimeError, match="bounds"):
        CommandEncodingReceipt.from_canonical([1.2, 0.25], clipped.action)
    state_canonical = clipped.state.to_canonical(state)
    targets = np.array([[1.2, 0.25]], np.float32)
    np.testing.assert_array_equal(
        strict.model_actions(state_canonical, targets),
        clipped.model_actions(state_canonical, targets),
    )


@pytest.mark.parametrize("mode", ["strict", "clip_bounds_v1"])
@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_nonfinite_model_coordinates_cannot_hide_in_virtual_or_padding(mode, index):
    profile = projection_profile(mode)
    for invalid in (np.nan, np.inf, -np.inf):
        action = np.zeros(4, np.float32)
        action[index] = invalid
        with pytest.raises(ValueError, match="finite"):
            profile.canonical_target(action, [0.9, 0])


def test_nonfinite_unused_state_and_inverse_overflow_reject_before_clipping():
    profile = projection_profile("clip_bounds_v1")
    with pytest.raises(ValueError, match="finite"):
        profile.canonical_target([0.3, 0, 0, 0], [0.9, np.nan])
    with pytest.raises(ValueError, match="after inverse"):
        profile.canonical_target([3e38, 0, 0, 0], [3e38, 0])


def test_gripper_projection_keeps_measured_endpoints():
    value = example_profile().descriptor()
    value["output_projection"] = "clip_bounds_v1"
    profile = RobotProfile.model_validate(value)
    action = np.zeros(profile.model_action_width, np.float32)
    action[3] = 1.2
    assert profile.native_target(action, [0, 0, 50])[-1] == 0
    assert profile.canonical_target(action, [0, 0, 50], project=False)[3] == np.float32(1.2)
    action[3] = -0.2
    assert profile.native_target(action, [0, 0, 50])[-1] == 100


def test_projection_tamper_invalidates_prepared_bindings(tmp_path):
    fixture = write_data_fixture(tmp_path / "source")
    profile = RobotProfile.load(fixture["profile"])
    result = prepare_dataset(
        fixture["dataset"],
        tmp_path / "prepared",
        input_space="native",
        dataset_repo_id="example/projection",
        profile=profile,
    )
    path = result.assets / "transform.json"
    descriptor = json.loads(path.read_text())
    descriptor["output_projection"] = "clip_bounds_v1"
    path.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError):
        verify_preparation(result.root)
