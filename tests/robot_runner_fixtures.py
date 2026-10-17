"""CPU-only installation adapters: explicit fixture state, never hardware evidence."""

from __future__ import annotations

import time

import numpy as np

from tenon_pi.data_fixture import example_profile
from tenon_pi.robot_runtime import CameraCapture, RobotCapture, RobotRunSettings


def runner_profile(layout):
    value = example_profile(layout).descriptor()
    value["action_horizon_steps"] = 10 if layout == "planar" else 9
    return type(example_profile()).model_validate(value)


def settings(profile, steps=5):
    lo, hi = profile.action.bounds("native")
    return RobotRunSettings(
        lower=tuple(lo),
        upper=tuple(hi),
        max_delta=tuple([0.2] * len(lo)),
        max_capture_age_ms=500,
        max_capture_skew_ms=25,
        max_steps=steps,
    )


class PlanarAdapter:
    def __init__(self, profile):
        self.profile = profile
        self.connected = self.armed = True
        self.stop_requested = False
        self.values = dict(base_angle=0.0, elbow_angle=0.0, jaw_open=50.0)
        self.commands, self.captures, self.stops = [], [], 0
        self.failure = None

    def last_applied_action(self):
        return dict(self.values)

    def capture(self, *, after_action_index):
        if self.failure == "capture":
            raise OSError("fixture capture failed")
        stamp = time.monotonic_ns()
        if self.failure == "stale":
            stamp -= 1_000_000_000
        camera = self.profile.cameras[0]
        record = RobotCapture(
            dict(self.values),
            stamp,
            (
                CameraCapture(
                    camera,
                    np.full((camera.height, camera.width, 3), 37, np.uint8),
                    stamp - (100_000_000 if self.failure == "skew" else 0),
                ),
            ),
            after_action_index + (1 if self.failure == "alignment" else 0),
            self.profile.calibration_digest,
        )
        self.captures.append(record)
        return record

    def send_action(self, action):
        self.values = dict(action)
        self.commands.append(dict(action))
        if self.failure == "apply":
            raise OSError("fixture command failed after partial write")
        return dict(action)

    def stop(self):
        self.stops += 1
        self.armed = False


class GantryAdapter:
    """Different implementation: motor bank, independent jaw and load telemetry."""

    def __init__(self, profile):
        self.profile = profile
        self.connected = self.armed = True
        self.stop_requested = False
        self.motor_bank = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.jaw = 20.0
        self.load = 0.25
        self.commands, self.captures, self.stopped_groups = [], [], []
        self.partial_failure = False

    def last_applied_action(self):
        return {**self.motor_bank, "jaw": self.jaw}

    def capture(self, *, after_action_index):
        stamp = time.monotonic_ns()
        cameras = tuple(
            CameraCapture(
                camera, np.full((camera.height, camera.width, 3), 40 + i * 30, np.uint8), stamp
            )
            for i, camera in enumerate(reversed(self.profile.cameras))
        )
        result = RobotCapture(
            {"load_sensor": self.load, "jaw": self.jaw, **self.motor_bank},
            stamp,
            cameras,
            after_action_index,
            self.profile.calibration_digest,
        )
        self.captures.append(result)
        return result

    def send_action(self, action):
        self.motor_bank.update({k: action[k] for k in self.motor_bank})
        self.commands.append(dict(action))
        if self.partial_failure:
            raise OSError("fixture jaw group failed after motor group wrote")
        self.jaw = action["jaw"]
        return self.last_applied_action()

    def stop(self):
        # Both owned groups are attempted; never return after only the first.
        self.stopped_groups.extend(["motor_bank", "jaw"])
        self.armed = False
