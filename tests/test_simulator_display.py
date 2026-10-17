from __future__ import annotations

import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from tenon_pi import cli
from tenon_pi.libero_runtime import LiberoAsyncRollout
from tenon_pi.profiles import libero_robot_schema
from tenon_pi.simulator_display import SimulatorDisplay
from tenon_pi.workflow import RunContext


def frames():
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    return {"agentview_image": pixels, "robot0_eye_in_hand_image": pixels + 10}


def test_headless_selection_is_explicit_and_window_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    auto = SimulatorDisplay("auto")
    assert auto.summary()["mode"] == "headless"
    assert auto.summary()["reason"] == "no_x11_display"
    assert auto.show(frames()) and auto.frames_presented == 0
    explicit = SimulatorDisplay("headless")
    assert explicit.reason == "explicit_headless"
    with pytest.raises(RuntimeError, match="window unavailable"):
        SimulatorDisplay("window")


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_window_presents_actual_oriented_camera_frames_on_main_thread(monkeypatch, platform):
    import PIL

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("DISPLAY", ":example")
    images, threads = [], []

    class Root:
        def title(self, value):
            assert "Live LIBERO" in value

        def protocol(self, name, handler):
            self.on_close = handler

        def update_idletasks(self):
            pass

        def update(self):
            threads.append(threading.get_ident())

        def destroy(self):
            pass

    class Label:
        def __init__(self, root, **kwargs):
            pass

        def pack(self):
            pass

        def configure(self, **kwargs):
            pass

    monkeypatch.setitem(
        sys.modules, "tkinter", SimpleNamespace(Tk=Root, Label=Label, TclError=RuntimeError)
    )
    image_tk = SimpleNamespace(
        PhotoImage=lambda image, **kw: images.append(np.asarray(image)) or object()
    )
    monkeypatch.setitem(sys.modules, "PIL.ImageTk", image_tk)
    monkeypatch.setattr(PIL, "ImageTk", image_tk, raising=False)
    display = SimulatorDisplay("window")
    raw = frames()
    assert display.show(raw)
    np.testing.assert_array_equal(images[-1][:, :6], raw["agentview_image"][::-1, ::-1])
    np.testing.assert_array_equal(images[-1][:, 6:], raw["robot0_eye_in_hand_image"][::-1, ::-1])
    assert threads and set(threads) == {threading.get_ident()}
    assert display.frames_presented == 1
    display._root.on_close()
    assert not display.show(raw)


@pytest.mark.parametrize("done_after", [1, 12])
def test_viewer_close_stops_rollout_before_another_environment_action(done_after):
    from test_libero_runtime import FakeEnv, raw_observation

    env = FakeEnv(done_after=done_after)
    metrics = SimpleNamespace(
        accepted_responses=0,
        dropped_responses=0,
        fallback_actions=0,
        applied_overrides=0,
        prefix_invalidations=0,
    )
    controller = SimpleNamespace(
        warm_start=lambda observations: None,
        next_action=lambda observation: np.zeros(7),
        record_applied=lambda action: None,
        metrics=metrics,
    )
    shown = []

    def present(raw):
        shown.append(raw)
        return len(shown) < 2

    result = LiberoAsyncRollout(env, controller, prompt="task", on_frame=present).run(
        raw_observation(), max_steps=10
    )
    assert result.steps == len(env.actions) == 1
    assert result.stopped_reason == "viewer_closed"
    assert result.success == (done_after == 1)


def test_agent_launch_forwards_display_and_keeps_key_out_of_arguments(tmp_path, monkeypatch):
    journal_path = tmp_path / "run.json"
    schema = libero_robot_schema()
    with RunContext(journal_path) as journal:
        journal.set("input", {"schema": schema.model_dump(mode="json")})
        journal.set("session_id", "ctl_example")
        journal.set("deployment_id", "dep_example")
        journal.set("session_spec", {"execution_mode": "async_aligned"})
        journal.set("dataset_version", {"schema_hash": "sha256:example"})

    class Client:
        base_url = "https://service.example.invalid/v1"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def me(self):
            return SimpleNamespace(id="user_example")

        def get_control_session(self, identifier):
            return SimpleNamespace(
                id=identifier,
                state="active",
                connection_state="never_connected",
                deployment_id="dep_example",
            )

    launched = []

    def launch(command, *, env, check):
        launched.append(command)
        assert env["TENON_API_KEY"] == "fixture-secret"
        assert "fixture-secret" not in command
        return SimpleNamespace(returncode=0)

    from contextlib import contextmanager

    @contextmanager
    def qualified(*args, **kwargs):
        yield {
            "python": "simulator-python",
            "environment": {},
            "receipt_path": tmp_path / "qualified-receipt.json",
        }

    monkeypatch.setattr("tenon_pi.simulator_receipt.qualified_simulator", qualified)
    monkeypatch.setattr(cli, "_client", lambda ctx: Client())
    monkeypatch.setattr("subprocess.run", launch)
    result = CliRunner().invoke(
        cli.app,
        [
            "--api-key",
            "fixture-secret",
            "agent",
            "start-simulator",
            "--context",
            str(journal_path),
            "--display",
            "window",
            "--python",
            "simulator-python",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert launched[0][:3] == ["simulator-python", "-m", "tenon_pi.simulator"]
    assert launched[0][launched[0].index("--display") + 1] == "window"
    assert launched[0][launched[0].index("--setup-receipt") + 1] == str(
        tmp_path / "qualified-receipt.json"
    )
    saved = json.loads(journal_path.read_text())
    assert saved["simulator_launch"]["display_requested"] == "window"
    assert "fixture-secret" not in journal_path.read_text()
