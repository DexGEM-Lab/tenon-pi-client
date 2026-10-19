from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tenon_pi import __version__, cli
from tenon_pi import simulator_receipt as receipts
from tenon_pi import simulator_setup as setup


def fake_installation(tmp_path, monkeypatch):
    prefix = tmp_path / "env"
    site = prefix / "lib/site-packages"
    site.mkdir(parents=True)
    source = tmp_path / "libero"
    source.mkdir()
    (source / "capture.py").write_text("capture = 1\n")
    versions = {
        "tenon-pi-client": __version__,
        "torch": "2.2.2+cpu",
        "libero": "0.1.0",
        "glfw": "1.0",
        "PyOpenGL": "1.0",
    }
    for line in Path("src/tenon_pi/resources/simulator-requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            versions[name] = version
    roots = {}
    for name in versions:
        package = "tenon_pi" if name == "tenon-pi-client" else name.replace("-", "_")
        root = site / package
        root.mkdir()
        (root / "__init__.py").write_text("value = 1\n")
        roots[name] = root
    (roots["tenon-pi-client"] / "simulator_receipt.py").write_text("fixture source\n")

    class Distribution:
        def __init__(self, name):
            self.name = name
            self.version = versions[name]
            self.files = [PurePosixPath(roots[name].name, "__init__.py")]

        def locate_file(self, relative):
            return site / str(relative)

        def read_text(self, name):
            # Only the archive hash may enter a receipt; never the URL.
            return json.dumps(
                {
                    "url": "https://user:fixture-secret@example.invalid/file",
                    "archive_info": {"hashes": {"sha256": "a" * 64}},
                }
            )

    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(
        receipts, "__file__", str(roots["tenon-pi-client"] / "simulator_receipt.py")
    )
    monkeypatch.setattr(receipts.metadata, "distribution", Distribution)
    monkeypatch.setattr(
        receipts,
        "_libero_inventory",
        lambda revision: {"revision": revision, "runtime_source": receipts._tree_inventory(source)},
    )
    return roots, versions, source


def make_receipt(tmp_path, monkeypatch):
    roots, versions, source = fake_installation(tmp_path, monkeypatch)
    current = receipts.runtime_inventory()
    path = tmp_path / "setup.json"
    receipts.write_receipt(
        path, receipts.seal_receipt({"preflight": "passed", "display": {"mode": "window"}}, current)
    )
    return path, roots, versions, source


@pytest.mark.parametrize(
    "mutation", ["sdk", "sdk_added_file", "dependency", "version", "source", "receipt"]
)
def test_actual_byte_inventory_detects_same_version_drift(tmp_path, monkeypatch, mutation):
    path, roots, versions, source = make_receipt(tmp_path, monkeypatch)
    assert receipts.validate_current_receipt(path, display="window")["preflight"] == "passed"
    assert "fixture-secret" not in path.read_text()
    if mutation == "sdk":
        (roots["tenon-pi-client"] / "__init__.py").write_text("value = 2\n")
    elif mutation == "sdk_added_file":
        (roots["tenon-pi-client"] / "new_module.py").write_text("new = True\n")
    elif mutation == "dependency":
        (roots["robosuite"] / "__init__.py").write_text("value = 2\n")
    elif mutation == "version":
        versions["mujoco"] = "9.9.9"
    elif mutation == "source":
        (source / "capture.py").write_text("capture = 2\n")
    else:
        value = json.loads(path.read_text())
        value["python"] = "changed-python"
        path.write_text(json.dumps(value))
    with pytest.raises(receipts.SetupReceiptError):
        receipts.validate_current_receipt(path, display="window")


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"version": "tenon-simulator-setup-v1", "preflight": "passed"},
        {"version": receipts.RECEIPT_VERSION, "preflight": "passed", "binding": {}},
    ],
)
def test_old_or_incomplete_receipts_are_not_trusted(tmp_path, value):
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(value))
    with pytest.raises(receipts.SetupReceiptError):
        receipts.read_receipt(path, display="headless")


@pytest.mark.parametrize("custom", [False, True])
def test_failed_qualification_precedes_any_api_use(tmp_path, monkeypatch, custom):
    context = tmp_path / "run.json"
    context.write_text("{}")
    api_calls = []
    monkeypatch.setattr(cli, "_client", lambda ctx: api_calls.append("called"))
    monkeypatch.setenv("TENON_API_KEY", "fixture-secret")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/source")
    child_calls = []

    def fail(command, **kwargs):
        child_calls.append(command)
        assert "TENON_API_KEY" not in kwargs["env"]
        assert "PYTHONPATH" not in kwargs["env"]
        assert "fixture-secret" not in command
        return SimpleNamespace(returncode=1, stdout="", stderr="fixture failure")

    monkeypatch.setattr(receipts.subprocess, "run", fail)
    args = [
        "agent",
        "start-simulator",
        "--context",
        str(context),
        "--setup",
        str(tmp_path / "missing.json"),
    ]
    if custom:
        args += ["--python", "custom-python"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code != 0 and not api_calls
    if custom:
        assert "--preflight" in child_calls[0]


def test_custom_python_runs_fresh_preflight_then_validates_before_launch(tmp_path, monkeypatch):
    path, *_ = make_receipt(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setenv("TENON_API_KEY", "fixture-secret")

    def run(command, **kwargs):
        assert "TENON_API_KEY" not in kwargs["env"]
        commands.append(command)
        if "--preflight" in command:
            destination = Path(command[command.index("--output") + 1])
            destination.write_bytes(path.read_bytes())
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(receipts.subprocess, "run", run)
    with receipts.qualified_simulator(
        tmp_path / "unused.json", python="custom-python", display="window"
    ) as selected:
        assert selected["receipt_path"].is_file()
        assert "--preflight" in commands[0] and "--validate-receipt" in commands[1]
        assert commands[0][0] == "custom-python"
    assert not selected["receipt_path"].exists()


def test_valid_receipt_is_checked_in_selected_interpreter_before_yield(tmp_path, monkeypatch):
    path, *_ = make_receipt(tmp_path, monkeypatch)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(receipts.subprocess, "run", run)
    with receipts.qualified_simulator(path, python=None, display="window") as selected:
        assert calls[0][0] == selected["python"]
        assert calls[0][3] == "--validate-receipt"


@pytest.mark.parametrize("session_id", ["ctl_fixture", None])
def test_child_revalidates_before_requesting_or_reading_token(tmp_path, monkeypatch, session_id):
    from tenon_pi import simulator

    actions = []

    def stale(*args, **kwargs):
        actions.append("validate")
        raise receipts.SetupReceiptError("fixture source changed")

    monkeypatch.setattr(simulator, "validate_current_receipt", stale)
    monkeypatch.setattr(simulator, "TenonClient", lambda *a, **kw: actions.append("API"))

    class TokenPath:
        def read_text(self):
            actions.append("token-read")
            raise AssertionError("must not read the token")

    args = SimpleNamespace(
        setup_receipt=tmp_path / "setup.json",
        display="window",
        session_id=session_id,
        connection_json=TokenPath(),
    )
    with pytest.raises(receipts.SetupReceiptError):
        simulator.run_connected_episode(args, None, None, None, None)
    assert actions == ["validate"]


def test_darwin_glx_rejects_before_inventory_or_renderer(tmp_path, monkeypatch):
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("MUJOCO_GL", "glx")
    monkeypatch.setattr(
        receipts, "runtime_inventory", lambda: pytest.fail("must reject before inventory")
    )
    with pytest.raises(ValueError):
        setup.simulator_preflight(display="window", output=tmp_path / "setup.json")


def test_editable_dependency_cannot_claim_installed_file_coverage(monkeypatch):
    distribution = SimpleNamespace(
        version="1.4.1",
        files=[],
        read_text=lambda name: json.dumps({"dir_info": {"editable": True}}),
    )
    monkeypatch.setattr(receipts.metadata, "distribution", lambda name: distribution)
    with pytest.raises(receipts.SetupReceiptError, match="editable"):
        receipts._distribution_inventory("robosuite", "1.4.1")
