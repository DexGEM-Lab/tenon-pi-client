from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tenon_pi.simulator_setup import LIBERO_REVISION, bootstrap_simulator, setup_environment


def test_platform_renderer_selection_is_cpu_and_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert setup_environment(tmp_path, display="headless")["MUJOCO_GL"] == "osmesa"
    monkeypatch.setenv("DISPLAY", ":fixture")
    assert setup_environment(tmp_path, display="window")["MUJOCO_GL"] == "glx"
    for renderer in ("egl", "glfw"):
        with pytest.raises(ValueError):
            setup_environment(tmp_path, display="headless", renderer=renderer)
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert setup_environment(tmp_path, display="window")["MUJOCO_GL"] == "cgl"
    for renderer in ("osmesa", "glx"):
        with pytest.raises(ValueError):
            setup_environment(tmp_path, display="headless", renderer=renderer)


def test_bootstrap_owns_pins_strips_credentials_and_requires_preflight(tmp_path, monkeypatch):
    source = tmp_path / "client"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nname="tenon-pi-client"\n')
    destination = tmp_path / "simulator"
    commands = []
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setenv("TENON_API_KEY", "fixture-secret")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/source")
    monkeypatch.setattr("venv.EnvBuilder.create", lambda self, path: Path(path).mkdir())

    def run(command, **kwargs):
        assert "TENON_API_KEY" not in kwargs["env"]
        assert "PYTHONPATH" not in kwargs["env"]
        assert "fixture-secret" not in command
        commands.append(command)
        if "--preflight" in command:
            output = Path(command[command.index("--output") + 1])
            from tenon_pi.simulator_receipt import seal_receipt

            binding = {
                "inventory_version": 1,
                "sdk": {},
                "dependencies": {},
                "graphics_bindings": {},
                "libero": {},
                "python": {},
                "environment": {},
            }
            result = seal_receipt(
                {
                    "preflight": "passed",
                    "display": {"mode": "headless"},
                    "libero_revision": LIBERO_REVISION,
                },
                binding,
            )
            output.write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", run)
    result = bootstrap_simulator(destination, client_source=source, display="headless")
    assert result["preflight"] == "passed" and result["libero_revision"] == LIBERO_REVISION
    assert any("3.11.11" in command for command in commands)
    assert any("uv==0.6.17" in command for command in commands)
    assert any("https://download.pytorch.org/whl/cpu" in command for command in commands)
    assert commands[-1][1:3] == ["-m", "tenon_pi.simulator_setup"]


def test_setup_does_not_overwrite_unowned_directory(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(ValueError, match="ownership"):
        bootstrap_simulator(target, client_source=tmp_path, display="headless")
