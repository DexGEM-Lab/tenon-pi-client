from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tenon_pi.cli import app
from tenon_pi.robot_profile import RobotProfile


def test_agent_discovery_skill_and_portable_profiles():
    root = Path(__file__).parents[1]
    skill_dir = root / "skills/tenon-pi-agent"
    text = (skill_dir / "SKILL.md").read_text()
    assert text.startswith("---\nname: tenon-pi-agent\n")
    assert "user gives an API key and robot/simulator data" in text.split("---", 2)[1]
    for path in (
        "../../docs/quickstart.md",
        "../../docs/quickstart.zh-CN.md",
        "../../docs/robot-preparation.md",
        "../../docs/libero-demo.md",
        "../../docs/control-runtime.md",
        "../../docs/control-runtime.zh-CN.md",
        "../../examples/robot_control_boundary.py",
    ):
        assert (skill_dir / path).resolve().is_file()
        assert path in text
    assert ".tenon/run.json" in text
    assert "Never present" in text
    assert "empty committed prefix" in text or "Empty-prefix warm probes" in text
    assert "skills/tenon-pi-agent/SKILL.md" in (root / "AGENTS.md").read_text()
    assert "@AGENTS.md" in (root / "CLAUDE.md").read_text()
    for layout in ("planar", "gantry"):
        profile = RobotProfile.load(root / f"examples/profiles/{layout}.json")
        assert "synthetic" in profile.profile_id
    schema = json.loads((root / "schemas/robot-profile.schema.json").read_text())
    assert schema == RobotProfile.model_json_schema()


def test_documented_agent_and_data_commands_exist():
    runner = CliRunner()
    for group, command in (
        ("data", "inspect"),
        ("data", "import-jsonl"),
        ("data", "prepare"),
        ("data", "verify"),
        ("data", "manifest"),
        ("agent", "train"),
        ("agent", "deploy"),
        ("agent", "start-simulator"),
    ):
        result = runner.invoke(app, [group, command, "--help"])
        assert result.exit_code == 0, result.output
