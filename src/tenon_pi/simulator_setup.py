"""Owned, isolated CPU simulator bootstrap and real renderer/viewer preflight.

No server, policy weights or hosted billable resource is created here. Operating
system packages are never installed silently; missing GL/Tk needs host permission.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import venv
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

LIBERO_REVISION = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
PYTHON_VERSION = "3.11.11"
UV_VERSION = "0.6.17"
TORCH_VERSION = "2.2.2"
SETUP_VERSION = "tenon-simulator-setup-v1"


def setup_environment(root: Path, *, display: str, renderer: str | None = None) -> dict[str, str]:
    if display not in {"window", "headless", "auto"}:
        raise ValueError("display must be window, headless or auto")
    system = platform.system()
    if system not in {"Linux", "Darwin"}:
        raise ValueError("the pinned simulator setup supports Linux/macOS only")
    selected = renderer or (
        "cgl" if system == "Darwin" else "glx" if os.environ.get("DISPLAY") else "osmesa"
    )
    allowed = {"cgl"} if system == "Darwin" else {"glx", "osmesa"}
    if selected not in allowed:
        raise ValueError(f"{system} simulator renderer must be one of {sorted(allowed)}")
    result = {
        "LIBERO_CONFIG_PATH": str(root / "libero-config"),
        "MUJOCO_GL": selected,
        "TOKENIZERS_PARALLELISM": "false",
    }
    if system == "Linux":
        result["LIBGL_ALWAYS_SOFTWARE"] = "1"
        result["__GLX_VENDOR_LIBRARY_NAME"] = "mesa"
    return result


def bootstrap_simulator(
    directory: str | Path,
    *,
    client_source: str | Path,
    display: str = "window",
    renderer: str | None = None,
    python: str | None = None,
) -> dict[str, Any]:
    root, source = (
        Path(directory).expanduser().resolve(),
        Path(client_source).expanduser().resolve(),
    )
    if not (source / "pyproject.toml").is_file():
        raise ValueError("run setup from the cloned client repository or provide --client-source")
    selected_env = setup_environment(root, display=display, renderer=renderer)
    marker = root / "owner.json"
    identity = {
        "version": SETUP_VERSION,
        "libero_revision": LIBERO_REVISION,
        "python_family": "3.11",
        "client_source": str(source),
    }
    if root.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise ValueError(
                "setup directory exists without matching ownership; do not overwrite it"
            )
    else:
        root.mkdir(parents=True)
        marker.write_text(json.dumps(identity, indent=2) + "\n")
    receipt = root / "setup.json"
    if receipt.exists():
        receipt.unlink()  # Any failed new attempt invalidates the prior success receipt.
    env = os.environ.copy()
    env.pop("TENON_API_KEY", None)
    env.pop("TENON_BASE_URL", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(selected_env)
    log_path = root / "setup.log"

    def run(command: list[str]) -> None:
        with log_path.open("a") as log:
            outcome = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        if outcome.returncode:
            raise RuntimeError(
                f"simulator setup step failed; inspect local {log_path.name} "
                "(no paid resource created)"
            )

    tools = root / "tools"
    if not (tools / "bin/python").is_file():
        venv.EnvBuilder(with_pip=True).create(tools)
    tool_python = str(tools / "bin/python")
    run([tool_python, "-m", "pip", "install", f"uv=={UV_VERSION}"])
    runtime = root / "venv"
    if not (runtime / "bin/python").is_file():
        run(
            [
                tool_python,
                "-m",
                "uv",
                "venv",
                "--seed",
                "--python",
                python or PYTHON_VERSION,
                str(runtime),
            ]
        )
    executable = str(runtime / "bin/python")
    run(
        [
            executable,
            "-c",
            "import sys; assert sys.version_info[:2] == (3, 11), 'Python 3.11 required'",
        ]
    )
    checkout = root / "libero-source"
    if not (checkout / ".git").exists():
        checkout.mkdir(exist_ok=True)
        run(["git", "init", str(checkout)])
        run(
            [
                "git",
                "-C",
                str(checkout),
                "remote",
                "add",
                "origin",
                "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
            ]
        )
    run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", LIBERO_REVISION])
    run(["git", "-C", str(checkout), "checkout", "--detach", LIBERO_REVISION])
    run(["git", "-C", str(checkout), "diff", "--exit-code", "HEAD"])
    run([executable, "-m", "pip", "install", "setuptools==75.3.0", "wheel==0.45.1"])
    torch_command = [executable, "-m", "pip", "install", f"torch=={TORCH_VERSION}"]
    if platform.system() == "Linux":
        torch_command += ["--index-url", "https://download.pytorch.org/whl/cpu"]
    run(torch_command)
    requirements = root / "simulator-requirements.txt"
    requirements.write_bytes(
        files("tenon_pi").joinpath("resources/simulator-requirements.txt").read_bytes()
    )
    run([executable, "-m", "pip", "install", "-r", str(requirements), f"{source}[data]"])
    run([executable, "-m", "pip", "install", "--no-deps", str(checkout)])
    # This pinned upstream uses a namespace root that find_packages omits from
    # its wheel. Expose the exact owned checkout, as its official source recipe
    # does, without making the Tenon SDK editable or inheriting caller PYTHONPATH.
    run(
        [
            executable,
            "-c",
            "import pathlib,site,sys; pathlib.Path(site.getsitepackages()[0], "
            "'tenon_libero_source.pth').write_text(sys.argv[1] + '\\n')",
            str(checkout),
        ]
    )
    # Prevent LIBERO's interactive first-import configuration and avoid ~/.libero.
    config_dir = root / "libero-config"
    config_dir.mkdir(exist_ok=True)
    base = checkout / "libero/libero"
    datasets = root / "datasets"
    datasets.mkdir(exist_ok=True)
    config = {
        "benchmark_root": str(base),
        "bddl_files": str(base / "bddl_files"),
        "init_states": str(base / "init_files"),
        "assets": str(base / "assets"),
        "datasets": str(datasets),
    }
    # JSON is valid YAML, the format read by this pinned LIBERO revision.
    (config_dir / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    run(
        [
            executable,
            "-m",
            "tenon_pi.simulator_setup",
            "--preflight",
            "--output",
            str(receipt),
            "--display",
            display,
        ]
    )
    from .simulator_receipt import read_receipt

    return read_receipt(receipt, display=display)


def simulator_preflight(*, display: str, output: Path) -> dict[str, Any]:
    """Create the actual selected LIBERO environment, render and step on CPU."""
    from .simulator_receipt import runtime_inventory, seal_receipt, write_receipt

    # Reject unsupported renderer arguments before simulator imports or API use.
    selected = setup_environment(Path.cwd(), display=display, renderer=os.environ.get("MUJOCO_GL"))
    os.environ.update(
        {key: value for key, value in selected.items() if key != "LIBERO_CONFIG_PATH"}
    )
    output.unlink(missing_ok=True)
    before = runtime_inventory()
    from libero.libero import benchmark

    from .libero_runtime import LIBERO_DUMMY_ACTION
    from .simulator import environment
    from .simulator_display import SimulatorDisplay

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(0)
    viewer = SimulatorDisplay(display)
    env = None
    try:
        env = environment(task)
        env.seed(7)
        env.reset()
        raw = env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(3):
            if not viewer.show(raw):
                raise RuntimeError("viewer closed during setup preflight")
            raw, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
        from OpenGL import GL

        renderer_bytes = GL.glGetString(GL.GL_RENDERER)
        renderer_name = renderer_bytes.decode() if renderer_bytes is not None else "unknown"
        software = any(
            name in renderer_name.lower() for name in ("llvmpipe", "softpipe", "software")
        )
        if platform.system() == "Linux" and not software:
            raise RuntimeError("CPU graphics preflight requires an observed software GL renderer")
        shapes = {
            name: list(raw[name].shape) for name in ("agentview_image", "robot0_eye_in_hand_image")
        }
        result = {
            "preflight": "passed",
            "task_suite": "libero_10",
            "task_id": 0,
            "camera_shapes": shapes,
            "display": viewer.summary(),
            "policy_loaded": False,
            "gl_renderer": renderer_name,
            "cpu_graphics_verified": software,
            "platform": platform.system(),
            "machine": platform.machine(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("tenon-pi-client", "libero", "robosuite", "mujoco", "torch", "numpy")
            },
        }
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            viewer.close()

    after = runtime_inventory()
    if before != after:
        raise ValueError(
            "simulator environment changed during preflight; no success receipt written"
        )
    result.update(qualified_at=datetime.now(UTC).isoformat(), libero_revision=LIBERO_REVISION)
    receipt = seal_receipt(result, after)
    write_receipt(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--preflight", action="store_true")
    operation.add_argument("--validate-receipt", type=Path)
    parser.add_argument("--display", choices=("window", "headless", "auto"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.preflight:
        if args.output is None:
            parser.error("--preflight requires --output")
        result = simulator_preflight(display=args.display, output=args.output)
        print(json.dumps(result, indent=2))
    else:
        from .simulator_receipt import validate_current_receipt

        validate_current_receipt(args.validate_receipt, display=args.display)
        print(json.dumps({"receipt": "current", "display_requested": args.display}))


if __name__ == "__main__":
    main()
