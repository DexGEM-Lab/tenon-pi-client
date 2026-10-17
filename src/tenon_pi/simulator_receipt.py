"""Byte-bound local simulator qualification; no API calls or per-frame hashing.

This detects ordinary environment drift, not a hostile owner who can rewrite
both code and receipts. OS drivers/stdlib and unselected transitive packages are
outside the inventory; actual renderer/window creation still precedes token use.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any

RECEIPT_VERSION = "tenon-simulator-receipt-v2"
ENVIRONMENT_FIELDS = frozenset(
    {
        "LIBERO_CONFIG_PATH",
        "MUJOCO_GL",
        "LIBGL_ALWAYS_SOFTWARE",
        "TOKENIZERS_PARALLELISM",
        "__GLX_VENDOR_LIBRARY_NAME",
    }
)


class SetupReceiptError(ValueError):
    """Local qualification is missing, edited or no longer describes installed bytes."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inventory(paths: Mapping[str, Path]) -> dict[str, Any]:
    records = []
    for name, path in sorted(paths.items()):
        if path.is_symlink() or not path.is_file():
            raise SetupReceiptError("inventory contains a missing, symlink or non-regular file")
        records.append([name, path.stat().st_size, _file_hash(path)])
    if not records:
        raise SetupReceiptError("empty inventory cannot qualify an environment")
    return {
        "sha256": _digest(records),
        "files": len(records),
        "size_bytes": sum(row[1] for row in records),
    }


def _tree_inventory(root: Path) -> dict[str, Any]:
    paths = {}
    for path in root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise SetupReceiptError("runtime source tree contains an unqualified symlink")
        if path.is_file():
            paths[path.relative_to(root).as_posix()] = path
    return _inventory(paths)


def _distribution_inventory(name: str, expected: str | None = None) -> dict[str, Any]:
    distribution = metadata.distribution(name)
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("dir_info", {}).get("editable") is True:
        raise SetupReceiptError(f"editable dependency is outside the receipt inventory: {name}")
    if expected is not None and distribution.version.split("+", 1)[0] != expected:
        raise SetupReceiptError(f"pinned dependency version differs: {name}")
    if distribution.files is None:
        raise SetupReceiptError(f"installed file inventory unavailable: {name}")
    paths = {}
    prefix = Path(sys.prefix).resolve()
    for relative in distribution.files:
        if relative.suffix in {".pyc", ".pyo"}:
            continue
        path = Path(distribution.locate_file(relative))
        if not path.resolve().is_relative_to(prefix):
            raise SetupReceiptError(f"dependency file escapes the selected environment: {name}")
        paths[str(relative)] = path
    return {"version": distribution.version, "contents": _inventory(paths)}


def _libero_inventory(revision: str) -> dict[str, Any]:
    import yaml

    config_file = (
        Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))) / "config.yaml"
    )
    if not config_file.is_file():
        raise SetupReceiptError("LIBERO configuration is missing; run agent setup-simulator")
    config = yaml.safe_load(config_file.read_text())
    if not isinstance(config, dict) or set(config) != {
        "benchmark_root",
        "bddl_files",
        "init_states",
        "assets",
        "datasets",
    }:
        raise SetupReceiptError(
            "LIBERO configuration does not have the supported path declarations"
        )
    base = Path(config["benchmark_root"]).resolve()
    checkout = base.parent.parent
    git_env = os.environ.copy()
    git_env.pop("TENON_API_KEY", None)
    git_env.pop("TENON_BASE_URL", None)

    def git(*args: str) -> str:
        process = subprocess.run(
            ["git", "-C", str(checkout), "-c", "core.fsmonitor=false", *args],
            env=git_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode:
            raise SetupReceiptError("pinned LIBERO source is missing or modified; rerun setup")
        return process.stdout.strip()

    if git("rev-parse", "HEAD") != revision:
        raise SetupReceiptError("LIBERO source revision differs from the supported pin")
    git("diff", "--no-ext-diff", "--quiet", "HEAD", "--")
    if base != checkout / "libero/libero":
        raise SetupReceiptError("LIBERO runtime root differs from its pinned source layout")
    for name, folder in (
        ("assets", "assets"),
        ("init_states", "init_files"),
        ("bddl_files", "bddl_files"),
    ):
        if Path(config[name]).resolve() != base / folder:
            raise SetupReceiptError("LIBERO resource paths differ from the pinned runtime source")
    tracked = set(git("ls-files", "-z", "--", "libero").split("\0")) - {""}
    actual = {
        path.relative_to(checkout).as_posix()
        for path in (checkout / "libero").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if actual != tracked:
        raise SetupReceiptError("LIBERO runtime source file set differs from its pinned checkout")
    spec = importlib.util.find_spec("libero.libero")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != base / "__init__.py":
        raise SetupReceiptError("imported LIBERO source differs from its configured source")
    return {
        "revision": revision,
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "runtime_source": _tree_inventory(checkout / "libero"),
        "configuration_sha256": _file_hash(config_file),
        "configuration_path": str(config_file.resolve()),
    }


def runtime_inventory() -> dict[str, Any]:
    """Read actual bytes once per setup/launch boundary, not RECORD hashes or mtimes."""
    from ._version import __version__
    from .simulator_setup import LIBERO_REVISION, TORCH_VERSION

    package_root = Path(__file__).resolve().parent
    distribution = metadata.distribution("tenon-pi-client")
    if Path(distribution.locate_file("tenon_pi")).resolve() != package_root:
        raise SetupReceiptError(
            "simulator SDK must be installed non-editably, without source-path injection"
        )
    sdk = _distribution_inventory("tenon-pi-client", __version__)
    sdk["package_tree"] = _tree_inventory(package_root)
    # Optional archive digest is only corroboration; actual package bytes above
    # remain authoritative. Never retain a direct_url URL or credentials.
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    archive_hash = direct.get("archive_info", {}).get("hashes", {}).get("sha256")
    sdk["reported_wheel_sha256"] = (
        archive_hash
        if isinstance(archive_hash, str) and re.fullmatch(r"[0-9a-f]{64}", archive_hash)
        else None
    )
    pins = {"torch": TORCH_VERSION, "libero": "0.1.0"}
    for line in (
        files("tenon_pi").joinpath("resources/simulator-requirements.txt").read_text().splitlines()
    ):
        if line.strip() and not line.startswith("#"):
            name, version = line.strip().split("==")
            pins[name] = version
    dependencies = {
        name: _distribution_inventory(name, version) for name, version in sorted(pins.items())
    }
    # These direct graphics bindings are installed transitively; bind their
    # observed versions/full files without claiming the whole environment is pinned.
    graphics = {name: _distribution_inventory(name) for name in ("glfw", "PyOpenGL")}
    return {
        "inventory_version": 1,
        "sdk": sdk,
        "dependencies": dependencies,
        "graphics_bindings": graphics,
        "libero": _libero_inventory(LIBERO_REVISION),
        "python": {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "version": sys.version,
            "binary_sha256": _file_hash(Path(sys.executable).resolve()),
        },
        "environment": {
            name: os.environ[name] for name in sorted(ENVIRONMENT_FIELDS) if name in os.environ
        },
    }


def seal_receipt(result: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    receipt = {
        **result,
        "version": RECEIPT_VERSION,
        "python": sys.executable,
        "environment": inventory["environment"],
        "binding": inventory,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".setup-receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(receipt, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_receipt(path: Path, *, display: str) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text())
        if (
            receipt["version"] != RECEIPT_VERSION
            or receipt["preflight"] != "passed"
            or not isinstance(receipt["binding"], dict)
            or receipt["binding"].get("inventory_version") != 1
            or set(receipt["binding"])
            != {
                "inventory_version",
                "sdk",
                "dependencies",
                "graphics_bindings",
                "libero",
                "python",
                "environment",
            }
            or not isinstance(receipt["environment"], dict)
            or set(receipt["environment"]) - ENVIRONMENT_FIELDS
            or not all(isinstance(v, str) for v in receipt["environment"].values())
            or not isinstance(receipt["python"], str)
        ):
            raise ValueError()
        unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        if receipt["receipt_sha256"] != _digest(unsigned):
            raise ValueError()
        if display == "window" and receipt["display"]["mode"] != "window":
            raise ValueError()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SetupReceiptError(
            "setup receipt is missing, incomplete or edited; rerun agent setup-simulator"
        ) from exc
    return receipt


def validate_current_receipt(path: Path, *, display: str) -> dict[str, Any]:
    receipt = read_receipt(path, display=display)
    if receipt["binding"] != runtime_inventory():
        raise SetupReceiptError(
            "simulator source/dependency/environment bytes changed; "
            "rerun setup and actual preflight"
        )
    return receipt


def credential_free_environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("TENON_API_KEY", "TENON_BASE_URL", "PYTHONPATH", "PYTHONHOME"):
        environment.pop(name, None)
    environment.update(overrides or {})
    return environment


def _check_process(command: list[str], environment: dict[str, str]) -> None:
    outcome = subprocess.run(
        command,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if outcome.returncode:
        # No arbitrary child output, environment or credential-bearing exception
        # arguments enter the agent log. The explicit local command can diagnose it.
        raise SetupReceiptError(
            "simulator qualification failed; rerun setup/preflight before any API use"
        )


@contextmanager
def qualified_simulator(
    setup_file: Path, *, python: str | None, display: str
) -> Iterator[dict[str, Any]]:
    """Qualify selected Python before the CLI constructs any REST client."""
    with tempfile.TemporaryDirectory(prefix="tenon-simulator-preflight-") as temporary:
        receipt_path = setup_file.resolve()
        if python is not None:
            receipt_path = Path(temporary) / "setup.json"
            _check_process(
                [
                    python,
                    "-m",
                    "tenon_pi.simulator_setup",
                    "--preflight",
                    "--output",
                    str(receipt_path),
                    "--display",
                    display,
                ],
                credential_free_environment(),
            )
        receipt = read_receipt(receipt_path, display=display)
        executable = receipt["python"]
        _check_process(
            [
                executable,
                "-m",
                "tenon_pi.simulator_setup",
                "--validate-receipt",
                str(receipt_path),
                "--display",
                display,
            ],
            credential_free_environment(receipt["environment"]),
        )
        yield {
            "python": executable,
            "environment": receipt["environment"],
            "receipt_path": receipt_path,
        }
