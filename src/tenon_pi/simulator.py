#!/usr/bin/env python3
"""Run one LIBERO episode through explicit async_aligned or guided_rtc control.

Run this inside an environment where the official OpenPI LIBERO example and its
pinned LIBERO dependency are installed. The connection JSON comes from:

    tenon-pi session connection ctl_... > connection.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from tenon_pi.client import TenonClient
from tenon_pi.control import ControlSocket, NativeWebSocketBackend
from tenon_pi.libero_runtime import LIBERO_ACTION_NAMES, LiberoAsyncRollout
from tenon_pi.models import (
    ControlConnection,
    ExecutionMode,
    FallbackPolicy,
    ZeroDeltaPreserve,
)
from tenon_pi.runtime import AsyncAlignedController, RuntimeStopped
from tenon_pi.simulator_display import SimulatorDisplay
from tenon_pi.simulator_receipt import validate_current_receipt


def environment(task, resolution: int = 256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=20,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    connection_args = parser.add_mutually_exclusive_group(required=True)
    connection_args.add_argument("--connection-json", type=pathlib.Path)
    connection_args.add_argument("--session-id")
    schema = parser.add_mutually_exclusive_group()
    schema.add_argument("--schema-hash")
    schema.add_argument("--deployment-json", type=pathlib.Path)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--setup-receipt", type=pathlib.Path, required=True)
    parser.add_argument(
        "--display",
        choices=("auto", "window", "headless"),
        default="auto",
        help="Live camera window; window requires working Tk/desktop, auto reports fallback.",
    )
    parser.add_argument(
        "--execution-mode",
        required=True,
        choices=(
            ExecutionMode.ASYNC_ALIGNED.value,
            ExecutionMode.GUIDED_RTC.value,
            ExecutionMode.TRAINED_RTC.value,
        ),
    )
    args = parser.parse_args()

    validate_current_receipt(args.setup_receipt, display=args.display)
    if sys.platform == "darwin" and os.environ.get("MUJOCO_GL") in {"egl", "osmesa", "glx"}:
        raise RuntimeError(
            "Linux EGL/OSMesa renderer settings cannot be reused on macOS; "
            "configure the simulator's supported native renderer"
        )
    display = SimulatorDisplay(args.display)
    try:
        print(json.dumps({"simulator_display": display.summary()}), file=sys.stderr)
        result = run_episode(args, display)
    finally:
        display.close()
    print(json.dumps({**result, "display": display.summary()}, indent=2, sort_keys=True))


def run_episode(args, display: SimulatorDisplay) -> dict:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    env = environment(task)
    try:
        env.seed(7)
        env.reset()
        raw = env.set_init_state(suite.get_task_init_states(args.task_id)[args.trial])
        if not display.show(raw):
            return {"steps": 0, "success": False, "stopped_reason": "viewer_closed"}
        return run_connected_episode(args, env, raw, task, display)
    finally:
        env.close()


def run_connected_episode(args, env, raw, task, display: SimulatorDisplay) -> dict:
    # Recheck after environment/viewer initialization and before any REST call
    # or reading a token file. Never rely only on the parent process check.
    validate_current_receipt(args.setup_receipt, display=args.display)
    if args.session_id:
        with TenonClient(
            os.environ["TENON_API_KEY"],
            base_url=os.environ.get("TENON_BASE_URL", "https://tenon.dex-gem.ai/v1"),
        ) as client:
            session = client.get_control_session(args.session_id)
            if session.state != "active":
                raise ValueError(f"session is {session.state}; no simulator actuation started")
            deployment = client.get_deployment(session.deployment_id)
            if args.schema_hash and args.schema_hash != deployment.schema_hash:
                raise ValueError("deployment schema hash differs from verified run context")
            args.schema_hash = deployment.schema_hash
            connection = client.create_control_connection(session.id)
    else:
        connection = ControlConnection.model_validate_json(args.connection_json.read_text())
    schema_hash = args.schema_hash
    if args.deployment_json is not None:
        schema_hash = json.loads(args.deployment_json.read_text())["schema_hash"]
    assert isinstance(schema_hash, str)
    execution_mode = ExecutionMode(args.execution_mode)
    fallback = FallbackPolicy(
        mode="drain_then_hold",
        hold_steps=5,
        hold_behavior=ZeroDeltaPreserve(preserve_channels=["gripper"]),
    )
    with ControlSocket(connection) as socket:
        hello = socket.native_handshake(
            schema_hash=schema_hash,
            expected_mode=execution_mode,
        )
        backend = NativeWebSocketBackend(
            socket,
            expected_mode=execution_mode,
            action_dim=7,
            max_committed_prefix_steps=int(hello["limits"]["max_committed_prefix_steps"]),
            authoritative_epoch=int(hello["epoch"]),
        )
        controller = AsyncAlignedController(
            backend,
            expected_mode=execution_mode,
            controller_hz=20,
            state_dim=8,
            action_dim=7,
            action_horizon_steps=10,
            action_names=LIBERO_ACTION_NAMES,
            action_semantics="cartesian_delta",
            fallback=fallback,
            action_lower=(-1.0,) * 7,
            action_upper=(1.0,) * 7,
            committed_steps=3,
            max_server_processing_ms=400,
        )
        try:
            rollout = LiberoAsyncRollout(
                env, controller, prompt=str(task.language), on_frame=display.show
            )
            raw = rollout.stabilize(raw, steps=10)
            result = rollout.run(raw, max_steps=args.max_steps)
        except RuntimeStopped as exc:
            return {
                "steps": 0,
                "success": False,
                "stopped_reason": str(exc),
                "execution_mode": execution_mode.value,
            }
        finally:
            controller.close()
    return {**result.__dict__, "execution_mode": execution_mode.value}


if __name__ == "__main__":
    main()
