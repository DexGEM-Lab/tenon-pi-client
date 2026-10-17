#!/usr/bin/env python3
"""Supplied-profile native control boundary for explicit async_aligned/guided_rtc.

This is integration code, not hardware commissioning. The caller still owns
synchronized PNG capture, real calibration, collision/workspace limits,
emergency stop, driver construction, and the control clock.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from tenon_pi.control import ControlSocket, NativeWebSocketBackend
from tenon_pi.models import (
    ControlConnection,
    ExecutionMode,
    FallbackPolicy,
    LastAppliedAbsoluteTarget,
    RobotSchema,
)
from tenon_pi.robot import CommandEncodingReceipt, RobotController
from tenon_pi.runtime import AsyncAlignedController


@contextmanager
def open_robot_runtime(
    connection: ControlConnection,
    *,
    schema_hash: str,
    schema: RobotSchema,
    execution_mode: ExecutionMode,
) -> Iterator[AsyncAlignedController]:
    """Open one explicitly negotiated native controller; no mode fallback."""

    if schema.action_semantics != "joint_position":
        raise ValueError("this boundary requires absolute joint-position semantics")
    fallback = FallbackPolicy(
        mode="drain_then_hold",
        hold_steps=5,
        hold_behavior=LastAppliedAbsoluteTarget(),
    )
    with ControlSocket(connection) as socket:
        hello = socket.native_handshake(
            schema_hash=schema_hash,
            expected_mode=execution_mode,
        )
        backend = NativeWebSocketBackend(
            socket,
            expected_mode=execution_mode,
            action_dim=len(schema.action_channels),
            max_committed_prefix_steps=int(hello["limits"]["max_committed_prefix_steps"]),
            authoritative_epoch=int(hello["epoch"]),
        )
        controller = AsyncAlignedController(
            backend,
            expected_mode=execution_mode,
            controller_hz=float(schema.controller_hz),
            state_dim=len(schema.state_channels),
            action_dim=len(schema.action_channels),
            action_horizon_steps=schema.action_horizon_steps,
            action_names=tuple(channel.name for channel in schema.action_channels),
            action_semantics=schema.action_semantics,
            fallback=fallback,
            action_lower=tuple(float(channel.lower) for channel in schema.action_channels),
            action_upper=tuple(float(channel.upper) for channel in schema.action_channels),
            committed_steps=schema.max_committed_prefix_steps,
            max_server_processing_ms=1_000,
        )
        try:
            yield controller
        finally:
            controller.close()


def apply_requested_to_hardware(
    runtime: AsyncAlignedController,
    hardware: RobotController,
    requested_canonical: np.ndarray,
) -> np.ndarray:
    """Apply through real local limits, then report driver-returned actuator truth.

    Exact native echo confirms the immutable canonical command token without a
    lossy units round trip. A genuine limiter/driver byte change is converted
    from the returned native payload and reported independently even if canonical
    inverse quantization aliases; it can therefore invalidate guided work. This receipt
    is not physical-state telemetry or a hardware tracking guarantee.
    """

    # Receipt snapshots both canonical bytes and their encoded native payload
    # before any limiter or driver receives mutable values.
    receipt = CommandEncodingReceipt.from_canonical(requested_canonical, hardware.mapping)
    trace = hardware.apply_with_trace(receipt.encoded_native())
    result = receipt.resolve_application(trace)
    applied_canonical = result.canonical_array()
    runtime.record_applied(
        applied_canonical,
        command_changed=result.native_changed,
    )
    return applied_canonical
