# Local async-aligned control runtime

`AsyncAlignedController` owns native `tenon-control-v1` action chronology while
the caller owns the control clock and final actuator/environment call. The name
is retained for API compatibility; construction now requires an explicit
`async_aligned`, `guided_rtc` or `trained_rtc` expected mode. Sampling remains
server functionality; the client implements the unchanged prefix wire, index
alignment, actuator-truth tracking and reset/rearm. Trained mode requires a
WAN-trained checkpoint and a qualified service/deployment. Mode relabeling and
fallback to a different sampler are rejected. `openpi-ws-v1` remains a separate raw-upstream
compatibility path.

## Admission before motion

`warm_start()` submits stationary observations before the first action. It
measures full turnaround and refuses to arm unless the conservative requirement
fits the declared horizon:

```text
required_steps = ceil(max_warm_turnaround × controller_hz) + safety_steps
2 × required_steps + 1 <= action_horizon_steps
```

No `env.step` or hardware command occurs before this passes.

This is a local admission check over observed warm samples, not a WAN latency
SLA. In either conditioned mode those probes have an empty committed prefix, so they measure
ordinary sampler turnaround; they do not bound the extra cost of nonempty prefix
conditioning. Simulator control frequency is its configured simulation timebase,
not proof of wall-clock real-time execution. Measure the actual operating case.

## Explicit modes and prefix wire

Handshake, backend and controller must receive the same expected mode. The
client rejects a different hello or response generation mode.

- `async_aligned` never sends `committed_prefix`; it preserves locally committed
  indices and splices only the still-future suffix.
- `guided_rtc` and `trained_rtc` send the existing frozen `committed_prefix` exactly covering
  `[target_start_action_index, committed_end_action_index_exclusive)`. Its hash
  is SHA-256 over row-major little-endian float32 action bytes. Even the warm
  zero-prefix request carries an explicit empty segment.
- A guided response is accepted only when mode, consumed count, exact prefix
  hash and `sampler_conditioning="guided"` match the request.
- A trained response requires the same exact count/hash evidence with
  `sampler_conditioning="trained"`; guided evidence cannot satisfy it.
- `synchronous_chunked`, missing prefixes, prefixes in async mode and mode or
  conditioning relabeling fail explicitly.

The REST client checks the specific ready checkpoint's `training_mode=wan_rtc`,
versioned RTC metadata/shape, artifact lineage, endpoint capability and deployed
mode. The service must verify the saved model and match an actual capable worker;
a global enum does not qualify a checkpoint. See [candidate workflow](training-candidate.md).

No private sampler, guidance weights, VJP, or fallback-to-baseline logic exists
in this client.

## Indexed alignment

At most one request is in flight. A request records:

- epoch and request/observation index;
- observation-aligned target start;
- locally committed future boundary;
- reply-needed and buffer-end indices.

On response, the splice index is:

```text
max(local_next_to_execute,
    request.committed_end_action_index_exclusive,
    response.start_action_index)
```

Elapsed entries are discarded, committed future actions are never overwritten,
and the remaining uncommitted future is atomically replaced. Wrong epoch,
request, observation, start, shape, dimension, generation mode or conditioned prefix
evidence, non-finite values, and actions outside deployment physical bounds are
dropped/faulted before execution. The client does not silently clip a native
response.

## Requested versus actually applied

`next_action()` returns a requested command. Before the next tick, the caller
must pass the actuator/environment-returned command to `record_applied()`. The
runtime cannot infer what hardware applied and makes no tracking guarantee
beyond this supplied value.

With a supplied profile, canonical and native units may use different
command representations. Their float32 round trip is intentionally not used as
an equality test. `CommandEncodingReceipt` snapshots the exact canonical
command token and its exact native encoding before any mutable limiter/driver
call. `NativeCommandTrace` separately snapshots the post-limiter native
payload submitted and the native payload returned by the driver:

- exact submitted-native echo of the receipt encoding confirms the original
  canonical command token without a second lossy conversion;
- a limiter-changed submitted payload or driver-changed returned payload is
  converted from that actual native value and sets an independent exact
  `native_changed` signal;
- the bridge passes that signal as `record_applied(..., command_changed=True)`,
  so conditioned work invalidates even when inverse quantization maps a distinct
  native float32 payload to the same canonical float32 vector;
- no numerical tolerance weakens prefix hashes or bounds, and no fake canonical
  vector is manufactured merely to force invalidation.

This receipt proves command representation continuity, not physical joint state.
Physical-state tracking still requires real telemetry owned by the installation.

For guided and trained modes, every future row sent as a committed prefix is retained until
its application is reported. If a local limiter or driver changes one of those
rows, the conditioned cloud suffix is causally invalid: the client clears it,
drops any affected in-flight result, enters `rearm_required`, and serializes an
existing reset behind the one in-flight request. Reset ACK must advance the
epoch before fresh-observation warm admission. The changed **applied** command
becomes fallback truth; the requested prediction is never reported as applied.
Async mode sends no value prefix, but still records applied truth for fallback.

## Serialized reset ownership

Ordinary reset and prefix-override reset use the same one-backend/socket lane. If
inference is still running, its `Future` remains owned and its response is
consumed by that original backend call. One reset task waits behind it, then
sends exactly one existing reset frame and receives exactly one ACK. Reset never
starts a concurrent socket `recv`.

`mark_disconnected()` never treats `Future.cancel()` as proof that a running
receive stopped. A receive that could not be cancelled stays explicitly owned;
warming is blocked, and any later reset waits for that owner even when the caller
injected a multiworker executor.

`reset(reason, timeout_seconds=5)` locally disarms immediately. If its caller
wait expires before the backend lane is free, it raises `ResetPending` and
`reset_pending` stays true; timeout is not treated as completion. Epoch remains
unchanged. Calling `reset` again while pending waits on the same reset index/task
and cannot emit a second frame. Only an ACK with exactly `old_epoch + 1`
finalizes reset and permits warm admission. A failed/invalid ACK leaves server
epoch unknown and requires recovery rather than forging local progress.

The default local wait is 5 seconds even when a backend receive timeout is
longer. This bounds caller waiting without abandoning in-flight ownership.

## Disconnect and rearm

Accepted actions may drain. Then the pinned fallback applies for at most
`hold_steps`:

- delta semantics: all deltas become zero while named channels such as gripper
  retain the last **actually applied** value;
- absolute joint position: repeat the last target actually returned by the local
  driver, not an unexecuted prediction.

Fallback ticks advance the local action index. Exhaustion stops. On reconnect,
the client first accepts the authoritative epoch from the new server `hello`
(including process-restart or multi-step epoch jumps), then clears old cloud
work/buffer, sends explicit fresh-observation rearm in that epoch, and runs warm
admission again. It never guesses the reconnect epoch from a local disconnect
count and never resumes from an old response.

## LIBERO

`LiberoAsyncRollout` is the only object that calls `env.step`; inference runs on
a separate worker. It:

- preserves official 20 Hz simulator control;
- rotates and resize-pads the two 256×256 views to 224×224;
- sends JPEG camera payloads and the exact 8D state;
- uses seven-dimensional Cartesian-delta actions strictly bounded to `[-1,1]`;
- falls back to zero six motion deltas plus the last applied gripper command.

The normal launcher revalidates the byte-bound setup receipt before API use,
checks the context-created active session and server-authoritative schema hash,
then obtains its one-time connection token in memory:

```bash
tenon-pi agent start-simulator --context .tenon/run.json \
  --setup .tenon/simulator/setup.json --display window \
  --task-suite libero_10 --task-id 0
```

For the WAN workflow use `agent deploy --execution-mode trained_rtc` only with
a verified WAN-trained checkpoint and qualified service; the launcher
reads the admitted mode from that context and never relabels it. For an explicit
custom interpreter, `--python` triggers fresh local qualification rather than
bypassing the receipt. Output is measured rollout evidence, not a benchmark claim.
See [setup and receipt coverage](simulator-setup.md).

## Supplied-profile robot boundary

`examples/robot_control_boundary.py` consumes the prepared schema and a
`RobotController` constructed with your action mapping, connected/armed driver,
calibration bounds and per-channel slew. Camera capture, acquisition skew limits,
control clock, workspace/collision rules and stop behavior belong to the actual
installation. The SDK does not supply device-specific defaults.

The bridge encodes a canonical command through `profile.action`, applies the
local limiter and driver, then records driver-returned commands through the
immutable receipt. Native state/action widths, canonical widths, virtual slots,
gripper endpoints and model delta mappings all come from the profile. State and
action dimensions need not match. See [preparation](robot-preparation.md).

Profile-based observation cameras use the supplied shapes and PNG encoding;
request-size admission comes from negotiated hello limits, never a new
RobotSchema field. A hosted adapter must explicitly implement the accepted
profile's model transforms and output bounds; client preparation alone does not
establish that adapter's existence.

Driver exceptions fault the local controller and call the supplied adapter's
`stop()`. Multi-actuator adapters must stop every owned actuator on partial
failure. Command receipts are not joint-state telemetry or collision guarantees.
