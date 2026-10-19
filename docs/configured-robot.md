# Run an explicitly configured robot

`agent start-robot` is the packaged capture → hosted inference → limited native
actuation loop. It consumes the existing active session in the run context; it
does not train a new model, arm hardware or select a device by name.

```bash
tenon-pi agent start-robot --context .tenon/run.json \
  --profile .tenon/prepared/assets/robot-profile-v1/transform.json \
  --settings ./robot-run.json --adapter my_installation.tenon_adapter:create \
  --adapter-config ./local-controller.json --prompt 'place the block in the tray'
```

The `module:factory` is **explicit trusted local code**, selected separately from
profile data. It is imported in the CLI process, receives `profile` and `config`
keyword arguments, and returns an adapter implementing `RobotAdapter` from
`tenon_pi.robot_runtime`. Neither a profile nor a downloaded dataset can name
code to execute. Keep configuration/credentials local; no key is passed through
subprocess arguments or written to the journal.

## Local preflight before spending

Use the same trusted adapter, profile and explicit limits before training or
opening a paid control session:

```bash
tenon-pi agent preflight-robot \
  --profile .tenon/prepared/assets/robot-profile-v1/transform.json \
  --settings ./robot-run.json --adapter my_installation.tenon_adapter:create \
  --adapter-config ./local-controller.json --prompt 'place the block in the tray'
```

This invokes no REST client and sends no policy/motion commands. It verifies
actual connected/explicitly armed state, initial applied target, calibrated
bounds/slew and three fresh aligned camera/state captures, then calls `stop()`
on every exit. The trusted adapter's import, capture and stop methods remain
installation-owned code. This is an I/O contract check, not physical commissioning
or a reusable arming receipt: the operator must explicitly rearm before the later
run, which repeats capture and measured network admission checks.

## Adapter contract

The supplied installation must implement:

- `connected`, `armed`: actual current device state, never unconditional flags.
  The runner does not connect or arm an unidentified machine on the user's behalf.
- `stop_requested`: the installation's cancellation/E-stop/window-stop request.
- `last_applied_action()`: all native **action** channel names mapped to the last
  driver-reported command. State and action dimensions may differ. This is not
  physical-position telemetry.
- `capture(after_action_index=...) -> RobotCapture`: all native state channels
  by name, state capture timestamp, exact camera records, calibration digest,
  and the actual action index after which capture occurred.
- `send_action(dict) -> dict`: send the supplied native action channels and return
  the actual driver-applied commands, including any driver modification.
- `stop()`: idempotently attempt to stop **every owned actuator**, even if an
  earlier actuator's stop fails. Surface an aggregate failure after all attempts.
  A multi-device adapter owns its partial-write and group-stop mechanism.

Each `CameraCapture` contains original RGB uint8 pixels of the declared dimensions,
its exact `CaptureCamera` provenance and an acquisition timestamp. Camera order
is resolved by explicit IDs; state order by explicit profile channel names.
`rotation_degrees` is **capture provenance**, not a runtime rotation instruction.
Pixels must already have the same orientation as the profile-bound recordings;
`orientation="profile"` attests that. The runner verifies metadata equality and
never guesses camera roles, rotates, resizes or flips ambiguous pixels.

All timestamps must be translated by the trusted adapter into this process's
`time.monotonic_ns()` domain using known clock calibration. The capture declares
`timestamp_clock="host_monotonic_ns"`; original device clock identity remains in
camera provenance. Do not replace stale device timestamps with the current time.
The runner checks freshness of every source, explicit age/skew limits, non-future
samples and capture after the reported action application. Unknown clock
alignment is an error, not a reason to trust coincident wall clocks.

## Explicit execution settings

`robot-run.json` contains `lower`, `upper` and `max_delta` arrays in the profile's
native **action** order, plus `max_capture_age_ms`, `max_capture_skew_ms`,
`max_steps` and optionally `max_server_processing_ms` (default 500). Bounds must
lie within measured profile calibration; slew is per control tick. Supply actual
installation limits rather than copying synthetic example values. The JSON model
rejects unknown fields. Source timing and the frozen RobotSchema are unchanged.

Three fresh stationary captures warm the runtime before any command application.
The existing measured admission gate must pass. The loop then captures, requests
an action, encodes an immutable canonical/native receipt, applies local limits,
records the driver-returned command and retains the independent native-change
signal. A guided or trained prefix override stops for explicit reset/rearm; the runner never
forges an epoch or launches a competing receive. Existing reset ownership is
retained while cleanup drains outstanding work.

Every exit—including partial actuation, capture failure, stale/skewed observation,
exhausted fallback, max steps, stop request, SIGINT, SIGTERM and exceptions—calls
adapter `stop()` before waiting for network cleanup. Stop errors are surfaced.
The result reports actual steps, observations, overrides and stop reason;
`physical_tracking_verified` remains false. No generic task-success evaluator or
physical commissioning is invented. Afterward close the control session using
its fresh public resource version to release admission/billing.

## Scope of the tests

CPU fixtures implement two different adapters: a direct rotary mechanism and an
independent motor-bank/jaw controller with additional load telemetry and two
camera shapes. They run complete loops over the public socket transport, with
failure injection and stop-all checks. These fixtures establish SDK integration,
not functioning physical hardware. Service model/profile support is separate
from the client's ability to run a supplied validated mapping.
