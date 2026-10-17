# Pinned LIBERO-10 mini conversion

The public source is LeRobot v2. Tenon accepts native v2 with
`tenon-native-v1`, so the immediate demo uses a lossless renumbered v2 artifact.
An optional canonical-v3 materializer through the public LeRobot 0.6.1 API is
also provided for workflows that need v3.

## Acquire the immutable source

Install the materialization extra. Native-v2 preparation works on the base
supported Python range; optional LeRobot 0.6.1 v3 conversion requires Python
3.12 or newer.

```bash
pip install '.[libero]'
tenon-pi libero schema -o libero-robot-schema.json
tenon-pi libero manifest -o libero10-mini-acquisition.json
tenon-pi libero acquire -o libero10-mini-source
```

The download is 348,445,346 episode bytes (332.3 MiB), 2,806 retained frames,
plus metadata. Every file is pinned to
`physical-intelligence/libero@a4336d589d589045d1c56423ffdf3b88a0e19b1f`
and verified with SHA-256. The source card declares CC-BY-4.0.

`tenon-pi libero schema` is generated from the shipped versioned profile:

```text
OpenPI revision:       215abfb217dbac7d5f1273282331b9b1866c0479
model/config:          openpi/pi05-libero / pi05_libero / horizon 10
normalization SHA-256: b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84
transform SHA-256:     81bd9293ace15b1a9209ddf514c79a88e1bf0c2a7b521a13b2449747644c20d6
raw descriptor SHA:    4157e65cc7d6ba97b585ac8ae5c63a21d4624f7322c865f83883eb1ca0cae58b
projection reference:  933f91496d2adb15fd591a3f8a04c950e364eab9
state/action:          8 / 7, cartesian_delta, 20 Hz
cameras:               base, left_wrist
image transform:       rotate source 180°, resize-pad 224×224 RGB uint8
```

The normalization JSON is copied byte-for-byte from the released
`pi05_libero` checkpoint asset. Tests recompute both hashes and compare the
command output with the shipped `schemas/libero-robot-schema.json`.

### Raw model output versus projected/applied command

The raw flow policy is preserved for numerical parity and may emit small finite
overshoots such as `-1.009` or `+1.009`. The strict native action schema remains
`[-1,1]`; its bounds are not widened.

The pinned deployment profile declares a separate, explicit physical output
projection after unnormalization and seven-dimension trimming:

```text
applied = componentwise_clip(raw_physical_action, -1, +1)
```

For the native `tenon-control-v1` protocol it applies to
`synchronous_chunked`, `async_aligned`, and `guided_rtc`, matching robosuite
1.4.1 controller saturation. Native local buffers, committed prefixes and
evaluation action sequences use the actually applied projected commands.

`openpi-ws-v1` remains raw upstream compatibility: the service returns the
unprojected OpenPI action chunk and the upstream client/actuator performs its
normal controller-side bounds handling. Raw compatibility output must never be
presented as native projected output.

Raw sampler output and native projected/applied commands are logged and compared
separately. The native public runtime never silently clips an out-of-bounds
response; if the deployment projection is absent or inconsistent, strict client
validation faults before `env.step`.

This describes only public projection math and provenance. Guided sampler
implementation remains server-private.

The rollout requires the admitted mode explicitly; it never relabels guided
responses as async:

```bash
tenon-pi agent start-simulator --context .tenon/run.json \
  --setup .tenon/simulator/setup.json --display window \
  --task-suite libero_10 --task-id 0
```

Select `agent deploy --execution-mode guided_rtc` only for a supported guided
session; the launcher reads that explicit mode from the run context. In that mode the
public client sends the existing exact committed-prefix count/actions/float32
hash and verifies the echoed conditioning evidence; all guidance math stays on
the server. `env.step` remains the application boundary, and only the command
reported to `record_applied` becomes fallback/prefix truth.

## Temporal meaning

The source isn't a uniformly sampled 10 Hz recording. Its provenance is:

1. a LIBERO environment configured at 20 Hz;
2. irregular removal of semantic no-op transitions during OpenVLA regeneration;
3. every retained step copied in order by OpenPI;
4. LeRobot timestamps generated as `frame_index / 10`.

Therefore the conversion preserves retained rows exactly and never applies a
second 20-to-10 Hz downsample. Original idle elapsed time cannot be recovered.

## Immediate native-v2 demo

```bash
tenon-pi libero materialize-v2 ./libero10-mini-source \
  -o ./libero10-mini-v2
tenon-pi dataset upload ./libero10-mini-v2 \
  --format lerobot_v2 \
  --compatibility-profile tenon-native-v1 \
  --schema ./libero-robot-schema.json \
  --name 'LIBERO-10 mini' \
  --wait
```

This path rewrites only episode/task/global indices, recomputes subset stats,
and verifies source/output Parquet image/state/action/timestamp/frame columns.
There is no temporal resampling. `lerobot_v2` cannot use the optional partner
data profile.

## Optional canonical v3

**Local export only for the native LIBERO recipe:** the current hosted native
LIBERO training route accepts v2, not this v3 export. Successful local
materialization or generic-format upload validation does not establish native
LIBERO-v3 training/deployment support. Use the v2 route for the one-prompt demo.
A future hosted v3 path requires explicit recipe/loader qualification.


```bash
tenon-pi libero materialize-v3 ./libero10-mini-source \
  -o ./libero10-mini-v3
```

The v3 converter:

- verifies every source size and SHA-256;
- selects source episodes `[0,1,2,6,7,8,10,12,14,27]`;
- decodes the source RGB and writes it with lossless v3 `image` features;
- preserves `state[8]`, `action[7]`, task text, timestamps, and row order;
- lets LeRobot v3 generate canonical contiguous episode/frame/global indices;
- finalizes the v3 dataset through `LeRobotDataset.finalize()`;
- reopens it with `LeRobotDataset` and compares every decoded RGB pixel and
  state/action row against the source;
- writes `meta/source_provenance.json` with source episode IDs, source file
  hashes, per-episode decoded-content hashes, revision, license, and explicit
  `implicit_resampling: false`.

The output is a local v3 artifact. A service may validate/store that format, but
the native LIBERO recipe remains v2-only. For an explicitly supported export/storage
workflow, never change its format label:

```bash
tenon-pi dataset manifest ./libero10-mini-v3 \
  -o ./upload-manifest.json
tenon-pi dataset upload ./libero10-mini-v3 \
  --compatibility-profile tenon-native-v1 \
  --schema ./libero-robot-schema.json \
  --name 'LIBERO-10 mini' \
  --wait
```

The service owns any later conversion from validated v3 to the pinned older
OpenPI runtime view. Native-v2 demo uploads need no client-side round trip.
Both uploaded artifacts remain immutable and independently digested.


## Agent-operated simulation

Use the short [demo prompt](demo-prompt.md). The agent owns setup **before paid
resources**; an existing simulator Python is not a user prerequisite:

```bash
tenon-pi agent setup-simulator --directory .tenon/simulator --display window
```

The [setup helper](simulator-setup.md) installs an isolated supported SDK/simulator,
pins the upstream source/dependencies, creates the real environment and verifies
its CPU renderer and requested window. Its success receipt binds actual installed
SDK/dependency/source bytes. Old, incomplete, edited or drifted receipts are
rejected before the launcher creates an API client; the simulator rechecks the
same binding before requesting or reading a one-time connection token. Rerun setup
and actual preflight after a code/environment change.

`agent train` selects the pinned recipe from the LIBERO schema; `agent deploy`
requires a verified ready checkpoint and active quoted session. Then launch using
the receipt—not an assumed executable path:

```bash
tenon-pi agent start-simulator --context .tenon/run.json \
  --setup .tenon/simulator/setup.json --display window --task-id 0
```

The official [OpenPI LIBERO example](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero)
is upstream provenance/reference, not a manual installation prerequisite for this
workflow. An empty-prefix guided warm probe does not measure nonempty conditioning
cost; no network-latency or wall-clock guarantee follows from that probe.

### Advanced explicit Python override

An installation owner may use `agent start-simulator --python /path/to/python ...`
with the same context and display options. That interpreter must already have the
current SDK, supported dependencies and pinned LIBERO configuration. The override
runs a **fresh credential-free real preflight** before any API use, creates a
private temporary byte-bound receipt and passes it to the child for revalidation.
It does not bypass qualification or trust a version string. Prefer the owned setup
receipt for normal use. Direct `python -m tenon_pi.simulator` calls require a
current `--setup-receipt`; manually supplied connection-token files are read only
after validation.

## Desktop window setup

For a visible local simulator, use the editable command:

```bash
tenon-pi agent start-simulator --context .tenon/run.json \
  --setup .tenon/simulator/setup.json --display window \
  --task-suite libero_10 --task-id 0
```

The window displays live agent/wrist camera frames from the actual local
simulator. Closing it stops local stepping. Policy inference stays hosted; no
model weights are loaded on the desktop. The viewer is a Tk/Pillow window on the
simulation thread, independent of the simulator's offscreen renderer.

- **macOS:** use a GUI-capable Python with matching Tcl/Tk (for example a suitable
  python.org distribution, or a matching Homebrew Python/Tk installation). Run
  `/path/to/simulator-python -m tkinter` on the logged-in desktop to test that
  interpreter's GUI. The SDK also needs Pillow (`.[data]`). Configure the pinned
  simulator's supported native renderer; do not inherit Linux-only
  `MUJOCO_GL=egl`, `osmesa` or `glx`. The launcher rejects those values on macOS instead
  of pretending that a Linux renderer configuration is portable. Native simulator
  support still depends on the installed LIBERO/robosuite/MuJoCo versions and
  architecture; this repository does not claim an executed macOS qualification.
- **Linux desktop:** use an interpreter with matching Tk libraries and an X11
  `DISPLAY` (including XWayland where configured). A distribution's `python3-tk`
  must match the interpreter actually selected; installing it for a different
  Python does not repair a managed environment. Test that Python with `-m tkinter`.
- **Headless Linux/SSH:** `--display headless` explicitly runs without a window.
  Use the setup helper's supported software GLX/OSMesa path. The viewer does not
  guess hardware support; setup verifies the actual Linux software renderer.
  `--display auto` attempts a desktop window and reports why it fell back; pure
  Wayland without an X11 display is not treated as working Tk/X11.

Startup and result JSON include `display.requested`, `mode`, `reason` and
`frames_presented`. A successful process exit alone is not proof that a window
opened or that the task succeeded. GUI rendering adds local overhead; no
wall-clock control guarantee follows. For media requiring a visible window, keep
`--display window`; a saved MP4 or composited window is a different artifact.
