# Profile-driven data preparation

`RobotProfile` is explicit JSON data, validated without executing expressions or
loading code. It defines a transformation, not a hosted model entitlement.
See [the schema](../schemas/robot-profile.schema.json) and synthetic
[planar](../examples/profiles/planar.json) / [gantry](../examples/profiles/gantry.json)
examples. Never use synthetic calibration on a physical machine.

## Profile fields

- `profile_version`, `profile_id`, actual `calibration_digest`.
- Independent `state` and `action` mappings. Each has ordered `native` channels
  with names, units and measured bounds, and ordered `canonical` coordinates.
  Each physical coordinate is retained exactly once; permutation and additional
  constant virtual coordinates are supported.
- Per-coordinate conversion is one of `identity`, `degrees_to_radians`,
  `millimeters_to_meters`, `gripper_closedness`, or `constant`. Identity requires
  matching units. Grippers require explicit distinct `native_closed` and
  `native_open` endpoints spanning their calibrated native range; output is
  closedness `[0,1]`. Unknown units are rejected, not inferred.
- `action_semantics` uses the existing RobotSchema enum. Explicit
  `model_delta_state_indices` associate absolute joint actions with the captured
  state coordinates subtracted in model space; `null` preserves an absolute
  coordinate. These mappings are distinct from the uploaded canonical actions.
- `model_state_width` / `model_action_width` allow zero padding without dropping
  source coordinates. Normalization is over the unpadded canonical widths.
- Cameras declare ID, actual width/height, serial, mount, capture clock,
  orientation and calibration digest. `image_transform`, `controller_hz`,
  `action_horizon_steps` and `max_committed_prefix_steps` are explicit.

Feature names/order must exactly match the supplied mapping. Preparation checks
calibration bounds and virtual values; it never clips, guesses a scale, resamples
time or silently changes image encoding. The resulting schema contains only
frozen `robot-schema-v1` fields and units.

## Supported source formats

| Source | Supported behavior |
|---|---|
| v2.0 RGB PNG | Native or canonical state/action; global `meta/stats.json` |
| v2.1 RGB PNG | Native or canonical state/action; `meta/episodes_stats.jsonl` |
| v3.0 RGB video | Canonical state/action only, default path templates, complete consecutive episodes |
| Explicit JSONL + PNG | Import the record schema below to v2.1, then prepare |

v2 requires one sorted Parquet per episode, contiguous frame/global indices,
matching task/episode metadata and embedded Arrow Image/Sequence/Value tags.
Native conversion rewrites only state/action vectors and their feature metadata,
then recomputes every numeric feature and real sampled-pixel statistic. Existing
optional statistics files are updated too; no empty cross-version file is added.
Consecutive timestamps are checked at the declared profile rate with the pinned
100-microsecond tolerance. The original timestamp values remain unchanged.

v3 preserves original rows, task/episode metadata, encoded videos and timebase.
Videos may be shared across episodes at non-keyframe offsets; data/video file
boundaries may differ. Metadata ownership, offsets, duration and per-row timing
are checked. Local verification binds video bytes and metadata, but does not
replace hosted container/decoded-frame validation. Native-v3 conversion,
v2 video/JPEG, mixed/depth variants and custom v3 templates are unsupported.
LIBERO has a separate supported import path with its own temporal provenance.

## Prepare and verify

```bash
tenon-pi data prepare ./source --profile ./profile.json \
  --input-space native --repo-id owner/recording -o .tenon/prepared
tenon-pi data verify .tenon/prepared
tenon-pi data manifest .tenon/prepared -o .tenon/bundle.json
```

Use `--input-space canonical` only when recorded vectors already have exactly
the profile's canonical layout/units. `--repo-id` is a stable logical
`namespace/name`, never a filesystem path, URL or cache locator. Source data
must be immutable during preparation. Output must be new and outside the source.

The bundle contains `dataset/`, `assets/robot-profile-v1/`, `robot-schema.json`
and `preparation-manifest.json`. Upload/train from the **whole bundle root**.
`verify_preparation` regenerates statistics, schema and all bindings, so changing
statistics and recomputing superficial hashes does not pass verification.

For each source row, action normalization uses consecutive same-episode horizon
targets; only the episode tail repeats its final action. Float32 transforms are
followed by one float64 state and one float64 action-window update per row.
Per-feature LeRobot metadata and model normalization are separate statistics.
Canonical JSON digests and exact normalization byte serialization are preserved.

## Raw JSONL import

Each line has exactly these keys (illustrative one-camera, three-native-coordinate
record; values and camera ID must match **your** profile):

```json
{"episode_index":0,"frame_index":0,"timestamp":0.0,"state":[0,0,50],"action":[1,2,40],"task":"place block","images":{"view_0":"frames/000.png"}}
```

```bash
tenon-pi data import-jsonl ./raw/rows.jsonl --profile ./profile.json \
  --input-space native -o .tenon/imported
tenon-pi data prepare .tenon/imported --profile ./profile.json \
  --input-space native --repo-id owner/recording -o .tenon/prepared
```

Episodes and frames must be consecutive from zero, in source order. PNG paths
are relative to the JSONL directory; traversal and symlinks are rejected. Original
PNG bytes are embedded unchanged. State/action numbers must be finite and exactly
float32-representable; other precision requires a separately approved conversion,
not an implicit downcast. Timestamp numbers retain float64 values. The importer
records original JSONL/image digests and compares the source again before publishing.

For other raw formats, an agent can write a converter after establishing the
source schema and conversion rules, then prove row/pixel/order preservation. A
successful file-format check alone is not that proof.

## Local actuation

`RobotController` consumes the action mapping, a supplied driver, explicit native
bounds and per-channel per-tick slew. It uses driver-returned command values as
truth. If a driver fails, it faults and calls the adapter's `stop()`; an adapter
controlling multiple actuators must stop every owned actuator after a partial
failure. No physical telemetry or group-stop mechanism is fabricated by the SDK.
See [control semantics](control-runtime.md) and the
[bridge example](../examples/robot_control_boundary.py).

## Explicit model-output projection

`output_projection` is `strict` (default) or `clip_bounds_v1`. It is serialized
in the canonical descriptor and changes the transform digest. It applies only
to unnormalized **model outputs** after canonical-width trimming, inverse deltas
against the captured state and restoration of virtual coordinates. All incoming
state/action coordinates must be finite before any trimming or overwrites;
inverse overflow must also be rejected before projection. `clip_bounds_v1`
clamps finite canonical outputs componentwise to the declared float32 action
bounds. `strict` leaves finite outputs unprojected for schema-bound rejection.
`native_target` refuses an out-of-bounds strict target; `canonical_target(...,
project=False)` retains raw finite compatibility output for comparison.

This never clips demonstrations, observations, ordinary native-unit conversions
or already-received canonical server commands. Regenerate unpublished candidate
bundles under the updated descriptor; historical artifact identities stay fixed.
