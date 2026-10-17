# Client 0.2 integration contract

## Wire boundary

REST, control MessagePack, `RobotSchema`, enums and request serialization are
unchanged from `spec/openapi.json`. A profile is a **local data artifact** carried
inside an ordinary whole-bundle dataset upload, never an added REST field.
Local preparation proves that data match a declared mapping. It does not prove
that the service offers a model or training recipe for that mapping.

## Python and CLI surface

- `RobotProfile.load(path)` / `.model_validate(value)` validate the data-only
  `tenon-robot-profile-v1` descriptor. Native state and action mappings may have
  different widths; each physical coordinate is mapped exactly once. Canonical
  slots may be reordered or constant. Units, calibration, gripper endpoints,
  camera dimensions, fps, model delta indices and padding widths are explicit.
- `prepare_dataset(source, output, input_space="native" | "canonical",
  dataset_repo_id="namespace/name", profile=profile)` produces a `PreparationResult`.
- `verify_preparation(root)` regenerates statistics, schema and digest bindings;
  `build_bundle_manifest(root)` enumerates the **whole** verified root.
- CLI: `data inspect`, `data fixture`, `data prepare --profile ...`, `data verify`,
  `data manifest`. Install the `data` extra for Arrow/Pillow.
- Control: `RobotController(driver, mapping, limiter, initial_applied)` requires
  a connected/armed supplied adapter and explicit per-channel bounds/slew.
  `CommandEncodingReceipt.from_canonical(command, profile.action)` preserves the
  canonical token on exact native echo; native byte changes remain independent
  of inverse float32 quantization. Pass `result.native_changed` to runtime
  `record_applied(..., command_changed=...)`.

## New bundle identity and service dispatch

```
preparation-manifest.json          version: tenon-preparation-v2
robot-schema.json                 frozen robot-schema-v1
assets/robot-profile-v1/
  transform.json                  canonical validated profile descriptor
  norm_stats.json                 exact OpenPI-compatible JSON bytes
  normalization-manifest.json     version: tenon-normalization-v2
dataset/                          original/canonical LeRobot source
```

`build_bundle_manifest` emits `tenon-prepared-bundle-v2`. The normalization
manifest declares `state_width` and `action_width` separately, actual horizon,
row counts, source content/transform/normalization digests and fixed statistics
schedule. Hashes use sorted compact UTF-8 JSON, `ensure_ascii=False`, no NaN.
Normalization JSON retains its independent exact pinned serializer.

Consumers must dispatch on the preparation **version**, validate the descriptor,
regenerate semantics with the matching client version and explicitly resolve an
accepted model/recipe. Do not accept arbitrary profile data merely because its
hash is internally consistent. New profile IDs are not model catalog entries.
`Capabilities` offers model/method information, not a profile catalog. Quote and
admission remain authoritative; unknown layouts must produce explicit findings.

Retain the previous release's verifier and physical adapters privately for
previously published bundles/checkpoints. Their markers, asset names, schema
hashes, normalization identity and checkpoint lineage must remain unchanged.
Do not rename old artifacts or regenerate old digests under the new descriptor.
The two bundle generations require separate dispatch, not a string alias.
Existing LIBERO assets and digests remain unchanged.

## Data semantics

- v2.0/global and v2.1/episode statistics stay distinct. Native conversion changes
  only state/action columns and corresponding feature metadata/statistics; PNG
  cells, timestamps, indices, tasks and episode order are preserved.
- v3.0 accepts canonical vectors and declared RGB video cameras under default
  templates. Source rows, media, task/episode metadata and offsets remain exact.
  Native-v3 conversion, depth, mixed image/video layouts and transcoding are not
  implemented. Local verification binds video bytes/metadata; the service must
  still decode/validate media before declaring readiness.
- Model normalization applies float32 transformations, then one float64 state
  update and one float64 horizon-action update per source row. Horizon windows
  contain consecutive actions from the same episode, with last-action padding
  only at its tail. Do not regroup histogram updates.

Abstract test layouts: a rotary planar mechanism (native state/action 3/3,
canonical 4/4 with a virtual slot, one camera, 20 Hz, horizon 4) and a linear
xyz mechanism with independent load sensing (native 5/4, canonical 5/4, two
camera shapes, 10 Hz, horizon 3). These synthetic examples establish mechanics,
not physical calibration or hosted model availability.

## Resumable agent operations

`agent train --context .tenon/run.json` binds local file manifests, schema,
account and endpoint identity, and the training spec. Mutation keys are written
before requests; exact file-completion bodies/ETags are written before completion.
Resuming re-reads public state and reuses IDs instead of uploading or training
again. An interrupted PUT without a complete receipt stops for reconciliation;
it is never silently repeated. Context ownership uses POSIX file locking and
atomic mode-0600 writes. Journals belong outside upload roots and Git.

`agent deploy --context ...` rechecks checkpoint lineage, creates/reuses one
deployment and quoted session, and requires actual `active` session state.
`agent start-simulator --context ... --python SIMULATOR_PYTHON` starts the real
installed LIBERO runner, using an in-memory one-time connection token. A fresh
session is required; terminal sessions are never resurrected. The supplied
simulator Python must have this client and the documented simulator dependencies.
No journal or local check establishes a WAN latency guarantee or policy quality.

A persisted training/session create key means its budget approval already ran.
If the response was lost, replay uses that same key/body without requiring the
already-reserved credits a second time. Ambiguous mutation replay is refused
after the contract's guaranteed 24-hour idempotency window (or when its creation
time is absent); reconcile public resource state rather than minting a new key.

## Configured-robot runner

`agent start-robot` consumes the verified profile, existing context session,
explicit `RobotRunSettings` and trusted `module:factory` adapter. No profile or
REST field changed. The adapter supplies synchronized named native state/RGB
captures with host-monotonic timestamps, exact capture provenance, actual native
command receipts and idempotent stop-all. Camera `rotation_degrees` is validated
provenance; pixels are not silently transformed. See `configured-robot.md`.

## Descriptor-bound output projection

The unpublished profile artifact now includes `output_projection`, default
`strict`, with optional `clip_bounds_v1`. It changes canonical descriptor/digest
identity; regenerate candidate generic bundles and retain historical identities.
Validate every incoming state/action coordinate before trim/virtual overwrite,
invert against captured state, restore virtual coordinates, reject inverse
NaN/Inf, then project finite native-protocol outputs only when explicitly selected.
Raw compatibility output remains unprojected. No demonstration/observation or
ordinary received command is clipped, and no RobotSchema/REST field is added.
