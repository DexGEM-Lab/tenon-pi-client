---
name: tenon-pi-agent
description: Inspect and convert demonstrations, upload, train, verify checkpoints, deploy and launch a simulator or configured controller with Tenon Pi. Use when a user gives an API key and robot/simulator data and asks for a working policy or embodied demo.
---

# Data → trained policy → real episode

Use only the public SDK/CLI. Run commands from the repository root. References
below resolve from this skill directory. Read `../../docs/quickstart.md` (or
`../../docs/quickstart.zh-CN.md`) for the exact installed commands.

This working tree is an unpublished training-contract candidate. Read
`../../docs/training-candidate.md` for the recommended WAN workflow and current
capability boundary. Installing it does not enable new server features. Stable
0.2 ordinary runs retain their original specs and modes.

## Establish the user's run

- Read the request and existing `.tenon/run.json` before starting resources.
  Keep one context per intended run; do not delete it to escape an error.
- Use `TENON_API_KEY` and configured `TENON_BASE_URL`. Do not print credentials,
  environment dumps, connection tokens or signed URLs. A default URL is not
  proof of service availability. Never request service/storage internals.
- Establish task meaning, data location, target simulator or supplied controller,
  method, user-selected effective batch and total spending ceiling. Ask only for missing information that cannot
  be established from data/configuration. Split the ceiling between training and
  control; do not independently reuse the whole ceiling for both. Effective batch
  is the user's choice: disclose and confirm a proposed value, never derive it
  from a dataset recipe. Use existing local data/access; do not ask for account
  handles or a download address. Shared service lifecycle belongs to its operator;
  colleague development uses branches and reviewed PRs.

## Inspect, prepare and preserve

1. Run `tenon-pi data inspect DATA` before creating remote resources. Read
   `meta/info.json`, feature names/shapes, task text, episode/index/timestamp
   metadata and representative rows/images. Do not relabel unknown data as v3.
2. For an existing bundle, reverify it and retain the whole root. For explicit
   profiles read `../../docs/robot-preparation.md` and install `.[data]`.
   Use `data import-jsonl` only for its documented ordered record schema, then
   `data prepare --profile PROFILE --input-space native|canonical --repo-id OWNER/TASK`.
   Other formats require an explicit converter and source/output comparisons;
   do not guess units, calibration, action meaning or image orientation.
3. Run `data verify BUNDLE` and `data manifest BUNDLE -o .tenon/bundle.json`.
   Inspect all files before upload. Preserve original rows/pixels/timestamps,
   action order, episode ownership and transformation provenance. Never upload
   model-space deltas in place of declared physical actions or upload only
   `BUNDLE/dataset` and discard its bound assets.
4. Local preparation supports the documented formats only. v3 keeps original
   video bytes/offset metadata; actual media decoding and hosted acceptance
   remain service checks. Synthetic examples prove mechanics, not calibration.

For LIBERO, **own the setup before paid resources**: read
`../../docs/simulator-setup.md`, run `agent setup-simulator --display window`, and
inspect its actual renderer/real-environment/window preflight receipt. It creates
an isolated Python3.11 SDK/simulator, pinned LIBERO source and CPU dependencies;
no preconfigured simulator Python is required. Reconcile missing OS GL/Tk/display
with the user's host permission, not a fabricated readiness claim. Reuse a
successful owned setup only while its v2 byte-bound receipt remains current.
Startup revalidates SDK/dependency/LIBERO bytes before API/token use. Old/incomplete
or drifted receipts require setup and actual preflight again; never patch hashes
by hand. An advanced custom Python always runs fresh preflight, not a bypass.

Read `../../docs/libero-demo.md`. Reuse a verified existing acquisition;
otherwise run `libero schema`, `libero acquire`, then `libero materialize-v2`.
Its source timestamps have special provenance: never resample the retained
transitions. Use native v2 for hosted LIBERO; optional local v3 export is not admitted by the
current native LIBERO recipe. Missing simulator dependencies are not a reason to
create another training job. Keep defaults (100 steps,7200s cap,checkpoint interval100)
in the run plan, not the user prompt. Split one approved total ceiling between
training/control caps and account for prior authorized context; never invent a budget.

For a supplied robot adapter, run `agent preflight-robot` with its profile,
settings/config and trusted factory before spending. This checks real local
captures and bounds without API or policy commands, and stops all owned actuators.
The operator must explicitly rearm for the later run; preflight is not permission
to move or proof of physical commissioning.

## Execute the hosted lifecycle

1. Read `account capabilities` and `account credits`. `agent train` resolves the
   pinned LIBERO recipe from its schema; other layouts use the service default
   and must pass quote/admission. No unadvertised model/profile is assumed.
2. Run `agent train DATA --schema SCHEMA --name NAME --method full_ft|lora`
   with `--training-mode wan_rtc --effective-batch-size USER_BATCH`, explicit
   step/runtime/checkpoint limits, `--max-cost-microcredits` and
   `--context .tenon/run.json`. Show the quote maximum, method, resources, expiry
   and available credits. Use `--yes` only when that spending is already authorized.
3. Repeat the same command/context to resume. The journal binds inputs/specs,
   resources and pre-request idempotency keys, and persists exact file-completion
   receipts. A PUT interruption without a complete receipt, expired quote or
   ambiguity beyond idempotency retention needs reconciliation, never blind
   reupload/retrain. Failed/canceled tasks stay failed/canceled. Changes to batch,
   objective, steps or budget require a new intended-run context. Coarse queued/
   running transitions remain the same logical job, not a cue to create another.
4. Inspect the reported checkpoint's ready state, producing job, dataset/model,
   schema/transform/normalization digests, runtime revision and artifacts. Check
   exact effective batch, `training_mode=wan_rtc` and versioned RTC metadata. LoRA
   additionally requires its named profile and trainable manifest. These checks
   do not independently download all weights or establish policy quality.
5. Run `agent deploy --context .tenon/run.json --name NAME --prompt TASK`
   with `--execution-mode trained_rtc`, session duration and the separate control
   budget. Require matching endpoint/checkpoint/deployment/worker capability;
   never relabel a standard checkpoint or fall back to another mode. It
   rechecks checkpoint lineage and reuses deployment/session IDs. Wait for an
   actual active session; deployment.ready alone is insufficient.

## Start the requested environment

- **LIBERO:** run `agent start-simulator --context .tenon/run.json --setup .tenon/simulator/setup.json --display window`
  with the user's task suite/ID. Preflight Tk with that same Python on the user's
  desktop; see the Mac/Linux guidance in `../../docs/libero-demo.md`. This opens
  a live camera window and starts the installed real simulator runner,
  obtains an in-memory one-time token and uses the negotiated WSS URL verbatim.
  Inspect actual episode output, display mode/frame count and process failure.
  `window` must fail if a GUI cannot open. Only use `auto` or `headless` when the
  user accepts that fallback; label headless results explicitly, never treat a
  composited video window as SDK GUI evidence. Do not claim macOS execution from
  Linux tests. Close the session afterward
  with its fresh `resource_version`. Do not resurrect an expired session.
- **Configured hardware:** read `../../docs/control-runtime.md` (中文：
  `../../docs/control-runtime.zh-CN.md`) and `../../examples/robot_control_boundary.py`.
  Run `agent start-robot` with the supplied trusted `module:factory`, profile,
  adapter config and explicit run settings; read `../../docs/configured-robot.md`.
  This is the packaged capture/inference/actuation loop, not a request to write one.
  Use the supplied trusted driver and actual profile/calibration, explicit limits,
  synchronized capture and local stop/collision controls. Require genuinely
  connected and armed state; no synthetic driver can count as hardware execution.
  Multi-actuator drivers must stop all owned actuators on partial failure.
- In either path, requested actions are not applied actions. Record the actual
  driver/environment result. `CommandEncodingReceipt` binds exact canonical and
  native bytes; pass `native_changed` independently to `record_applied` so a
  quantization alias cannot preserve invalid conditioned work. On `ResetPending`,
  retain the same pending reset and receive owner; wait for its ACK before warm
  admission. Disconnect/reconnect never permits stale buffers or concurrent recv.

`async_aligned`, `guided_rtc` and `trained_rtc` are distinct. Trained replies must
carry `sampler_conditioning="trained"` and the unchanged exact prefix count/hash;
guided receipts cannot satisfy this. Empty-prefix warm probes measure ordinary
sampler turnaround, not the cost of nonempty conditioning or a WAN SLA.
Do not weaken the frozen timing protocol to make an episode look successful.

Report resource lineage, real task outcomes and concrete failures. Never present
repository fixtures, mocked responses, queued jobs or manifests as deployed
success. `evaluation.ready` means structural validation, not improved policy skill.
