# Data-first quickstart

Install from the repository with `pip install '.[data]'` (Python 3.11+). Set
`TENON_API_KEY` without echoing it and configure `TENON_BASE_URL` for your invited
endpoint. An API key does not establish endpoint availability or model support.
Use `tenon-pi --help` for exact command options.

## 1. Inspect and prepare

```bash
tenon-pi data inspect ./my-data
```

Choose from evidence, not a filename guess:

- **Prepared bundle:** inspection verifies it. Upload the entire root, including
  `dataset/`, assets, schema and preparation manifest.
- **LeRobot v2.0/v2.1 RGB PNG or v3.0 canonical RGB video:** use an explicitly
  supplied profile as described in [preparation](robot-preparation.md).
- **Explicit ordered JSONL plus PNGs:** `data import-jsonl` imports the supported
  record schema; then prepare the imported data with the same profile.
- **Other raw data:** inspect metadata/examples and propose an explicit converter.
  Preserve originals; compare rows, pixels, timestamps and episode/action order.
  Unknown units or missing calibration need the user, not a heuristic.
- **LIBERO:** follow [the public acquisition example](libero-demo.md). Its pinned
  schema and materializer are separate from arbitrary robot profiles.

```bash
tenon-pi data prepare ./my-data --profile ./profile.json \
  --input-space native --repo-id owner/task -o .tenon/prepared
tenon-pi data verify .tenon/prepared
tenon-pi data manifest .tenon/prepared -o .tenon/bundle.json
```

Inspect the manifest before upload. Profile IDs describe transforms, not hosted
model entitlements. The service may reject a locally valid layout.

## 2. Set up the simulator before spending

For a simulator target, the agent runs the owned [pinned setup](simulator-setup.md)
now—not after creating a training job or session:

```bash
tenon-pi agent setup-simulator --directory .tenon/simulator --display window
```

This provisions the isolated Python/dependencies/source, creates the real CPU
simulator, renders both cameras and verifies the requested window. Its v2 receipt
binds actual installed SDK/dependency/LIBERO bytes; startup refuses old or drifted
receipts before API use. An advanced `--python` override still runs fresh preflight. Continue only
with a successful setup receipt; ask for missing OS/display permission rather
than assuming a preconfigured simulator Python. For configured hardware, verify
the supplied trusted adapter and calibration instead; see [runner](configured-robot.md).

## 3. Train once; resume the same run

```bash
tenon-pi account capabilities
tenon-pi account credits
tenon-pi agent train .tenon/prepared \
  --schema .tenon/prepared/robot-schema.json --name 'my task' \
  --method full_ft --max-steps 100 --max-runtime-seconds 7200 \
  --checkpoint-interval-steps 100 --max-cost-microcredits 5000000 \
  --context .tenon/run.json
```

The numbers are an example budget/configuration, not a performance prescription.
For LIBERO, supply its materialized root and generated schema instead. The agent
command recognizes that schema and selects the pinned public recipe. Other
profiles use the service's `balanced-v1` default; quote/admission must accept it.
Explicit service-supplied `--base-model`/`--preset` overrides remain available.
No profile catalog endpoint is invented.

The command uploads, waits for actual dataset readiness, displays the quote and
asks before spending. `--yes` is only for a budget already approved by the user.
For LoRA use `--method lora`; the service must advertise the named compatible
profile. No arbitrary rank/alpha fields are added.

Re-run the **same command/context** after interruption. The private context binds
account/endpoint, input manifest/schema, specs, IDs, quotes and mutation keys.
It retains exact completed-file receipts before sending completion, then resumes
against fresh public state. It does not store API keys or signed URLs. Keep it
outside upload roots and Git. POSIX file locking prevents concurrent run writers.
An interruption during a PUT without a complete receipt, an expired quote or an
ambiguous operation beyond the service's idempotency retention requires explicit
reconciliation; do not delete the journal and start another paid job.

A failed/canceled job remains failed/canceled. Successful jobs must reference
ready checkpoints with matching dataset, schema, transform, normalization,
method, model and artifact metadata before this workflow continues. That is a
server-readiness/manifest check, not an independent download of all weight bytes.

## 4. Deploy and launch

Choose a separate control-session budget within the total authorized spending:

```bash
tenon-pi agent deploy --context .tenon/run.json --name 'my policy' \
  --prompt 'place the block in the tray' --execution-mode async_aligned \
  --duration-seconds 300 --max-cost-microcredits 1000000
```

This rechecks checkpoint lineage and reuses deployment/session IDs. A ready
deployment does not mean a warm runtime; the command requires an actually active
session. Failed/expired sessions are reported, never silently replaced.

For LIBERO, use the setup receipt created before spending:

```bash
tenon-pi agent start-simulator --context .tenon/run.json \
  --setup .tenon/simulator/setup.json --display window --task-suite libero_10 --task-id 0
```

`--display window` opens a live local Tk window showing the simulator's agent
and wrist camera frames. Closing it stops local stepping. It fails explicitly if
GUI setup is unavailable; it never substitutes a saved video. The environment
and viewer initialize before consuming the one-time connection token.

Use `--display auto` to try a desktop window and report a headless fallback, or
`--display headless` to deliberately run without a window. Startup and result JSON
report the selected mode, reason and presented-frame count. For a visible demo,
keep `window`; a headless result does not satisfy that requirement.

The launched process obtains a one-time connection token in memory and uses the
service-returned WSS URL. Missing simulator dependencies fail explicitly. A new
run requires a deliberately admitted fresh session; old buffers never resume.
After the episode, close the session with its current resource version using
`session get` then `session close --resource-version VERSION`.

For hardware, supply a trusted local driver, actual calibration, per-channel
bounds/slew, synchronized capture and installation stop/collision controls.
[The packaged runner](configured-robot.md), `agent start-robot`, consumes the supplied
profile, trusted adapter factory and explicit limits. It captures synchronized
observations and runs the full hosted-control loop; the adapter must be actually
connected and explicitly armed. No hardware connection is fabricated.

Record real episode outcomes and lineage. `evaluation.state=ready` certifies
structural validation, not task success or policy improvement.
