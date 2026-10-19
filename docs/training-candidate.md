# Candidate training contract

This unpublished SDK candidate requires an endpoint implementing the amended
training contract. The stable 0.2 client and its ordinary workflow remain usable;
new training/capability features are not claimed live by installing this candidate.

`TrainingSpec` and quote/job create bodies accept optional:

- `effective_batch_size`: a positive integer selected by the user, describing
  samples per optimizer update. The quoted/requested value must remain unchanged
  in job and checkpoint provenance. Execution planning belongs to the service.
- `training_mode`: `standard` or `wan_rtc`. The latter requests a genuinely
  prefix-conditioned training objective and requires matching service capability.
  A deployment flag cannot relabel an ordinary checkpoint.

The CLI exposes these as `--effective-batch-size` and `--training-mode` on
`training spec`, `training quote`, `training run` and `agent train`. Omitting them
preserves the legacy request shape; the SDK does not inject new null/default
fields into old requests. Explicit null remains distinct in low-level SDK wire
serialization, as permitted by the amended schema.

For a new WAN workflow, provide the user's explicit effective batch. If a default
is proposed, state it and obtain the user's approval; do not derive it silently
from a dataset or recipe name. An unavailable service capability is an explicit
error, not a reason to change the batch or training objective.

Continue an existing run using the same context and the same arguments. Legacy
contexts without the new fields keep their original bound input/spec shapes and
pending mutation keys/bodies. Changing batch, training mode, steps or budget
requires a new intended-run context, never editing the old journal or reviving a
failed job. Coarse execution segments, when supported by the service, remain one
logical job; queued progress is not a new training attempt created by the agent.

## Recommended WAN workflow

Use one local asynchronous control path: explicit user batch → `wan_rtc` training
→ verified matching checkpoint → `trained_rtc` deployment/session → local runner.
Do not switch objectives or modes when capacity is unavailable. The endpoint must
support the amended contract and the actual checkpoint/model/worker combination;
a catalog enum by itself is insufficient. These features are **not live merely
because the candidate is installed**.

1. Inspect the user's **local** dataset, prepare the supplied profile and verify
   the whole bundle. Use existing configured access; do not ask for a dataset
   download address or an account handle.
2. Before spending, run `agent setup-simulator` for the simulator, or
   `agent preflight-robot` for the supplied calibrated adapter as documented in
   [configured robot](configured-robot.md). Fix genuine local capture/driver
   problems before opening a paid control session. Shared service lifecycle
   belongs to its operator; colleague development uses reviewed PRs.
3. Obtain the total spending ceiling, separate training/control caps, and the
   user's effective batch. The example below uses **8 only if the user selected
   or explicitly approved 8**; its steps and caps are also examples, not advice
   about convergence or throughput.

```bash
tenon-pi agent train .tenon/prepared \
  --schema .tenon/prepared/robot-schema.json --name 'my task' \
  --method full_ft --max-steps 100 --max-runtime-seconds 7200 \
  --checkpoint-interval-steps 100 --effective-batch-size 8 \
  --training-mode wan_rtc --max-cost-microcredits 5000000 \
  --context .tenon/wan-run.json

tenon-pi agent deploy --context .tenon/wan-run.json \
  --name 'my task' --prompt 'place the object in the tray' \
  --execution-mode trained_rtc --duration-seconds 300 \
  --max-cost-microcredits 1000000
```

4. The journal verifies ready artifact lineage, exact user batch/objective,
   schema/transform/normalization and versioned RTC metadata. Deployment checks
   compatible horizon/prefix capacity. The service must validate the saved model
   and match an actual qualified worker. Wait for an **active** session; a queued
   response is not serving readiness. Retain the same journal during coarse
   queued/running progress rather than creating another logical job.
5. Launch `agent start-robot` or `agent start-simulator` using that context and
   current local preflight. The launcher uses the admitted mode; unchanged warm
   turnaround, horizon/frequency, request deadline and bounded fallback checks
   still apply. Empty-prefix warm samples cannot certify nonempty-prefix WAN
   latency. Prefix modifications invalidate conditioned work and require reset/
   rearm; never weaken timing limits to make a run appear successful.
6. Report real observations, applied commands and stop reason, then close the
   session with its current resource version. Local stop-all remains the supplied
   adapter's responsibility. CPU fixtures are neither physical commissioning nor
   trained-policy quality evidence.

## One prompt

> Read AGENTS.md. My demonstrations are in `./my-data` and my API key is already
> configured. Use my supplied calibrated adapter, or the simulator I requested.
> My effective batch is **[BATCH]** and total budget is **[TOTAL] microcredits**.
> Inspect/prepare locally and pass local adapter/simulator preflight before
> spending. Use the recommended WAN-trained workflow only with a qualified
> endpoint/checkpoint; show separate quotes within my total cap and retain one
> run context. Do not restart shared services or replace a failed job. Ask only
> for physical semantics or permissions genuinely missing from local metadata,
> and report actual run outcomes rather than queued or fixture success.

Existing ordinary runs remain on their original journal/spec/mode. Their omitted
new fields stay omitted; use the stable 0.2 workflow when no new capability is
requested. Updating the SDK invalidates byte-bound simulator setup receipts:
rerun setup/preflight instead of editing receipts.
