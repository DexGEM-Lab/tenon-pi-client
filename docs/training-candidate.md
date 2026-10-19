# Candidate training contract

This unpublished SDK candidate requires an endpoint implementing the amended
training contract. The stable0.2 client and its ordinary workflow remain usable;
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
