# Tenon Pi: coding-agent entrypoint

For a user who supplies demonstrations and an API key and wants a trained policy
running in a simulator or configured robot, read
`skills/tenon-pi-agent/SKILL.md` and execute it. This is an SDK workflow, not a
request to build a service.

- Start with `tenon-pi data inspect DATA`; preserve source data.
- Own simulator installation and CPU renderer/window preflight with `agent setup-simulator`
  before paid resources; follow `docs/simulator-setup.md`, not a preconfigured-Python assumption.
- For a supplied calibrated controller, run `agent preflight-robot` before spending,
  then `agent start-robot`; see `docs/configured-robot.md`.
- API key: environment only. Service endpoint must be configured/available;
  never claim access merely because a default URL exists.
- Infer supported import paths from metadata. Ask for unknown units/calibration,
  hardware setup, task meaning, user effective batch and a spending ceiling; do not guess them.
- Candidate WAN workflow: `docs/training-candidate.md`. Require actual endpoint/
  checkpoint qualification for `wan_rtc` → `trained_rtc`; never relabel or fall back.
  New features are not claimed live. Existing ordinary contexts retain their mode.
- Shared service lifecycle belongs to its operator. Development uses reviewed PRs,
  not edits/restarts of the shared service. Local data/access stay with the user.
- Use `.tenon/run.json` to resume the same upload/job/deployment/session. Do not
  replace failures with a fresh run or silently spend again.
- Use current capabilities and quotes. Local profile validity does not imply
  hosted model support. Ready checkpoint, active session and successful episode
  are separate evidence boundaries.
- Real hardware needs supplied calibration/controller limits and a genuinely
  connected, armed driver. Profiles are data, never executable adapters.
- Keep user data, journals, tokens and credentials outside committed source.

English instructions: `docs/quickstart.md`. 中文：`docs/quickstart.zh-CN.md`。
Hardware-free prompt: `docs/demo-prompt.md`. For code changes, preserve
`spec/openapi.json` and run `pytest`; never add fields to the frozen wire casually.
