# API snapshot provenance

`openapi.json` is the approved Tenon Pi v0.1 client wire surface from
`b644badcd8edc2a97170755a5bb10ead20879107`, with the narrow canonical browser
bootstrap clarification:

- `POST /auth/activate` and `POST /auth/login` retain their original `User`
  responses;
- `GET /me` returns a **flat** `SessionBootstrap`: all `User` fields plus the
  top-level `csrf_token` (`null` for API-key authentication);
- `GET /me` returns `Cache-Control: no-store`.

The explicitly approved `b474c653` extension accepts `lerobot_v2` and
`lerobot_v3`. Native v2 is restricted to `tenon-native-v1`; the optional partner
profile remains v3-only. No other request or response field changed.

The snapshot retains the approved `max_cost_microcredits`, `TrainingSpec`,
`CreditBalance`, `Capabilities`, and dataset-version shapes. It is the public
client projection, so non-client topology and worker-transport vendor extensions
are omitted. Further wire changes require separate approval.
