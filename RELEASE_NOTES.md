# v0.1.0-alpha.1

First runnable FastPath AI reference release.

## Included

- Pydantic-native `Choice`, `Score`, and `Noul` decision contracts
- Contract fingerprinting and fail-closed artifact compatibility
- Shared deterministic encoder with parallel linear heads
- Temperature scaling, Brier score, expected calibration error, and abstention
- Reproducible ticket-triage training example
- CLI, latency benchmark, tests, and GitHub Actions workflows

## Explicit non-claims

- This release does not contain a 3B foundation model or enterprise-ready weights.
- The included hashing encoder is a reference kernel, not a transformer.
- Calibration is empirical and distribution-dependent. Brier loss does not guarantee that a
  reported 0.82 confidence is correct exactly 82 percent of the time.
- Sub-30 ms is a benchmark target, not a universal guarantee. Hardware, input length, encoder,
  and concurrency determine latency.

