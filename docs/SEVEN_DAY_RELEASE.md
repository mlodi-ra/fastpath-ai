# Seven-day release plan

The release is intentionally scoped to `v0.1.0-alpha`.

| Day | Exit criterion |
|---|---|
| 1 | Contract syntax, artifact format, architecture record, and non-claims agreed |
| 2 | Runtime supports Choice, Score, Noul, fingerprinting, and abstention |
| 3 | Reproducible trainer and synthetic enterprise triage example run end to end |
| 4 | Calibration metrics, holdout logic, malformed-input tests, and latency harness pass |
| 5 | Documentation, security guidance, packaging, and CI are complete |
| 6 | Fresh-environment install, benchmark, API review, and release-candidate freeze |
| 7 | Tag `v0.1.0-alpha.1`, publish artifacts, open known-limitations issues |

## Definition of done

- `python -m pip install -e ".[dev,train]"` succeeds on Python 3.10 through 3.13.
- The demo trains from source and returns only contract-valid outputs.
- All automated tests and lint checks pass.
- Warm benchmark reports median, p95, and p99 with host details recorded.
- Release notes state the hashing reference backbone and exclude a 3B weight claim.
- The repository contains no customer data, credentials, generated model secrets, or restricted
  teacher-model outputs.

## After alpha

The first material proof point is not Rust or TypeScript bindings. It is a head-to-head evaluation
against a prompted LLM, rules, and a conventional classifier on a real enterprise routing dataset.
If FastPath does not win on selective risk, latency, and total operating cost, further SDK work is
premature.

