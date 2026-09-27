# Architecture decision record

## Decision

FastPath v0.1 separates four concerns:

1. A Pydantic contract compiler defines the only valid output space.
2. A shared encoder converts input state into one fixed-size vector.
3. Independent projection heads produce choice, bounded score, or binary outputs.
4. Calibration and abstention convert raw logits into an operational decision envelope.

The application owns thresholds and actions. FastPath never executes a side effect.

## Why the alpha uses feature hashing

Training and shipping a credible 3B transformer is outside the scope of this alpha. It would
require licensed data, substantial compute, evaluation across multiple domains, weight
distribution, and security review. The reference hashing encoder lets maintainers validate
contracts, artifact format, calibration, abstention, packaging, and reference-kernel performance.

The next backbone should be a small encoder-only transformer exported to ONNX. Calling the model
"non-autoregressive" is technically correct but not novel by itself. The product contribution is
the schema compiler, multi-head artifact, deployment boundary, and evaluation discipline.

## Runtime invariants

- One encoding call per state.
- No vocabulary decoder and no output-generation loop.
- No network access during evaluation.
- Contract mismatch fails before inference.
- Every head carries a confidence or uncertainty measure.
- Low-confidence results can be rejected rather than silently routed.

## Threat model

The runtime reduces invalid-output risk but does not remove adversarial input, poisoning, drift,
model extraction, or biased labels. Model artifacts must be treated as trusted executable inputs.
Production deployments should add signed artifacts, digest pinning, input limits, audit events,
rate limits, and per-head rollback.
