# Reference benchmark results

Recorded 2026-09-27 in the ChatGPT Work Linux environment using Python 3.12.14,
NumPy 2.3.5, a 512-dimensional hashing encoder, three heads, one short text state,
100 warm-up iterations, and 10,000 measured iterations.

| Metric | Result |
|---|---:|
| Median | 0.3063 ms |
| p95 | 0.5467 ms |
| p99 | 1.3003 ms |

Run `python -m benchmarks.latency --iterations 10000` to reproduce on another host.
These figures cover in-process Python inference only. They exclude service transport,
serialization, queueing, model loading, and application side effects. They must not be
presented as results for an ONNX transformer or as a universal sub-30 ms guarantee.

