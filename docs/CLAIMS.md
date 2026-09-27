# Claims register

This register separates hypotheses from demonstrated properties.

| Claim | Status in v0.1 | Evidence required |
|---|---|---|
| Outputs conform to the schema | Demonstrated by construction and tests | Contract tests |
| No output tokens are generated | Demonstrated by architecture | Runtime inspection |
| Deterministic decision values | Demonstrated for reference kernel | Repeatability tests |
| Sub-30 ms inference | Target, hardware-dependent | Named-host p50/p95/p99 benchmark |
| Well calibrated confidence | Model and dataset-dependent | Held-out Brier/ECE and reliability plot |
| Lower cost than an LLM | Workload-dependent hypothesis | End-to-end cost comparison |
| Better routing accuracy than an LLM | Unproven | Representative blinded evaluation |
| CPU/edge capable | Reference kernel is CPU capable | Target device benchmark |

The phrase "mathematically calibrated" is excluded. Calibration methods improve empirical
alignment on evaluation data; they cannot guarantee future correctness under distribution shift.

