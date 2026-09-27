from __future__ import annotations

import argparse
import statistics
import time

from examples.ticket_triage.schema import InboundTicketTriage
from fastpath import FastPathEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="examples/ticket_triage/fastpath-ticket-triage-v0.json")
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    engine = FastPathEngine(args.model)
    state = "Production DB CPU spiked to 99%. Connections exhausted in us-east-1."
    for _ in range(100):
        engine.evaluate(state, InboundTicketTriage)
    timings = []
    for _ in range(args.iterations):
        started = time.perf_counter_ns()
        engine.evaluate(state, InboundTicketTriage)
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    timings.sort()
    print(f"iterations={len(timings)}")
    print(f"median_ms={statistics.median(timings):.4f}")
    print(f"p95_ms={timings[int(len(timings) * 0.95)]:.4f}")
    print(f"p99_ms={timings[int(len(timings) * 0.99)]:.4f}")


if __name__ == "__main__":
    main()
