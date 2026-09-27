from __future__ import annotations

import argparse

import numpy as np

from fastpath import FastPathEngine
from fastpath.metrics import brier_score, expected_calibration_error
from fastpath.training import load_jsonl

from .schema import InboundTicketTriage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="examples/ticket_triage/tickets.jsonl")
    parser.add_argument("--model", default="examples/ticket_triage/fastpath-ticket-triage-v0.json")
    args = parser.parse_args()
    records = load_jsonl(args.data)
    records = records[int(len(records) * 0.8) :]
    engine = FastPathEngine(args.model, abstain_threshold=0.0)
    choice_options = ["Infrastructure", "Billing", "Security", "General"]
    choice_index = {value: index for index, value in enumerate(choice_options)}
    outputs = [engine.evaluate(record["state"], InboundTicketTriage) for record in records]

    predicted_categories = np.asarray([choice_index[item.category.value] for item in outputs])
    category_labels = np.asarray([choice_index[record["category"]] for record in records])
    category_probabilities = np.asarray(
        [[item.category.probabilities[option] for option in choice_options] for item in outputs]
    )
    page_probabilities = np.asarray([item.requires_tier3_page.prob for item in outputs])
    page_labels = np.asarray([record["requires_tier3_page"] for record in records], dtype=int)
    urgency_predictions = np.asarray([item.urgency.value for item in outputs])
    urgency_labels = np.asarray([record["urgency"] for record in records])

    choice_confidence = category_probabilities.max(axis=1)
    print(f"records={len(records)} split=calibration")
    print(f"category_accuracy={(predicted_categories == category_labels).mean():.4f}")
    print(f"category_brier={brier_score(category_probabilities, category_labels):.4f}")
    print(
        "category_ece="
        f"{expected_calibration_error(choice_confidence, predicted_categories == category_labels):.4f}"
    )
    print(f"page_accuracy={((page_probabilities >= 0.5) == page_labels).mean():.4f}")
    print(f"page_brier={brier_score(page_probabilities, page_labels):.4f}")
    print(f"urgency_mae={np.abs(urgency_predictions - urgency_labels).mean():.4f}")
    print("warning=synthetic calibration split; not evidence of production generalization")


if __name__ == "__main__":
    main()
