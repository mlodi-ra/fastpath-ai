"""Generate a deterministic toy dataset. It proves plumbing, not production fitness."""

import json
import random
from pathlib import Path

TEMPLATES = {
    "Infrastructure": [
        "Production {asset} CPU at {level}% and connections exhausted in {region}",
        "{asset} is unavailable; timeout rate is {level}% in {region}",
        "Network packet loss affecting {asset} in {region}",
    ],
    "Billing": [
        "Customer reports duplicate invoice {asset} for ${level}",
        "Payment was charged twice on account {asset}",
        "Refund missing for billing account {asset}",
    ],
    "Security": [
        "Suspicious login from {region} for privileged account {asset}",
        "Malware alert detected on endpoint {asset}",
        "Possible credential theft and unauthorized access to {asset}",
    ],
    "General": [
        "How do I update the display name for {asset}",
        "Request documentation for service {asset}",
        "User asks where to find training for {asset}",
    ],
}


def main() -> None:
    random.seed(7)
    records = []
    for category, templates in TEMPLATES.items():
        for index in range(80):
            level = random.randint(10, 100)
            urgent = category in {"Infrastructure", "Security"} and index % 2 == 0
            priority_signal = "critical immediate page" if urgent else "routine standard queue"
            records.append(
                {
                    "state": priority_signal
                    + ". "
                    + random.choice(templates).format(
                        asset=f"svc-{index % 13}",
                        level=level,
                        region=random.choice(["us-east-1", "west", "EU"]),
                    ),
                    "category": category,
                    "urgency": 5
                    if urgent
                    else (3 if category in {"Infrastructure", "Security"} else 2),
                    "requires_tier3_page": urgent,
                }
            )
    random.shuffle(records)
    output = Path(__file__).with_name("tickets.jsonl")
    output.write_text("\n".join(json.dumps(item) for item in records) + "\n")


if __name__ == "__main__":
    main()
