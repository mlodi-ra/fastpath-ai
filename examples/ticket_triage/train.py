from pathlib import Path

from fastpath.training import load_jsonl, train_artifact

from .schema import InboundTicketTriage


def main() -> None:
    root = Path(__file__).parent
    train_artifact(
        load_jsonl(root / "tickets.jsonl"),
        InboundTicketTriage,
        root / "fastpath-ticket-triage-v0.json",
        model_id="fastpath-ticket-triage-v0",
        dimensions=512,
    )


if __name__ == "__main__":
    main()
