from __future__ import annotations

import argparse
import importlib
import json

from .engine import FastPathEngine


def _object(reference: str):
    module_name, object_name = reference.split(":", 1)
    return getattr(importlib.import_module(module_name), object_name)


def main() -> None:
    parser = argparse.ArgumentParser(prog="fastpath")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="evaluate one state")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--schema", required=True, help="module:Class")
    evaluate.add_argument("--state", required=True)
    evaluate.add_argument("--abstain-threshold", type=float, default=0.60)
    args = parser.parse_args()
    if args.command == "evaluate":
        result = FastPathEngine(args.model, args.abstain_threshold).evaluate(
            args.state, _object(args.schema)
        )
        print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
