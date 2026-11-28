"""Run ``python -m ocsid {features,train,attribute,evaluate,fusion} --help``."""

import importlib
import sys

COMMANDS = {
    "features": "operation_builder",
    "train": "training",
    "attribute": "audit",
    "fusion": "fusion",
    "evaluate": "evaluation",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        return 0
    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}. Choose: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    module = importlib.import_module(f"ocsid.{COMMANDS[command]}")
    return module.main(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
