"""Run the recipe with a shell-quoted args file, without evaluating a shell."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args-file", type=Path, required=True)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    recipe = Path(__file__).resolve().parents[1] / "run_qwen3_agentcore_math.py"
    command = [
        sys.executable, str(recipe),
        *shlex.split(args.args_file.read_text(), comments=True), *extra,
    ]
    if args.print_command:
        print(shlex.join(command))
    else:
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
