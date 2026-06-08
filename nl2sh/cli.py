"""Command-line interface for NL2SH."""

from __future__ import annotations

import argparse
import subprocess
import sys

from nl2sh.converter import Converter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nl2sh",
        description="Convert natural language to shell commands using an LLM.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        metavar="QUERY",
        help="Natural-language description of the command to generate.",
    )
    parser.add_argument(
        "-e",
        "--execute",
        action="store_true",
        help="Execute the generated command after displaying it.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start an interactive session (REPL).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=None,
        help="OpenAI model to use (default: gpt-4o-mini or NL2SH_MODEL env var).",
    )
    return parser


def _run_command(command: str) -> int:
    """Execute *command* in a subshell and return its exit code."""
    result = subprocess.run(command, shell=True)
    return result.returncode


def _single_shot(converter: Converter, query: str, execute: bool) -> int:
    """Handle a single-query invocation. Returns an exit code."""
    try:
        command = converter.convert(query)
    except ValueError as exc:
        print(f"nl2sh: {exc}", file=sys.stderr)
        return 1

    print(command)
    if execute:
        return _run_command(command)
    return 0


def _interactive(converter: Converter, execute: bool) -> int:
    """Run an interactive REPL. Returns an exit code."""
    print("NL2SH interactive mode. Type 'exit' or press Ctrl-D to quit.")
    while True:
        try:
            query = input("nl2sh> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        try:
            command = converter.convert(query)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        print(command)
        if execute:
            _run_command(command)

    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``nl2sh`` command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.query and not args.interactive:
        parser.print_help()
        return 0

    try:
        converter = Converter(model=args.model)
    except ValueError as exc:
        print(f"nl2sh: {exc}", file=sys.stderr)
        return 1

    if args.interactive:
        return _interactive(converter, execute=args.execute)

    return _single_shot(converter, args.query, execute=args.execute)


if __name__ == "__main__":
    sys.exit(main())
