"""Command-line interface for report generation and publication."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .config import ReportConfig
from .pipeline import ReportPipeline
from .schedule import run_scheduled
from .window import ReportWindow


DEFAULT_CONFIG = "configs/megatron-lm.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="megatron-activity-report",
        description="Generate bilingual, theme-oriented Megatron-LM activity reports.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--cutoff", required=True, help="inclusive YYYY-MM-DD cutoff")
    run_parser.add_argument("--final", action="store_true", default=None)
    _publication_flags(run_parser)

    backfill_parser = subparsers.add_parser("backfill")
    backfill_parser.add_argument("--month", required=True, help="YYYY-MM month")
    _publication_flags(backfill_parser)

    scheduled_parser = subparsers.add_parser("scheduled")
    _publication_flags(scheduled_parser)
    subparsers.add_parser("status")

    args = parser.parse_args(argv)
    config = ReportConfig.load(args.config)
    if args.command == "status":
        _print(ReportPipeline(config).status())
        return
    publish = bool(args.publish)
    if args.command == "run":
        window = ReportWindow.for_cutoff(
            args.cutoff, timezone_name=config.timezone, final=args.final
        )
        result = ReportPipeline(config).run(window, publish=publish)
    elif args.command == "backfill":
        window = ReportWindow.for_month(args.month, timezone_name=config.timezone)
        result = ReportPipeline(config).run(window, publish=publish)
    elif args.command == "scheduled":
        result = run_scheduled(config, publish=publish)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _print(result)


def _publication_flags(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true", help="commit reports and update the Issue")
    mode.add_argument("--dry-run", action="store_true", help="write runtime previews only")


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
