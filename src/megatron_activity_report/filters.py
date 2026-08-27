"""Conservative deterministic exclusions for report candidates."""

from __future__ import annotations

import collections
import re
from typing import Any, Iterable

from .config import ReportConfig


FORMAT_SIGNAL = re.compile(
    r"(?:^|\b)(?:auto[- ]?format|formatting|format|style-only|style only)(?:\b|:)",
    re.IGNORECASE,
)


def classify_records(
    records: Iterable[dict[str, Any]], config: ReportConfig
) -> list[dict[str, Any]]:
    return [classify_record(record, config) for record in records]


def classify_record(record: dict[str, Any], config: ReportConfig) -> dict[str, Any]:
    number = int(record["number"])
    paths = [str(file["path"]) for file in record.get("files", []) if file.get("path")]
    evidence: dict[str, Any] = {"paths": paths}
    if paths:
        in_tests = [_matches(path, config.test_prefixes) for path in paths]
        in_ci = [_matches(path, config.ci_prefixes) for path in paths]
        if all(in_ci):
            return _excluded(number, "ci_only", "all changed files are CI files", evidence)
        if all(in_tests):
            return _excluded(number, "test_only", "all changed files are test files", evidence)
        if all(test or ci for test, ci in zip(in_tests, in_ci)):
            return _excluded(
                number, "test_ci_only", "all changed files are test or CI files", evidence
            )

    labels = " ".join(str(label) for label in record.get("labels", []))
    signal_text = f"{record.get('title', '')} {labels}"
    if FORMAT_SIGNAL.search(signal_text) and _whitespace_only(record.get("files", [])):
        evidence["format_signal"] = signal_text
        return _excluded(
            number,
            "format_only",
            "explicit format signal and whitespace-equivalent patch",
            evidence,
        )
    return {
        "number": number,
        "category": "include",
        "excluded": False,
        "reason": "contains report-eligible changes",
        "evidence": evidence,
    }


def _excluded(
    number: int, category: str, reason: str, evidence: dict[str, Any]
) -> dict[str, Any]:
    return {
        "number": number,
        "category": category,
        "excluded": True,
        "reason": reason,
        "evidence": evidence,
    }


def _matches(path: str, prefixes: tuple[str, ...]) -> bool:
    normalized = path[2:] if path.startswith("./") else path
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in prefixes
    )


def _whitespace_only(files: Iterable[dict[str, Any]]) -> bool:
    saw_change = False
    for file in files:
        patch = file.get("patch")
        if not patch:
            return False
        removed: collections.Counter[str] = collections.Counter()
        added: collections.Counter[str] = collections.Counter()
        for line in str(patch).splitlines():
            if line.startswith(("+++", "---", "@@", "\\")):
                continue
            if line.startswith("+"):
                added[_normalize_whitespace(line[1:])] += 1
                saw_change = True
            elif line.startswith("-"):
                removed[_normalize_whitespace(line[1:])] += 1
                saw_change = True
        if removed != added:
            return False
    return saw_change


def _normalize_whitespace(line: str) -> str:
    return re.sub(r"\s+", "", line)
