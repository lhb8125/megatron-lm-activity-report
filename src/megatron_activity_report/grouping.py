"""High-confidence grouping of related dev/main pull requests."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Iterable


PR_REFERENCE = re.compile(r"(?<![A-Za-z0-9])#(\d+)\b")
CONTEXT_SIGNAL = re.compile(
    r"(?:main\s+pr|dev\s+pr|cherry[- ]?pick|ported?|reappl(?:y|ied)|mirror|counterpart|supersed)",
    re.IGNORECASE,
)
PREFIXES = re.compile(
    r"^(?:(?:\[[^\]]+\]\s*)|(?:cp|cherry[- ]?pick|reapply|port|dev|main|chore|chroe)\s*[:\-]\s*)+",
    re.IGNORECASE,
)


def build_change_groups(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    items = {int(record["number"]): record for record in records}
    parent = {number: number for number in items}
    evidence: list[dict[str, Any]] = []

    def find(number: int) -> int:
        while parent[number] != number:
            parent[number] = parent[parent[number]]
            number = parent[number]
        return number

    def union(left: int, right: int, reason: str, score: float) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        parent[max(root_left, root_right)] = min(root_left, root_right)
        evidence.append({"left": left, "right": right, "reason": reason, "score": score})

    for number, record in items.items():
        body = str(record.get("body") or "")
        for match in PR_REFERENCE.finditer(body):
            other = int(match.group(1))
            if other not in items or not _opposite_release_bases(record, items[other]):
                continue
            window = body[max(0, match.start() - 100) : match.end() + 100]
            similarity = _title_similarity(record, items[other])
            if CONTEXT_SIGNAL.search(window) or similarity >= 0.82:
                union(number, other, "explicit-reference", max(0.95, similarity))

    numbers = sorted(items)
    fingerprints: dict[str, list[int]] = defaultdict(list)
    for number in numbers:
        fingerprint = _patch_fingerprint(items[number])
        if fingerprint:
            fingerprints[fingerprint].append(number)
    for candidates in fingerprints.values():
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                if _opposite_release_bases(items[left], items[right]):
                    union(left, right, "patch-fingerprint", 1.0)

    for index, left in enumerate(numbers):
        for right in numbers[index + 1 :]:
            left_record, right_record = items[left], items[right]
            if not _opposite_release_bases(left_record, right_record):
                continue
            overlap = _file_overlap(left_record, right_record)
            if overlap < 0.5:
                continue
            left_title = _normalize_title(left_record.get("title", ""))
            right_title = _normalize_title(right_record.get("title", ""))
            similarity = difflib.SequenceMatcher(None, left_title, right_title).ratio()
            if left_title and left_title == right_title:
                union(left, right, "normalized-title", 0.98)
            elif similarity >= 0.94 and overlap >= 0.75:
                union(left, right, "title-files", round(similarity * overlap, 4))

    grouped: dict[int, list[int]] = defaultdict(list)
    for number in numbers:
        grouped[find(number)].append(number)
    result = []
    for _, members in sorted(grouped.items()):
        member_set = set(members)
        group_evidence = [
            item for item in evidence
            if item["left"] in member_set and item["right"] in member_set
        ]
        group = {
            "group_id": f"pr-{min(members)}",
            "numbers": sorted(members),
            "evidence": group_evidence,
        }
        group["content_hash"] = group_content_hash(group, items)
        result.append(group)
    return result


def group_content_hash(
    group: dict[str, Any], by_number: dict[int, dict[str, Any]]
) -> str:
    payload = []
    for number in group["numbers"]:
        record = by_number[int(number)]
        payload.append(
            {
                "number": int(number),
                "title": record.get("title", ""),
                "body": record.get("body", ""),
                "labels": record.get("labels", []),
                "base_ref": record.get("base_ref", ""),
                "opened": bool(record.get("opened")),
                "committed": bool(record.get("committed")),
                "merged": bool(record.get("merged")),
                "state_at_cutoff": record.get("state_at_cutoff", ""),
                "files": [file.get("path", "") for file in record.get("files", [])],
                "commits": [commit.get("sha", "") for commit in record.get("commits", [])],
            }
        )
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _opposite_release_bases(left: dict[str, Any], right: dict[str, Any]) -> bool:
    bases = {
        str(left.get("base_ref") or "").lower(),
        str(right.get("base_ref") or "").lower(),
    }
    return bases == {"dev", "main"}


def _normalize_title(value: Any) -> str:
    text = PREFIXES.sub("", str(value or "").strip().lower())
    text = re.sub(r"#\d+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _title_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    return difflib.SequenceMatcher(
        None, _normalize_title(left.get("title")), _normalize_title(right.get("title"))
    ).ratio()


def _file_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_paths = {file["path"] for file in left.get("files", [])}
    right_paths = {file["path"] for file in right.get("files", [])}
    union = left_paths | right_paths
    return len(left_paths & right_paths) / len(union) if union else 0.0


def _patch_fingerprint(record: dict[str, Any]) -> str | None:
    rows = []
    for file in record.get("files", []):
        patch = file.get("patch")
        if not patch:
            return None
        normalized_lines = []
        for line in str(patch).splitlines():
            if line.startswith("@@"):
                continue
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                normalized_lines.append(line[0] + re.sub(r"\s+", "", line[1:]))
        rows.append(f"{file['path']}\n" + "\n".join(normalized_lines))
    if not rows:
        return None
    return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()
