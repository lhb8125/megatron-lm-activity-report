"""Collect PR snapshots and replay their state at a reporting cutoff."""

from __future__ import annotations

import concurrent.futures
import dataclasses
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator

from .github import GitHubClient, GitHubError
from .storage import ActivityStore
from .window import ReportWindow, parse_github_time


Progress = Callable[[str], None]


@dataclasses.dataclass(frozen=True)
class CollectedPullRequest:
    pull: dict[str, Any]
    files: list[dict[str, Any]]
    events: list[dict[str, Any]]
    commits: list[dict[str, Any]]


class ActivityCollector:
    def __init__(
        self,
        client: GitHubClient,
        store: ActivityStore,
        *,
        source_repo: str,
        workers: int = 8,
        progress: Progress = print,
    ):
        self.client = client
        self.store = store
        self.source_repo = source_repo
        self.workers = workers
        self.progress = progress

    def collect(self, window: ReportWindow) -> dict[str, int]:
        candidates = self._discover_candidates(window)
        changed = {
            number
            for number, updated_at in candidates.items()
            if self.store.cached_updated_at(self.source_repo, number) != updated_at
        }
        self.progress(
            f"discovered {len(candidates)} candidate PRs; fetching {len(changed)} changed snapshots"
        )
        for index, item in enumerate(self._fetch_all(changed), 1):
            self.store.replace_pr(
                self.source_repo, item.pull, item.files, item.events, item.commits
            )
            if index % 50 == 0 or index == len(changed):
                self.progress(f"stored {index}/{len(changed)} changed PR snapshots")

        rows: list[dict[str, Any]] = []
        for number in sorted(candidates):
            bundle = self.store.raw_bundle(self.source_repo, number)
            if bundle is None:
                raise RuntimeError(f"missing collected snapshot for PR #{number}")
            row = _window_row(
                bundle["pull"], bundle["events"], bundle["commits"], window
            )
            if row is not None:
                rows.append(row)
        self.store.replace_window_rows(self.source_repo, window.key, rows)
        return _counts(rows)

    def _discover_candidates(self, window: ReportWindow) -> dict[int, str]:
        start = window.start - timedelta(days=1)
        end = window.cutoff_exclusive + timedelta(days=1)
        candidates: dict[int, str] = {}
        for qualifier in ("created", "updated", "merged", "closed"):
            found = self._search_window(qualifier, start, end)
            self.progress(f"{qualifier}: {len(found)} candidates")
            candidates.update(found)
        return candidates

    def _search_window(
        self, qualifier: str, start: datetime, end: datetime
    ) -> dict[int, str]:
        query = _search_query(self.source_repo, qualifier, start, end)
        first = self.client.get(
            "/search/issues",
            params={"q": query, "per_page": 100, "page": 1, "sort": "updated"},
        )
        total = int(first.get("total_count") or 0)
        if first.get("incomplete_results") or total > 950:
            span = end - start
            if span <= timedelta(minutes=1):
                raise GitHubError(
                    f"GitHub search cap reached for {qualifier} in one-minute window"
                )
            midpoint = start + span / 2
            left = self._search_window(qualifier, start, midpoint)
            left.update(self._search_window(qualifier, midpoint, end))
            return left

        items = list(first.get("items") or [])
        page = 2
        while len(items) < total:
            response = self.client.get(
                "/search/issues",
                params={"q": query, "per_page": 100, "page": page, "sort": "updated"},
            )
            batch = list(response.get("items") or [])
            if not batch:
                break
            items.extend(batch)
            page += 1
        if len(items) < total:
            raise GitHubError(
                f"GitHub search returned {len(items)} of {total} {qualifier} PRs"
            )
        return {
            int(item["number"]): str(item.get("updated_at") or "") for item in items
        }

    def _fetch_all(self, numbers: set[int]) -> Iterator[CollectedPullRequest]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._fetch_one, number): number for number in numbers}
            for future in concurrent.futures.as_completed(futures):
                number = futures[future]
                try:
                    yield future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise GitHubError(f"failed to fetch PR #{number}: {exc}") from exc

    def _fetch_one(self, number: int) -> CollectedPullRequest:
        base = f"/repos/{self.source_repo}"
        return CollectedPullRequest(
            pull=self.client.get(f"{base}/pulls/{number}"),
            files=self.client.paginate(f"{base}/pulls/{number}/files"),
            events=self.client.paginate(
                f"{base}/issues/{number}/timeline",
                accept="application/vnd.github+json",
            ),
            commits=self.client.paginate(f"{base}/pulls/{number}/commits"),
        )


def _search_query(repo: str, qualifier: str, start: datetime, end: datetime) -> str:
    upper = end - timedelta(microseconds=1)
    return (
        f"repo:{repo} is:pr {qualifier}:"
        f"{_search_time(start)}..{_search_time(upper)}"
    )


def _search_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _window_row(
    pull: dict[str, Any],
    events: list[dict[str, Any]],
    commits: list[dict[str, Any]],
    window: ReportWindow,
) -> dict[str, Any] | None:
    created = parse_github_time(pull.get("created_at"))
    merged = parse_github_time(pull.get("merged_at"))
    state_events: list[tuple[datetime, str]] = []
    if created:
        state_events.append((created, "opened"))
    for event in events:
        kind = event.get("event")
        at = parse_github_time(event.get("created_at"))
        if kind in {"closed", "reopened", "merged"} and at:
            state_events.append((at, str(kind)))
    if merged and not any(kind == "merged" and at == merged for at, kind in state_events):
        state_events.append((merged, "merged"))

    commit_times = [at for commit in commits if (at := _commit_time(commit)) is not None]
    ordering = {"opened": 0, "closed": 1, "reopened": 2, "merged": 3}
    state_events.sort(key=lambda item: (item[0], ordering[item[1]]))
    relevant_states = [(at, kind) for at, kind in state_events if window.contains(at)]
    relevant_commits = [at for at in commit_times if window.contains(at)]
    if not relevant_states and not relevant_commits:
        return None

    state = "not_created"
    for at, kind in state_events:
        if at >= window.cutoff_exclusive:
            break
        if kind in {"opened", "reopened"}:
            state = "open"
        elif kind == "closed":
            state = "closed"
        elif kind == "merged":
            state = "merged"

    return {
        "number": int(pull["number"]),
        "opened": window.contains(created),
        "committed": bool(relevant_commits),
        "merged": any(kind == "merged" for _, kind in relevant_states),
        "closed_unmerged": any(kind == "closed" for _, kind in relevant_states)
        and state == "closed",
        "state_at_cutoff": state,
    }


def _commit_time(commit: dict[str, Any]) -> datetime | None:
    if commit.get("committed_at"):
        return parse_github_time(str(commit["committed_at"]))
    metadata = commit.get("commit") or {}
    value = (
        (metadata.get("committer") or {}).get("date")
        or (metadata.get("author") or {}).get("date")
    )
    return parse_github_time(value)


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "active_prs": len(rows),
        "opened": sum(bool(row["opened"]) for row in rows),
        "committed": sum(bool(row["committed"]) for row in rows),
        "merged": sum(bool(row["merged"]) for row in rows),
        "closed_unmerged": sum(bool(row["closed_unmerged"]) for row in rows),
        "ongoing": sum(row["state_at_cutoff"] == "open" for row in rows),
    }
