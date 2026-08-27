"""DuckDB persistence for the rebuildable raw ledger and generated reports."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb


SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL);

CREATE TABLE IF NOT EXISTS report_runs (
    run_id VARCHAR PRIMARY KEY,
    source_repo VARCHAR NOT NULL,
    window_key VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    stage VARCHAR NOT NULL,
    config_hash VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    error VARCHAR
);

CREATE TABLE IF NOT EXISTS pull_requests (
    source_repo VARCHAR NOT NULL,
    number BIGINT NOT NULL,
    title VARCHAR NOT NULL,
    body VARCHAR NOT NULL,
    url VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    draft BOOLEAN NOT NULL,
    author VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    closed_at VARCHAR,
    merged_at VARCHAR,
    base_ref VARCHAR NOT NULL,
    head_ref VARCHAR NOT NULL,
    labels_json VARCHAR NOT NULL,
    additions BIGINT NOT NULL,
    deletions BIGINT NOT NULL,
    changed_files BIGINT NOT NULL,
    raw_json VARCHAR NOT NULL,
    fetched_at VARCHAR NOT NULL,
    PRIMARY KEY (source_repo, number)
);

CREATE TABLE IF NOT EXISTS pr_files (
    source_repo VARCHAR NOT NULL,
    number BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    additions BIGINT NOT NULL,
    deletions BIGINT NOT NULL,
    patch VARCHAR,
    PRIMARY KEY (source_repo, number, path)
);

CREATE TABLE IF NOT EXISTS pr_events (
    source_repo VARCHAR NOT NULL,
    number BIGINT NOT NULL,
    event_type VARCHAR NOT NULL,
    event_at VARCHAR NOT NULL,
    event_key VARCHAR NOT NULL,
    raw_json VARCHAR NOT NULL,
    PRIMARY KEY (source_repo, number, event_key)
);

CREATE TABLE IF NOT EXISTS pr_commits (
    source_repo VARCHAR NOT NULL,
    number BIGINT NOT NULL,
    sha VARCHAR NOT NULL,
    committed_at VARCHAR NOT NULL,
    message VARCHAR NOT NULL,
    raw_json VARCHAR NOT NULL,
    PRIMARY KEY (source_repo, number, sha)
);

CREATE TABLE IF NOT EXISTS window_prs (
    source_repo VARCHAR NOT NULL,
    window_key VARCHAR NOT NULL,
    number BIGINT NOT NULL,
    opened BOOLEAN NOT NULL,
    committed BOOLEAN NOT NULL,
    merged BOOLEAN NOT NULL,
    closed_unmerged BOOLEAN NOT NULL,
    state_at_cutoff VARCHAR NOT NULL,
    PRIMARY KEY (source_repo, window_key, number)
);

CREATE TABLE IF NOT EXISTS classifications (
    source_repo VARCHAR NOT NULL,
    window_key VARCHAR NOT NULL,
    number BIGINT NOT NULL,
    category VARCHAR NOT NULL,
    excluded BOOLEAN NOT NULL,
    reason VARCHAR NOT NULL,
    evidence_json VARCHAR NOT NULL,
    PRIMARY KEY (source_repo, window_key, number)
);

CREATE TABLE IF NOT EXISTS change_groups (
    source_repo VARCHAR NOT NULL,
    window_key VARCHAR NOT NULL,
    group_id VARCHAR NOT NULL,
    numbers_json VARCHAR NOT NULL,
    evidence_json VARCHAR NOT NULL,
    content_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_repo, window_key, group_id)
);

CREATE TABLE IF NOT EXISTS group_fact_cache (
    source_repo VARCHAR NOT NULL,
    group_id VARCHAR NOT NULL,
    content_hash VARCHAR NOT NULL,
    fact_json VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    PRIMARY KEY (source_repo, group_id, content_hash)
);

CREATE TABLE IF NOT EXISTS reports (
    source_repo VARCHAR NOT NULL,
    window_key VARCHAR NOT NULL,
    month_key VARCHAR NOT NULL,
    cutoff_date VARCHAR NOT NULL,
    final BOOLEAN NOT NULL,
    english_title VARCHAR NOT NULL,
    chinese_title VARCHAR NOT NULL,
    english_summary_json VARCHAR NOT NULL,
    chinese_summary_json VARCHAR NOT NULL,
    english_markdown VARCHAR NOT NULL,
    chinese_markdown VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    PRIMARY KEY (source_repo, window_key)
);
"""


class ActivityStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.path))
        self.connection.execute(SCHEMA)
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO metadata VALUES ('schema_version', ?)",
                [str(SCHEMA_VERSION)],
            )
        elif int(row[0]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema {row[0]} is incompatible with {SCHEMA_VERSION}; "
                "remove the rebuildable runtime database"
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ActivityStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def start_run(self, source_repo: str, window_key: str, config_hash: str) -> str:
        run_id = uuid.uuid4().hex
        self.connection.execute(
            "INSERT INTO report_runs VALUES (?, ?, ?, 'running', 'collect', ?, ?, NULL, NULL)",
            [run_id, source_repo, window_key, config_hash, _now()],
        )
        return run_id

    def update_run(
        self,
        run_id: str,
        *,
        stage: str,
        status: str = "running",
        error: str | None = None,
    ) -> None:
        finished = _now() if status in {"failed", "completed"} else None
        self.connection.execute(
            """UPDATE report_runs SET stage = ?, status = ?, error = ?,
               finished_at = COALESCE(?, finished_at) WHERE run_id = ?""",
            [stage, status, error, finished, run_id],
        )

    def cached_updated_at(self, source_repo: str, number: int) -> str | None:
        row = self.connection.execute(
            "SELECT updated_at FROM pull_requests WHERE source_repo = ? AND number = ?",
            [source_repo, number],
        ).fetchone()
        return None if row is None else str(row[0])

    def replace_pr(
        self,
        source_repo: str,
        pr: dict[str, Any],
        files: Iterable[dict[str, Any]],
        events: Iterable[dict[str, Any]],
        commits: Iterable[dict[str, Any]],
    ) -> None:
        number = int(pr["number"])
        labels = [label.get("name", "") for label in pr.get("labels", [])]
        values = [
            source_repo,
            number,
            pr.get("title") or "",
            pr.get("body") or "",
            pr.get("html_url") or f"https://github.com/{source_repo}/pull/{number}",
            pr.get("state") or "unknown",
            bool(pr.get("draft", False)),
            (pr.get("user") or {}).get("login") or "",
            pr.get("created_at") or "",
            pr.get("updated_at") or "",
            pr.get("closed_at"),
            pr.get("merged_at"),
            (pr.get("base") or {}).get("ref") or "",
            (pr.get("head") or {}).get("ref") or "",
            json.dumps(labels, ensure_ascii=False),
            int(pr.get("additions") or 0),
            int(pr.get("deletions") or 0),
            int(pr.get("changed_files") or 0),
            json.dumps(pr, ensure_ascii=False),
            _now(),
        ]
        self.connection.execute(
            "INSERT OR REPLACE INTO pull_requests VALUES (" + ",".join(["?"] * 20) + ")",
            values,
        )
        self.connection.execute(
            "DELETE FROM pr_files WHERE source_repo = ? AND number = ?",
            [source_repo, number],
        )
        for file in files:
            self.connection.execute(
                "INSERT INTO pr_files VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    source_repo,
                    number,
                    file.get("filename") or "",
                    file.get("status") or "modified",
                    int(file.get("additions") or 0),
                    int(file.get("deletions") or 0),
                    file.get("patch"),
                ],
            )
        self.connection.execute(
            "DELETE FROM pr_events WHERE source_repo = ? AND number = ?",
            [source_repo, number],
        )
        for event in events:
            event_type = str(event.get("event") or "")
            event_at = str(event.get("created_at") or "")
            if event_type not in {"closed", "reopened", "merged"} or not event_at:
                continue
            event_key = str(event.get("id") or f"{event_type}:{event_at}")
            self.connection.execute(
                "INSERT INTO pr_events VALUES (?, ?, ?, ?, ?, ?)",
                [
                    source_repo,
                    number,
                    event_type,
                    event_at,
                    event_key,
                    json.dumps(event, ensure_ascii=False),
                ],
            )
        self.connection.execute(
            "DELETE FROM pr_commits WHERE source_repo = ? AND number = ?",
            [source_repo, number],
        )
        for commit in commits:
            metadata = commit.get("commit") or {}
            committed_at = (
                (metadata.get("committer") or {}).get("date")
                or (metadata.get("author") or {}).get("date")
                or ""
            )
            if not commit.get("sha") or not committed_at:
                continue
            self.connection.execute(
                "INSERT INTO pr_commits VALUES (?, ?, ?, ?, ?, ?)",
                [
                    source_repo,
                    number,
                    str(commit["sha"]),
                    str(committed_at),
                    str(metadata.get("message") or ""),
                    json.dumps(commit, ensure_ascii=False),
                ],
            )

    def raw_bundle(self, source_repo: str, number: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT raw_json FROM pull_requests WHERE source_repo = ? AND number = ?",
            [source_repo, number],
        ).fetchone()
        if row is None:
            return None
        return {
            "pull": json.loads(row[0]),
            "files": self.files(source_repo, number),
            "events": self.events(source_repo, number),
            "commits": self.commits(source_repo, number),
        }

    def replace_window_rows(
        self, source_repo: str, window_key: str, rows: Iterable[dict[str, Any]]
    ) -> None:
        self.connection.execute(
            "DELETE FROM window_prs WHERE source_repo = ? AND window_key = ?",
            [source_repo, window_key],
        )
        for row in rows:
            self.connection.execute(
                "INSERT INTO window_prs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    source_repo,
                    window_key,
                    row["number"],
                    row["opened"],
                    row["committed"],
                    row["merged"],
                    row["closed_unmerged"],
                    row["state_at_cutoff"],
                ],
            )

    def window_records(self, source_repo: str, window_key: str) -> list[dict[str, Any]]:
        columns = [
            "number", "title", "body", "url", "state", "draft", "author",
            "created_at", "updated_at", "closed_at", "merged_at", "base_ref",
            "head_ref", "labels_json", "additions", "deletions", "changed_files",
            "opened", "committed", "merged", "closed_unmerged", "state_at_cutoff",
        ]
        rows = self.connection.execute(
            """SELECT p.number, p.title, p.body, p.url, p.state, p.draft, p.author,
                      p.created_at, p.updated_at, p.closed_at, p.merged_at,
                      p.base_ref, p.head_ref, p.labels_json, p.additions,
                      p.deletions, p.changed_files, w.opened, w.committed, w.merged,
                      w.closed_unmerged, w.state_at_cutoff
               FROM window_prs w JOIN pull_requests p USING (source_repo, number)
               WHERE w.source_repo = ? AND w.window_key = ? ORDER BY p.number""",
            [source_repo, window_key],
        ).fetchall()
        records = [dict(zip(columns, row)) for row in rows]
        for record in records:
            record["labels"] = json.loads(record.pop("labels_json"))
            number = int(record["number"])
            record["files"] = self.files(source_repo, number)
            record["events"] = self.events(source_repo, number)
            record["commits"] = self.commits(source_repo, number)
        return records

    def files(self, source_repo: str, number: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT path, status, additions, deletions, patch FROM pr_files
               WHERE source_repo = ? AND number = ? ORDER BY path""",
            [source_repo, number],
        ).fetchall()
        return [dict(zip(("path", "status", "additions", "deletions", "patch"), row)) for row in rows]

    def events(self, source_repo: str, number: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT event_type, event_at, event_key FROM pr_events
               WHERE source_repo = ? AND number = ? ORDER BY event_at, event_key""",
            [source_repo, number],
        ).fetchall()
        return [dict(zip(("event", "created_at", "event_key"), row)) for row in rows]

    def commits(self, source_repo: str, number: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT sha, committed_at, message FROM pr_commits
               WHERE source_repo = ? AND number = ? ORDER BY committed_at, sha""",
            [source_repo, number],
        ).fetchall()
        return [dict(zip(("sha", "committed_at", "message"), row)) for row in rows]

    def replace_classifications(
        self, source_repo: str, window_key: str, rows: Iterable[dict[str, Any]]
    ) -> None:
        self.connection.execute(
            "DELETE FROM classifications WHERE source_repo = ? AND window_key = ?",
            [source_repo, window_key],
        )
        for row in rows:
            self.connection.execute(
                "INSERT INTO classifications VALUES (?, ?, ?, ?, ?, ?, ?)",
                [source_repo, window_key, row["number"], row["category"], row["excluded"],
                 row["reason"], json.dumps(row.get("evidence", {}), ensure_ascii=False)],
            )

    def replace_groups(
        self, source_repo: str, window_key: str, groups: Iterable[dict[str, Any]]
    ) -> None:
        self.connection.execute(
            "DELETE FROM change_groups WHERE source_repo = ? AND window_key = ?",
            [source_repo, window_key],
        )
        for group in groups:
            self.connection.execute(
                "INSERT INTO change_groups VALUES (?, ?, ?, ?, ?, ?)",
                [source_repo, window_key, group["group_id"], json.dumps(group["numbers"]),
                 json.dumps(group.get("evidence", []), ensure_ascii=False), group["content_hash"]],
            )

    def cached_fact(
        self, source_repo: str, group_id: str, content_hash: str
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            """SELECT fact_json FROM group_fact_cache
               WHERE source_repo = ? AND group_id = ? AND content_hash = ?""",
            [source_repo, group_id, content_hash],
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def cache_fact(
        self, source_repo: str, group_id: str, content_hash: str, fact: dict[str, Any]
    ) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO group_fact_cache VALUES (?, ?, ?, ?, ?)",
            [source_repo, group_id, content_hash, json.dumps(fact, ensure_ascii=False), _now()],
        )

    def save_report(
        self,
        source_repo: str,
        *,
        window_key: str,
        month_key: str,
        cutoff_date: str,
        final: bool,
        english_title: str,
        chinese_title: str,
        english_summary: dict[str, Any],
        chinese_summary: dict[str, Any],
        english_markdown: str,
        chinese_markdown: str,
        status: str = "draft",
    ) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [source_repo, window_key, month_key, cutoff_date, final, english_title,
             chinese_title, json.dumps(english_summary, ensure_ascii=False),
             json.dumps(chinese_summary, ensure_ascii=False), english_markdown,
             chinese_markdown, status, _now()],
        )

    def report(self, source_repo: str, window_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """SELECT month_key, cutoff_date, final, english_title, chinese_title,
                      english_summary_json, chinese_summary_json, english_markdown,
                      chinese_markdown, status, updated_at FROM reports
               WHERE source_repo = ? AND window_key = ?""",
            [source_repo, window_key],
        ).fetchone()
        if row is None:
            return None
        keys = ("month_key", "cutoff_date", "final", "english_title", "chinese_title",
                "english_summary", "chinese_summary", "english_markdown",
                "chinese_markdown", "status", "updated_at")
        result = dict(zip(keys, row))
        result["english_summary"] = json.loads(result["english_summary"])
        result["chinese_summary"] = json.loads(result["chinese_summary"])
        return result

    def mark_report_published(self, source_repo: str, window_key: str) -> None:
        self.connection.execute(
            "UPDATE reports SET status = 'published', updated_at = ? "
            "WHERE source_repo = ? AND window_key = ?",
            [_now(), source_repo, window_key],
        )

    def latest_runs(self, source_repo: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT run_id, window_key, status, stage, started_at, finished_at, error
               FROM report_runs WHERE source_repo = ?
               ORDER BY started_at DESC LIMIT ?""",
            [source_repo, limit],
        ).fetchall()
        keys = (
            "run_id", "window_key", "status", "stage", "started_at",
            "finished_at", "error",
        )
        return [dict(zip(keys, row)) for row in rows]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
