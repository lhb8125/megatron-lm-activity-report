"""Idempotent official Issue publication and committed publication state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import ReportConfig
from .github import GitHubClient, GitHubError
from .report import report_marker
from .repository import GitRepository
from .window import ReportWindow


class ReportPublisher:
    def __init__(
        self,
        config: ReportConfig,
        client: GitHubClient,
        *,
        progress=print,
    ):
        self.config = config
        self.client = client
        self.progress = progress
        self.repository = GitRepository(config.project_root, branch=config.report_branch)

    def publish(
        self,
        window: ReportWindow,
        *,
        english_title: str,
        english_markdown: str,
        chinese_markdown: str,
    ) -> dict[str, Any]:
        english_path, chinese_path = report_paths(self.config, window)
        english_path.parent.mkdir(parents=True, exist_ok=True)
        chinese_path.parent.mkdir(parents=True, exist_ok=True)
        english_path.write_text(english_markdown, encoding="utf-8")
        chinese_path.write_text(chinese_markdown, encoding="utf-8")
        self.progress("committing bilingual report before updating the official Issue")
        report_commit = self.repository.commit_and_push(
            [english_path, chinese_path],
            message=f"Update {window.month_key} report through {window.cutoff_date.isoformat()}",
        )

        state = load_state(self.config.state_path)
        state_key = f"{self.config.source_repo}:{window.month_key}"
        known = state["reports"].get(state_key) or {}
        known_number = (
            int(known["issue_number"])
            if known.get("destination_repo") == self.config.destination_repo
            and known.get("issue_number")
            else None
        )
        issue = publish_issue(
            self.client,
            self.config,
            month_key=window.month_key,
            title=english_title,
            body=english_markdown,
            known_issue_number=known_number,
            progress=self.progress,
        )
        issue_number = int(issue["number"])
        issue_url = str(
            issue.get("html_url")
            or f"https://github.com/{self.config.destination_repo}/issues/{issue_number}"
        )
        state["reports"][state_key] = {
            "destination_repo": self.config.destination_repo,
            "issue_number": issue_number,
            "issue_url": issue_url,
            "cutoff_date": window.cutoff_date.isoformat(),
            "final": window.final,
            "english_sha256": _sha256(english_markdown),
            "chinese_sha256": _sha256(chinese_markdown),
        }
        save_state(self.config.state_path, state)
        state_commit = self.repository.commit_and_push(
            [self.config.state_path],
            message=f"Record {window.month_key} Issue publication",
        )
        return {
            "issue_number": issue_number,
            "issue_url": issue_url,
            "report_commit": report_commit,
            "state_commit": state_commit,
            "english_report": str(english_path),
            "chinese_report": str(chinese_path),
        }


def publish_issue(
    client: GitHubClient,
    config: ReportConfig,
    *,
    month_key: str,
    title: str,
    body: str,
    known_issue_number: int | None = None,
    progress=print,
) -> dict[str, Any]:
    repository = client.get(f"/repos/{config.destination_repo}")
    if not repository.get("has_issues"):
        raise GitHubError(f"Issues are disabled on {config.destination_repo}")
    marker = report_marker(config.source_repo, month_key)
    accepted_markers = (marker, _legacy_report_marker(config.source_repo, month_key))
    issue = None
    if known_issue_number is not None:
        try:
            candidate = client.get(
                f"/repos/{config.destination_repo}/issues/{known_issue_number}"
            )
            if "pull_request" not in candidate and _has_marker(candidate, accepted_markers):
                issue = candidate
        except GitHubError:
            issue = None
    if issue is None:
        issue = _find_existing(client, config, accepted_markers)
    if issue is None:
        progress(f"creating Issue in {config.destination_repo}: {title}")
        return client.post(
            f"/repos/{config.destination_repo}/issues", {"title": title, "body": body}
        )
    progress(f"updating Issue {config.destination_repo}#{issue['number']}: {title}")
    return client.patch(
        f"/repos/{config.destination_repo}/issues/{issue['number']}",
        {"title": title, "body": body, "state": "open"},
    )


def _find_existing(
    client: GitHubClient, config: ReportConfig, markers: tuple[str, ...]
) -> dict[str, Any] | None:
    recent = client.paginate(
        f"/repos/{config.destination_repo}/issues",
        params={"state": "all", "sort": "created", "direction": "desc"},
        max_pages=2,
    )
    for issue in recent:
        if "pull_request" not in issue and _has_marker(issue, markers):
            return issue
    response = client.get(
        "/search/issues",
        params={
            "q": f'repo:{config.destination_repo} is:issue in:title "Megatron-LM Monthly Activity Report"',
            "per_page": 100,
        },
    )
    for issue in response.get("items") or []:
        if _has_marker(issue, markers):
            return issue
    return None


def _has_marker(issue: dict[str, Any], markers: tuple[str, ...]) -> bool:
    body = str(issue.get("body") or "")
    return any(marker in body for marker in markers)


def _legacy_report_marker(source_repo: str, month_key: str) -> str:
    return f"<!-- pr-activity-report source={source_repo} period={month_key} -->"


def report_paths(config: ReportConfig, window: ReportWindow) -> tuple[Path, Path]:
    relative = Path(f"{window.year:04d}") / f"{window.month:02d}.md"
    return (
        config.reports_dir / "en-US" / relative,
        config.reports_dir / "zh-CN" / relative,
    )


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "reports": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("reports"), dict):
        raise ValueError("state/issues.json has an unsupported schema")
    return payload


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def is_published_cutoff(config: ReportConfig, window: ReportWindow) -> bool:
    state = load_state(config.state_path)
    entry = state["reports"].get(f"{config.source_repo}:{window.month_key}") or {}
    return (
        entry.get("destination_repo") == config.destination_repo
        and entry.get("cutoff_date") == window.cutoff_date.isoformat()
        and bool(entry.get("final")) == window.final
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
