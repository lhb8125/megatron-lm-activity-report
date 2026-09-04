"""End-to-end collection, generation, validation, and publication pipeline."""

from __future__ import annotations

import contextlib
import fcntl
import json
from pathlib import Path
from typing import Any, Iterator

from .ai import ThemeSummarizer, make_runner
from .collector import ActivityCollector, activity_counts
from .config import ReportConfig
from .filters import classify_records
from .github import GitHubClient
from .grouping import build_change_groups
from .publisher import ReportPublisher
from .report import render_reports
from .storage import ActivityStore
from .window import ReportWindow
from .window import parse_github_time


class ReportPipeline:
    def __init__(self, config: ReportConfig, *, progress=print):
        self.config = config
        self.progress = progress

    def run(self, window: ReportWindow, *, publish: bool = False) -> dict[str, Any]:
        with self._locked(), ActivityStore(self.config.database_path) as store:
            run_id = store.start_run(
                self.config.source_repo, window.key, self.config.fingerprint()
            )
            stage = "collect"
            try:
                if store.has_completed_collection(self.config.source_repo, window.key):
                    records = store.window_records(
                        self.config.source_repo, window.key
                    )
                    counts = activity_counts(records)
                    self.progress(
                        f"reusing frozen collection for {window.key}: "
                        f"{len(records)} active PRs"
                    )
                else:
                    source_client = GitHubClient(
                        timeout=self.config.request_timeout_seconds
                    )
                    counts = ActivityCollector(
                        source_client,
                        store,
                        source_repo=self.config.source_repo,
                        workers=self.config.github_workers,
                        progress=self.progress,
                    ).collect(window)
                    records = store.window_records(
                        self.config.source_repo, window.key
                    )

                stage = "classify"
                store.update_run(run_id, stage=stage)
                if not records:
                    raise RuntimeError(
                        f"no PR activity found for {self.config.source_repo} {window.key}"
                    )
                classification_rows = classify_records(records, self.config)
                store.replace_classifications(
                    self.config.source_repo, window.key, classification_rows
                )
                classifications = {
                    int(row["number"]): row for row in classification_rows
                }
                eligible = [
                    record
                    for record in records
                    if not classifications[int(record["number"])]["excluded"]
                    and _eligible_for_narrative(record, window)
                ]
                groups = build_change_groups(eligible)
                store.replace_groups(self.config.source_repo, window.key, groups)
                self.progress(
                    f"eligible PRs: {len(eligible)}; change groups: {len(groups)}"
                )

                stage = "summarize"
                store.update_run(run_id, stage=stage)
                run_dir = self._run_dir(window, run_id)
                if groups:
                    _, english, chinese = ThemeSummarizer(
                        self.config,
                        make_runner(self.config, run_dir),
                        store,
                        progress=self.progress,
                    ).summarize(eligible, groups, window)
                else:
                    english = {
                        "overview": "No report-eligible feature, optimization, or reliability theme was active in this window.",
                        "delivered_themes": [],
                        "ongoing_themes": [],
                    }
                    chinese = {
                        "overview": "本周期没有符合报告口径的功能、优化或可靠性主题。",
                        "delivered_themes": [],
                        "ongoing_themes": [],
                    }
                (
                    english_title,
                    chinese_title,
                    english_markdown,
                    chinese_markdown,
                ) = render_reports(
                    self.config,
                    window,
                    records,
                    classifications,
                    groups,
                    english,
                    chinese,
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                preview_en = run_dir / "report.en-US.md"
                preview_zh = run_dir / "report.zh-CN.md"
                preview_en.write_text(english_markdown, encoding="utf-8")
                preview_zh.write_text(chinese_markdown, encoding="utf-8")
                (run_dir / "summary.en-US.json").write_text(
                    json.dumps(english, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (run_dir / "summary.zh-CN.json").write_text(
                    json.dumps(chinese, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                store.save_report(
                    self.config.source_repo,
                    window_key=window.key,
                    month_key=window.month_key,
                    cutoff_date=window.cutoff_date.isoformat(),
                    final=window.final,
                    english_title=english_title,
                    chinese_title=chinese_title,
                    english_summary=english,
                    chinese_summary=chinese,
                    english_markdown=english_markdown,
                    chinese_markdown=chinese_markdown,
                )
                result: dict[str, Any] = {
                    "run_id": run_id,
                    "window": window.key,
                    "final": window.final,
                    "counts": counts,
                    "eligible_prs": len(eligible),
                    "change_groups": len(groups),
                    "english_preview": str(preview_en),
                    "chinese_preview": str(preview_zh),
                }

                if publish:
                    stage = "publish"
                    store.update_run(run_id, stage=stage)
                    publication_client = GitHubClient(
                        timeout=self.config.request_timeout_seconds
                    )
                    publication = ReportPublisher(
                        self.config, publication_client, progress=self.progress
                    ).publish(
                        window,
                        english_title=english_title,
                        english_markdown=english_markdown,
                        chinese_markdown=chinese_markdown,
                    )
                    store.mark_report_published(self.config.source_repo, window.key)
                    result.update(publication)
                store.update_run(
                    run_id,
                    stage="publish" if publish else "draft",
                    status="completed",
                )
                return result
            except Exception as exc:
                store.update_run(run_id, stage=stage, status="failed", error=str(exc))
                raise

    def status(self) -> dict[str, Any]:
        with ActivityStore(self.config.database_path) as store:
            return {
                "source_repo": self.config.source_repo,
                "destination_repo": self.config.destination_repo,
                "database_path": str(self.config.database_path),
                "runs": store.latest_runs(self.config.source_repo),
            }

    def _run_dir(self, window: ReportWindow, run_id: str) -> Path:
        return self.config.artifacts_dir / window.month_key / window.cutoff_date.isoformat() / run_id

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.config.artifacts_dir.parent / "pipeline.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another report process is running") from exc
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _eligible_for_narrative(
    record: dict[str, Any], window: ReportWindow
) -> bool:
    """Reject carry-over PRs that have no activity in the current month."""

    if bool(record.get("merged")):
        return True
    if record.get("state_at_cutoff") != "open":
        return False
    if bool(record.get("opened") or record.get("committed")):
        return True
    return any(
        event.get("event") == "reopened"
        and window.contains(parse_github_time(event.get("created_at")))
        for event in record.get("events", [])
    )
