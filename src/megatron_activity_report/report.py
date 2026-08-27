"""Deterministic Markdown rendering for validated bilingual summaries."""

from __future__ import annotations

import calendar
import re
from collections import Counter
from typing import Any

from .config import ReportConfig
from .window import ReportWindow


def render_reports(
    config: ReportConfig,
    window: ReportWindow,
    records: list[dict[str, Any]],
    classifications: dict[int, dict[str, Any]],
    groups: list[dict[str, Any]],
    english: dict[str, Any],
    chinese: dict[str, Any],
) -> tuple[str, str, str, str]:
    english_title = issue_title(window)
    chinese_title = _chinese_title(window)
    marker = report_marker(config.source_repo, window.month_key)
    substantive = [
        record
        for record in records
        if not classifications[int(record["number"])]["excluded"]
    ]
    counts = {
        "opened": sum(bool(record["opened"]) for record in substantive),
        "merged": sum(bool(record["merged"]) for record in substantive),
        "closed_unmerged": sum(bool(record["closed_unmerged"]) for record in substantive),
        "ongoing": sum(
            record["state_at_cutoff"] == "open"
            and bool(record["opened"] or record["committed"])
            for record in substantive
        ),
    }
    excluded = Counter(
        item["category"] for item in classifications.values() if item["excluded"]
    )
    by_number = {int(record["number"]): record for record in records}
    group_numbers = {group["group_id"]: group["numbers"] for group in groups}
    chinese_url = _report_url(config, "zh-CN", window)
    english_url = _report_url(config, "en-US", window)

    english_lines = [
        marker,
        f"# {english_title}",
        "",
        f"> Chinese version: [中文版本]({chinese_url})",
        "",
        f"> Reporting window: {window.year:04d}-{window.month:02d}-01 through "
        f"{window.cutoff_date.isoformat()} ({config.timezone}); source: "
        f"[`{config.source_repo}`](https://github.com/{config.source_repo}).",
        "",
        "## Overview",
        "",
        str(english["overview"]).strip(),
        "",
        f"- Substantive PR activity after filtering: opened {counts['opened']}; "
        f"merged {counts['merged']}; "
        f"closed without merge {counts['closed_unmerged']}; active at cutoff "
        f"{counts['ongoing']}.",
        f"- Excluded from the report and statistics: test-only {excluded['test_only']}, "
        f"CI-only {excluded['ci_only']}, test/CI-only {excluded['test_ci_only']}, "
        f"format-only {excluded['format_only']}.",
        "",
    ]
    _render_section(
        english_lines,
        "## Delivered",
        english["delivered_themes"],
        group_numbers,
        by_number,
        empty_text="No major delivered theme was selected for this window.",
        related_label="Related PRs: ",
        separator=", ",
    )
    _render_section(
        english_lines,
        "## In Progress",
        english["ongoing_themes"],
        group_numbers,
        by_number,
        empty_text="No major ongoing theme was selected for this window.",
        related_label="Related PRs: ",
        separator=", ",
    )
    english_lines.extend(
        [
            "## Scope and Method",
            "",
            "The narrative is grouped by technical theme rather than PR chronology. "
            "Related `dev` and `main` PRs are described once. The rebuildable raw "
            "ledger, exclusion evidence, commit activity, and state events are retained "
            "in the workflow database artifact. PRs closed without merge are counted "
            "but not narrated.",
            "",
        ]
    )

    chinese_lines = [
        marker,
        f"# {chinese_title}",
        "",
        f"> English version: [English]({english_url})",
        "",
        f"> 统计周期：{window.year:04d}-{window.month:02d}-01 至 "
        f"{window.cutoff_date.isoformat()}（{config.timezone}）；源仓库："
        f"[`{config.source_repo}`](https://github.com/{config.source_repo})。",
        "",
        "## 概览",
        "",
        str(chinese["overview"]).strip(),
        "",
        f"- 过滤后的实质性 PR 活动：新开 {counts['opened']}；"
        f"合并 {counts['merged']}；关闭未合并 "
        f"{counts['closed_unmerged']}；截止时仍在进行 {counts['ongoing']}。",
        f"- 不计入报告正文和统计：纯 UT {excluded['test_only']}，纯 CI "
        f"{excluded['ci_only']}，UT/CI-only {excluded['test_ci_only']}，"
        f"纯格式化 {excluded['format_only']}。",
        "",
    ]
    _render_section(
        chinese_lines,
        "## 已交付成果",
        chinese["delivered_themes"],
        group_numbers,
        by_number,
        empty_text="本周期没有进入重点报告的已交付主题。",
        related_label="相关 PR：",
        separator="、",
    )
    _render_section(
        chinese_lines,
        "## 进行中的方向",
        chinese["ongoing_themes"],
        group_numbers,
        by_number,
        empty_text="本周期没有进入重点报告的进行中主题。",
        related_label="相关 PR：",
        separator="、",
    )
    chinese_lines.extend(
        [
            "## 口径说明",
            "",
            "正文按技术主题组织，而不是按 PR 时间顺序罗列。同一变更的 `dev` / "
            "`main` PR 只描述一次；可重建的完整流水账、过滤证据、提交活动和状态事件"
            "保存在工作流数据库 artifact 中。关闭未合并的 PR 只计入统计。",
            "",
        ]
    )
    english_markdown = "\n".join(english_lines)
    chinese_markdown = "\n".join(chinese_lines)
    if re.search(r"[\u3400-\u9fff]", english_markdown.replace("中文版本", "")):
        raise ValueError("English report contains unexpected Han characters")
    validate_markdown_parity(english_markdown, chinese_markdown)
    return english_title, chinese_title, english_markdown, chinese_markdown


def issue_title(window: ReportWindow) -> str:
    month = calendar.month_name[window.month]
    cutoff = f"{calendar.month_name[window.cutoff_date.month]} {window.cutoff_date.day}, {window.year}"
    final = "Final, " if window.final else ""
    return f"Megatron-LM Monthly Activity Report — {month} {window.year} ({final}through {cutoff})"


def report_marker(source_repo: str, month_key: str) -> str:
    return f"<!-- megatron-activity-report source={source_repo} month={month_key} -->"


def validate_markdown_parity(english: str, chinese: str) -> None:
    pattern = re.compile(r"https://github\.com/NVIDIA/Megatron-LM/pull/(\d+)")
    english_prs = pattern.findall(english)
    chinese_prs = pattern.findall(chinese)
    if english_prs != chinese_prs:
        raise ValueError("English and Chinese report PR citations differ")


def _chinese_title(window: ReportWindow) -> str:
    suffix = "最终版，" if window.final else ""
    return (
        f"Megatron-LM 月度活动报告 — {window.year}年{window.month}月"
        f"（{suffix}截至{window.cutoff_date.isoformat()}）"
    )


def _report_url(config: ReportConfig, language: str, window: ReportWindow) -> str:
    return (
        f"https://github.com/{config.report_repo}/blob/{config.report_branch}/reports/"
        f"{language}/{window.year:04d}/{window.month:02d}.md"
    )


def _render_section(
    lines: list[str],
    heading: str,
    themes: list[dict[str, Any]],
    group_numbers: dict[str, list[int]],
    by_number: dict[int, dict[str, Any]],
    *,
    empty_text: str,
    related_label: str,
    separator: str,
) -> None:
    lines.extend([heading, ""])
    if not themes:
        lines.extend([empty_text, ""])
        return
    for theme in themes:
        lines.extend([f"### {str(theme['title']).strip()}", "", str(theme["summary"]).strip(), ""])
        for highlight in theme.get("highlights") or []:
            lines.append(f"- {str(highlight).strip()}")
        if theme.get("highlights"):
            lines.append("")
        numbers = sorted(
            {
                int(number)
                for group_id in theme["group_ids"]
                for number in group_numbers[group_id]
            }
        )
        links = [
            f"[#{number}]({by_number[number]['url']})"
            for number in numbers
            if number in by_number
        ]
        lines.extend([related_label + separator.join(links), ""])
