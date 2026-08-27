from pathlib import Path

from megatron_activity_report.config import ReportConfig
from megatron_activity_report.report import issue_title, render_reports
from megatron_activity_report.window import ReportWindow


def config(tmp_path: Path) -> ReportConfig:
    return ReportConfig(
        source_repo="NVIDIA/Megatron-LM",
        destination_repo="NVIDIA/Megatron-LM",
        report_repo="lhb8125/megatron-lm-activity-report",
        database_path=tmp_path / "db",
        artifacts_dir=tmp_path / "runs",
        state_path=tmp_path / "state.json",
        reports_dir=tmp_path / "reports",
        project_root=tmp_path,
    )


def test_title_contains_cutoff_and_final():
    interim = ReportWindow.for_cutoff("2026-08-23", timezone_name="Asia/Shanghai")
    final = ReportWindow.for_month("2026-07", timezone_name="Asia/Shanghai")
    assert issue_title(interim) == "Megatron-LM Monthly Activity Report — August 2026 (through August 23, 2026)"
    assert issue_title(final) == "Megatron-LM Monthly Activity Report — July 2026 (Final, through July 31, 2026)"


def test_report_has_stable_chinese_link_and_identical_prs(tmp_path):
    window = ReportWindow.for_cutoff("2026-08-23", timezone_name="Asia/Shanghai")
    records = [
        {
            "number": 10,
            "url": "https://github.com/NVIDIA/Megatron-LM/pull/10",
            "opened": True,
            "committed": True,
            "merged": True,
            "closed_unmerged": False,
            "state_at_cutoff": "merged",
        }
    ]
    english = {
        "overview": "THD support landed.",
        "delivered_themes": [{"theme_id": "d-1", "title": "THD", "summary": "Added THD.", "highlights": [], "group_ids": ["pr-10"]}],
        "ongoing_themes": [],
    }
    chinese = {
        "overview": "THD 支持已经落地。",
        "delivered_themes": [{"theme_id": "d-1", "title": "THD", "summary": "新增 THD。", "highlights": [], "group_ids": ["pr-10"]}],
        "ongoing_themes": [],
    }
    _, _, en_body, zh_body = render_reports(
        config(tmp_path), window, records, {10: {"excluded": False, "category": "include"}},
        [{"group_id": "pr-10", "numbers": [10]}], english, chinese
    )
    assert "reports/zh-CN/2026/08.md" in en_body
    assert en_body.count("/pull/10") == zh_body.count("/pull/10") == 1
    assert "implementation details" not in en_body
