from pathlib import Path

from megatron_activity_report.config import ReportConfig
from megatron_activity_report.filters import classify_record
from megatron_activity_report.grouping import build_change_groups


def config(tmp_path: Path) -> ReportConfig:
    return ReportConfig(
        source_repo="NVIDIA/Megatron-LM",
        destination_repo="NVIDIA/Megatron-LM",
        report_repo="me/reports",
        database_path=tmp_path / "db",
        artifacts_dir=tmp_path / "runs",
        state_path=tmp_path / "state.json",
        reports_dir=tmp_path / "reports",
        project_root=tmp_path,
    )


def record(number, title, base, files, body=""):
    return {
        "number": number,
        "title": title,
        "body": body,
        "base_ref": base,
        "labels": [],
        "files": files,
        "commits": [],
        "merged": True,
        "opened": True,
        "committed": True,
        "state_at_cutoff": "merged",
    }


def test_pure_test_ci_and_format_are_excluded(tmp_path):
    test = record(1, "add coverage", "main", [{"path": "tests/unit_tests/test_x.py"}])
    ci = record(2, "fix CI", "main", [{"path": ".github/workflows/ci.yml"}])
    formatted = record(
        3,
        "format: autoformat",
        "main",
        [{"path": "megatron/core/x.py", "patch": "@@\n-def f( x ):\n+def f(x):"}],
    )
    assert classify_record(test, config(tmp_path))["category"] == "test_only"
    assert classify_record(ci, config(tmp_path))["category"] == "ci_only"
    assert classify_record(formatted, config(tmp_path))["category"] == "format_only"


def test_implementation_plus_tests_is_retained(tmp_path):
    item = record(
        4,
        "support THD",
        "main",
        [
            {"path": "megatron/core/transformer/attention.py"},
            {"path": "tests/unit_tests/transformer/test_attention.py"},
        ],
    )
    assert not classify_record(item, config(tmp_path))["excluded"]


def test_dev_main_counterparts_group_but_same_base_does_not():
    files = [{"path": "megatron/core/cic.py", "patch": "@@\n-old\n+new"}]
    records = [
        record(100, "[Dev] Add CIC", "dev", files),
        record(200, "cp: Add CIC", "main", files, body="Cherry-pick of #100"),
    ]
    groups = build_change_groups(records)
    assert [100, 200] in [group["numbers"] for group in groups]
    assert all(group["content_hash"] for group in groups)

    same_base = [
        record(300, "Add CIC", "main", files),
        record(301, "Add CIC", "main", files),
    ]
    assert [group["numbers"] for group in build_change_groups(same_base)] == [[300], [301]]
