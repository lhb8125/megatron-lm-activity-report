from pathlib import Path

from megatron_activity_report.config import ReportConfig
from megatron_activity_report.publisher import is_published_cutoff, publish_issue, save_state
from megatron_activity_report.report import report_marker
from megatron_activity_report.window import ReportWindow


class FakeGitHub:
    def __init__(self, issues=None):
        self.issues = list(issues or [])
        self.created = []
        self.updated = []

    def get(self, path, params=None):
        if path == "/repos/NVIDIA/Megatron-LM":
            return {"has_issues": True}
        if path.startswith("/repos/NVIDIA/Megatron-LM/issues/"):
            number = int(path.rsplit("/", 1)[1])
            return next(issue for issue in self.issues if issue["number"] == number)
        if path == "/search/issues":
            return {"items": self.issues}
        raise AssertionError(path)

    def paginate(self, path, params=None, max_pages=None):
        return self.issues

    def post(self, path, payload):
        issue = {"number": 7000, "html_url": "https://github.com/NVIDIA/Megatron-LM/issues/7000", **payload}
        self.created.append(issue)
        return issue

    def patch(self, path, payload):
        number = int(path.rsplit("/", 1)[1])
        issue = {"number": number, "html_url": f"https://github.com/NVIDIA/Megatron-LM/issues/{number}", **payload}
        self.updated.append(issue)
        return issue


def config(tmp_path: Path) -> ReportConfig:
    return ReportConfig(
        source_repo="NVIDIA/Megatron-LM",
        destination_repo="NVIDIA/Megatron-LM",
        report_repo="me/reports",
        state_path=tmp_path / "issues.json",
        project_root=tmp_path,
    )


def test_title_change_still_updates_same_marked_issue(tmp_path):
    marker = report_marker("NVIDIA/Megatron-LM", "2026-08")
    client = FakeGitHub([{"number": 9, "title": "old", "body": marker}])
    issue = publish_issue(
        client, config(tmp_path), month_key="2026-08", title="new cutoff", body=marker,
        progress=lambda _message: None,
    )
    assert issue["number"] == 9
    assert not client.created
    assert client.updated[0]["title"] == "new cutoff"


def test_legacy_month_marker_updates_existing_issue_during_migration(tmp_path):
    legacy = "<!-- pr-activity-report source=NVIDIA/Megatron-LM period=2026-07 -->"
    client = FakeGitHub([{"number": 6607, "title": "old", "body": legacy}])
    issue = publish_issue(
        client,
        config(tmp_path),
        month_key="2026-07",
        title="final",
        body=report_marker("NVIDIA/Megatron-LM", "2026-07"),
        progress=lambda _message: None,
    )
    assert issue["number"] == 6607
    assert not client.created


def test_wrong_known_issue_marker_is_not_modified(tmp_path):
    marker = report_marker("NVIDIA/Megatron-LM", "2026-08")
    client = FakeGitHub([{"number": 9, "title": "unrelated", "body": "other"}])
    issue = publish_issue(
        client, config(tmp_path), month_key="2026-08", title="new", body=marker,
        known_issue_number=9, progress=lambda _message: None,
    )
    assert issue["number"] == 7000
    assert client.created


def test_committed_state_deduplicates_same_scheduled_cutoff(tmp_path):
    cfg = config(tmp_path)
    window = ReportWindow.for_cutoff("2026-08-23", timezone_name="Asia/Shanghai")
    save_state(
        cfg.state_path,
        {
            "schema_version": 1,
            "reports": {
                "NVIDIA/Megatron-LM:2026-08": {
                    "destination_repo": "NVIDIA/Megatron-LM",
                    "issue_number": 9,
                    "cutoff_date": "2026-08-23",
                    "final": False,
                }
            },
        },
    )
    assert is_published_cutoff(cfg, window)
