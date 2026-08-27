from megatron_activity_report.pipeline import _eligible_for_narrative
from megatron_activity_report.window import ReportWindow


def test_narrative_eligibility_requires_current_month_activity():
    window = ReportWindow.for_cutoff("2026-08-23", timezone_name="UTC")
    stale = {
        "state_at_cutoff": "open",
        "opened": False,
        "committed": False,
        "merged": False,
        "events": [],
    }
    reopened = {
        **stale,
        "events": [{"event": "reopened", "created_at": "2026-08-04T00:00:00Z"}],
    }
    assert not _eligible_for_narrative(stale, window)
    assert _eligible_for_narrative(reopened, window)
