from megatron_activity_report.collector import _window_row
from megatron_activity_report.window import ReportWindow


def test_replays_close_and_reopen_at_cutoff():
    window = ReportWindow.for_cutoff("2026-07-12", timezone_name="UTC")
    pull = {"number": 7, "created_at": "2026-07-02T00:00:00Z", "merged_at": None}
    events = [
        {"event": "closed", "created_at": "2026-07-10T00:00:00Z"},
        {"event": "reopened", "created_at": "2026-07-11T00:00:00Z"},
    ]
    row = _window_row(pull, events, [], window)
    assert row == {
        "number": 7,
        "opened": True,
        "committed": False,
        "merged": False,
        "closed_unmerged": False,
        "state_at_cutoff": "open",
    }


def test_old_open_pr_with_commit_is_active_but_comment_only_is_not():
    window = ReportWindow.for_cutoff("2026-07-12", timezone_name="UTC")
    pull = {"number": 8, "created_at": "2026-05-02T00:00:00Z", "merged_at": None}
    commit = {"committed_at": "2026-07-05T00:00:00Z", "sha": "abc"}
    row = _window_row(pull, [], [commit], window)
    assert row is not None
    assert row["committed"]
    assert row["state_at_cutoff"] == "open"
    assert _window_row(pull, [], [], window) is None


def test_previous_month_activity_does_not_carry_into_current_report():
    window = ReportWindow.for_cutoff("2026-08-23", timezone_name="UTC")
    pull = {"number": 10, "created_at": "2026-06-02T00:00:00Z", "merged_at": None}
    old_commit = {"committed_at": "2026-07-31T23:59:59Z", "sha": "old"}
    assert _window_row(pull, [], [old_commit], window) is None


def test_future_merge_does_not_change_historical_cutoff_state():
    window = ReportWindow.for_cutoff("2026-07-12", timezone_name="UTC")
    pull = {
        "number": 9,
        "created_at": "2026-07-02T00:00:00Z",
        "merged_at": "2026-07-20T00:00:00Z",
    }
    row = _window_row(pull, [], [], window)
    assert row is not None
    assert row["state_at_cutoff"] == "open"
    assert not row["merged"]
