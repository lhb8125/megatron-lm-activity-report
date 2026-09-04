import pytest

from megatron_activity_report.collector import (
    ActivityCollector,
    CollectedPullRequest,
    _window_row,
)
from megatron_activity_report.github import GitHubError
from megatron_activity_report.storage import ActivityStore
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


def test_completed_snapshots_are_durable_when_a_later_fetch_fails(tmp_path, monkeypatch):
    updated_at = "2026-08-04T00:00:00Z"
    item = CollectedPullRequest(
        pull={
            "number": 1,
            "title": "THD",
            "body": "",
            "html_url": "https://github.com/NVIDIA/Megatron-LM/pull/1",
            "state": "open",
            "created_at": updated_at,
            "updated_at": updated_at,
        },
        files=[],
        events=[],
        commits=[],
    )

    def fetch_then_fail(_numbers):
        yield item
        raise GitHubError("rate limited")

    with ActivityStore(tmp_path / "activity.duckdb") as store:
        collector = ActivityCollector(
            object(), store, source_repo="NVIDIA/Megatron-LM", workers=1
        )
        monkeypatch.setattr(
            collector, "_discover_candidates", lambda _window: {1: updated_at, 2: updated_at}
        )
        monkeypatch.setattr(collector, "_fetch_all", fetch_then_fail)
        with pytest.raises(GitHubError, match="rate limited"):
            collector.collect(ReportWindow.for_cutoff("2026-08-23", timezone_name="UTC"))
        assert store.cached_updated_at("NVIDIA/Megatron-LM", 1) == updated_at
        assert store.cached_updated_at("NVIDIA/Megatron-LM", 2) is None


def test_completed_collection_is_frozen_for_an_exact_window(tmp_path):
    with ActivityStore(tmp_path / "activity.duckdb") as store:
        run_id = store.start_run("NVIDIA/Megatron-LM", "2026-08-31-final", "hash")
        assert not store.has_completed_collection(
            "NVIDIA/Megatron-LM", "2026-08-31-final"
        )

        store.update_run(run_id, stage="classify")
        assert store.has_completed_collection(
            "NVIDIA/Megatron-LM", "2026-08-31-final"
        )
        assert not store.has_completed_collection(
            "NVIDIA/Megatron-LM", "2026-08-24"
        )
