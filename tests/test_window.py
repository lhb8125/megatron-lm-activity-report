from datetime import datetime, timezone

import pytest

from megatron_activity_report.window import ReportWindow, scheduled_window


def test_cutoff_uses_shanghai_half_open_boundaries():
    window = ReportWindow.for_cutoff("2026-07-12", timezone_name="Asia/Shanghai")
    assert window.start.isoformat() == "2026-06-30T16:00:00+00:00"
    assert window.cutoff_exclusive.isoformat() == "2026-07-12T16:00:00+00:00"
    assert window.month_key == "2026-07"
    assert not window.final


def test_backfill_is_final_and_handles_leap_year():
    window = ReportWindow.for_month("2024-02", timezone_name="UTC")
    assert window.cutoff_date.isoformat() == "2024-02-29"
    assert window.final


def test_final_before_month_end_is_rejected():
    with pytest.raises(ValueError, match="last day"):
        ReportWindow.for_cutoff("2026-07-12", timezone_name="UTC", final=True)


def test_month_start_wins_over_monday_and_returns_one_final_window():
    window = scheduled_window(
        timezone_name="Asia/Shanghai",
        now=datetime(2026, 5, 31, 17, 0, tzinfo=timezone.utc),  # Monday June 1 01:00
    )
    assert window is not None
    assert window.key == "2026-05@2026-05-31"
    assert window.final


def test_regular_monday_cuts_off_sunday():
    window = scheduled_window(
        timezone_name="Asia/Shanghai",
        now=datetime(2026, 8, 24, 1, 0, tzinfo=timezone.utc),
    )
    assert window is not None
    assert window.cutoff_date.isoformat() == "2026-08-23"
    assert not window.final
