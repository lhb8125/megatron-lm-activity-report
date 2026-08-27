"""Month-to-date reporting windows with timezone-aware cutoff boundaries."""

from __future__ import annotations

import calendar
import dataclasses
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclasses.dataclass(frozen=True)
class ReportWindow:
    """A month-to-date half-open interval ending after ``cutoff_date``."""

    month_key: str
    timezone_name: str
    cutoff_date: date
    start: datetime
    cutoff_exclusive: datetime
    final: bool

    @classmethod
    def for_cutoff(
        cls,
        value: str | date,
        *,
        timezone_name: str,
        final: bool | None = None,
    ) -> "ReportWindow":
        cutoff = date.fromisoformat(value) if isinstance(value, str) else value
        zone = ZoneInfo(timezone_name)
        last_day = calendar.monthrange(cutoff.year, cutoff.month)[1]
        natural_final = cutoff.day == last_day
        if final is True and not natural_final:
            raise ValueError("a final report cutoff must be the last day of its month")
        is_final = natural_final if final is None else final
        start_local = datetime.combine(
            cutoff.replace(day=1), time.min, tzinfo=zone
        )
        end_local = datetime.combine(cutoff + timedelta(days=1), time.min, tzinfo=zone)
        return cls(
            month_key=f"{cutoff.year:04d}-{cutoff.month:02d}",
            timezone_name=timezone_name,
            cutoff_date=cutoff,
            start=start_local.astimezone(timezone.utc),
            cutoff_exclusive=end_local.astimezone(timezone.utc),
            final=is_final,
        )

    @classmethod
    def for_month(cls, month: str, *, timezone_name: str) -> "ReportWindow":
        try:
            year_text, month_text = month.split("-", 1)
            year, month_number = int(year_text), int(month_text)
        except (TypeError, ValueError) as exc:
            raise ValueError("month must use YYYY-MM format") from exc
        if not 1 <= month_number <= 12:
            raise ValueError("month must use YYYY-MM format")
        last_day = calendar.monthrange(year, month_number)[1]
        return cls.for_cutoff(
            date(year, month_number, last_day),
            timezone_name=timezone_name,
            final=True,
        )

    @property
    def key(self) -> str:
        return f"{self.month_key}@{self.cutoff_date.isoformat()}"

    @property
    def year(self) -> int:
        return self.cutoff_date.year

    @property
    def month(self) -> int:
        return self.cutoff_date.month

    def contains(self, timestamp: datetime | None) -> bool:
        if timestamp is None:
            return False
        return self.start <= timestamp.astimezone(timezone.utc) < self.cutoff_exclusive


def scheduled_window(
    *, timezone_name: str, now: datetime | None = None
) -> ReportWindow | None:
    """Return the due weekly/month-final window for the current local date."""

    zone = ZoneInfo(timezone_name)
    current = (now or datetime.now(timezone.utc)).astimezone(zone)
    today = current.date()
    if today.day == 1:
        return ReportWindow.for_cutoff(
            today - timedelta(days=1), timezone_name=timezone_name, final=True
        )
    if today.weekday() == 0:
        return ReportWindow.for_cutoff(
            today - timedelta(days=1), timezone_name=timezone_name
        )
    return None


def parse_github_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
