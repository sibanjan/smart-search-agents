"""Cron schedule for jobs (APScheduler inside the daemon).

The watcher catches most changes live; the schedule is the safety net for changes made
while the daemon was not running. On start, a job whose last successful run is older than
one schedule interval is queued once to catch up.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .jobs import KINDS, JobQueue

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, jobs: JobQueue, schedule: dict[str, str]):
        self.jobs = jobs
        # coalesce + one instance: a laptop that slept through several slots runs once on wake
        self._sched = BackgroundScheduler(
            job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 12 * 3600}
        )
        self._load(schedule)

    def _load(self, schedule: dict[str, str]) -> None:
        self.schedule = {k: v for k, v in schedule.items() if k in KINDS}
        for k in set(schedule) - set(self.schedule):
            log.warning("schedule: unknown job kind %r ignored", k)
        self.triggers = {k: CronTrigger.from_crontab(expr) for k, expr in self.schedule.items()}

    def _add_jobs(self) -> None:
        for kind, trigger in self.triggers.items():
            self._sched.add_job(
                self.jobs.submit, trigger, args=[kind, {}, "schedule"], id=f"schedule-{kind}"
            )
            log.info("schedule: %s at '%s'", kind, self.schedule[kind])

    def start(self) -> None:
        self._add_jobs()
        self._sched.start()

    def reschedule(self, schedule: dict[str, str]) -> None:
        """Replace the cron jobs while running (Settings page)."""
        self._load(schedule)
        self._sched.remove_all_jobs()
        self._add_jobs()

    def stop(self) -> None:
        if self._sched.running:
            self._sched.shutdown(wait=False)

    def catch_up(self, now: datetime | None = None) -> list[str]:
        """Queue each scheduled job whose last success is older than its interval."""
        now = now or datetime.now(timezone.utc)
        queued = []
        for kind, trigger in self.triggers.items():
            interval = self.interval(trigger, now)
            last = self.jobs.last_done(kind)
            if last is None or last["finished"] < time.time() - interval.total_seconds():
                self.jobs.submit(kind, {}, "startup")
                queued.append(kind)
        return queued

    def next_runs(self) -> dict[str, str | None]:
        out = {}
        for kind in self.triggers:
            job = self._sched.get_job(f"schedule-{kind}") if self._sched.running else None
            out[kind] = job.next_run_time.isoformat() if job and job.next_run_time else None
        return out

    @staticmethod
    def interval(trigger: CronTrigger, now: datetime) -> timedelta:
        first = trigger.get_next_fire_time(None, now)
        second = trigger.get_next_fire_time(first, first + timedelta(seconds=1))
        return second - first
