"""Privacy-safe usage counts and per-result feedback, kept in one SQLite file (data/usage.sqlite by default).

Stored:
- daily_counts: one row per (UTC date, input kind, verdict) with a running total. No per-check rows.
- checks: for 24 hours only, a random check_id with the UTC date, input kind, verdict, shown probability,
  model version and whether the judge answered, so feedback can be tied to the verdict it is about.
- feedback: the answer (right/wrong/unsure), the optional real label and comment, and the verdict fields above.

Never stored: IP addresses or IP hashes, browser or device details, images or image hashes, links, or anything else per person.
"""
import logging
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("sixfingers")
TTL_S = 24 * 3600  # how long a check_id can receive feedback
MIN_FEEDBACK = 20  # the scorecard shows agreement only from this many answers up

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_counts (day TEXT, kind TEXT, verdict TEXT, n INTEGER NOT NULL,
                                         PRIMARY KEY (day, kind, verdict));
CREATE TABLE IF NOT EXISTS checks (check_id TEXT PRIMARY KEY, day TEXT, kind TEXT, verdict TEXT, probability REAL,
                                   model_version TEXT, judge_used INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS feedback (check_id TEXT PRIMARY KEY, day TEXT, kind TEXT, verdict TEXT, probability REAL,
                                     model_version TEXT, judge_used INTEGER, answer TEXT, actual TEXT, comment TEXT);
"""


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class Usage:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=5)
        con.executescript(SCHEMA)
        return con

    def record_check(self, kind: str, verdict: str, probability: float, model_version: str, judge_used: bool) -> str | None:
        """Count one completed check and remember its verdict for 24 h.

        Every check route calls this once. Returns the new check_id, or None if the write failed:
        a storage problem is logged and never fails the check itself.
        """
        check_id = str(uuid.uuid4())
        try:
            with closing(self._connect()) as con, con:
                con.execute("DELETE FROM checks WHERE created < ?", (time.time() - TTL_S,))
                con.execute("INSERT INTO daily_counts VALUES (?, ?, ?, 1) "
                            "ON CONFLICT (day, kind, verdict) DO UPDATE SET n = n + 1", (_today(), kind, verdict))
                con.execute("INSERT INTO checks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (check_id, _today(), kind, verdict, probability, model_version, int(judge_used), time.time()))
            return check_id
        except Exception:
            log.exception("usage write failed")
            return None

    def add_feedback(self, check_id: str, answer: str, actual: str | None, comment: str | None) -> str:
        """Store feedback for a check. Returns "ok", "duplicate" (already answered) or "unknown" (no such id, or expired)."""
        with closing(self._connect()) as con, con:
            if con.execute("SELECT 1 FROM feedback WHERE check_id = ?", (check_id,)).fetchone():
                return "duplicate"
            row = con.execute("SELECT day, kind, verdict, probability, model_version, judge_used FROM checks "
                              "WHERE check_id = ? AND created >= ?", (check_id, time.time() - TTL_S)).fetchone()
            if row is None:
                return "unknown"
            con.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (check_id, *row, answer, actual, comment))
            return "ok"

    def summary(self) -> dict | None:
        """Totals for the scorecard, read fresh on every call. None if the file cannot be read."""
        since = (datetime.now(timezone.utc).date() - timedelta(days=29)).isoformat()  # today plus the 29 days before
        try:
            with closing(self._connect()) as con:
                total, last30 = con.execute("SELECT COALESCE(SUM(n), 0), COALESCE(SUM(CASE WHEN day >= ? THEN n END), 0) "
                                            "FROM daily_counts", (since,)).fetchone()
                answers, right = con.execute("SELECT COUNT(*), COALESCE(SUM(answer = 'right'), 0) FROM feedback").fetchone()
        except Exception:
            log.exception("usage read failed")
            return None
        return {"checks_total": total, "checks_last_30_days": last30, "feedback_answers": answers,
                "feedback_right": right if answers >= MIN_FEEDBACK else None}
