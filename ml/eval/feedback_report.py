"""Print daily check totals and export feedback rows to CSV.

    python -m ml.eval.feedback_report      (or: make feedback)

Reads data/usage.sqlite (written by the API) and writes data/feedback.csv. Both live in the
git-ignored data dir. The file never holds IPs or images; see api/usage.py for what it does hold.
"""
import csv
import sqlite3
from contextlib import closing

from ml.common.paths import data_dir

COLUMNS = ["day", "kind", "verdict", "probability", "model_version", "judge_used", "answer", "actual", "comment"]


def _cell(v):
    # comments are visitor text: a leading = + - @ would run as a formula when the CSV is opened in a spreadsheet
    return "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v


def main():
    db, out = data_dir() / "usage.sqlite", data_dir() / "feedback.csv"
    if not db.exists():
        print(f"No usage data yet ({db} does not exist).")
        return
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as con:
        days = con.execute("SELECT day, kind, verdict, n FROM daily_counts ORDER BY day, kind, verdict").fetchall()
        rows = con.execute(f"SELECT {', '.join(COLUMNS)} FROM feedback ORDER BY day").fetchall()
    print(f"{'day':<12}{'kind':<8}{'verdict':<14}{'checks':>7}")
    for day, kind, verdict, n in days:
        print(f"{day:<12}{kind:<8}{verdict:<14}{n:>7}")
    print(f"total checks: {sum(r[3] for r in days)}")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows([_cell(v) for v in r] for r in rows)
    print(f"feedback rows: {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
