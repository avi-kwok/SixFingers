"""Usage counter, per-result feedback and the scorecard usage block."""
import json
import sqlite3
import time

from fastapi.testclient import TestClient

from api.main import create_app
from tests.api.conftest import StubEmbedder, img_bytes, post


def _rows(tmp_path, sql):
    with sqlite3.connect(tmp_path / "usage.sqlite") as con:
        return con.execute(sql).fetchall()


def _fb(c, check_id, **kw):
    return c.post("/v1/feedback", json={"check_id": check_id, "answer": "right", **kw})


def test_counter_increments_per_check_and_stores_no_ip(build, tmp_path):
    c = build()
    ids = {post(c, img_bytes(255)).json()["check_id"] for _ in range(2)}
    ids.add(post(c, img_bytes(0)).json()["check_id"])
    assert len(ids) == 3 and None not in ids
    counts = {(k, v): n for _, k, v, n in _rows(tmp_path, "SELECT * FROM daily_counts")}
    assert counts == {("file", "likely_ai"): 2, ("file", "likely_real"): 1}
    raw = (tmp_path / "usage.sqlite").read_bytes()
    assert b"testclient" not in raw  # the TestClient's client host, i.e. the "IP"
    limiter = c.app.state.limiter
    assert limiter.ip_hash("testclient").encode() not in raw
    cols = {r[1] for t in ("daily_counts", "checks", "feedback") for r in _rows(tmp_path, f"PRAGMA table_info({t})")}
    assert not {"ip", "ip_hash", "user_agent", "url", "host", "image"} & cols


def test_feedback_happy_path_and_repeat(build, tmp_path):
    c = build()
    cid = post(c, img_bytes(255)).json()["check_id"]
    r = _fb(c, cid, actual="ai", comment="  looks\x00 right\x1b\n ok  ")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    (row,) = _rows(tmp_path, "SELECT kind, verdict, judge_used, answer, actual, comment FROM feedback")
    assert row == ("file", "likely_ai", 0, "right", "ai", "looks right\n ok")
    assert _fb(c, cid).status_code == 409


def test_feedback_unknown_id_and_bad_input(build):
    c = build()
    assert _fb(c, "123e4567-e89b-42d3-a456-426614174000").status_code == 404
    cid = post(c, img_bytes(255)).json()["check_id"]
    assert c.post("/v1/feedback", json={"check_id": cid, "answer": "yes"}).status_code == 422
    assert _fb(c, cid, actual="maybe").status_code == 422
    assert _fb(c, "not-a-uuid").status_code == 422
    assert _fb(c, cid, extra=1).status_code == 422
    assert _fb(c, cid, comment="x" * 501).status_code == 422
    assert _fb(c, cid, comment="x" * 500).status_code == 200


def test_feedback_rate_limited(build):
    c = build()
    codes = {_fb(c, "123e4567-e89b-42d3-a456-426614174000").status_code for _ in range(21)}
    assert codes == {404, 429}


def test_expired_check_ids_are_purged(build, tmp_path, monkeypatch):
    c = build()
    old = post(c, img_bytes(255)).json()["check_id"]
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 25 * 3600)
    assert _fb(c, old).status_code == 404  # expired even before the purge runs
    post(c, img_bytes(255))  # the next write purges it
    left = [r[0] for r in _rows(tmp_path, "SELECT check_id FROM checks")]
    assert len(left) == 1 and old not in left


def test_scorecard_usage_block(build, models_dir):
    (models_dir / "v1" / "scorecard.json").write_text(json.dumps({"model_version": "v1", "usage": "stale"}))
    c = build(rate_limit=100)
    ids = [post(c, img_bytes(255)).json()["check_id"] for _ in range(20)]
    for cid in ids[:19]:
        assert _fb(c, cid, answer="right" if cid != ids[0] else "wrong").status_code == 200
    u = c.get("/v1/scorecard").json()["usage"]
    assert u == {"checks_total": 20, "checks_last_30_days": 20, "feedback_answers": 19, "feedback_right": None}
    c.app.state.usage.add_feedback(ids[19], "unsure", None, None)  # direct call: the endpoint allows 20 posts per hour
    post(c, img_bytes(0))
    u = c.get("/v1/scorecard").json()["usage"]  # not frozen: reflects the new check and the 20th answer
    assert u == {"checks_total": 21, "checks_last_30_days": 21, "feedback_answers": 20, "feedback_right": 18}


def test_old_days_count_in_total_only(build, tmp_path, models_dir):
    (models_dir / "v1" / "scorecard.json").write_text(json.dumps({"model_version": "v1"}))
    c = build()
    post(c, img_bytes(255))
    with sqlite3.connect(tmp_path / "usage.sqlite") as con:
        con.execute("INSERT INTO daily_counts VALUES ('2020-01-01', 'file', 'likely_ai', 5)")
    u = c.get("/v1/scorecard").json()["usage"]
    assert (u["checks_total"], u["checks_last_30_days"]) == (6, 1)


def test_check_succeeds_when_db_unwritable(tmp_path, models_dir):
    blocker = tmp_path / "file"
    blocker.write_text("")  # a file where the DB folder should be, so no write can succeed
    app = create_app(StubEmbedder(), None, models_dir=models_dir,
                     budget_path=tmp_path / "budget.json", usage_path=blocker / "usage.sqlite")
    c = TestClient(app)
    r = post(c, img_bytes(255))
    assert r.status_code == 200 and r.json()["check_id"] is None and r.json()["verdict"] == "likely_ai"


def test_feedback_report_csv(build, tmp_path, monkeypatch, capsys):
    from ml.eval import feedback_report
    c = build()
    cid = post(c, img_bytes(255)).json()["check_id"]
    assert _fb(c, cid, answer="wrong", actual="real", comment="=1+1, a \"quote\"").status_code == 200
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path))
    feedback_report.main()
    assert "total checks: 1" in capsys.readouterr().out
    lines = (tmp_path / "feedback.csv").read_text().splitlines()
    assert lines[0].startswith("day,kind,verdict") and ",wrong,real,\"'=1+1, a \"\"quote\"\"\"" in lines[1]
