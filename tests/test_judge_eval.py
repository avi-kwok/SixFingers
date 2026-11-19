import json
import time

from tests.api.conftest import JUDGE_JSON, command_script
from tests.test_ml_pipeline import env, FakeBackbone  # noqa: F401


def _judge_script(tmp_path, fail_at: int, fail_stdout: str | None) -> str:
    """Command backend that exits 75 on call number fail_at, otherwise prints a valid verdict."""
    counter = tmp_path / "calls.txt"
    fail_print = f"    print({fail_stdout!r})\n" if fail_stdout is not None else ""
    body = ("import pathlib, sys\n"
            f"p = pathlib.Path({str(counter)!r})\n"
            "n = int(p.read_text()) + 1 if p.exists() else 1\n"
            "p.write_text(str(n))\n"
            "sys.stdin.read()\n"
            f"if n == {fail_at}:\n{fail_print}    sys.exit(75)\n"
            f"print({JUDGE_JSON!r})\n")
    return command_script(tmp_path, body)


def _prepare(monkeypatch, tmp_path, cmd):
    from ml.features import extract
    from ml.probes.train import train
    extract.run(["clip_vitl14"], limit=200)
    train(["clip_vitl14"], promote=True)
    sleeps = []
    monkeypatch.setenv("SIXFINGERS_JUDGE_BACKEND", "command")
    monkeypatch.setenv("SIXFINGERS_JUDGE_CMD", cmd)
    real_sleep = time.sleep

    def fake_sleep(s):  # time.sleep is global: subprocess polling also sleeps, so record only retry waits
        if s >= 1:
            sleeps.append(s)
        else:
            real_sleep(s)

    monkeypatch.setattr("ml.judge_eval.run.time.sleep", fake_sleep)
    return sleeps


def test_command_backend_waits_on_75_and_resumes(env, tmp_path, monkeypatch):
    from ml.common.paths import data_dir
    from ml.features import store
    from ml.judge_eval import run as j
    cmd = _judge_script(tmp_path, fail_at=3, fail_stdout='{"retry_after_s": 5}')
    sleeps = _prepare(monkeypatch, tmp_path, cmd)
    j.main(["--approve-spend", "10", "--n", "4"])
    assert sleeps == [5]
    sl = j.stratified(store.load_manifest().query("split == 'report_test'"), 4)
    arms = [a for a, m, _ in j.ARMS if m is not None]
    rows = [json.loads(line) for line in (data_dir() / "eval" / "judge_cache_prod.jsonl").read_text().splitlines()]
    assert {(r["arm"], r["image_id"]) for r in rows} == {(a, i) for a in arms for i in sl.image_id}
    assert all(r["verdict"] != "error" for r in rows)
    assert "Backend: command" in (env / "results.md").read_text()


def test_exit_75_without_stdout_waits_default(env, tmp_path, monkeypatch):
    from ml.judge_eval import run as j
    cmd = _judge_script(tmp_path, fail_at=1, fail_stdout=None)
    sleeps = _prepare(monkeypatch, tmp_path, cmd)
    j.main(["--approve-spend", "10", "--n", "4"])
    assert sleeps == [1800]


def test_bad_image_caches_error_row_and_run_continues(env, tmp_path, monkeypatch):
    from api.validate import UploadError, validate_upload
    from ml.common.paths import data_dir
    from ml.judge_eval import run as j
    _prepare(monkeypatch, tmp_path, _judge_script(tmp_path, fail_at=0, fail_stdout=None))
    seen = []

    def flaky(data):
        seen.append(1)
        if len(seen) == 1:
            raise UploadError(400, "Could not read this image.")
        return validate_upload(data)
    monkeypatch.setattr(j, "validate_upload", flaky)
    j.main(["--approve-spend", "10", "--n", "4"])
    rows = [json.loads(line) for line in (data_dir() / "eval" / "judge_cache_prod.jsonl").read_text().splitlines()]
    assert sum(r.get("error") == "invalid_image" for r in rows) == 1
    assert len(rows) == len(seen) > 1  # every other image was still judged
