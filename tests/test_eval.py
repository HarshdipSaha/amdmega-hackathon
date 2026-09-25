import json, os, subprocess, sys, time
from pathlib import Path
from PIL import Image

def _worker(tmp_path, port, reply):
    env = {**os.environ, "ROADREAD_ENGINE": "fake", "ROADREAD_FAKE_REPLY": reply, "ROADREAD_PORT": str(port),
           "ROADREAD_READY_FILE": str(tmp_path / f"ready{port}"), "PYTHONPATH": str(Path.cwd())}
    w = subprocess.Popen([sys.executable, "-m", "roadread.worker"], env=env)
    for _ in range(200):
        if (tmp_path / f"ready{port}").exists(): break
        time.sleep(0.1)
    return w, env

def test_run_eval_scores_times_and_overhead(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png"); Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    rows = [{"image": "a.png", "gold": "stop", "slice": "word_sign"}, {"image": "b.png", "gold": "35", "slice": "advisory_plaque"}]
    (tmp_path / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    w, env = _worker(tmp_path, 47903, "KIND: sign\nTEXT: STOP")
    try:
        r = subprocess.run([sys.executable, "eval/run_eval.py", "--manifest", str(tmp_path / "m.jsonl"),
                            "--out", str(tmp_path / "res.jsonl")], env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        s = json.loads(r.stdout.strip().splitlines()[-1])
        assert s["correct"] == 1 and s["total"] == 2 and s["by_slice"]["word_sign"] == [1, 1]
        assert s["max_s"] < 30 and s["violations"] == 0 and s["max_overhead_s"] < 5
        res = [json.loads(l) for l in (tmp_path / "res.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [x["ok"] for x in res] == [True, False] and res[1]["pred"] == "STOP"
    finally:
        w.terminate()

def test_run_eval_records_missing_worker_as_failure(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    (tmp_path / "m.jsonl").write_text(json.dumps({"image": "a.png", "gold": "STOP"}) + "\n", encoding="utf-8")
    env = {**os.environ, "ROADREAD_PORT": "47998", "PYTHONPATH": str(Path.cwd())}
    r = subprocess.run([sys.executable, "eval/run_eval.py", "--manifest", str(tmp_path / "m.jsonl"),
                        "--out", str(tmp_path / "res.jsonl")], env=env, capture_output=True, text=True, timeout=120)
    s = json.loads(r.stdout.strip().splitlines()[-1])
    assert r.returncode == 0 and s["correct"] == 0 and s["total"] == 1
