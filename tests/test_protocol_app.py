import json, os, socket, subprocess, sys, threading, time
from pathlib import Path
from roadread import protocol

def _stub_server(port, reply):
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port)); srv.listen()
    def loop():
        while True:
            c, _ = srv.accept()
            req = protocol.recv_line(c)
            protocol.send_line(c, {"id": req["id"], **reply}); c.close()
    threading.Thread(target=loop, daemon=True).start()
    return srv

def _run_app(tmp_path, port, name="image_01.png", extra_env=None):
    img = tmp_path / name; img.write_bytes(b"x")
    env = {**os.environ, "ROADREAD_PORT": str(port), "ROADREAD_OUTPUT_DIR": str(tmp_path / "out"),
           "PYTHONPATH": str(Path.cwd()), **(extra_env or {})}
    t = time.time()
    r = subprocess.run([sys.executable, "app/app.py", "--input-image", str(img)], env=env, capture_output=True, text=True, timeout=60)
    return r, time.time() - t, tmp_path / "out" / (Path(name).stem + "_output.json")

def test_app_writes_named_json_utf8(tmp_path):
    _stub_server(47901, {"text": "沪B·88888", "confidence": 0.9, "diag": {"elapsed_s": 0.1}})
    r, _, out = _run_app(tmp_path, 47901, "image_07.tiff")
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == {"text": "沪B·88888", "confidence": 0.9}

def test_app_clamps_confidence_and_logs_diag(tmp_path):
    _stub_server(47904, {"text": "STOP", "confidence": 7, "diag": {"elapsed_s": 0.2}})
    log = tmp_path / "diag.jsonl"
    r, _, out = _run_app(tmp_path, 47904, extra_env={"ROADREAD_DIAG_LOG": str(log)})
    assert json.loads(out.read_text(encoding="utf-8"))["confidence"] == 1.0
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert row["image"] == "image_01.png" and row["diag"]["elapsed_s"] == 0.2 and row["client_wall_s"] > 0

def test_app_removes_stale_output_and_writes_empty_when_worker_absent(tmp_path):
    stale = tmp_path / "out" / "image_01_output.json"; stale.parent.mkdir(parents=True); stale.write_text('{"text":"OLD"}')
    r, dt, out = _run_app(tmp_path, 47999)
    assert r.returncode == 0 and json.loads(out.read_text(encoding="utf-8"))["text"] == "" and dt < 10

def test_app_does_not_import_torch():
    code = "import sys; import app.app; print('torch' in sys.modules, 'transformers' in sys.modules, 'PIL' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path.cwd()).stdout
    assert out.strip() == "False False False"

def test_worker_roundtrip_and_survives_bad_request(tmp_path):
    from PIL import Image
    img = tmp_path / "s.png"; Image.new("RGB", (32, 16)).save(img)
    ready = tmp_path / "ready"
    env = {**os.environ, "ROADREAD_ENGINE": "fake", "ROADREAD_FAKE_REPLY": "KIND: sign\nTEXT: STOP",
           "ROADREAD_PORT": "47902", "ROADREAD_READY_FILE": str(ready), "PYTHONPATH": str(Path.cwd())}
    p = subprocess.Popen([sys.executable, "-m", "roadread.worker"], env=env)
    try:
        for _ in range(100):
            if ready.exists(): break
            time.sleep(0.1)
        assert json.loads(ready.read_text())["engine"] == "fake"
        assert protocol.request(str(tmp_path / "missing.png"), 10, port=47902)["text"] == ""   # error -> empty
        assert protocol.request(str(img), 10, port=47902)["text"] == "STOP"                     # still serving
    finally:
        p.terminate()

