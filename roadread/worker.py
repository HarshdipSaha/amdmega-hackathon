"""Resident worker: load engine once, warm up, write ready file, serve requests sequentially."""
import json, os, socket, time, traceback
from PIL import Image
from . import protocol
from .decode import load_rgb
from .pipeline import Pipeline

def make_engine():
    kind = os.environ.get("ROADREAD_ENGINE", "qwen")
    if kind == "fake":
        from .engine_fake import FakeEngine
        return FakeEngine([os.environ.get("ROADREAD_FAKE_REPLY", "KIND: sign\nTEXT: ")])
    from .engine_qwen import QwenEngine
    return QwenEngine(os.environ.get("ROADREAD_MODEL", "/models/current"), attn=os.environ.get("ROADREAD_ATTN", "sdpa"))

def main() -> None:
    t0 = time.monotonic()
    engine = make_engine()
    pipe = Pipeline(engine)
    pipe.run(Image.new("RGB", (256, 128), "white"), deadline_s=60)            # warm-up: kernels, allocator, caches
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", protocol.PORT)); srv.listen(8)
    ready = os.environ.get("ROADREAD_READY_FILE", "/tmp/roadread.ready")
    with open(ready, "w") as f:
        json.dump({"pid": os.getpid(), "startup_s": round(time.monotonic() - t0, 2), "engine": engine.name}, f)
    print(f"roadread worker ready in {time.monotonic() - t0:.1f}s engine={engine.name}", flush=True)
    while True:
        conn, _ = srv.accept()
        req = None
        try:
            req = protocol.recv_line(conn)
            r = pipe.run(load_rgb(req["image"]), deadline_s=float(req.get("deadline_s", 25)))
            protocol.send_line(conn, {"id": req["id"], "text": r.text, "confidence": r.confidence, "diag": r.diag})
            print(json.dumps({"image": req["image"], "text": r.text, "elapsed_s": r.diag["elapsed_s"]}, ensure_ascii=False), flush=True)
        except Exception:
            traceback.print_exc()
            try:
                protocol.send_line(conn, {"id": (req or {}).get("id"), "text": "", "diag": {"error": traceback.format_exc(limit=2)}})
            except Exception:
                pass
        finally:
            conn.close()

if __name__ == "__main__":
    main()
