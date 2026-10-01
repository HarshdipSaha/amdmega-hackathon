"""Resident worker: owns the engine and the index, serves index / query / status requests (spec §2)."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import traceback
from pathlib import Path

from . import pipeline, protocol
from .index import Index, build_index

INDEX_BUDGET_S = float(os.environ.get("SB_INDEX_BUDGET_S", "540"))
MIN_INDEX_S = 120.0


class Worker:
    def __init__(self, engine, index_dir: Path, started: float | None = None, opener=open,
                 parse_workers: int = int(os.environ.get("SB_PARSE_WORKERS", "4"))):
        self.engine, self.index_dir = engine, Path(index_dir)
        self.started = started or time.time()
        self.opener, self.parse_workers = opener, parse_workers
        self.index = Index.load(self.index_dir)
        self.indexing = False
        self.indexed_once = False        # only the first index is charged to the container's startup budget
        self.index_lock = threading.Lock()
        self.engine_lock = threading.Lock()

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "status":
            return {"ok": True, "engine": getattr(self.engine, "name", "?"), "indexing": self.indexing,
                    "indexed": self.index is not None, "stats": self.index.stats if self.index else {}}
        if op == "index":
            return self._index(req)
        if op == "query":
            return self._query(req)
        return {"ok": False, "error": f"unknown op {op!r}"}

    def _index(self, req: dict) -> dict:
        if self.indexed_once:
            remaining = INDEX_BUDGET_S
        else:
            remaining = self.started + INDEX_BUDGET_S - time.time()
        if req.get("deadline_epoch"):
            remaining = min(remaining, float(req["deadline_epoch"]) - time.time())
        deadline = time.monotonic() + max(MIN_INDEX_S, remaining)
        with self.index_lock:
            self.indexing = True
            try:
                with self.engine_lock:
                    idx = build_index(Path(req["corpus"]), self.engine, self.index_dir, deadline,
                                      workers=self.parse_workers, opener=self.opener)
                self.index = idx
                self.indexed_once = True
            finally:
                self.indexing = False
        return {"ok": True, "stats": idx.stats, "skipped": idx.ledger}

    def _query(self, req: dict) -> dict:
        def refuse(reason):
            return {"ok": True, "answer": "", "citations": [], "confidence": 0.0, "diag": {"reason": reason}}
        if self.indexing:
            return refuse("indexing in progress")
        if self.index is None:
            self.index = Index.load(self.index_dir)
        if self.index is None:
            return refuse("no index")
        with self.engine_lock:
            res = pipeline.answer(self.index, self.engine, str(req.get("query", "")), float(req.get("deadline_s", 24)))
        return {"ok": True, **res}


def _serve_conn(worker: Worker, conn: socket.socket) -> None:
    with conn:
        conn.settimeout(900)
        try:
            resp = worker.handle(protocol.recv_line(conn))
        except Exception:  # noqa: BLE001 - a bad request must never kill the worker
            resp = {"ok": False, "error": traceback.format_exc(limit=3)}
        try:
            protocol.send_line(conn, resp)
        except OSError:
            pass


def serve(worker: Worker, addr: str | None = None, ready_file: str | None = None,
          stop: threading.Event | None = None) -> None:
    srv = protocol.listen(addr)
    srv.settimeout(0.5)
    if ready_file:
        Path(ready_file).write_text(json.dumps({"pid": os.getpid(), "addr": addr or protocol.address(),
                                                "startup_s": round(time.time() - worker.started, 2)}))
    try:
        while not (stop and stop.is_set()):
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(target=_serve_conn, args=(worker, conn), daemon=True).start()
    finally:
        srv.close()


def make_engine():
    if os.environ.get("SB_ENGINE", "qwen") == "fake":
        from .engine import FakeEngine
        return FakeEngine.from_env()
    from .engine_qwen import QwenEngine
    return QwenEngine(os.environ.get("SB_READER", "/models/reader"), os.environ.get("SB_EMBEDDER", "/models/embedder"))


def main() -> None:
    started = float(os.environ.get("SB_STARTED_EPOCH", time.time()))
    engine = make_engine()
    engine.warm_up()
    worker = Worker(engine, Path(os.environ.get("SB_INDEX_DIR", "/app/index")), started=started)
    print(f"sourcebound worker ready in {time.time() - started:.1f}s engine={getattr(engine, 'name', '?')}", flush=True)
    serve(worker, ready_file=os.environ.get("SB_READY_FILE", "/tmp/sourcebound.ready"))


if __name__ == "__main__":
    main()
