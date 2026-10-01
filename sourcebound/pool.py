"""A small process pool whose stuck workers can be killed one at a time (spec §3, per-file timeout).

concurrent.futures cannot cancel a running task, so a hung PDF would hold a worker forever. Here each child
serves jobs over a Pipe; a child that overruns its timeout is terminated and replaced, and the job is recorded
as a failure. Children are started with "spawn" so they never inherit the GPU worker's HIP state.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import time
from multiprocessing.connection import wait


def _resolve(fn_path: str):
    mod, name = fn_path.split(":")
    return getattr(importlib.import_module(mod), name)


def _child(conn, fn_path: str) -> None:
    fn = _resolve(fn_path)
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        try:
            conn.send(("ok", fn(*job)))
        except BaseException as e:  # noqa: BLE001 - report every failure to the parent
            conn.send(("err", f"{type(e).__name__}: {e}"))


class ParserPool:
    def __init__(self, fn_path: str = "sourcebound.parsers:parse_file", workers: int = 4, timeout_s: float = 20.0):
        self.fn_path, self.workers, self.timeout_s = fn_path, max(1, workers), timeout_s
        self.ctx = mp.get_context("spawn")

    def _spawn(self) -> dict:
        parent, child = self.ctx.Pipe()
        p = self.ctx.Process(target=_child, args=(child, self.fn_path), daemon=True)
        p.start()
        child.close()
        return {"p": p, "c": parent, "job": None, "t": 0.0}

    def _replace(self, k: dict) -> None:
        try:
            k["p"].terminate()
            k["p"].join(2)
            k["c"].close()
        except Exception:
            pass
        k.update(self._spawn())

    def run(self, jobs: list[tuple[str, tuple]], deadline: float | None = None) -> dict[str, tuple[str, object]]:
        """jobs: [(key, args)]. Returns {key: ("ok", result) | ("err", reason)}. deadline is time.monotonic()."""
        results: dict[str, tuple[str, object]] = {}
        pending = list(jobs)
        kids = [self._spawn() for _ in range(min(self.workers, len(pending)))]
        try:
            while pending or any(k["job"] is not None for k in kids):
                if deadline is not None and time.monotonic() > deadline:
                    for key, _ in pending:
                        results[key] = ("err", "index deadline reached before parsing")
                    pending.clear()
                    for k in kids:
                        if k["job"] is not None:
                            results[k["job"]] = ("err", "index deadline reached during parsing")
                            self._replace(k)
                            k["job"] = None
                    break
                for k in kids:
                    if k["job"] is None and pending:
                        key, args = pending.pop(0)
                        try:
                            k["c"].send(args)
                        except (BrokenPipeError, OSError):
                            self._replace(k)
                            k["c"].send(args)
                        k["job"], k["t"] = key, time.monotonic()
                busy = [k["c"] for k in kids if k["job"] is not None]
                ready = wait(busy, timeout=0.1) if busy else []
                for k in kids:
                    if k["job"] is None:
                        continue
                    if k["c"] in ready:
                        try:
                            results[k["job"]] = k["c"].recv()
                        except (EOFError, OSError):
                            results[k["job"]] = ("err", "parser process died")
                            self._replace(k)
                        k["job"] = None
                    elif time.monotonic() - k["t"] > self.timeout_s:
                        results[k["job"]] = ("err", f"parse timeout after {self.timeout_s:.0f}s")
                        self._replace(k)
                        k["job"] = None
        finally:
            for k in kids:
                try:
                    k["c"].send(None)
                except Exception:
                    pass
            for k in kids:
                k["p"].join(1)
                if k["p"].is_alive():
                    k["p"].terminate()
        return results
