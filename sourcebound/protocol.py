"""Newline-delimited JSON between the thin client and the worker. Standard library only.

Address: SB_ADDR = "unix:/path" or "tcp:host:port". Default: a unix socket on POSIX, localhost TCP on Windows
(Windows CPython has no AF_UNIX).
"""
from __future__ import annotations

import json
import os
import socket

MAX_LINE = 8 * 2**20
DEFAULT_ADDR = "unix:/tmp/sourcebound.sock" if hasattr(socket, "AF_UNIX") else "tcp:127.0.0.1:47823"


def address() -> str:
    return os.environ.get("SB_ADDR", DEFAULT_ADDR)


def _parse(addr: str):
    kind, _, rest = addr.partition(":")
    if kind == "unix":
        return socket.AF_UNIX, rest
    host, _, port = rest.rpartition(":")
    return socket.AF_INET, (host, int(port))


def connect(addr: str | None = None, timeout: float = 2.0) -> socket.socket:
    fam, target = _parse(addr or address())
    s = socket.socket(fam, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(target)
    except BaseException:
        s.close()
        raise
    return s


def listen(addr: str | None = None) -> socket.socket:
    fam, target = _parse(addr or address())
    s = socket.socket(fam, socket.SOCK_STREAM)
    if fam == getattr(socket, "AF_UNIX", None):
        try:
            os.unlink(target)
        except FileNotFoundError:
            pass
    else:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(target)
    s.listen(16)
    return s


def send_line(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def recv_line(sock: socket.socket) -> dict:
    buf = bytearray()
    while not buf.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("connection closed before a full message")
        buf += chunk
        if len(buf) > MAX_LINE:
            raise ValueError("message too large")
    return json.loads(buf.decode("utf-8"))


def request(obj: dict, timeout: float, addr: str | None = None) -> dict:
    with connect(addr, timeout=min(timeout, 2.0)) as s:
        s.settimeout(timeout)
        send_line(s, obj)
        return recv_line(s)
