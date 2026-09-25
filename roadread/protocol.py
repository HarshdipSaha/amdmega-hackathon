"""Newline-delimited JSON over localhost TCP. Standard library only (imported by the thin client)."""
import json, os, socket, uuid

PORT = int(os.environ.get("ROADREAD_PORT", "47811"))

def send_line(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))

def recv_line(sock: socket.socket) -> dict:
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("closed before newline")
        buf += chunk
    return json.loads(buf.decode("utf-8"))

def request(image: str, deadline_s: float, port: int = PORT) -> dict:
    rid = uuid.uuid4().hex
    with socket.create_connection(("127.0.0.1", port), timeout=2) as s:
        s.settimeout(deadline_s)
        send_line(s, {"id": rid, "image": image, "deadline_s": deadline_s})
        resp = recv_line(s)
    if resp.get("id") != rid:
        raise ValueError("stale response id")
    return resp
