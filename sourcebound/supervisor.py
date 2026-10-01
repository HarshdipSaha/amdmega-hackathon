"""Container CMD: keep a worker running and never exit (spec §2, user stories 5 and 45)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

MAX_RESTARTS = int(os.environ.get("SB_MAX_RESTARTS", "5"))


def main() -> None:
    os.environ.setdefault("SB_STARTED_EPOCH", str(time.time()))
    log_path = os.environ.get("SB_WORKER_LOG", "/tmp/sourcebound-worker.log")
    child: list[subprocess.Popen] = []

    def stop(signum, frame):
        for c in child:
            c.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    restarts = 0
    while True:
        with open(log_path, "a", encoding="utf-8") as log:
            p = subprocess.Popen([sys.executable, "-m", "sourcebound.worker"], stdout=log, stderr=subprocess.STDOUT)
            child[:] = [p]
            rc = p.wait()
        print(f"sourcebound supervisor: worker exited rc={rc}", flush=True)
        restarts += 1
        if restarts > MAX_RESTARTS:
            print("sourcebound supervisor: restart limit reached; staying alive", flush=True)
            while True:
                time.sleep(3600)
        time.sleep(min(30, 2 ** restarts))


if __name__ == "__main__":
    main()
