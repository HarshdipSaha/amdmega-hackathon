#!/usr/bin/env bash
# Start or stop the resident worker on the GPU notebook:
#   bash eval_mc3/worker_ctl.sh start [nodac] | stop
# This lives in a file on purpose. gpu.js runs commands as `bash -lc "<command>"`; if that command line itself
# contained "sourcebound.worker", `pkill -f` would match and kill the shell running it. Here the only processes
# whose command lines contain the name are the worker and the supervisor.
set -u
NODAC="setpriv --bounding-set=-dac_override,-dac_read_search --inh-caps=-dac_override,-dac_read_search"
LOG=/workspace/mc3-worker.log

stop() {
  pkill -f '[s]ourcebound.supervisor'   # a supervisor started by app.py would otherwise respawn the worker
  pkill -f '[s]ourcebound.worker'
  sleep 1
  rm -f /tmp/sourcebound.ready /tmp/sourcebound.sock
}

case "${1:-}" in
  stop)
    stop
    echo stopped
    ;;
  start)
    stop
    rm -rf "${SB_INDEX_DIR:-/tmp/sb-index}"        # fresh index and transcript cache for every worker
    PREFIX=""
    [ "${2:-}" = nodac ] && PREFIX="$NODAC"
    nohup $PREFIX python -m sourcebound.worker > "$LOG" 2>&1 &
    WPID=$!
    echo "spawned worker pid $WPID"
    for _ in $(seq 1 300); do
      if [ -f /tmp/sourcebound.ready ]; then cat /tmp/sourcebound.ready; echo; exit 0; fi
      if ! kill -0 "$WPID" 2>/dev/null; then echo "worker exited early:"; tail -40 "$LOG"; exit 1; fi
      sleep 2
    done
    echo "timeout waiting for ready file"; tail -40 "$LOG"; exit 1
    ;;
  *)
    echo "usage: worker_ctl.sh start [nodac] | stop"; exit 2
    ;;
esac
