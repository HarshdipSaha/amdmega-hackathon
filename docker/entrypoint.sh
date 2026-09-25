#!/bin/sh
set -e

# Download-at-startup fallback: only runs if the baked model is missing
if [ -n "$ROADREAD_MODEL_REPO" ] && [ ! -f /models/current/config.json ]; then
    echo "roadread: downloading model $ROADREAD_MODEL_REPO ..."
    HF_HUB_OFFLINE=0 python3 -c "
import os, time
from huggingface_hub import snapshot_download as d
t = time.time()
d(os.environ['ROADREAD_MODEL_REPO'],
  revision=os.environ.get('ROADREAD_MODEL_REV'),
  local_dir='/models/current')
print('download_s', round(time.time()-t, 1))"
fi

# Start the resident worker in background
python3 -m roadread.worker > /tmp/worker.log 2>&1 &
WORKER_PID=$!

# Wait for the ready file (max 600 s startup budget)
i=0
while [ ! -f /tmp/roadread.ready ]; do
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo "roadread worker exited:"
        cat /tmp/worker.log
        exit 1
    fi
    i=$((i+1))
    [ $i -lt 600 ] || { echo "timeout waiting for worker ready"; cat /tmp/worker.log; exit 1; }
    sleep 1
done

echo "roadread ready: $(cat /tmp/roadread.ready)"
exec tail -f /tmp/worker.log
