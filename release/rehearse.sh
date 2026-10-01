#!/usr/bin/env bash
# Release rehearsal on the GPU notebook (spec §10). The notebook pod runs the mandated base, so extracting ONLY
# our appended layers of the pushed image onto / reproduces the image filesystem. Then run the CMD and the harness.
# Usage (via remote-mc3.js rehearse): MC3_IMAGE=docker.io/<user>/<repo>:<tag> bash release/rehearse.sh
# Fallback when the pod cannot reach the registry (gate G5): LOCAL_LAYERS=1 bash release/rehearse.sh rebuilds the
# app and deps layers on the pod from the same commit and links /models to the notebook's model directories.
set -euo pipefail
LOCAL_LAYERS=${LOCAL_LAYERS:-0}
[ "$LOCAL_LAYERS" = 1 ] || : "${MC3_IMAGE:?}"
R=/workspace/mc3
BASE=docker.io/rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
CRANE=/workspace/bin/crane
if [ "$LOCAL_LAYERS" != 1 ] && [ ! -x "$CRANE" ]; then
  mkdir -p /workspace/bin
  curl -sL https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz \
    | tar -xz -C /workspace/bin crane
fi
pkill -f '[s]ourcebound' || true
rm -rf /app /models
if [ "$LOCAL_LAYERS" = 1 ]; then
  (cd $R && python release/build_layers.py deps --out /tmp/sb-layers && python release/build_layers.py app --out /tmp/sb-layers)
  tar -xf /tmp/sb-layers/deps.tar -C / && tar -xf /tmp/sb-layers/app.tar -C /
  mkdir -p /models && ln -sfn "$(readlink -f /workspace/models/mc3-reader)" /models/reader     && ln -sfn "$(readlink -f /workspace/models/mc3-embedder)" /models/embedder
else
  if ! $CRANE manifest --platform linux/amd64 "$MC3_IMAGE" > /dev/null; then
    echo "registry unreachable from the pod (gate G5): rerun with LOCAL_LAYERS=1"; exit 2
  fi
  nb=$($CRANE manifest --platform linux/amd64 "$BASE" | python -c "import json,sys; print(len(json.load(sys.stdin)['layers']))")
  repo=${MC3_IMAGE%:*}
  for d in $($CRANE manifest --platform linux/amd64 "$MC3_IMAGE" | python -c "import json,sys; print(' '.join(l['digest'] for l in json.load(sys.stdin)['layers'][$nb:]))"); do
    $CRANE blob "$repo@$d" | tar -xz -C /
  done
fi
du -shL /app /models/*

# The image's environment (check_image.py verifies the same values in the pushed config).
export PYTHONPATH=/app/pylib:/app HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SB_READER=/models/reader \
       SB_EMBEDDER=/models/embedder SB_INDEX_DIR=/app/index SB_OUTPUT_DIR=/app/output
cd /app
python3 -c "import torch; assert 'rocm' in torch.__version__, torch.__version__; print('torch', torch.__version__)"

T0=$(date +%s)
# Drop DAC_OVERRIDE like the grader when the pod allows it (gate G4), so the mode-000 file is really unreadable.
NODAC=""
if setpriv --bounding-set=-dac_override,-dac_read_search --inh-caps=-dac_override,-dac_read_search true 2>/dev/null; then
  NODAC="setpriv --bounding-set=-dac_override,-dac_read_search --inh-caps=-dac_override,-dac_read_search"
fi
nohup $NODAC python3 -m sourcebound.supervisor > /tmp/rehearse-cmd.log 2>&1 &
cp -r $R/eval_mc3/kit/mc3-corpus/. /app/corpus/ && mkdir -p /app/corpus/archive && chmod 000 /app/corpus/vendor/internal_audit.txt
python3 $R/eval_mc3/run_eval.py --questions $R/eval_mc3/kit/sample-questions.json --corpus /app/corpus \
  --out $R/results/rehearse-kit.jsonl --app /app/app.py | tail -1 | tee $R/results/rehearse-kit.summary
echo "container-start -> end of kit run: $(( $(date +%s) - T0 )) s"

# Liveness: kill the worker; the supervisor must bring it back and answer again.
pkill -9 -f '[s]ourcebound.worker'
sleep 5
for i in $(seq 1 120); do python3 -c "import sys; sys.path[:0]=['/app']; from sourcebound import protocol; protocol.request({'op':'status'},2)" 2>/dev/null && break; sleep 3; done
python3 /app/app.py --corpus /app/corpus --query-id liveness --query "What error code is logged when the thermal throttle engages?"
cat /app/output/liveness_output.json; echo
pgrep -f sourcebound.supervisor > /dev/null && echo "supervisor alive"
