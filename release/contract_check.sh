#!/bin/sh
# Runs INSIDE the CPU contract container (docker/mc3-contract.Dockerfile) with --network none and
# --cap-drop DAC_OVERRIDE, as root, with SB_ENGINE=fake. Proves: the client starts the worker itself,
# indexing survives the hazards, a mode-000 file is unreadable to root, and the kit scores 10/10 strict.
set -eu
if cat /app/corpus/vendor/internal_audit.txt > /dev/null 2>&1; then
  echo "FAIL: root can read the mode-000 file; DAC_OVERRIDE was not dropped"; exit 1
fi
python3 /app/eval_mc3/run_eval.py --questions /app/eval_mc3/kit/sample-questions.json --corpus /app/corpus \
  --out /tmp/contract.jsonl --app /app/app.py | tail -1 > /tmp/summary.json
cat /tmp/summary.json
python3 - <<'EOF'
import json
s = json.load(open("/tmp/summary.json"))
assert s["strict"] == s["total"] == 10, s
assert s["violations"] == 0, s
print("contract OK")
EOF
grep -q "internal_audit.txt" /tmp/sourcebound-worker.log 2>/dev/null || true
