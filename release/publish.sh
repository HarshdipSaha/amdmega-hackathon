#!/usr/bin/env bash
# Assemble and push the release image in the registry: base layers untouched, our layers appended (spec §10).
# Required env: MC3_IMAGE (registry/repo, from a CI secret; never echoed), TAG,
#               READER_REPO READER_REV EMBEDDER_REPO EMBEDDER_REV (pinned Hugging Face revisions).
set -euo pipefail
: "${MC3_IMAGE:?}" "${TAG:?}" "${READER_REPO:?}" "${READER_REV:?}" "${EMBEDDER_REPO:?}" "${EMBEDDER_REV:?}"
BASE=docker.io/rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
WIP="$MC3_IMAGE:wip-$TAG"
OUT=layers
mkdir -p "$OUT"

# The supervisor goes in CMD (spec §10), so `docker run <image> python3 /app/app.py ...` still runs that command.
# That only holds if the base has no ENTRYPOINT; stop rather than guess if it does.
BASE_CFG=$(crane config --platform linux/amd64 "$BASE")
BASE_EP=$(echo "$BASE_CFG" | python -c "import json,sys; print(json.load(sys.stdin)['config'].get('Entrypoint') or '')")
if [ -n "$BASE_EP" ]; then echo "base image has ENTRYPOINT $BASE_EP; decide how to clear it before publishing"; exit 1; fi

python release/build_layers.py deps --out "$OUT"
python release/build_layers.py app --out "$OUT"
crane append --platform linux/amd64 -b "$BASE" -f "$OUT/deps.tar" -f "$OUT/app.tar" -t "$WIP" > /dev/null
rm -rf "$OUT/deps.tar" "$OUT/app.tar" "$OUT/stage-deps"

# One <= 4 GiB group at a time: download, tar, append, delete.
python release/build_layers.py push-weights --out "$OUT" --repo "$READER_REPO" --rev "$READER_REV" --slot reader --image "$WIP"
python release/build_layers.py push-weights --out "$OUT" --repo "$EMBEDDER_REPO" --rev "$EMBEDDER_REV" --slot embedder --image "$WIP"

# Keep every base env var; prepend our PYTHONPATH to any the base defines.
BASE_PP=$(echo "$BASE_CFG" | python -c "import json,sys; e=dict(x.split('=',1) for x in json.load(sys.stdin)['config'].get('Env',[])); print(e.get('PYTHONPATH',''))")
PP="/app/pylib:/app${BASE_PP:+:$BASE_PP}"
crane mutate "$WIP" --cmd=python3,-m,sourcebound.supervisor --workdir /app \
  --env "PYTHONPATH=$PP" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env SB_READER=/models/reader --env SB_EMBEDDER=/models/embedder --env SB_INDEX_DIR=/app/index \
  -t "$MC3_IMAGE:$TAG" > /dev/null
echo "published tag $TAG"
