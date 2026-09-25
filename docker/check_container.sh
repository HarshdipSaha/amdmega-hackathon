#!/bin/sh
# usage: docker/check_container.sh <image> <dir-with-image_01..10>
# Run this on the Linux docker builder with the AMD GPU after `docker build`.
set -e
IMG=$1; IN=$2

echo "=== size check ==="
SIZE=$(docker image inspect --format '{{.Size}}' "$IMG")
echo "size_bytes=$SIZE"
[ "$SIZE" -le 64424509440 ] || { echo "FAIL: image exceeds 60 GiB"; exit 1; }

echo "=== base layer check ==="
BASE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
docker image inspect --format '{{json .RootFS.Layers}}' "$BASE" > /tmp/base.json
docker image inspect --format '{{json .RootFS.Layers}}' "$IMG" | \
    python3 -c "import json,sys
b=json.load(open('/tmp/base.json'))
i=json.load(sys.stdin)
assert i[:len(b)]==b,'base layers not an ordered prefix of image layers'
print('layers ok',len(b),'base layers')"

echo "=== startup timing ==="
T0=$(date +%s)
C=$(docker run -d --device=/dev/kfd --device=/dev/dri --group-add video \
    -v "$IN":/app/input "$IMG")
trap "docker rm -f $C > /dev/null" EXIT
until docker logs "$C" 2>&1 | grep -q "roadread ready"; do
    sleep 2
    [ $(( $(date +%s) - T0 )) -lt 600 ] || { echo "FAIL: startup timeout"; exit 1; }
done
echo "startup_s=$(( $(date +%s) - T0 ))"

echo "=== per-image inference ==="
for f in $(ls "$IN"/*.png "$IN"/*.jpg "$IN"/*.tiff "$IN"/*.tif 2>/dev/null | sort); do
    bn=$(basename "$f")
    s=$(date +%s%N)
    docker exec "$C" python3 /app/app.py --input-image /app/input/$bn
    elapsed=$(( ($(date +%s%N) - s) / 1000000 ))
    out=$(docker exec "$C" cat /app/output/${bn%.*}_output.json 2>/dev/null || echo '{"text":""}')
    echo "$bn ${elapsed}ms $out"
    [ $elapsed -le 25000 ] || echo "WARN: $bn exceeded 25s"
done

echo "=== all checks passed ==="
