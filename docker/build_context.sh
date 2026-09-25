#!/bin/sh
# Run on the Linux AMD GPU builder to prepare the Docker build context.
# It resolves /workspace/models/current (a symlink) into a real directory.
set -e
cd "$(dirname "$0")/.."   # repo root

echo "Preparing Docker build context..."
rm -rf models
mkdir -p models

if [ -L /workspace/models/current ]; then
    cp -rL /workspace/models/current models/current
elif [ -d /workspace/models/current ]; then
    cp -r /workspace/models/current models/current
else
    echo "ERROR: /workspace/models/current not found"
    exit 1
fi

SIZE=$(du -sh models/current | cut -f1)
echo "Copied model: models/current ($SIZE)"
echo "Build context ready. Now run:"
echo "  docker build -f docker/Dockerfile -t roadread:rc1 ."
echo "  sh docker/check_container.sh roadread:rc1 eval/samples"
