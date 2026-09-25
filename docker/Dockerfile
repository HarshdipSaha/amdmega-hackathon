# syntax=docker/dockerfile:1
# ROADREAD — license-plate + road-sign OCR on AMD ROCm
# Base: rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
# Model baked in: Qwen/Qwen3-VL-4B-Instruct (8.3 GB)
# Uncompressed size target: ≤ 60 GiB (64,424,509,440 bytes)
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

# 1. Install Python dependencies (pinned; no torch/numpy/PIL — already in the base)
COPY app/requirements.lock /app/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt \
 && python3 -c "import torch; assert 'rocm' in torch.__version__, f'not ROCm: {torch.__file__}'"

# 2. Copy application code
COPY roadread /app/roadread
COPY app/app.py /app/app.py
COPY docker/entrypoint.sh /app/entrypoint.sh

# 3. Bake model weights (symlink resolved to real directory at build time)
#    On the Linux builder: cp -rL /workspace/models/current models/current
COPY models/current /models/current

# 4. Environment
ENV ROADREAD_MODEL=/models/current \
    ROADREAD_OUTPUT_DIR=/app/output \
    ROADREAD_READY_FILE=/tmp/roadread.ready \
    HF_HUB_OFFLINE=1 \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1

# 5. Ensure directories and permissions
RUN mkdir -p /app/input /app/output && chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
