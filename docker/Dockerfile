# syntax=docker/dockerfile:1
# ROADREAD — license-plate + road-sign OCR on AMD ROCm
# Base: rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
# Uncompressed size target: <= 60 GiB (64,424,509,440 bytes)
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

# 1. Install Python dependencies (pinned; no torch/numpy/PIL — already in the base)
COPY app/requirements.lock /app/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt \
 && python3 -c "import torch; assert 'rocm' in torch.__version__, f'not ROCm: {torch.__file__}'"

# 2. Copy application code
COPY roadread /app/roadread
COPY app/app.py /app/app.py
COPY docker/entrypoint.sh /app/entrypoint.sh

# 3. Environment
# If local model weights are present at build time, they are used;
# otherwise downloads Qwen/Qwen3-VL-4B-Instruct at startup under the 600s startup budget
ENV ROADREAD_MODEL=/models/current \
    ROADREAD_MODEL_REPO=Qwen/Qwen3-VL-4B-Instruct \
    ROADREAD_OUTPUT_DIR=/app/output \
    ROADREAD_READY_FILE=/tmp/roadread.ready \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1

# 4. Ensure directories and permissions
RUN mkdir -p /models/current /app/input /app/output && chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
