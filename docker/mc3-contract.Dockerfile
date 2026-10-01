# CPU-only contract image for CI (NOT the submission). Same code paths as the release, FakeEngine instead of
# the GPU engine, so the grader's isolation flags (--network none, --cap-drop DAC_OVERRIDE, root) can be
# reproduced on a GitHub-hosted runner. See release/contract_check.sh.
FROM python:3.14-slim
RUN pip install --no-cache-dir pymupdf openpyxl numpy pillow
WORKDIR /app
COPY sourcebound /app/sourcebound
COPY mc3/app.py /app/app.py
COPY eval_mc3/__init__.py eval_mc3/score.py eval_mc3/run_eval.py eval_mc3/vram.py /app/eval_mc3/
COPY eval_mc3/kit /app/eval_mc3/kit
COPY release/contract_check.sh /app/release/contract_check.sh
ENV PYTHONPATH=/app SB_ENGINE=fake SB_INDEX_DIR=/tmp/sb-index SB_OUTPUT_DIR=/app/output \
    SB_FAKE_REPLIES=/fake/replies.json SB_FAKE_TRANSCRIPTS=/fake/transcripts.json
CMD ["sh", "/app/release/contract_check.sh"]
