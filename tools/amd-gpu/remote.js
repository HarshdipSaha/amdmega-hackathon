#!/usr/bin/env node
// Remote ROADREAD workflow on the AMD notebook GPU (built on gpu.js; see GPUWEBSKILL.md).
//   node remote.js sync                  tar roadread/ app/ eval/ (not eval/data) -> /workspace/roadread
//   node remote.js data <dir>            tar a local data dir (e.g. eval/data/dev) -> /workspace/roadread/<dir> (<= ~40 MB per call)
//   node remote.js setup                 pip --target /workspace/pylib, purge torch shadows, freeze -> app/requirements.lock
//   node remote.js model <hf_repo>       snapshot_download -> /workspace/models/<name>; symlink /workspace/models/current
//   node remote.js gates                 eval/gates.py -> results/gates-<ts>.json
//   node remote.js worker-start          (re)start the resident worker, wait for ready
//   node remote.js worker-stop | worker-log
//   node remote.js eval <manifest> <tag> run eval detached; then `tail <tag>` until the summary appears
//   node remote.js tail <tag>            progress + GPU memory (never launches a pod)
//   node remote.js pull <tag>            results/<tag>.{jsonl,summary,diag.jsonl} -> ./results (never launches)
//   node remote.js end                   stop worker/eval + kernels, then Turn-off Session (never launches)
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { withLab, sh, put, get, killKernels, stop } = require('./gpu');
const { openBrowser, OUT_DIR } = require('./lib');

const REPO = path.resolve(__dirname, '..', '..');
const R = '/workspace/roadread';
const ENV = `export PYTHONPATH=/workspace/pylib:${R} HF_HOME=/workspace/.cache/huggingface ROADREAD_MODEL=/workspace/models/current ROADREAD_OUTPUT_DIR=${R}/results/app_output;`;
const TAR = process.platform === 'win32' ? 'C:\\Windows\\System32\\tar.exe' : 'tar'; // bsdtar; Git's GNU tar breaks on C:
const [cmd, a, b] = process.argv.slice(2);
const NO_LAUNCH = new Set(['tail', 'pull', 'worker-stop', 'worker-log', 'end']);
if (NO_LAUNCH.has(cmd)) process.env.AUTO_LAUNCH = '0';

async function run(shell, timeoutMs) {
  return withLab(async (page, base) => {
    const r = await sh(page, base, shell, timeoutMs);
    process.stdout.write(r.output);
    if (!r.ok || /\[exit [1-9]/.test(r.output)) process.exitCode = 1;
    return r.output;
  });
}

async function upload(relPaths, remoteDir, { excludes = ['__pycache__', 'eval/data'] } = {}) {
  const rel = path.relative(REPO, path.join(OUT_DIR, 'upload.tgz'));
  const excludeArgs = excludes.flatMap(e => ['--exclude=' + e]);
  execFileSync(TAR, ['-czf', rel, ...excludeArgs, ...relPaths], { cwd: REPO });
  const size = fs.statSync(path.join(REPO, rel)).size;
  if (size > 40e6) throw new Error(`archive is ${(size / 1e6).toFixed(1)} MB; split it (put goes through page.evaluate)`);
  return withLab(async (page, base) => {
    await put(page, base, path.join(REPO, rel), 'upload.tgz');
    const r = await sh(page, base, `mkdir -p ${remoteDir} && tar -xzf /workspace/upload.tgz --no-same-owner -C ${remoteDir} && rm /workspace/upload.tgz && ls ${remoteDir}`);
    process.stdout.write(r.output);
  });
}

const cmds = {
  sync: () => upload(['roadread', 'app', 'eval', 'pyproject.toml'], R),
  data: () => upload([a], R, { excludes: ['__pycache__'] }),
  async setup() {
    await run(`${ENV} cd ${R} && H=$(sha1sum app/requirements.txt | cut -c1-12); \
if [ "$(cat /workspace/pylib/.req 2>/dev/null)" != "$H" ]; then rm -rf /workspace/pylib && \
pip install -q --target /workspace/pylib -r app/requirements.txt && \
(cd /workspace/pylib && rm -rf torch torch-* torchgen functorch torchvision* torchaudio* triton* numpy numpy-* numpy.libs nvidia* PIL pillow* bin/torch*) && \
echo $H > /workspace/pylib/.req; fi; \
python -c "import torch,transformers,PIL; assert 'rocm' in torch.__version__, torch.__file__; assert '/opt/venv' in torch.__file__; print('torch', torch.__version__, torch.__file__); print('transformers', transformers.__version__)" && \
pip freeze --path /workspace/pylib > app/requirements.lock && cat app/requirements.lock`, 30 * 60_000);
    if (process.exitCode) return;
    await withLab((page, base) => get(page, base, 'roadread/app/requirements.lock', path.join(REPO, 'app', 'requirements.lock')));
    console.log('pulled app/requirements.lock');
  },
  model: () => run(`${ENV} N=$(basename ${a}); python -c "from huggingface_hub import snapshot_download as d; print(d('${a}', local_dir='/workspace/models/$N'))" && \
ln -sfn /workspace/models/$N /workspace/models/current && du -sh /workspace/models/$N && df -h /workspace | tail -1`, 30 * 60_000),
  async gates() {
    const out = await run(`${ENV} cd ${R} && python eval/gates.py`, 15 * 60_000);
    const line = out.split('\n').find((l) => l.startsWith('{')) || out;
    fs.mkdirSync(path.join(REPO, 'results'), { recursive: true });
    const file = path.join(REPO, 'results', `gates-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
    fs.writeFileSync(file, line);
    console.log('saved', file);
  },
  'worker-start': () => run(`${ENV} cd ${R} && [ -f /tmp/roadread.pid ] && kill -9 $(cat /tmp/roadread.pid 2>/dev/null) 2>/dev/null || true; sleep 1; rm -f /tmp/roadread.ready /tmp/roadread.pid; \
nohup python -m roadread.worker > /workspace/worker.log 2>&1 & \
WPID=$!; echo "spawned worker pid $WPID"; \
for i in $(seq 1 120); do \
  if [ -f /tmp/roadread.ready ]; then cat /tmp/roadread.ready; echo; exit 0; fi; \
  if ! kill -0 $WPID 2>/dev/null; then echo "worker exited early:"; tail -40 /workspace/worker.log; exit 1; fi; \
  sleep 2; \
done; echo "timeout waiting for ready file"; tail -40 /workspace/worker.log; exit 1`, 12 * 60_000),
  'worker-stop': () => run(`[ -f /tmp/roadread.pid ] && kill -9 $(cat /tmp/roadread.pid 2>/dev/null) 2>/dev/null || true; rm -f /tmp/roadread.ready /tmp/roadread.pid; echo stopped`),
  'worker-log': () => run(`tail -60 /workspace/worker.log`),
  eval: () => run(`${ENV} cd ${R} && mkdir -p results && rm -f results/${b}.jsonl results/${b}.diag.jsonl && \
(nohup python eval/run_eval.py --manifest ${a} --out results/${b}.jsonl > results/${b}.summary 2> results/${b}.err &) && echo started ${b}`),
  tail: () => run(`cd ${R}/results && echo "done: $(wc -l < ${a}.jsonl 2>/dev/null)"; tail -2 ${a}.jsonl 2>/dev/null | cut -c1-300; \
echo "summary: $(cat ${a}.summary 2>/dev/null)"; tail -3 ${a}.err 2>/dev/null; rocm-smi --showmeminfo vram | grep -E 'Used|Total'`),
  pull: () => withLab(async (page, base) => {
    fs.mkdirSync(path.join(REPO, 'results'), { recursive: true });
    for (const ext of ['jsonl', 'summary', 'diag.jsonl']) {
      const local = path.join(REPO, 'results', `${a}.${ext}`);
      await get(page, base, `roadread/results/${a}.${ext}`, local).then(() => console.log('pulled', local)).catch((e) => console.error(ext, e.message));
    }
  }),
  async end() {
    await withLab(async (page, base) => {
      await sh(page, base, 'pkill -f roadread.worker; pkill -f run_eval.py; true');
      await killKernels(page, base);
    }).catch((e) => console.log('no running pod or cleanup skipped:', e.message));
    const { context, page } = await openBrowser();
    try { await stop(page); } catch (e) { console.log('stop:', e.message); } finally { await context.close(); }
  },
};

if (!cmds[cmd]) { console.log(fs.readFileSync(__filename, 'utf8').split('\n').slice(1, 15).join('\n')); process.exit(2); }
if (['eval', 'tail', 'pull'].includes(cmd) && !(cmd === 'eval' ? b : a)) { console.error(`${cmd}: missing tag`); process.exit(2); }
cmds[cmd]().catch((e) => { console.error('ERROR:', e.message); process.exit(1); });
