#!/usr/bin/env node
// Remote SOURCEBOUND (Mini-Challenge 3) workflow on the AMD notebook GPU (built on gpu.js; see GPUWEBSKILL.md).
//   node remote-mc3.js sync                         tar sourcebound/ mc3/ eval_mc3/ (not eval_mc3/data) release/ -> /workspace/mc3
//   node remote-mc3.js data <dir>                   tar a local dir (e.g. eval_mc3/data/dev) -> /workspace/mc3/<dir> (<= ~40 MB)
//   node remote-mc3.js setup                        pip --target /workspace/pylib-mc3, purge torch shadows, freeze -> mc3/requirements.lock
//   node remote-mc3.js model <hf_repo> <slot> [tmp] snapshot_download -> /workspace/models/<name> (or /root/models with tmp); link mc3-<slot>
//   node remote-mc3.js gates                        eval_mc3/gates.py -> results/mc3-gates-<ts>.json
//   node remote-mc3.js worker-start [nodac]         (re)start the resident worker; nodac drops DAC_OVERRIDE like the grader
//   node remote-mc3.js worker-stop | worker-log
//   node remote-mc3.js suite <split_dir> <tag>     every corpus in the split (or the kit): index + one process per question, detached
//   node remote-mc3.js tail <tag> | pull <tag>      progress / fetch results (never launch a pod)
//   node remote-mc3.js rehearse [local]             release/rehearse.sh against $MC3_IMAGE (env var; never committed)
//   node remote-mc3.js end                          stop worker/eval + kernels, then Turn-off Session (never launches)
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { withLab, sh, put, get, killKernels, stop } = require('./gpu');
const { openBrowser, OUT_DIR } = require('./lib');

const REPO = path.resolve(__dirname, '..', '..');
const R = '/workspace/mc3';
// pkill patterns are written '[s]ourcebound...' so they never match the bash -lc that runs them.
const ENV = `export PYTHONPATH=/workspace/pylib-mc3:${R} HF_HOME=/workspace/.cache/huggingface ` +
  `SB_READER=/workspace/models/mc3-reader SB_EMBEDDER=/workspace/models/mc3-embedder SB_INDEX_DIR=/tmp/sb-index ` +
  `SB_OUTPUT_DIR=${R}/results/app_output HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1;`;
const TAR = process.platform === 'win32' ? 'C:\\Windows\\System32\\tar.exe' : 'tar'; // bsdtar; Git's GNU tar breaks on C:
const [cmd, a, b, c] = process.argv.slice(2);
const NO_LAUNCH = new Set(['tail', 'pull', 'worker-stop', 'worker-log', 'end']);
if (NO_LAUNCH.has(cmd)) process.env.AUTO_LAUNCH = '0';

async function run(shell, timeoutMs) {
  return withLab(async (page, base) => {
    const r = await sh(page, base, shell, timeoutMs);
    process.stdout.write(r.output);
    if (!r.ok || /\[exit -?[1-9]/.test(r.output)) process.exitCode = 1;   // -15: killed by a signal
    return r.output;
  });
}

async function upload(relPaths, remoteDir, { excludes = ['__pycache__', 'eval_mc3/data', '*.pyc'] } = {}) {
  const rel = path.relative(REPO, path.join(OUT_DIR, 'upload-mc3.tgz'));
  execFileSync(TAR, ['-czf', rel, ...excludes.map((e) => '--exclude=' + e), ...relPaths], { cwd: REPO });
  const size = fs.statSync(path.join(REPO, rel)).size;
  if (size > 40e6) throw new Error(`archive is ${(size / 1e6).toFixed(1)} MB; split it (put goes through page.evaluate)`);
  return withLab(async (page, base) => {
    await put(page, base, path.join(REPO, rel), 'upload-mc3.tgz');
    const r = await sh(page, base, `mkdir -p ${remoteDir} && tar -xzf /workspace/upload-mc3.tgz --no-same-owner -C ${remoteDir} && rm /workspace/upload-mc3.tgz && ls ${remoteDir}`);
    process.stdout.write(r.output);
  });
}

const cmds = {
  sync: () => upload(['sourcebound', 'mc3', 'eval_mc3', 'release', 'tests/fixtures'], R),
  data: () => upload([a], R, { excludes: ['__pycache__'] }),
  async setup() {
    await run(`${ENV} cd ${R} && H=$(sha1sum mc3/requirements.txt | cut -c1-12); \
if [ "$(cat /workspace/pylib-mc3/.req 2>/dev/null)" != "$H" ]; then rm -rf /workspace/pylib-mc3 && \
pip install -q --target /workspace/pylib-mc3 -r mc3/requirements.txt && \
(cd /workspace/pylib-mc3 && rm -rf torch torch-* torchgen functorch torchvision* torchaudio* triton* numpy numpy-* numpy.libs nvidia* PIL pillow* bin/torch*) && \
echo $H > /workspace/pylib-mc3/.req; fi; \
python -c "import torch,transformers,fitz,openpyxl,PIL,numpy; assert 'rocm' in torch.__version__, torch.__file__; assert '/opt/venv' in torch.__file__; print('torch', torch.__version__, torch.__file__); print('transformers', transformers.__version__, 'fitz', fitz.VersionBind, 'openpyxl', openpyxl.__version__)" && \
pip freeze --path /workspace/pylib-mc3 > mc3/requirements.lock && cat mc3/requirements.lock`, 30 * 60_000);
    if (process.exitCode) return;
    await withLab((page, base) => get(page, base, 'mc3/mc3/requirements.lock', path.join(REPO, 'mc3', 'requirements.lock')));
    console.log('pulled mc3/requirements.lock');
  },
  model: () => {
    if (!['reader', 'embedder'].includes(b)) throw new Error('model <hf_repo> <reader|embedder> [tmp]');
    const dir = c === 'tmp' ? '/root/models' : '/workspace/models';
    return run(`${ENV} N=$(basename ${a}); mkdir -p ${dir} && HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download as d; print(d('${a}', local_dir='${dir}/$N'))" && \
ln -sfn ${dir}/$N /workspace/models/mc3-${b} && du -sh ${dir}/$N && df -h /workspace / | tail -2`, 40 * 60_000);
  },
  async gates() {
    const out = await run(`${ENV} cd ${R} && python eval_mc3/gates.py`, 10 * 60_000);
    const line = out.split('\n').find((l) => l.startsWith('{')) || out;
    fs.mkdirSync(path.join(REPO, 'results'), { recursive: true });
    const file = path.join(REPO, 'results', `mc3-gates-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
    fs.writeFileSync(file, line);
    console.log('saved', file);
  },
  // Start/stop live in eval_mc3/worker_ctl.sh so that no `bash -lc` command line here contains the process name.
  'worker-start': () => run(`${ENV} cd ${R} && bash eval_mc3/worker_ctl.sh start ${a === 'nodac' ? 'nodac' : ''}`, 15 * 60_000),
  'worker-stop': () => run(`cd ${R} && bash eval_mc3/worker_ctl.sh stop`),
  'worker-log': () => run(`tail -60 /workspace/mc3-worker.log`),
  suite: () => {
    if (!a || !b) throw new Error('suite <split_dir relative to /workspace/mc3, e.g. eval_mc3/kit> <tag>');
    return run(`${ENV} cd ${R} && mkdir -p results && (nohup python eval_mc3/run_suite.py --split ${a} --tag ${b} > results/${b}.log 2>&1 &) && echo started ${b}`);
  },
  tail: () => run(`cd ${R}/results && echo "answered: $(wc -l < ${a}.jsonl 2>/dev/null)"; tail -2 ${a}.jsonl 2>/dev/null | cut -c1-400; \
tail -3 ${a}.log 2>/dev/null | cut -c1-600; echo "summary: $(tail -1 ${a}.summary 2>/dev/null)"; rocm-smi --showmeminfo vram | grep -E 'Used|Total'`),
  pull: () => withLab(async (page, base) => {
    fs.mkdirSync(path.join(REPO, 'results'), { recursive: true });
    for (const ext of ['jsonl', 'diag.jsonl', 'summary', 'log']) {
      const local = path.join(REPO, 'results', `${a}.${ext}`);
      await get(page, base, `mc3/results/${a}.${ext}`, local).then(() => console.log('pulled', local)).catch((e) => console.error(ext, e.message));
    }
  }),
  rehearse: () => {
    if (a === 'local') return run(`cd ${R} && LOCAL_LAYERS=1 bash release/rehearse.sh`, 55 * 60_000);
    if (!process.env.MC3_IMAGE) throw new Error('set MC3_IMAGE in your shell (never commit it), or use: rehearse local');
    return run(`cd ${R} && MC3_IMAGE='${process.env.MC3_IMAGE}' bash release/rehearse.sh`, 55 * 60_000);
  },
  async end() {
    await withLab(async (page, base) => {
      await sh(page, base, "pkill -f '[s]ourcebound'; pkill -f '[r]un_eval.py'; pkill -f '[r]un_suite.py'; true");
      await killKernels(page, base);
    }).catch((e) => console.log('no running pod or cleanup skipped:', e.message));
    const { context, page } = await openBrowser();
    try { await stop(page); } catch (e) { console.log('stop:', e.message); } finally { await context.close(); }
  },
};

if (!cmds[cmd]) { console.log(fs.readFileSync(__filename, 'utf8').split('\n').slice(1, 15).join('\n')); process.exit(2); }
if (['tail', 'pull'].includes(cmd) && !a) { console.error(`${cmd}: missing tag`); process.exit(2); }
Promise.resolve().then(() => cmds[cmd]()).catch((e) => { console.error('ERROR:', e.message); process.exit(1); });
