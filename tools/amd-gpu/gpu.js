#!/usr/bin/env node
// CLI + library for the AMD hackathon GPU notebook (notebooks.amd.com -> JupyterLab on jupyter.anruicloud.com).
//
//   node gpu.js status                 hub status + quota (JSON)
//   node gpu.js launch                 start a session (no-op if one is running) and wait until ready
//   node gpu.js url                    print a fresh JupyterLab URL (single-use ticket) — open it in a browser
//   node gpu.js py  "<python code>"    run Python in a kernel on the GPU box, print output
//   node gpu.js pyfile local.py        same, code read from a local file
//   node gpu.js sh  "<shell command>"  run a shell command on the GPU box (via the kernel, bash -lc)
//   node gpu.js term "<shell command>" run a shell command through a real Jupyter terminal (websocket)
//   node gpu.js put local remote       upload a file (remote path relative to the Jupyter root = /workspace, e.g. data/x.py)
//   node gpu.js get remote local       download a file
//   node gpu.js ls [remote_dir]        list a directory via the contents API
//   node gpu.js kernels                list running kernels;  node gpu.js kill-kernels  deletes them all
//   node gpu.js stop                   press "Turn-off Session" on the hub (stops the quota clock)
//
// Env: HEADLESS=1 to hide the browser, AMD_PW_PROFILE to use another profile dir, AUTO_LAUNCH=0 to fail instead
// of launching when no session is running, PW_CDP_URL=http://127.0.0.1:9333 to attach to a Chrome started by browser.js.
const fs = require('node:fs');
const { HUB_URL, log, openBrowser, isSignedIn, dump } = require('./lib');

// ---------- hub (notebooks.amd.com) ----------
async function hubApi(page) {
  if (!page.url().startsWith(HUB_URL)) await page.goto(HUB_URL, { waitUntil: 'domcontentloaded' });
  if (!(await isSignedIn(page))) {
    await page.waitForTimeout(3000);
    if (!(await isSignedIn(page))) throw new Error('NOT_SIGNED_IN: run `node login.js` first');
  }
  const get = async (p) => (await page.request.get(`${HUB_URL}${p}`)).json();
  const post = async (p, data) => (await page.request.post(`${HUB_URL}${p}`, { data })).json();
  const team = await get('/my-team');
  return { team, get, post, teamId: team.team_id };
}

async function status(page) {
  const h = await hubApi(page);
  const verify = await h.post('/verify-team', { team_id: h.teamId });
  const st = await h.get(`/status?team_id=${h.teamId}`);
  return { team: h.team, quota: verify, status: st };
}

async function launch(page, { timeoutMs = 15 * 60 * 1000 } = {}) {
  const h = await hubApi(page);
  let st = await h.get(`/status?team_id=${h.teamId}`);
  if (st.ready) { log('SESSION_ALREADY_READY', st.instance_id); return st; }
  if (st.status !== 'launching') {
    // Use the real button so the page sends exactly the body it expects (team + selected image).
    await page.goto(HUB_URL, { waitUntil: 'domcontentloaded' });
    const btn = page.getByRole('button', { name: /^(Request|Launch) Notebook$/ });
    await btn.waitFor({ timeout: 30_000 });
    await btn.click();
    log('LAUNCH_CLICKED');
  }
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    st = await h.get(`/status?team_id=${h.teamId}`);
    if (st.ready) { log('SESSION_READY', st.instance_id); return st; }
    if (/error|fail|stopped/i.test(st.status)) throw new Error('LAUNCH_FAILED ' + JSON.stringify(st));
    await page.waitForTimeout(5000);
  }
  throw new Error('LAUNCH_TIMEOUT');
}

// Returns { url, direct_url }. `url` carries a single-use #ticket; open it once to get the Jupyter auth cookie.
async function openInfo(page) {
  const h = await hubApi(page);
  const st = await h.get(`/status?team_id=${h.teamId}`);
  if (!st.ready) throw new Error('NO_RUNNING_SESSION ' + JSON.stringify(st));
  return h.get(`/open?team_id=${h.teamId}`);
}

// Navigates `page` into JupyterLab and returns the server base URL (".../instances/<id>/").
async function openLab(page) {
  const info = await openInfo(page);
  await page.goto(info.url, { waitUntil: 'domcontentloaded' });
  await page.waitForURL(/\/instances\/[^/]+\/lab/, { timeout: 90_000 });
  await page.locator('#jp-main-dock-panel, .jp-LabShell').first().waitFor({ timeout: 90_000 });
  const base = page.url().replace(/\/lab.*$/, '/');
  log('LAB_OPEN', base);
  return base;
}

async function stop(page) {
  await page.goto(HUB_URL, { waitUntil: 'domcontentloaded' });
  page.on('dialog', (d) => { log('DIALOG', d.type(), d.message()); d.accept(); }); // in case the hub uses window.confirm
  page.on('request', (r) => r.method() !== 'GET' && /notebooks\.amd\.com\/hackathon\//.test(r.url()) && log('HUB_REQ', r.method(), r.url(), r.postData() || ''));
  const btn = page.getByRole('button', { name: /Turn-off Session/i });
  await btn.waitFor({ timeout: 30_000 });
  await btn.click();
  // The hub then reveals an inline "Confirm Turn-off" button (not a modal, not window.confirm).
  await page.getByRole('button', { name: 'Confirm Turn-off' }).click({ timeout: 15_000 });
  const h = await hubApi(page);
  for (let i = 0; i < 24; i++) {
    const st = await h.get(`/status?team_id=${h.teamId}`);
    if (!st.ready && st.status !== 'launching') { log('SESSION_STOPPED', JSON.stringify(st)); return st; }
    await page.waitForTimeout(5000);
  }
  throw new Error('STOP_NOT_CONFIRMED');
}

// ---------- JupyterLab (in-page helpers: run inside the lab tab so cookies + XSRF are automatic) ----------
const IN_PAGE = {
  // Execute Python in a (reused) kernel over the Jupyter kernel websocket protocol.
  async exec({ base, code, timeoutMs, kid }) {
    const xsrf = (document.cookie.match(/(?:^|; )_xsrf=([^;]+)/) || [])[1] || '';
    const H = { 'Content-Type': 'application/json', 'X-XSRFToken': decodeURIComponent(xsrf) };
    if (kid) {
      const r = await fetch(`${base}api/kernels/${kid}`);
      if (!r.ok) kid = null;
    }
    if (!kid) {
      const r = await fetch(`${base}api/kernels`, { method: 'POST', headers: H, body: JSON.stringify({ name: 'python3' }) });
      if (!r.ok) throw new Error(`kernel create ${r.status} ${await r.text()}`);
      kid = (await r.json()).id;
    }
    const session = crypto.randomUUID();
    const ws = new WebSocket(`${base.replace(/^http/, 'ws')}api/kernels/${kid}/channels?session_id=${session}`);
    const msgId = crypto.randomUUID();
    const out = [];
    return await new Promise((resolve, reject) => {
      const timer = setTimeout(() => { ws.close(); resolve({ ok: false, kernel: kid, output: out.join(''), error: 'TIMEOUT' }); }, timeoutMs);
      ws.onerror = () => { clearTimeout(timer); reject(new Error('websocket error')); };
      ws.onopen = () => ws.send(JSON.stringify({
        header: { msg_id: msgId, username: 'pw', session, msg_type: 'execute_request', version: '5.3', date: new Date().toISOString() },
        parent_header: {}, metadata: {}, channel: 'shell', buffers: [],
        content: { code, silent: false, store_history: false, user_expressions: {}, allow_stdin: false, stop_on_error: true },
      }));
      let err = null;
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data);
        if (m.parent_header?.msg_id !== msgId) return;
        const t = m.header.msg_type;
        if (t === 'stream') out.push(m.content.text);
        else if (t === 'execute_result' || t === 'display_data') out.push((m.content.data['text/plain'] || '') + '\n');
        else if (t === 'error') { err = `${m.content.ename}: ${m.content.evalue}`; out.push(m.content.traceback.join('\n').replace(/\x1b\[[0-9;]*m/g, '') + '\n'); }
        else if (t === 'status' && m.content.execution_state === 'idle') {
          clearTimeout(timer); ws.close(); resolve({ ok: !err, kernel: kid, output: out.join(''), error: err });
        }
      };
    });
  },

  // Run a command in a real Jupyter terminal (terminado websocket). Ends when the marker is echoed back.
  async term({ base, cmd, timeoutMs }) {
    const xsrf = (document.cookie.match(/(?:^|; )_xsrf=([^;]+)/) || [])[1] || '';
    const H = { 'Content-Type': 'application/json', 'X-XSRFToken': decodeURIComponent(xsrf) };
    const t = await (await fetch(`${base}api/terminals`, { method: 'POST', headers: H, body: '{}' })).json();
    const ws = new WebSocket(`${base.replace(/^http/, 'ws')}terminals/websocket/${t.name}`);
    const mark = `__PW_DONE_${Date.now()}__`;
    let buf = '';
    const res = await new Promise((resolve) => {
      const timer = setTimeout(() => resolve({ ok: false, output: buf, error: 'TIMEOUT' }), timeoutMs);
      ws.onopen = () => ws.send(JSON.stringify(['stdin', `${cmd}; echo ${mark.slice(0, 5)}""${mark.slice(5)} $?\r`]));
      ws.onmessage = (ev) => {
        const [kind, data] = JSON.parse(ev.data);
        if (kind !== 'stdout') return;
        buf += data;
        const m = buf.match(new RegExp(mark + ' (\\d+)'));
        if (m) { clearTimeout(timer); resolve({ ok: m[1] === '0', exit: Number(m[1]), output: buf }); }
      };
    });
    ws.close();
    await fetch(`${base}api/terminals/${t.name}`, { method: 'DELETE', headers: H });
    res.output = res.output.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, '').replace(/\r/g, '');
    return res;
  },

  async contents({ base, method, path, body }) {
    const xsrf = (document.cookie.match(/(?:^|; )_xsrf=([^;]+)/) || [])[1] || '';
    const H = { 'Content-Type': 'application/json', 'X-XSRFToken': decodeURIComponent(xsrf) };
    const url = `${base}api/contents/${path.split('/').map(encodeURIComponent).join('/')}${method === 'GET' ? '?content=1' : ''}`;
    const r = await fetch(url, { method, headers: H, body: body ? JSON.stringify(body) : undefined });
    if (!r.ok) throw new Error(`contents ${method} ${path}: ${r.status} ${await r.text()}`);
    return r.status === 204 ? null : r.json();
  },
};

// The kernel id is remembered in out/kernel.json so variables survive between CLI calls (and kernels don't pile up).
const KFILE = require('node:path').join(require('./lib').OUT_DIR, 'kernel.json');
async function py(page, base, code, timeoutMs = 30 * 60 * 1000) {
  let saved = {};
  try { saved = JSON.parse(fs.readFileSync(KFILE, 'utf8')); } catch {}
  const kid = saved.base === base ? saved.kid : null;
  const r = await page.evaluate(IN_PAGE.exec, { base, code, timeoutMs, kid });
  fs.writeFileSync(KFILE, JSON.stringify({ base, kid: r.kernel }));
  return r;
}
// List / delete kernels on the server (clean up after yourself; each kernel holds GPU memory once torch is imported).
const kernels = (page, base) => page.evaluate(async (base) => (await fetch(`${base}api/kernels`)).json(), base);
const killKernels = (page, base) => page.evaluate(async (base) => {
  const xsrf = decodeURIComponent((document.cookie.match(/(?:^|; )_xsrf=([^;]+)/) || [])[1] || '');
  const ks = await (await fetch(`${base}api/kernels`)).json();
  for (const k of ks) await fetch(`${base}api/kernels/${k.id}`, { method: 'DELETE', headers: { 'X-XSRFToken': xsrf } });
  return { deleted: ks.map((k) => k.id) };
}, base);
const term = (page, base, cmd, timeoutMs = 30 * 60 * 1000) => page.evaluate(IN_PAGE.term, { base, cmd, timeoutMs });
const sh = (page, base, cmd, timeoutMs) =>
  py(page, base, `import subprocess,sys\nr=subprocess.run(['bash','-lc',${JSON.stringify(cmd)}],capture_output=True,text=True)\nsys.stdout.write(r.stdout); sys.stdout.write(r.stderr); print('[exit', r.returncode, ']')`, timeoutMs);

async function put(page, base, local, remote) {
  const content = fs.readFileSync(local).toString('base64');
  // The contents API does not create parent folders; create them one level at a time (409/400 = exists).
  const parts = remote.split('/').slice(0, -1);
  for (let i = 1; i <= parts.length; i++)
    await page.evaluate(IN_PAGE.contents, { base, method: 'PUT', path: parts.slice(0, i).join('/'), body: { type: 'directory' } }).catch(() => {});
  return page.evaluate(IN_PAGE.contents, { base, method: 'PUT', path: remote, body: { type: 'file', format: 'base64', content } });
}
async function get(page, base, remote, local) {
  const m = await page.evaluate(IN_PAGE.contents, { base, method: 'GET', path: remote });
  fs.writeFileSync(local, m.format === 'base64' ? Buffer.from(m.content, 'base64') : m.format === 'json' ? JSON.stringify(m.content, null, 1) : m.content);
  return { path: m.path, size: m.size };
}
const ls = (page, base, dir = '') => page.evaluate(IN_PAGE.contents, { base, method: 'GET', path: dir })
  .then((m) => m.content.map((c) => `${c.type.padEnd(9)} ${String(c.size ?? '').padStart(10)}  ${c.path}`).join('\n'));

// Opens browser -> ensures a session -> opens lab -> runs fn(page, base) -> closes browser.
async function withLab(fn) {
  const { context, page } = await openBrowser();
  try {
    const st = (await status(page)).status;
    if (!st.ready) {
      if (process.env.AUTO_LAUNCH === '0') throw new Error('NO_RUNNING_SESSION');
      await launch(page);
    }
    const base = await openLab(page);
    return await fn(page, base);
  } catch (e) {
    await dump(page, 'error');
    throw e;
  } finally {
    await context.close();
  }
}

module.exports = { status, launch, openInfo, openLab, stop, py, sh, term, put, get, ls, kernels, killKernels, withLab, IN_PAGE };

// ---------- CLI ----------
if (require.main === module) {
  const [cmd, a, b] = process.argv.slice(2);
  const print = (r) => {
    if (r && typeof r === 'object' && 'output' in r) { process.stdout.write(r.output); if (!r.ok) { console.error('FAILED:', r.error || `exit ${r.exit}`); process.exitCode = 1; } }
    else console.log(typeof r === 'string' ? r : JSON.stringify(r, null, 2));
  };
  const hubOnly = async (fn) => { const { context, page } = await openBrowser(); try { return await fn(page); } finally { await context.close(); } };
  const cmds = {
    status: () => hubOnly(status),
    launch: () => hubOnly(launch),
    url: () => hubOnly(openInfo),
    stop: () => hubOnly(stop),
    py: () => withLab((p, base) => py(p, base, a)),
    pyfile: () => withLab((p, base) => py(p, base, fs.readFileSync(a, 'utf8'))),
    sh: () => withLab((p, base) => sh(p, base, a)),
    term: () => withLab((p, base) => term(p, base, a)),
    put: () => withLab((p, base) => put(p, base, a, b)).then((m) => ({ uploaded: m.path, size: m.size })),
    get: () => withLab((p, base) => get(p, base, a, b)),
    ls: () => withLab((p, base) => ls(p, base, a || '')),
    kernels: () => withLab(kernels),
    'kill-kernels': () => withLab(killKernels),
  };
  if (!cmds[cmd]) { const lines = fs.readFileSync(__filename, 'utf8').split('\n').slice(1);
    console.log(lines.slice(0, lines.findIndex((l) => !l.startsWith('//'))).join('\n')); process.exit(2); }
  cmds[cmd]().then(print).catch((e) => { console.error('ERROR:', e.message); process.exit(1); });
}
