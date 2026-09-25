# GPUWEBSKILL — using the AMD hackathon GPU from Playwright, end to end

This guide is for an agent that has **no browser plugin**, only a shell with Node.js. By the end you can
sign in, start a GPU session, run Python or shell commands on an AMD Radeon Pro W7900D, move files in both
directions, drive the JupyterLab UI, and turn the session off. Everything here was run and checked against
the live service on **2026-09-24**. Where something was not tested, the guide says so.

Tool code lives in `tools/amd-gpu/`. Section 12 has a single-file version for building it from scratch.

---

## 0. Quick start (the five commands)

```bash
cd tools/amd-gpu
npm install                                  # once; installs Playwright (uses your installed Chrome, no browser download)
node gpu.js status                           # signed in? session running? quota left?
node gpu.js sh "rocm-smi && python -c 'import torch;print(torch.cuda.get_device_name(0))'"   # auto-launches a session if none
node gpu.js stop                             # ALWAYS turn the session off when done (quota clock)
```

If `status` fails with `NOT_SIGNED_IN`, a **human** runs `node login.js` (section 3). Every other step can run
unattended, and adding `HEADLESS=1` hides the browser window.

---

## 1. How the system is put together

```
 you (Node + Playwright)
   │  Chrome, persistent profile  tools/amd-gpu/.profile   (holds the AMD SSO cookies)
   ▼
 notebooks.amd.com/hackathon        ← "hub": sign-in, Request/Launch Notebook, quota, Turn-off Session
   │  SSO: login.amd.com (Okta SAML, "AMD AI Developer Program" account)
   │  GET /hackathon/open → single-use ticket URL
   ▼
 jupyter.anruicloud.com/_auth/start#ticket=…   → sets a Jupyter auth cookie, redirects to
 jupyter.anruicloud.com/instances/<pod>/lab     ← JupyterLab 4.6 on the GPU pod
   │  REST  /instances/<pod>/api/{kernels,sessions,contents,terminals}
   │  WS    /instances/<pod>/api/kernels/<id>/channels , /instances/<pod>/terminals/websocket/<name>
   ▼
 GPU pod: AMD Radeon Pro W7900D (gfx1100, 48 GB), root, /workspace = 25 GB persistent NFS
```

The key design choice is that **Playwright is used only to get authenticated. All the actual work goes
through Jupyter's own REST and WebSocket APIs, called from inside the lab tab with `page.evaluate`.** Inside
that tab the browser already holds the auth cookie and the `_xsrf` cookie, so nothing else is needed.
Clicking cells is slow and brittle. The API is exact and returns clean text output.

---

## 2. What the GPU box is (measured 2026-09-24)

| Item | Value |
|---|---|
| GPU | **AMD Radeon Pro W7900D**, `gfx1100` (RDNA3), 96 CUs, **48 GiB** VRAM, 241 W cap |
| fp16 matmul (8192², torch) | ~**57 TFLOPS** measured |
| CPU / RAM (cgroup limit) | AMD EPYC 9334; cgroup 16 CPUs, **55 GiB** memory limit (host shows 503 GB) |
| OS / Python | Ubuntu 26.04, Python 3.14.4 at `/opt/venv/bin/python` |
| torch | `2.13.0+rocm10.0.0` (HIP 7.15), torchvision 0.28, torchaudio 2.11, triton 3.8 rocm |
| ROCm tools | `rocm-smi`, `amd-smi`, `rocminfo`, `hipcc`, `rocprofv3` (no `/opt/rocm`, ROCm ships inside the pip wheels) |
| Not installed | vllm, transformers, git-lfs, docker, uv, conda, tmux, htop, ffmpeg, `nvidia-smi` (install with pip/apt as needed) |
| User | `root`, `sudo` works |
| Network | pypi, github, huggingface all reachable. HF download measured at **~100 MB/s** |
| HF config | `HF_ENDPOINT=https://hf-mirror.com`, `HF_HOME=/workspace/.cache/huggingface`, `HF_HUB_DISABLE_XET=1` |
| Jupyter | JupyterLab 4.6.4, jupyter_server 2.21, one kernelspec `python3`; AMD extensions `@amd-oneclick/resource-status`, `@amd-oneclick/progress` |
| Storage | `/workspace` = 25 GB NFS, **persistent**, and is the **Jupyter root**. `/` is a large overlay but **not persistent**. `/persistent` does **not** exist on this pod type |
| Pod name | `rgapi-hackathon-<n>-<hash>`. This is the "rgapi" pod type from COMPUTE.md, so persistence is `/workspace` |
| Team | `team-1161`, 1 member; image "Radeon Hackathon (default)" = `…/rocm/pytorch@sha256:8ccd7b…` |

Only `/workspace` survives a turn-off. Anything installed with `pip install` into `/opt/venv` is gone next
session. See recipe 9.3.

---

## 3. Authentication: the saved Chrome profile

### 3.1 The profile

- Location: `tools/amd-gpu/.profile/`. This is a Chrome `--user-data-dir`, gitignored. Override it with
  `AMD_PW_PROFILE=<dir>`.
- It holds the AMD SSO session, so any script that opens this profile is already signed in.
- It only works **on this machine, for this Windows user**, because Chrome encrypts cookies with the OS user
  key. Copying it to another machine will not keep you signed in. Run `login.js` there instead.
- **Do not point Playwright at the person's real Chrome profile** (`%LOCALAPPDATA%\Google\Chrome\User Data`).
  Chrome 136 and later refuses automation on the default profile, and copying it exposes everything in it.
  Use the dedicated `.profile` directory.
- Only one process can open a profile at a time. If a launch fails with "profile in use", another
  `gpu.js`/`login.js`/`browser.js` is still running. Stop it, or use shared mode (section 4).

### 3.2 Signing in (`node login.js`)

1. It opens `https://notebooks.amd.com/hackathon`. If already signed in it prints `ALREADY_LOGGED_IN` and exits.
2. Otherwise the hub redirects to `https://login.amd.com/app/amd_amdnotebook_1/…/sso/saml?...`. On some loads
   it instead shows a **"Sign in with AMD AI Developer Program"** link (`/auth/saml/login?next=/hackathon`), and
   the script clicks it.
3. The Okta page has the textboxes **"E-mail Address"** and **"Password"** and a button **"Sign in"**.
   - If `.env` exists (in `tools/amd-gpu/` or the repo root) with `username=`/`password=` (also accepted:
     `AMD_USERNAME`, `AMD_EMAIL`, `EMAIL`, `AMD_PASSWORD`), the script fills them in. It parses the file
     itself, because on Windows `process.env.USERNAME` is the OS user. It never logs the values.
   - Without `.env`, a human types them in the visible window.
4. MFA or captcha, if shown, is approved by the human in the window. The script waits up to 15 minutes
   (`LOGIN_TIMEOUT_MS`).
5. Success is `login.amd.com/login/token/redirect?stateToken=…` → `notebooks.amd.com/hackathon`, with title
   **"Dashboard - AMD AI Notebooks"** and heading **"Welcome back!"**. The script prints `LOGIN_OK`.

The account is the **AMD AI Developer Program** account, the same one lablab.ai links for the challenge.
The route the person took the first time was: sign in on `lablab.ai`, open
`lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge`, then go to the AMD hub. Signing in to lablab.ai
is **not** needed for the notebooks, which only need the AMD SSO.

**Rules for agents:** do not type or print the password yourself, do not `cat` the `.env` file, and never
commit `.env` or `.profile/` (both are gitignored). If the session expires, ask the human to run
`! cd tools/amd-gpu && node login.js`.

### 3.3 Checking you are signed in

The hub keeps the `/hackathon` URL even when signed out, so **do not check the URL**. Check the DOM instead:
signed out means the link "Sign in with AMD AI Developer Program" is visible, or the URL is on `login.amd.com`.
The function is `isSignedIn()` in `lib.js`.

---

## 4. Two ways to open the browser

| Mode | How | Use when |
|---|---|---|
| **Owned** (default) | Each command runs `chromium.launchPersistentContext('.profile', {channel:'chrome'})`, does its work, and closes Chrome. `HEADLESS=1` hides it. | Unattended agent runs, one command at a time |
| **Shared** (remote session) | `node browser.js` starts a visible Chrome on `.profile` with CDP on `127.0.0.1:9333` and keeps it running. Commands run with `PW_CDP_URL=http://127.0.0.1:9333`; each one opens its own tab and closes only that tab. | A human wants to watch, or many quick commands without relaunching Chrome |

```bash
node browser.js &                                   # terminal 1 (or run_in_background)
PW_CDP_URL=http://127.0.0.1:9333 node gpu.js status # terminal 2, any number of times
```

Gotchas that came up while testing:
- Port 9222 was already taken on this machine by another process on 127.0.0.1, and Chrome bound only IPv6,
  so `connectOverCDP` got a 404. The fix is port **9333** with `--remote-debugging-address=127.0.0.1`.
  Override it with `CDP_PORT`.
- With `connectOverCDP`, call `browser.close()` when done. On an attached browser it only **disconnects**,
  but without it Node never exits.
- Owned and shared mode cannot use the profile at the same time, because Chrome locks it.

---

## 5. The hub (`notebooks.amd.com/hackathon`)

### 5.1 What the UI shows

Signed-in dashboard (ARIA roles/names, stable for `getByRole`):
- heading "Welcome back!", text with the username, "Team: team-1161 (1 member)"
- combobox for the image: option "Radeon Hackathon (default)". While a pod runs it is disabled with the
  label "Image cannot be changed while a pod is running".
- button **"Request Notebook"** when no pod exists; it becomes **"Launch Notebook"** once one exists
- quota text, e.g. "2hrs 34min remaining", "Quota resets in 23h 33m"
- button **"Turn-off Session"** (id `hackathon-turnoff`, only shown while a pod exists). Clicking it shows
  the inline text "This will stop your GPU session. Your remaining GPU time will be preserved…" and a
  button **"Confirm Turn-off"**. This is not a modal and not `window.confirm`.
- link "Sign out" (`/auth/logout`). **Don't click it**, because it drops the saved session.

Launch progress text, observed timing (click → ready took **~80 s**):
`0%` → `32% Allocating GPU resources...` → `67% Starting notebook server...` → `90% Please be patient — you
will be redirected shortly.` → `100%` → the page redirects itself into JupyterLab.

### 5.2 The hub JSON API (called with the profile's cookies via `page.request`)

| Call | Body / query | Observed response |
|---|---|---|
| `GET /hackathon/my-team` | — | `{"team_id":"team-1161","team_members":[…],"team_size":1,"platform":"radeon","platform_images":[{"label":"Radeon Hackathon (default)","value":"…/rocm/pytorch@sha256:…"}]}` |
| `POST /hackathon/verify-team` | `{"team_id":"team-1161"}` | `{"success":true,"redirect_url":null or "/ready","pod_name":…,"quota_remaining_seconds":9211,"quota_resets_at":"…Z","message":"Team verified" or "Resuming existing session"}` |
| `POST /hackathon/launch` | sent by the button (body not captured; `gpu.js` clicks the button instead) | `{"launching":true,"instance_id":"radeon-hack-team-1161","quota_remaining_seconds":10800,…}` |
| `GET /hackathon/status?team_id=…` | — | `{"status":"launching"\|"ready"\|"not_found","url":"/ready"\|null,"ready":bool,"instance_id":"rgapi-hackathon-19245-8fcec3e4"\|null}` |
| `GET /hackathon/open?team_id=…` | — | `{"url":"https://jupyter.anruicloud.com/_auth/start#ticket=<single-use>","direct_url":"https://jupyter.anruicloud.com/instances/<pod>/lab"}` |
| `POST /hackathon/turn-off` | `{"team_id":"team-1161"}` | then `status` → `not_found` within ~10 s |

```js
// inside any Playwright script with the profile open and page on the hub:
const team = await (await page.request.get('https://notebooks.amd.com/hackathon/my-team')).json();
const st   = await (await page.request.get(`https://notebooks.amd.com/hackathon/status?team_id=${team.team_id}`)).json();
```

**The ticket in `open.url` works once and expires quickly.** Call `/open` again each time you need to get
into the lab. That is what `openLab()` does, and it is why COMPUTE.md says not to bookmark the URL.

### 5.3 Quota rules (from COMPUTE.md, confirmed by the API)

- 3 h (10800 s) of pod time per 24 h. Quota counts **while the pod exists**, including idle time.
- "Turn-off Session" keeps the remaining time. `quota_remaining_seconds` in `verify-team` is the number
  to check.
- **Always run `node gpu.js stop` when you finish**, including after errors.

---

## 6. JupyterLab (`jupyter.anruicloud.com/instances/<pod>/lab`)

### 6.1 What the lab looks like

- **AMD status bar** along the top (ARIA `status` "AMD Radeon Cloud resource status"): Workspace used/25 GiB
  · GPU % · VRAM % · RAM % · CPU % · Runtime · Consumed Credits. It showed "Status unavailable · retrying",
  which is harmless. It has buttons **"Help"** (clicking it showed nothing capturable) and **"Delete
  instance"**, shown in red. **Do not press "Delete instance".** It was not tested and probably destroys the
  pod. Use the hub's Turn-off Session instead.
- Menu bar: File, Edit, View, Run, Kernel, Tabs, Settings, Help.
- Left sidebar tabs: "File Browser (Ctrl+Shift+F)", "Running Terminals and Kernels", "Table of Contents",
  "Extension Manager".
- Launcher cards (`.jp-LauncherCard`): Notebook "Python 3 (ipykernel)", Console "Python 3 (ipykernel)",
  Terminal, Text File, Markdown File, Python File, Show Contextual Help. JupyterLab **restores the previous
  layout**, so the Launcher may not be open. Don't rely on it (see 7.4).
- A toast "Would you like to get notified about official Jupyter news?" with Yes/No. Click **No** or ignore it.
- The file browser root is **`/workspace`**. It already contains `starteramd.ipynb` (`print("hello world")`).
- Status bar at the bottom: "Python 3 (ipykernel) | Idle" and similar.

### 6.2 URL layout and the APIs (base = `https://jupyter.anruicloud.com/instances/<pod>/`)

| Purpose | Endpoint |
|---|---|
| open a file in the UI | `GET {base}lab/tree/<path>` |
| kernels | `GET/POST {base}api/kernels`, `DELETE {base}api/kernels/<id>`, WS `{base}api/kernels/<id>/channels` |
| sessions (notebook↔kernel) | `GET {base}api/sessions` |
| files | `GET/PUT/PATCH/DELETE {base}api/contents/<path>` (path relative to `/workspace`) |
| terminals | `POST {base}api/terminals` → `{name}`, WS `{base}terminals/websocket/<name>`, `DELETE {base}api/terminals/<name>` |
| identity | `GET {base}api/me` |

For every non-GET request, send the header `X-XSRFToken: <value of the _xsrf cookie>`. The cookie can be
read with `document.cookie` inside the lab tab. **Run these calls with `page.evaluate` in the lab tab**; the
browser supplies the auth cookie for you.

---

## 7. Running code: four methods, all tested

### 7.1 `gpu.js` command-line interface (use this first)

```bash
node gpu.js status                     # JSON: team, quota, status
node gpu.js launch                     # start + wait until ready (no-op if running)
node gpu.js url                        # fresh single-use lab URL for a human to open
node gpu.js py "import torch; print(torch.cuda.is_available())"
node gpu.js pyfile examples/inspect-env.py      # full environment inventory (section 2 came from this)
node gpu.js sh "rocm-smi; df -h /workspace"      # bash -lc via the kernel; prints "[exit N ]"
node gpu.js term "cd /workspace && ls"           # real Jupyter terminal (shell is /usr/bin/sh, not bash)
node gpu.js put local.txt data/local.txt         # remote path is relative to /workspace; parent dirs are created
node gpu.js get starteramd.ipynb out/starter.ipynb
node gpu.js ls ""                                # list /workspace
node gpu.js kernels | node gpu.js kill-kernels   # (two separate commands) list / delete all kernels
node gpu.js stop                                 # Turn-off Session + Confirm, waits for status not_found
```

Each command opens the browser, gets a fresh ticket, enters the lab, does one thing and closes. That is
about 10 s of overhead per call. Behaviour:
- `py`, `pyfile` and `sh` **reuse one kernel** between calls. Its id is stored in `out/kernel.json`, so
  variables persist: `py "x=41"` then `py "print(x+1)"` → `42` was tested. `kill-kernels` resets it.
- A command fails with exit code 1 when Python raises an error or the shell exit code is not 0. The
  traceback is printed without colour codes.
- If no pod is running, the `withLab` commands **launch one automatically**. Set `AUTO_LAUNCH=0` to fail
  instead. `status`, `url` and `stop` never launch.
- Every error saves `out/error.png` and `out/error.aria.yml` (a screenshot and the accessibility tree).

As a library:

```js
const { withLab, py, sh, put, get } = require('./tools/amd-gpu/gpu');
await withLab(async (page, base) => {
  await put(page, base, 'train.py', 'proj/train.py');
  const r = await sh(page, base, 'cd /workspace/proj && python train.py', 60 * 60 * 1000);
  console.log(r.ok, r.output);
  await get(page, base, 'proj/result.json', 'result.json');
});
```

### 7.2 How kernel execution works (so you can rebuild it)

Inside the lab tab (`page.evaluate`):
1. `POST {base}api/kernels` with body `{"name":"python3"}` and the XSRF header → `{id}`.
2. `new WebSocket(ws(s)://…/api/kernels/<id>/channels?session_id=<uuid>)`.
3. Send this message:
   ```json
   {"header":{"msg_id":"<uuid>","username":"pw","session":"<uuid>","msg_type":"execute_request","version":"5.3","date":"<iso>"},
    "parent_header":{},"metadata":{},"channel":"shell","buffers":[],
    "content":{"code":"<python>","silent":false,"store_history":false,"user_expressions":{},"allow_stdin":false,"stop_on_error":true}}
   ```
4. Keep only messages where `parent_header.msg_id` is your id. From those, collect `stream` → `content.text`,
   `execute_result`/`display_data` → `content.data["text/plain"]`, and `error` → `ename`, `evalue`, `traceback`.
   The run is **done** when a `status` message has `execution_state: "idle"`.
5. When you're done, delete the kernel with `DELETE {base}api/kernels/<id>`. A kernel that has imported
   torch holds GPU memory.

To run shell commands this way, execute
`subprocess.run(['bash','-lc',cmd],capture_output=True,text=True)` in the kernel. That is what `sh` does.

### 7.3 Terminal WebSocket

`POST {base}api/terminals` → `{name}`. Open the WS `…/terminals/websocket/<name>` and send
`["stdin","<cmd>; echo __DONE__ $?\r"]`. The replies are `["stdout","…"]`; read them until the marker
appears, then `DELETE` the terminal. Strip ANSI codes from the output. The echoed marker is split with `""`
in `gpu.js` so the echo of the typed command doesn't match. Use this when you need a real TTY. Otherwise
`sh` is simpler.

### 7.4 Driving the notebook UI (when a real `.ipynb` must be left behind)

Tested with `node examples/ui-notebook.js [name.ipynb]`. The pattern that works reliably:
1. `openLab(page)`.
2. Create the notebook with the contents API instead of the Launcher, because the layout is restored:
   `PUT api/contents/<name>` with `{"type":"notebook","content":{"cells":[],"metadata":{"kernelspec":{"name":"python3",…}},"nbformat":4,"nbformat_minor":5}}`.
3. `page.goto(base + 'lab/tree/' + name)` and wait for `.jp-NotebookPanel:visible .jp-CodeCell`.
4. **Wait for the kernel:** `.jp-StatusBar-Widget` containing text `Idle`. If you press Shift+Enter before
   this, the execution is silently dropped and the prompt stays `[ ]:`. That happened during testing.
5. For each cell: click `.cm-content` in cell *i*, press **Ctrl+A**, then use `page.keyboard.insertText(code)`.
   Don't use `.type()`, which triggers auto-closing brackets. Without Ctrl+A the text was inserted into the
   middle of existing text. Check the editor text matches, then press **Shift+Enter**.
6. The cell is done when `.jp-InputPrompt` shows `[<number>]`. The output is in `.jp-OutputArea`.
7. Save with **Ctrl+S**. Shut the kernel down from the Kernel menu → "Shut Down Kernel".

A tested run printed `2.13.0+rocm10.0.0 True AMD Radeon Pro W7900D` and then `106565.8125`.

### 7.5 Opening the lab for a human

`node gpu.js url` prints a single-use `url`. Open it within about a minute, or run `node browser.js` and
navigate there.

---

## 8. Turning things on and off

| Thing | Turn on | Turn off | Why it matters |
|---|---|---|---|
| GPU pod (quota clock) | `gpu.js launch`, or any lab command | **`gpu.js stop`** | Quota burns while the pod exists, even idle |
| Kernels | created by `py`/`sh` or opening a notebook | `gpu.js kill-kernels`, or Kernel → Shut Down | Each torch kernel holds VRAM |
| Terminals | `term`, or Launcher → Terminal | `DELETE api/terminals/<name>` (`term` does this itself) | Leftover processes |
| Shared Chrome | `node browser.js` | close the window or kill the process | Locks the profile |
| Background jobs on the pod | `nohup … &` (recipe 9.4) | `pkill -f …` via `sh`, or stop the pod | They keep running between CLI calls |
| "Delete instance" (lab) | — | **never press** | Untested; the hub Turn-off is the supported path |
| "Sign out" (hub) | — | **never press** | Clears the saved SSO session |

---

## 9. Recipes

**9.1 GPU sanity check**
```bash
node gpu.js py "import torch;print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)"
```

**9.2 Upload a project, run it, fetch the results**
```bash
node gpu.js put train.py proj/train.py
node gpu.js sh "cd /workspace/proj && python train.py > run.log 2>&1; tail -20 run.log"
node gpu.js get proj/run.log out/run.log
```
`put` sends one file at a time (base64 through the contents API). For many files, tar them locally, `put`
the archive, then `sh "cd /workspace && tar xzf proj.tgz"`. For code in a git repo, `sh "git clone …"`
works because github is reachable.

**9.3 Python packages that survive a restart.** `/opt/venv` is reset every session. Either reinstall each
time with `sh "pip install -q transformers accelerate"`, or keep a target directory in /workspace:
`sh "pip install -q --target /workspace/pylib transformers"` and start scripts with
`import sys; sys.path.insert(0,'/workspace/pylib')`. Neither has been timed across a restart yet.

**9.4 Long jobs.** Kernel calls wait by default (up to 30 min per call; the fourth argument to `py`/`sh`
changes it). For anything longer, detach the job and poll it:
```bash
node gpu.js sh "cd /workspace/proj && nohup python train.py > train.log 2>&1 & echo started"
node gpu.js sh "tail -5 /workspace/proj/train.log; rocm-smi --showuse"
```
The job dies when the pod stops, so write checkpoints under `/workspace`.

**9.5 Hugging Face models.** The environment already points `HF_HOME` at `/workspace/.cache/huggingface`
(persistent) and uses the `hf-mirror.com` endpoint. Downloads ran at about 100 MB/s. Keep an eye on the
25 GB `/workspace` limit.

---

## 10. Troubleshooting (all of these happened during testing)

| Symptom | Cause | Fix |
|---|---|---|
| Script says "logged in" while signed out | the hub keeps the `/hackathon` URL when signed out | check the DOM with `isSignedIn()` |
| `NOT_SIGNED_IN` | SSO cookie expired, or a new machine | human runs `node login.js` |
| Launch fails: profile in use | another process has `.profile` open | stop it, or use `PW_CDP_URL` shared mode |
| `connectOverCDP … 404 /json/version` | port 9222 owned by another process; Chrome on IPv6 only | use port 9333 and `--remote-debugging-address=127.0.0.1` |
| CDP script never exits | still connected | `browser.close()` (only disconnects) |
| `contents PUT … 500 No such file or directory: '/workspace/workspace/…'` | contents paths are relative to `/workspace`, and parents aren't created | drop the `workspace/` prefix; `put` creates parents |
| Notebook cell stays `[ ]:` | Shift+Enter before the kernel connected | wait for status bar `Idle` |
| Typed code duplicated or garbled in a cell | cursor placed inside existing text | Ctrl+A before `insertText` |
| Launcher cards not found | the lab restored the previous layout | create the notebook through the API and open `lab/tree/<name>` |
| Turn-off click does nothing | the hub needs a second click on the inline "Confirm Turn-off" | `stop()` clicks both |
| Kernels pile up | every new page load created a kernel | the kernel id is saved in `out/kernel.json`; `kill-kernels` |
| `nvidia-smi: not found` | it's an AMD box | `rocm-smi`, `amd-smi` |
| `/persistent` missing | this is an `rgapi-*` pod | persistence is `/workspace` |

Debug artifacts go to `tools/amd-gpu/out/`: `run.log` (every step), `*.png` and `*.aria.yml` snapshots,
`net.log` from `explore/explore-launch.js` (full hub traffic), and `kernel.json`.

---

## 11. Files

```
tools/amd-gpu/
  package.json          playwright ^1.63 (npm install; set PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1, Chrome channel is used)
  lib.js                openBrowser (owned / CDP), isSignedIn, loadCreds (.env), dump, log, paths
  login.js              one-time sign-in (auto-fill from .env or human), saves .profile
  gpu.js                CLI + library: status/launch/url/stop, py/pyfile/sh/term, put/get/ls, kernels
  browser.js            shared visible Chrome with CDP :9333 on .profile
  examples/ui-notebook.js   drive the notebook UI (create, type, run, read, save, shut down)
  examples/inspect-env.py   environment inventory (GPU, torch, pkgs, tools, env, net, limits)
  explore/explore-launch.js records hub UI states + network during a launch
  explore/explore-hub.js    hub buttons/state + direct API calls
  .profile/  out/  node_modules/   (gitignored)
```

Requirements: Node 18+ (tested on 22.23), Google Chrome installed (tested on 153). For Chromium instead of
Chrome, run `npx playwright install chromium` and set `PW_CHANNEL=chromium`.

---

## 12. Building it from scratch (single file, no repo)

Save this as `amd.js`, then run `npm i playwright` and `node amd.js "print(1+1)"`. **Tested 2026-09-24:** it printed
`SCRATCH_OK AMD Radeon Pro W7900D` and left the session off. It assumes a profile
directory that has already been signed in once (with a human completing the SSO in the window if needed).

```js
const { chromium } = require('playwright');
const HUB = 'https://notebooks.amd.com/hackathon';
const code = process.argv[2] || 'import torch; print(torch.cuda.get_device_name(0))';

(async () => {
  const ctx = await chromium.launchPersistentContext('./.profile', { channel: 'chrome', headless: false });
  const page = ctx.pages()[0] || (await ctx.newPage());
  try {
    await page.goto(HUB);
    // 1) signed in? (a human finishes the SSO in the window if not; wait up to 10 min)
    await page.getByRole('heading', { name: 'Welcome back!' }).waitFor({ timeout: 600_000 });
    const api = async (p, data) => (await (data ? page.request.post(HUB + p, { data }) : page.request.get(HUB + p))).json();
    const { team_id } = await api('/my-team');
    // 2) launch if needed
    let st = await api(`/status?team_id=${team_id}`);
    if (!st.ready) {
      if (st.status !== 'launching') await page.getByRole('button', { name: /^(Request|Launch) Notebook$/ }).click();
      while (!(st = await api(`/status?team_id=${team_id}`)).ready) await page.waitForTimeout(5000);
    }
    // 3) enter the lab with a fresh single-use ticket
    const { url } = await api(`/open?team_id=${team_id}`);
    await page.goto(url);
    await page.waitForURL(/\/instances\/[^/]+\/lab/);
    const base = page.url().replace(/\/lab.*$/, '/');
    // 4) execute code over the kernel websocket, inside the lab tab
    const out = await page.evaluate(async ({ base, code }) => {
      const xsrf = decodeURIComponent((document.cookie.match(/(?:^|; )_xsrf=([^;]+)/) || [])[1] || '');
      const H = { 'Content-Type': 'application/json', 'X-XSRFToken': xsrf };
      const k = await (await fetch(base + 'api/kernels', { method: 'POST', headers: H, body: '{"name":"python3"}' })).json();
      const ws = new WebSocket(base.replace(/^http/, 'ws') + `api/kernels/${k.id}/channels`);
      const id = crypto.randomUUID(), text = [];
      await new Promise((res) => {
        ws.onopen = () => ws.send(JSON.stringify({ header: { msg_id: id, username: 'pw', session: crypto.randomUUID(), msg_type: 'execute_request', version: '5.3' },
          parent_header: {}, metadata: {}, channel: 'shell', content: { code, silent: false, store_history: false, user_expressions: {}, allow_stdin: false } }));
        ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.parent_header?.msg_id !== id) return;
          const t = m.header.msg_type;
          if (t === 'stream') text.push(m.content.text);
          if (t === 'execute_result') text.push(m.content.data['text/plain'] + '\n');
          if (t === 'error') text.push(m.content.traceback.join('\n'));
          if (t === 'status' && m.content.execution_state === 'idle') res(); };
      });
      ws.close();
      await fetch(base + `api/kernels/${k.id}`, { method: 'DELETE', headers: H });
      return text.join('');
    }, { base, code });
    console.log(out);
  } finally {
    // 5) turn the session off (remove these lines if you want to keep working)
    await page.goto(HUB);
    await page.getByRole('button', { name: 'Turn-off Session' }).click().catch(() => {});
    await page.getByRole('button', { name: 'Confirm Turn-off' }).click().catch(() => {});
    await page.waitForTimeout(5000);
    await ctx.close();
  }
})();
```

---

## 13. Session record, for resuming (2026-09-24, UTC)

- 20:47: dedicated profile created at `tools/amd-gpu/.profile`. The person signed in to lablab.ai in it
  (Google), then `login.js` auto-filled the AMD SSO from the root `.env` → `LOGIN_OK` at 20:55. The root
  `.env` was **not gitignored** before this session; `.env` and `.env.*` were added to `.gitignore`.
- 20:55:51: first "Request Notebook" → ready at about 20:57:11. Pod `rgapi-hackathon-19245-8fcec3e4`.
- 20:59–21:19: tested `status`, `sh`, `pyfile` (inventory → section 2), `py` with kernel reuse,
  `put`/`get`/`ls`, `term`, `kill-kernels`, the UI notebook example, and a torch matmul benchmark. All test
  files were removed. `/workspace` contains only `starteramd.ipynb` (plus `.ipynb_checkpoints`).
- 21:22:56: `POST /hackathon/turn-off` → `status: not_found`. **The session is OFF.** Quota left:
  **9211 s (≈2 h 33 min)** until the reset at `2026-09-25T20:55:49Z`.
- The shared Chrome (`browser.js`) was tested on port 9333 and then closed.
- About 21:30: the section 12 single-file script was run end to end (launch → exec → turn-off) → OK. It ended
  with the **session OFF** and `quota_remaining_seconds` = **9191 (≈2 h 33 min)**. Nothing is left running.
- Open questions: the `POST /launch` body was not captured (the button is used instead). The lab's "Delete
  instance" and "Help" buttons are untested. Whether a `pip --target /workspace/...` install survives a
  restart has not been checked. How long the SSO cookie lasts is unknown (re-run `login.js` when it expires).
