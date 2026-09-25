// Shared helpers for driving notebooks.amd.com with Playwright.
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const ROOT = __dirname;
const PROFILE_DIR = process.env.AMD_PW_PROFILE || path.join(ROOT, '.profile');
const OUT_DIR = process.env.AMD_PW_OUT || path.join(ROOT, 'out');
const HUB_URL = 'https://notebooks.amd.com/hackathon';
fs.mkdirSync(OUT_DIR, { recursive: true });

function log(...args) {
  const line = `[${new Date().toISOString()}] ${args.join(' ')}`;
  console.log(line);
  fs.appendFileSync(path.join(OUT_DIR, 'run.log'), line + '\n');
}

// Persistent context = cookies/localStorage survive between runs in PROFILE_DIR.
// channel 'chrome' uses the installed Google Chrome, so no browser download is needed.
async function openBrowser({ headless = process.env.HEADLESS === '1' } = {}) {
  // Attach mode: reuse an already-running Chrome (started by browser.js). We open our own tab and
  // "closing" only closes that tab, so the shared browser keeps running.
  if (process.env.PW_CDP_URL) {
    const browser = await chromium.connectOverCDP(process.env.PW_CDP_URL);
    const shared = browser.contexts()[0];
    const page = await shared.newPage();
    const context = new Proxy(shared, {
      // close = close our tab + disconnect (browser.close() on a CDP-attached browser only disconnects).
      get: (t, k) => (k === 'close' ? async () => { await page.close().catch(() => {}); await browser.close(); } : typeof t[k] === 'function' ? t[k].bind(t) : t[k]),
    });
    return { context, page };
  }
  const context = await chromium.launchPersistentContext(PROFILE_DIR, {
    channel: process.env.PW_CHANNEL || 'chrome',
    headless,
    viewport: { width: 1400, height: 900 },
    args: ['--disable-blink-features=AutomationControlled'],
  });
  const page = context.pages()[0] || (await context.newPage());
  return { context, page };
}

async function dump(page, name) {
  try {
    fs.writeFileSync(path.join(OUT_DIR, `${name}.aria.yml`), await page.locator('body').ariaSnapshot());
    await page.screenshot({ path: path.join(OUT_DIR, `${name}.png`) });
  } catch (e) {
    log('dump failed', name, e.message);
  }
}

const isLoginPage = (url) => /login\.amd\.com|okta|\/auth\/|signin|sign-in/i.test(url);

// The hub keeps the /hackathon URL when logged out, so check the DOM, not the URL.
async function isSignedIn(page) {
  if (!page.url().startsWith(HUB_URL) || isLoginPage(page.url())) return false;
  const signIn = page.getByRole('link', { name: /Sign in with AMD AI Developer Program/i });
  return !(await signIn.isVisible().catch(() => false)) && (await page.locator('main').count()) > 0;
}

// Reads AMD SSO credentials from a .env file (tools/amd-gpu/.env first, then the repo root).
// Parsed from the file on purpose: on Windows process.env.USERNAME is the OS user.
// Accepted keys (case-insensitive): AMD_USERNAME | AMD_EMAIL | USERNAME | EMAIL, AMD_PASSWORD | PASSWORD.
function loadCreds() {
  const candidates = [path.join(ROOT, '.env'), path.join(ROOT, '..', '..', '.env')];
  for (const file of candidates) {
    if (!fs.existsSync(file)) continue;
    const kv = {};
    for (const raw of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
      const m = raw.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
      if (m) kv[m[1].toLowerCase()] = m[2].replace(/^(['"])(.*)\1$/, '$2');
    }
    const user = kv.amd_username || kv.amd_email || kv.username || kv.email;
    const pass = kv.amd_password || kv.password;
    if (user && pass) return { user, pass, file };
  }
  return null;
}

module.exports = { ROOT, PROFILE_DIR, OUT_DIR, HUB_URL, log, openBrowser, dump, isLoginPage, isSignedIn, loadCreds };
