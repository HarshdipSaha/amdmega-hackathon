// Sign in to notebooks.amd.com once; the session is saved in PROFILE_DIR for every later run.
// Usage: node login.js
//   - If already signed in: exits immediately (ALREADY_LOGGED_IN).
//   - If a .env with credentials exists: fills the AMD SSO (login.amd.com) form automatically.
//   - Otherwise (or if MFA/captcha appears): waits for a human to finish in the visible window.
const { HUB_URL, log, openBrowser, dump, isSignedIn, loadCreds } = require('./lib');

const TIMEOUT_MS = Number(process.env.LOGIN_TIMEOUT_MS || 15 * 60 * 1000);

async function waitForSignedIn(context) {
  const deadline = Date.now() + TIMEOUT_MS;
  while (Date.now() < deadline) {
    for (const p of context.pages()) if (await isSignedIn(p)) return p;
    await new Promise((r) => setTimeout(r, 2000));
  }
  return null;
}

async function fillAmdSso(page, creds) {
  // The hub either auto-redirects to login.amd.com or shows a sign-in link.
  if (!/login\.amd\.com/.test(page.url())) {
    const link = page.getByRole('link', { name: /Sign in with AMD AI Developer Program/i });
    if (await link.isVisible().catch(() => false)) await link.click();
  }
  await page.waitForURL(/login\.amd\.com/, { timeout: 60_000 });
  const email = page.getByRole('textbox', { name: 'E-mail Address' });
  await email.waitFor({ timeout: 60_000 });
  await email.fill(creds.user);
  // Some Okta flows ask for the e-mail first and show the password on a second step.
  const password = page.getByRole('textbox', { name: 'Password' });
  if (!(await password.isVisible().catch(() => false))) {
    await page.getByRole('button', { name: /next|sign in/i }).first().click();
    await password.waitFor({ timeout: 30_000 });
  }
  await password.fill(creds.pass);
  await page.getByRole('button', { name: 'Sign in' }).click();
  log('SSO_SUBMITTED (if MFA is enabled, approve it in the browser window)');
}

(async () => {
  const { context, page } = await openBrowser({ headless: false });
  page.on('framenavigated', (f) => f === page.mainFrame() && log('nav', f.url().slice(0, 120)));

  await page.goto(HUB_URL, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(3000);
  if (await isSignedIn(page)) {
    log('ALREADY_LOGGED_IN', page.url());
    await dump(page, 'hub');
    await context.close();
    return;
  }

  const creds = loadCreds();
  if (creds) {
    log('USING_CREDENTIALS_FROM', creds.file); // never log the values
    try {
      await fillAmdSso(page, creds);
    } catch (e) {
      log('AUTO_FILL_FAILED', e.message.split('\n')[0], '- finish the login by hand in the window');
      await dump(page, 'login-failed');
    }
  } else {
    log('NEED_HUMAN_LOGIN: no .env credentials; sign in manually in the window');
  }

  const hub = await waitForSignedIn(context);
  if (!hub) {
    log('LOGIN_TIMEOUT');
    await dump(page, 'login-timeout');
    await context.close();
    process.exit(1);
  }
  await hub.waitForTimeout(3000);
  log('LOGIN_OK', hub.url(), await hub.title());
  await dump(hub, 'hub');
  await context.close();
})();
