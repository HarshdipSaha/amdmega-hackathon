// Exploration: press "Request Notebook", record UI state changes + hub API traffic until ready.
const fs = require('node:fs');
const path = require('node:path');
const { HUB_URL, OUT_DIR, log, openBrowser, dump, isSignedIn } = require('../lib');

(async () => {
  const { context, page } = await openBrowser();
  const net = path.join(OUT_DIR, 'net.log');
  context.on('response', async (r) => {
    const u = r.url();
    const t = r.request().resourceType();
    if (!/notebooks\.amd\.com|jupyter|rgapi/.test(u) || !['xhr', 'fetch', 'document', 'eventsource'].includes(t)) return;
    let body = '';
    try { if (t !== 'document') body = (await r.text()).slice(0, 600); } catch {}
    fs.appendFileSync(net, `${r.request().method()} ${r.status()} ${t} ${u}\n  ${body.replace(/\n/g, ' ')}\n`);
  });
  context.on('page', (p) => log('NEW_TAB', p.url()));

  await page.goto(HUB_URL, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(3000);
  if (!(await isSignedIn(page))) throw new Error('not signed in; run login.js');
  await dump(page, 'launch-00-before');

  const req = page.getByRole('button', { name: /Request Notebook/i });
  if (await req.isVisible().catch(() => false)) {
    await req.click();
    log('CLICKED Request Notebook');
  } else {
    log('No Request button; current main text:', (await page.locator('main').innerText()).replace(/\s+/g, ' '));
  }

  let last = '';
  let n = 1;
  const deadline = Date.now() + 12 * 60 * 1000;
  while (Date.now() < deadline) {
    const text = (await page.locator('main').innerText().catch(() => '')).replace(/\s+/g, ' ').trim();
    const stable = text.replace(/\d+\s*(s|sec|seconds|min|m)\b/gi, '#'); // ignore ticking timers
    if (stable !== last) {
      last = stable;
      log(`STATE ${n}:`, text.slice(0, 400));
      await dump(page, `launch-${String(n++).padStart(2, '0')}`);
    }
    const ready = page.getByRole('link', { name: /open|launch|go to notebook|jupyter/i })
      .or(page.getByRole('button', { name: /open|launch notebook|go to notebook/i }));
    if (await ready.first().isVisible().catch(() => false)) {
      log('READY_CONTROL', await ready.first().innerText(), await ready.first().getAttribute('href'));
      await dump(page, 'launch-ready');
      break;
    }
    await page.waitForTimeout(4000);
  }
  await context.close();
})().catch((e) => { log('ERROR', e.message); process.exit(1); });
