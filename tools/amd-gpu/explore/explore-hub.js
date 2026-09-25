// Exploration: hub while a session is running + direct API calls via the context's cookies.
const { HUB_URL, log, openBrowser, dump } = require('../lib');

(async () => {
  const { context, page } = await openBrowser();
  context.on('request', (r) => {
    if (/notebooks\.amd\.com\/hackathon\//.test(r.url()) && r.method() !== 'GET')
      log('REQ', r.method(), r.url(), 'body=', r.postData(), 'ct=', r.headers()['content-type']);
  });
  await page.goto(HUB_URL, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(5000);
  log('URL after hub load:', page.url());
  await dump(page, 'hub-running');
  log('MAIN:', (await page.locator('main').innerText().catch(() => '')).replace(/\s+/g, ' '));
  const team = await (await page.request.get(`${HUB_URL}/my-team`)).json();
  log('my-team', JSON.stringify(team).slice(0, 200));
  const st = await page.request.get(`${HUB_URL}/status?team_id=${team.team_id}`);
  log('status', st.status(), await st.text());
  const btns = await page.getByRole('button').allInnerTexts();
  log('BUTTONS:', JSON.stringify(btns));
  await context.close();
})().catch((e) => { log('ERROR', e.message); process.exit(1); });
