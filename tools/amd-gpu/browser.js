// Start a long-lived, visible Chrome on the saved profile with a CDP port, so any number of
// agent commands can attach to it (PW_CDP_URL=http://127.0.0.1:9333 node gpu.js ...) while a human watches.
// Usage: node browser.js            (leave running; Ctrl+C or close the window to stop)
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const { PROFILE_DIR, log } = require('./lib');

const PORT = Number(process.env.CDP_PORT || 9333);
const CHROME = process.env.CHROME_PATH || [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
].find((p) => fs.existsSync(p));
if (!CHROME) throw new Error('Chrome not found; set CHROME_PATH');

// Must be a NON-default user-data-dir: Chrome 136+ ignores --remote-debugging-port on the default profile.
const child = spawn(CHROME, [
  `--user-data-dir=${PROFILE_DIR}`, `--remote-debugging-port=${PORT}`, '--remote-debugging-address=127.0.0.1',
  '--no-first-run', '--no-default-browser-check', 'https://notebooks.amd.com/hackathon',
], { stdio: 'ignore' });
log(`CHROME_STARTED pid=${child.pid} cdp=http://127.0.0.1:${PORT} profile=${PROFILE_DIR}`);
child.on('exit', (c) => { log('CHROME_EXITED', c); process.exit(0); });
