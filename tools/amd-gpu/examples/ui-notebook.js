// Example: drive the JupyterLab UI like a human — new notebook, type cells, run, read outputs, save.
// Usage: node examples/ui-notebook.js [notebook-name.ipynb]
const { openBrowser, dump, log } = require('../lib');
const { openLab } = require('../gpu');

const NAME = process.argv[2] || 'pw_ui_demo.ipynb';
const CELLS = [
  'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))',
  'x = torch.randn(4096, 4096, device="cuda"); (x @ x).sum().item()',
];

(async () => {
  const { context, page } = await openBrowser();
  try {
    const base = await openLab(page);
    // JupyterLab restores the last layout (the Launcher may not be open), so don't rely on Launcher cards.
    // Deterministic route: create an empty notebook via the contents API, then open it by URL.
    await page.evaluate(async ({ base, name }) => {
      const xsrf = decodeURIComponent((document.cookie.match(/(?:^|; )_xsrf=([^;]+)/) || [])[1] || '');
      await fetch(`${base}api/contents/${encodeURIComponent(name)}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json', 'X-XSRFToken': xsrf },
        body: JSON.stringify({ type: 'notebook', content: { cells: [], metadata: { kernelspec: { name: 'python3', display_name: 'Python 3 (ipykernel)', language: 'python' } }, nbformat: 4, nbformat_minor: 5 } }),
      });
    }, { base, name: NAME });
    await page.goto(`${base}lab/tree/${encodeURIComponent(NAME)}`, { waitUntil: 'domcontentloaded' });
    const nb = page.locator('.jp-NotebookPanel:visible');
    await nb.locator('.jp-CodeCell').first().waitFor({ timeout: 90_000 });
    // Typing before the kernel is connected silently drops the execution. Wait for "| Idle" in the status bar.
    await page.locator('.jp-StatusBar-Widget', { hasText: /Idle/ }).first().waitFor({ timeout: 120_000 });
    await page.getByRole('button', { name: 'No' }).click({ timeout: 2000 }).catch(() => {}); // "Jupyter news?" toast

    for (let i = 0; i < CELLS.length; i++) {
      const cell = nb.locator('.jp-CodeCell').nth(i);
      await cell.waitFor();
      const editor = cell.locator('.cm-content');
      await editor.click();
      await page.keyboard.press('Control+a'); // replace whatever is there (restored drafts, autocompletes)
      await page.keyboard.insertText(CELLS[i]); // insertText avoids the auto-closing brackets that .type() triggers
      const typed = await editor.innerText();
      if (typed.trim() !== CELLS[i]) throw new Error(`editor mismatch in cell ${i + 1}: ${JSON.stringify(typed)}`);
      await page.keyboard.press('Shift+Enter'); // run + move to (or create) the next cell
      // Done when the prompt shows a number: "[1]:" instead of "[*]:" / "[ ]:".
      await cell.locator('.jp-InputPrompt').filter({ hasText: /\[\d+\]/ }).waitFor({ timeout: 10 * 60_000 });
      log(`CELL ${i + 1} OUTPUT:`, (await cell.locator('.jp-OutputArea').innerText()).trim());
    }

    await page.keyboard.press('Control+s');
    await page.locator('.jp-mod-current .jp-mod-dirty').waitFor({ state: 'detached', timeout: 15_000 }).catch(() => {});
    log('SAVED', `/workspace/${NAME}`);
    await dump(page, 'ui-notebook');
    // Shut the notebook's kernel down so it releases GPU memory (Kernel menu -> Shut Down Kernel).
    await page.getByRole('menuitem', { name: 'Kernel' }).click();
    await page.getByRole('menuitem', { name: /Shut Down Kernel/ }).click();
    await page.getByRole('dialog').getByRole('button', { name: /Shut Down/ }).click({ timeout: 3000 }).catch(() => {});
    log('KERNEL_SHUT_DOWN');
  } finally {
    await context.close();
  }
})().catch((e) => { log('ERROR', e.message); process.exit(1); });
