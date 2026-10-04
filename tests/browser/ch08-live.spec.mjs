import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { spawn, execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import readline from 'node:readline';
import { test, before, after } from 'node:test';

const require = createRequire(new URL('../../.runtime/browser/package.json', import.meta.url));
const { chromium } = require('playwright');
const repository = new URL('../..', import.meta.url).pathname;
let browser, serverProcess, baseUrl, dataDir, databasePath;

async function freePort() {
  const server = createServer();
  await new Promise((resolve, reject) => server.listen(0, '127.0.0.1', error => error ? reject(error) : resolve()));
  const { port } = server.address();
  await new Promise(resolve => server.close(resolve));
  return port;
}

async function waitForService(port, startupLine, processRef) {
  const started = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('fixture did not announce startup')), 15000);
    startupLine.once('line', line => { clearTimeout(timer); resolve(JSON.parse(line)); });
    processRef.once('exit', code => { clearTimeout(timer); reject(new Error(`fixture exited early (${code})`)); });
  });
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/health`);
      if (response.ok) return started;
    } catch { /* Wait for Uvicorn to bind after announcing its configuration. */ }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error('fixture health endpoint did not become ready');
}

before(async () => {
  assert.ok(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE, 'PLAYWRIGHT_CHROMIUM_EXECUTABLE must point to the cached Chromium binary');
  dataDir = await mkdtemp(join(tmpdir(), 'shophelper-ch08-browser-'));
  const port = await freePort();
  baseUrl = `http://127.0.0.1:${port}`;
  serverProcess = spawn('python3', [
    'tests/browser/ch08_fixture.py', '--data-dir', dataDir, '--port', String(port),
  ], { cwd: repository, stdio: ['ignore', 'pipe', 'inherit'] });
  const startupLine = readline.createInterface({ input: serverProcess.stdout });
  const started = await waitForService(port, startupLine, serverProcess);
  databasePath = started.db;
  startupLine.close();
  browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE });
});

after(async () => {
  await browser?.close();
  if (serverProcess && serverProcess.exitCode === null) {
    serverProcess.kill('SIGTERM');
    await new Promise(resolve => {
      const timer = setTimeout(() => { serverProcess.kill('SIGKILL'); resolve(); }, 5000);
      serverProcess.once('exit', () => { clearTimeout(timer); resolve(); });
    });
  }
  if (dataDir) await rm(dataDir, { recursive: true, force: true });
});

async function createPage(t) {
  const context = await browser.newContext();
  t.after(() => context.close());
  const page = await context.newPage();
  page.setDefaultTimeout(10000);
  await page.goto(baseUrl);
  return page;
}

async function createPendingPreview(page) {
  await page.locator('#message-input').fill('帮我建工单');
  await page.locator('#chat-form button[type=submit]').click();
  await page.getByText('请补充问题描述。', { exact: true }).waitFor();
  await page.locator('#message-input').fill('耳机无法充电');
  await page.locator('#chat-form button[type=submit]').click();
  await page.locator('.ticket-preview').waitFor();
  await page.locator('.ticket-preview p').getByText('耳机无法充电', { exact: true }).waitFor();
  return await page.evaluate(() => localStorage.getItem('mewhelp-conversation-id'));
}

function inspectConversation(conversationId) {
  const output = execFileSync('python3', [
    'tests/browser/ch08_fixture.py', '--inspect', databasePath, '--conversation-id', conversationId,
  ], { cwd: repository, encoding: 'utf8' });
  return JSON.parse(output);
}

test('real HTTP ticket preview survives refresh and approval creates one SQL ticket', async t => {
  const page = await createPage(t);
  const confirmationRequests = [];
  page.on('request', request => {
    if (request.url().endsWith('/api/v1/chat/ticket-confirmation')) confirmationRequests.push(request);
  });
  const conversationId = await createPendingPreview(page);

  await page.reload();
  await page.locator('.ticket-preview').waitFor();
  assert.equal(await page.locator('.ticket-preview p').textContent(), '耳机无法充电');
  const confirm = page.getByRole('button', { name: '确认提交', exact: true });
  const cancel = page.getByRole('button', { name: '取消', exact: true });
  await confirm.click();
  await page.getByText(/工单已创建，工单号：TKT-[A-F0-9]+。/, { exact: false }).waitFor();
  assert.equal(await confirm.isDisabled(), true);
  assert.equal(await cancel.isDisabled(), true);
  await confirm.evaluate(button => button.click());
  await page.waitForTimeout(100);

  assert.equal(confirmationRequests.length, 1);
  assert.equal(confirmationRequests[0].postDataJSON().approve, true);
  const visibleAnswer = await page.locator('#messages').innerText();
  assert.match(visibleAnswer, /工单已创建，工单号：TKT-[A-F0-9]+。/);
  const rows = inspectConversation(conversationId);
  assert.equal(rows.tickets.length, 1);
  assert.equal(rows.tickets[0][1], '耳机无法充电');
  assert.equal(rows.audits.filter(([, status]) => status === 'success').length, 1);
});

test('real HTTP cancellation records denial and never creates a ticket', async t => {
  const page = await createPage(t);
  const confirmationRequests = [];
  page.on('request', request => {
    if (request.url().endsWith('/api/v1/chat/ticket-confirmation')) confirmationRequests.push(request);
  });
  const conversationId = await createPendingPreview(page);
  await page.getByRole('button', { name: '取消', exact: true }).click();
  await page.getByText('已取消本次建单，没有创建工单。', { exact: true }).waitFor();
  assert.equal(await page.getByRole('button', { name: '确认提交', exact: true }).isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: '取消', exact: true }).isDisabled(), true);
  await page.getByRole('button', { name: '取消', exact: true }).evaluate(button => button.click());
  await page.waitForTimeout(100);

  assert.equal(confirmationRequests.length, 1);
  assert.equal(confirmationRequests[0].postDataJSON().approve, false);
  const rows = inspectConversation(conversationId);
  assert.equal(rows.tickets.length, 0);
  assert.ok(rows.audits.some(([toolName, status]) => toolName === 'create_ticket' && status === 'permission_denied'));
});
