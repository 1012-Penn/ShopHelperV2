import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { test, before, after } from "node:test";

const browserRequire = createRequire(new URL("../../.runtime/browser/package.json", import.meta.url));
const { chromium } = browserRequire("playwright");
const pageHtml = await readFile(new URL("../../app/static/index.html", import.meta.url), "utf8");
let browser;
let server;
let baseUrl;

before(async () => {
  browser = await chromium.launch({ headless: true });
  server = createServer((_request, response) => {
    response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    response.end(pageHtml);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  baseUrl = `http://127.0.0.1:${server.address().port}`;
});

after(async () => {
  await browser?.close();
  await new Promise((resolve) => server?.close(resolve));
});

async function openPage(t, streams) {
  const context = await browser.newContext();
  const page = await context.newPage();
  page.setDefaultTimeout(3000);
  t.after(() => context.close());
  let streamIndex = 0;
  await page.route("**/api/v1/chat/stream", async (route) => {
    const frames = streams[streamIndex++];
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: frames.join("\r\n\r\n") + "\r\n\r\n",
    });
  });
  await page.goto(baseUrl);
  return page;
}

const frame = (event, data) => `event: ${event}\r\ndata: ${JSON.stringify(data)}`;

test("actions stay bound to their message and ticket and handoff act independently", async (t) => {
  const page = await openPage(t, [[
    frame("tool_status", { tool_name: "query_order", status: "running" }),
    frame("tool_status", { tool_name: "query_logistics", status: "running" }),
    frame("token", { content: "我可以帮你处理。" }),
    frame("actions", { items: [{ type: "handoff", label: "转人工" }, { type: "create_ticket", label: "建工单" }], message_id: 41 }),
    frame("done", { message_id: 41, conversation_id: "c1" }),
  ], [
    frame("token", { content: "第二轮回复。" }),
    frame("done", { message_id: 42, conversation_id: "c1" }),
    frame("actions", { items: [{ type: "create_ticket", label: "建工单" }], message_id: 42 }),
  ]]);
  let ticketCount = 0;
  let ticketPayload;
  let finishTicket;
  const ticketStarted = new Promise((resolve) => { finishTicket = resolve; });
  await page.route("**/api/v1/tickets", async (route) => {
    ticketCount += 1;
    ticketPayload = route.request().postDataJSON();
    finishTicket();
    await new Promise((resolve) => setTimeout(resolve, 250));
    await route.fulfill({ json: { status: "created", ticket_no: "TKT-ABC123" } });
  });

  await page.locator("#message-input").fill("我要投诉");
  await page.locator("#chat-form button[type=submit]").click();
  const firstAnswer = page.locator("#messages .message.assistant").nth(0);
  await firstAnswer.getByRole("button", { name: "转人工", exact: true }).waitFor();
  assert.equal(await firstAnswer.locator(".badge").count(), 2, "each tool gets its own status badge");

  page.once("dialog", (dialog) => dialog.accept());
  await firstAnswer.getByRole("button", { name: "建工单", exact: true }).click();
  await ticketStarted;
  const ticketButton = firstAnswer.getByRole("button", { name: "建工单", exact: true });
  assert.equal(await ticketButton.isDisabled(), true, "only the submitting ticket action is disabled");
  const handoffButton = firstAnswer.getByRole("button", { name: "转人工", exact: true });
  assert.equal(await handoffButton.isDisabled(), false);
  page.once("dialog", (dialog) => dialog.accept());
  await handoffButton.click();
  await page.locator("#messages").getByText("已转接人工客服", { exact: true }).waitFor();
  await page.locator("#messages").getByText("您好，我是客服小猫，请问有什么可以帮您的", { exact: true }).waitFor();
  assert.equal(ticketCount, 1, "handoff is local and does not submit another ticket");

  await firstAnswer.getByText("TKT-ABC123", { exact: false }).waitFor();
  const conversationId = await page.evaluate(() => localStorage.getItem("mewhelp-conversation-id"));
  assert.deepEqual(ticketPayload, { conversation_id: conversationId, user_id: "demo-user", message_id: 41 });

  await page.locator("#message-input").fill("继续问物流");
  await page.locator("#chat-form button[type=submit]").click();
  const secondAnswer = page.locator("#messages .message.assistant").last();
  await secondAnswer.getByRole("button", { name: "建工单", exact: true }).waitFor();
  page.once("dialog", (dialog) => dialog.accept());
  await secondAnswer.getByRole("button", { name: "建工单", exact: true }).click();
  await page.waitForTimeout(300);
  assert.equal(ticketCount, 2, "late actions bind to the second message, not the first one");
  assert.equal(ticketPayload.message_id, 42);
});

test("cancelled actions have no side effect and unknown ticket results are not retried", async (t) => {
  const page = await openPage(t, [[
    frame("token", { content: "需要进一步协助吗？" }),
    frame("done", { message_id: 51, conversation_id: "c2" }),
    frame("actions", { items: [{ type: "handoff", label: "转人工" }, { type: "create_ticket", label: "建工单" }], message_id: 51 }),
  ], [
    frame("token", { content: "我会继续帮你查询。" }),
    frame("done", { message_id: 52, conversation_id: "c2" }),
  ]]);
  let ticketCount = 0;
  await page.route("**/api/v1/tickets", async (route) => {
    ticketCount += 1;
    await route.fulfill({ status: 409, json: { status: "unknown", message: "请联系人工核查" } });
  });
  await page.locator("#message-input").fill("我要投诉");
  await page.locator("#chat-form button[type=submit]").click();
  const firstAnswer = page.locator("#messages .message.assistant").last();
  const cancel = (dialog) => dialog.dismiss();
  page.once("dialog", cancel);
  await firstAnswer.getByRole("button", { name: "转人工", exact: true }).click();
  page.once("dialog", cancel);
  await firstAnswer.getByRole("button", { name: "建工单", exact: true }).click();
  assert.equal(ticketCount, 0);

  page.once("dialog", (dialog) => dialog.accept());
  await firstAnswer.getByRole("button", { name: "建工单", exact: true }).click();
  await firstAnswer.getByText("请联系人工核查", { exact: false }).waitFor();
  await page.waitForTimeout(100);
  assert.equal(await firstAnswer.getByRole("button", { name: "建工单", exact: true }).isDisabled(), true);
  assert.equal(ticketCount, 1, "an unknown outcome is not automatically resubmitted");

  await page.locator("#message-input").fill("继续聊天");
  await page.locator("#chat-form button[type=submit]").click();
  await page.locator("#messages .message.assistant").last().getByText("我会继续帮你查询。", { exact: true }).waitFor();
  assert.equal(ticketCount, 1, "chat continues without another ticket request");
});

test("citation snapshots open the safe source link and feedback stays in localStorage", async (t) => {
  const page = await openPage(t, [[
    frame("token", { content: "退货期限为七天[1]。<img src=x onerror=alert(1)>" }),
    frame("citations", { items: [{ n: 1, chunk_id: 7, section_path: ["售后", "退货"], answer: "符合条件可申请退货。", source_url: "/api/v1/knowledge/source?source=policy#section-1" }] }),
    frame("done", { message_id: 61, conversation_id: "c3" }),
  ]]);
  await page.locator("#message-input").fill("退货政策");
  await page.locator("#chat-form button[type=submit]").click();
  const answer = page.locator("#messages .message.assistant").last();
  await answer.getByRole("button", { name: "查看引用 1" }).waitFor();
  assert.equal(await answer.locator("img").count(), 0, "model HTML is rendered as text");
  await answer.getByRole("button", { name: "查看引用 1" }).click();
  const citation = page.getByRole("dialog", { name: "引用来源" });
  await citation.getByText("符合条件可申请退货。", { exact: true }).waitFor();
  assert.equal(await citation.getByRole("link", { name: "跳回原文章节" }).getAttribute("href"), "/api/v1/knowledge/source?source=policy#section-1");
  await page.getByRole("button", { name: "关闭来源" }).click();

  await answer.getByRole("button", { name: "满意", exact: true }).click();
  const feedback = await page.evaluate(() => JSON.parse(localStorage.getItem("mewhelp.feedback.v1")));
  const conversationId = await page.evaluate(() => localStorage.getItem("mewhelp-conversation-id"));
  assert.equal(feedback.length, 1);
  assert.deepEqual(feedback[0], {
    conversation_id: conversationId,
    message_id: 61,
    choice: "up",
    time: feedback[0].time,
  });
});
