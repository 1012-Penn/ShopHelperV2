import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { test, before, after } from "node:test";

const browserModuleRoot = process.env.BROWSER_MODULE_ROOT;
const browserRequire = createRequire(browserModuleRoot
  ? pathToFileURL(resolve(browserModuleRoot, "package.json"))
  : new URL("../../.runtime/browser/package.json", import.meta.url));
const { chromium } = browserRequire(browserModuleRoot ? "." : "playwright");
const pageHtml = await readFile(new URL("../../app/static/index.html", import.meta.url), "utf8");
let browser;
let server;
let baseUrl;

before(async () => {
  browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined });
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

async function openPage(t, { conversationId = "c-home", feedback = [], list = [], listStatus = 200, listWait, onListRequested, histories = {}, onHistoryRequested, onChat } = {}) {
  const context = await browser.newContext();
  const page = await context.newPage();
  page.setDefaultTimeout(3000);
  t.after(() => context.close());
  await page.addInitScript((id) => localStorage.setItem("mewhelp-conversation-id", id), conversationId);
  await page.addInitScript((records) => localStorage.setItem("mewhelp.feedback.v1", JSON.stringify(records)), feedback);
  await page.route(/\/api\/conversations\?user_id=demo-user$/, async (route) => {
    onListRequested?.();
    if (listWait) await listWait;
    await route.fulfill({ status: listStatus, json: list });
  });
  await page.route(/\/api\/conversations\/[^/]+\/messages\?user_id=demo-user$/, async (route) => {
    onHistoryRequested?.(route);
    const id = new URL(route.request().url()).pathname.split("/")[3];
    const history = histories[id];
    if (typeof history === "function") return history(route);
    await route.fulfill({ json: history || [] });
  });
  await page.route("**/api/v1/chat/stream", async (route) => {
    await onChat?.(route);
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: [
        `event: token\r\ndata: ${JSON.stringify({ content: "继续完成。" })}`,
        `event: done\r\ndata: ${JSON.stringify({ message_id: 900 })}`,
    ].join("\r\n\r\n") + "\r\n\r\n",
    });
  });
  await page.goto(baseUrl);
  return page;
}

test("new chat gets a new ID while the previous conversation remains in the sidebar", async (t) => {
  let sent;
  const page = await openPage(t, {
    conversationId: "c-old",
    list: { items: [{ conversation_id: "c-old", preview: "之前的问题", has_summary: true, created_at: "2026-10-04" }] },
    onChat: async (route) => { sent = route.request().postDataJSON(); },
  });

  await page.getByRole("button", { name: "新对话" }).click();
  await page.locator("#message-input").fill("新问题");
  await page.locator("#chat-form button[type=submit]").click();
  await page.locator("#messages .message.assistant").last().getByText("继续完成。", { exact: true }).waitFor();

  assert.ok(sent.conversation_id);
  assert.notEqual(sent.conversation_id, "c-old");
  assert.equal(sent.message, "新问题");
  await page.getByRole("button", { name: /之前的问题/ }).waitFor();
  assert.equal(await page.evaluate(() => localStorage.getItem("mewhelp-conversation-id")), sent.conversation_id);
});

test("opening a conversation restores original messages, citations and actions, then continues on its ID", async (t) => {
  let sent;
  const page = await openPage(t, {
    conversationId: "c-home",
    feedback: [{ conversation_id: "c-older", message_id: 82, choice: "up", time: "2026-10-03T00:02:00" }],
    list: { items: [{ conversation_id: "c-older", preview: "订单1001怎么了", has_summary: true, created_at: "2026-10-03" }] },
    histories: {
      "c-older": { items: [
        { id: 81, role: "user", content: "订单1001怎么了", citations: null, actions: null, created_at: "2026-10-03T00:00:00" },
        { id: 82, role: "assistant", content: "物流预计明天送达[1]。", citations: [{ n: 1, chunk_id: 17, section_path: ["物流"], answer: "预计明天送达", source_url: "/api/v1/knowledge/source?source=ship" }], actions: { items: ["handoff", "create_ticket"], question: "订单1001怎么了", ticket: { status: "offered" } }, created_at: "2026-10-03T00:01:00" },
      ] },
    },
    onChat: async (route) => { sent = route.request().postDataJSON(); },
  });

  await page.getByRole("button", { name: /订单1001怎么了/ }).click();
  const user = page.locator("#messages .message.user").last();
  await user.getByText("订单1001怎么了", { exact: true }).waitFor();
  const answer = page.locator("#messages .message.assistant").last();
  await answer.getByRole("button", { name: "查看引用 1" }).click();
  await page.getByRole("dialog", { name: "引用来源" }).getByText("预计明天送达", { exact: true }).waitFor();
  await page.getByRole("button", { name: "关闭来源" }).click();
  const satisfied = answer.getByRole("button", { name: "满意", exact: true });
  assert.equal(await satisfied.isDisabled(), true);
  assert.equal(await satisfied.getAttribute("aria-pressed"), "true");
  await answer.getByRole("button", { name: "转人工", exact: true }).waitFor();
  await answer.getByRole("button", { name: "建工单", exact: true }).waitFor();

  await page.locator("#message-input").fill("还要多久？");
  await page.locator("#chat-form button[type=submit]").click();
  await page.locator("#messages .message.assistant").last().getByText("继续完成。", { exact: true }).waitFor();
  assert.deepEqual(sent, { conversation_id: "c-older", user_id: "demo-user", message: "还要多久？" });
});

test("history restores pending chapter-six order and refund actions", async (t) => {
  const sent = [];
  let refundPayload;
  const page = await openPage(t, {
    conversationId: "c-home",
    list: { items: [
      { conversation_id: "c-order", preview: "要查订单政策", has_summary: false },
      { conversation_id: "c-order-done", preview: "已选择订单", has_summary: false },
      { conversation_id: "c-refund", preview: "申请退款", has_summary: false },
    ] },
    histories: {
      "c-order": { items: [
        { id: 101, role: "assistant", content: "请先选择要查询政策的订单（演示数据）：", actions: {
          items: ["order_choices"], order_choice: { request_id: "order-offer-101", order_ids: ["DEMO-1001"] },
        } },
      ] },
      "c-order-done": { items: [
        { id: 103, role: "assistant", content: "请先选择订单。", actions: {
          items: ["order_choices"], order_choice: { request_id: "order-offer-103", order_ids: ["DEMO-1001"] },
        } },
        { id: 104, role: "assistant", content: "该订单符合演示政策的进一步核验条件。", actions: null },
      ] },
      "c-refund": { items: [
        { id: 102, role: "assistant", content: "可以提交演示申请。", actions: {
          items: ["refund_form"], refund_form: { request_id: "refund-offer-102", order_id: "DEMO-1001", request_type: "退款" },
        } },
      ] },
    },
    onChat: async (route) => {
      sent.push(route.request().postDataJSON());
    },
  });
  await page.route("**/api/v1/refund-applications", async (route) => {
    refundPayload = route.request().postDataJSON();
    await route.fulfill({ json: { status: "recorded", message: "演示申请已记录；未执行真实退款。" } });
  });

  await page.getByRole("button", { name: /要查订单政策/ }).click();
  const orderCard = page.getByRole("button", { name: /DEMO-1001/ });
  await orderCard.waitFor();
  await orderCard.click();
  await page.locator("#messages .message.assistant").last().waitFor();
  assert.deepEqual(sent[0], {
    conversation_id: "c-order", user_id: "demo-user", selected_order_id: "DEMO-1001",
    selection_message_id: 101, request_id: "order-offer-101",
  });

  await page.getByRole("button", { name: /已选择订单/ }).click();
  await page.locator("#messages").getByText("该订单符合演示政策的进一步核验条件。", { exact: true }).waitFor();
  assert.equal(await page.getByRole("button", { name: /DEMO-1001/ }).count(), 0);

  await page.getByRole("button", { name: /申请退款/ }).click();
  const refundForm = page.locator(".refund-form");
  await refundForm.waitFor();
  await refundForm.locator("select").selectOption("商品质量问题");
  await refundForm.getByRole("button", { name: "提交演示申请" }).click();
  await refundForm.getByText("演示申请已记录；未执行真实退款。", { exact: true }).waitFor();
  assert.deepEqual(refundPayload, {
    conversation_id: "c-refund", user_id: "demo-user", message_id: 102,
    request_id: "refund-offer-102", order_id: "DEMO-1001", request_type: "退款", reason: "商品质量问题",
  });
});

test("history restores a recorded refund as read-only with its original reason", async (t) => {
  let refundRequests = 0;
  const page = await openPage(t, {
    conversationId: "c-home",
    list: { items: [{ conversation_id: "c-refund-recorded", preview: "已提交退款申请", has_summary: false }] },
    histories: {
      "c-refund-recorded": { items: [{ id: 111, role: "assistant", content: "可以提交演示申请。", actions: {
        items: ["refund_form"], refund_form: {
          request_id: "refund-offer-111", order_id: "DEMO-1001", request_type: "退款",
          status: "recorded", reason: "商品质量问题", application_id: "refund-offer-111",
        },
      } }] },
    },
  });
  await page.route("**/api/v1/refund-applications", async (route) => {
    refundRequests += 1;
    await route.fulfill({ json: { status: "recorded", message: "演示申请已记录。" } });
  });

  await page.getByRole("button", { name: /已提交退款申请/ }).click();
  const refundForm = page.locator(".refund-form");
  await refundForm.waitFor();
  assert.equal(await refundForm.locator("select").inputValue(), "商品质量问题");
  assert.equal(await refundForm.locator("select").isDisabled(), true);
  assert.equal(await refundForm.getByRole("button", { name: "提交演示申请" }).isDisabled(), true);
  await refundForm.getByText(/已记录/).waitFor();
  assert.equal(refundRequests, 0);
});

test("conversation list failure stays quiet and leaves chat usable", async (t) => {
  const page = await openPage(t, { listStatus: 503, list: { detail: "unavailable" } });
  await page.locator("#message-input").fill("仍可聊天");
  await page.locator("#chat-form button[type=submit]").click();
  await page.locator("#messages .message.assistant").last().getByText("继续完成。", { exact: true }).waitFor();
  assert.equal(await page.locator("#messages .message.user").last().textContent(), "你仍可聊天");
  assert.equal(await page.locator("[role=alert]").count(), 0);
});

test("a slow stale history response cannot replace the latest selected conversation", async (t) => {
  let slowStarted;
  const slowStartedPromise = new Promise((resolve) => { slowStarted = resolve; });
  let sent;
  const page = await openPage(t, {
    conversationId: "c-home",
    list: { items: [
      { conversation_id: "c-a", preview: "会话A问题", has_summary: false, created_at: "2026-10-02" },
      { conversation_id: "c-b", preview: "会话B问题", has_summary: false, created_at: "2026-10-01" },
    ] },
    histories: {
      "c-a": async (route) => {
        slowStarted();
        await new Promise((resolve) => setTimeout(resolve, 250));
        await route.fulfill({ json: { items: [{ id: 1, role: "user", content: "A 的旧记录", citations: null, actions: null }] } });
      },
      "c-b": { items: [{ id: 2, role: "user", content: "B 的最新记录", citations: null, actions: null }] },
    },
    onChat: async (route) => { sent = route.request().postDataJSON(); },
  });

  await page.getByRole("button", { name: /会话A问题/ }).click();
  await slowStartedPromise;
  await page.getByRole("button", { name: /会话B问题/ }).click();
  await page.locator("#messages").getByText("B 的最新记录", { exact: true }).waitFor();
  await page.waitForTimeout(300);
  assert.equal(await page.locator("#messages").getByText("A 的旧记录", { exact: true }).count(), 0);

  await page.locator("#message-input").fill("发给B");
  await page.locator("#chat-form button[type=submit]").click();
  await page.locator("#messages .message.assistant").last().getByText("继续完成。", { exact: true }).waitFor();
  assert.equal(sent.conversation_id, "c-b");
});

test("sending during history load prevents the delayed history response from erasing new messages", async (t) => {
  let finishHistory;
  let historyStarted;
  const historyStartedPromise = new Promise((resolve) => { historyStarted = resolve; });
  const delayedHistory = new Promise((resolve) => { finishHistory = resolve; });
  let sent;
  const page = await openPage(t, {
    conversationId: "c-home",
    list: { items: [{ conversation_id: "c-target", preview: "要继续的会话", has_summary: false }] },
    histories: {
      "c-target": async (route) => {
        historyStarted();
        await delayedHistory;
        await route.fulfill({ json: { items: [{ id: 70, role: "user", content: "迟到的旧历史", citations: null, actions: null }] } });
      },
    },
    onChat: async (route) => { sent = route.request().postDataJSON(); },
  });

  await page.getByRole("button", { name: /要继续的会话/ }).click();
  await historyStartedPromise;
  await page.locator("#message-input").fill("正在回载时发送");
  await page.locator("#chat-form button[type=submit]").click();
  await page.locator("#messages .message.assistant").last().getByText("继续完成。", { exact: true }).waitFor();
  finishHistory();
  await page.waitForTimeout(100);

  assert.equal(sent.conversation_id, "c-target");
  assert.equal(await page.locator("#messages").getByText("正在回载时发送", { exact: true }).count(), 1);
  assert.equal(await page.locator("#messages").getByText("迟到的旧历史", { exact: true }).count(), 0);
});

test("restored submitted and unknown ticket actions stay disabled", async (t) => {
  let ticketRequests = 0;
  const page = await openPage(t, {
    conversationId: "c-home",
    list: { items: [
      { conversation_id: "c-created", preview: "已建工单", has_summary: false },
      { conversation_id: "c-submitting", preview: "提交中的工单", has_summary: false },
      { conversation_id: "c-unknown", preview: "结果待核查", has_summary: false },
    ] },
    histories: {
      "c-created": { items: [{ id: 91, role: "assistant", content: "已处理", citations: null, actions: { items: ["create_ticket"], question: "订单", ticket: { status: "created", ticket_no: "TKT-OLD" } } }] },
      "c-submitting": { items: [{ id: 92, role: "assistant", content: "处理中", citations: null, actions: { items: ["create_ticket"], question: "订单", ticket: { status: "submitting" } } }] },
      "c-unknown": { items: [{ id: 93, role: "assistant", content: "待核查", citations: null, actions: { items: ["create_ticket"], question: "订单", ticket: { status: "unknown", message: "请联系人工核查" } } }] },
    },
  });
  await page.route("**/api/v1/tickets", async (route) => {
    ticketRequests += 1;
    await route.fulfill({ json: { status: "created", ticket_no: "TKT-DUPLICATE" } });
  });

  for (const [conversationId, preview, statusText] of [
    ["c-created", "已建工单", "TKT-OLD"],
    ["c-submitting", "提交中的工单", "工单已提交"],
    ["c-unknown", "结果待核查", "请联系人工核查"],
  ]) {
    await page.getByRole("button", { name: new RegExp(preview) }).click();
    const answer = page.locator("#messages .message.assistant").last();
    const ticket = answer.getByRole("button", { name: "建工单", exact: true });
    await ticket.waitFor();
    assert.equal(await ticket.isDisabled(), true, `${conversationId} cannot resubmit a ticket`);
    await answer.getByText(statusText, { exact: false }).waitFor();
  }
  assert.equal(ticketRequests, 0);
});

test("a delayed startup list cannot start history loading after the user has sent a message", async (t) => {
  let finishList;
  let finishChat;
  let listStarted;
  let chatStarted;
  let sent;
  let historyRequests = 0;
  const listWait = new Promise((resolve) => { finishList = resolve; });
  const chatWait = new Promise((resolve) => { finishChat = resolve; });
  const listStartedPromise = new Promise((resolve) => { listStarted = resolve; });
  const chatStartedPromise = new Promise((resolve) => { chatStarted = resolve; });
  const page = await openPage(t, {
    conversationId: "c-startup",
    list: { items: [{ conversation_id: "c-startup", preview: "已有对话", has_summary: false }] },
    listWait,
    onListRequested: () => listStarted(),
    onHistoryRequested: () => { historyRequests += 1; },
    onChat: async (route) => {
      sent = route.request().postDataJSON();
      chatStarted();
      await chatWait;
    },
  });

  try {
    await listStartedPromise;
    await page.locator("#message-input").fill("启动列表等待时发送");
    await page.locator("#chat-form button[type=submit]").click();
    await chatStartedPromise;
    finishList();
    await page.waitForTimeout(100);

    assert.equal(historyRequests, 0);
    assert.equal(sent.conversation_id, "c-startup");
    assert.equal(await page.locator("#messages").getByText("启动列表等待时发送", { exact: true }).count(), 1);
    assert.equal(await page.locator("#messages .message.assistant").count(), 1);
  } finally {
    finishList();
    finishChat();
  }
  await page.locator("#messages .message.assistant").last().getByText("继续完成。", { exact: true }).waitFor();
  assert.equal(await page.locator("#messages").getByText("启动列表等待时发送", { exact: true }).count(), 1);
});
