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

const frame = (event, data) => `event: ${event}\r\ndata: ${JSON.stringify(data)}`;

async function openPage(t) {
  const context = await browser.newContext();
  const page = await context.newPage();
  page.setDefaultTimeout(4000);
  t.after(() => context.close());
  await page.goto(baseUrl);
  return page;
}

test("selecting an inline demo order resumes the same chat without creating a second user question", async (t) => {
  const page = await openPage(t);
  let streamCount = 0;
  let resumePayload;
  await page.route("**/api/v1/chat/stream", async (route) => {
    streamCount += 1;
    if (streamCount === 1) {
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: frame("order_choices", {
          conversation_id: "server-conversation",
          message_id: 81,
          request_id: "order-offer-81",
          content: "请先选择订单（演示数据）：",
          choices: [{
            order_id: "DEMO-1001", product: "耳机 <img src=x>", ordered_at: "2026-09-18",
            status: "已签收", total: 399, source: "模拟数据",
          }],
        }) + "\r\n\r\n",
      });
      return;
    }
    resumePayload = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: [
        frame("token", { content: "该订单符合演示政策的进一步核验条件。" }),
        frame("done", { message_id: 82, conversation_id: resumePayload.conversation_id }),
      ].join("\r\n\r\n") + "\r\n\r\n",
    });
  });

  await page.locator("#message-input").fill("这个能退吗");
  await page.locator("#chat-form button[type=submit]").click();
  const card = page.getByRole("button", { name: /DEMO-1001/ });
  await card.waitFor();
  assert.equal(await card.locator("img").count(), 0, "order fields are rendered as text");
  await card.click();
  await page.locator("#messages").getByText("该订单符合演示政策的进一步核验条件。", { exact: true }).waitFor();
  const conversationId = await page.evaluate(() => localStorage.getItem("mewhelp-conversation-id"));
  assert.deepEqual(resumePayload, {
    conversation_id: conversationId,
    user_id: "demo-user",
    selected_order_id: "DEMO-1001",
    selection_message_id: 81,
    request_id: "order-offer-81",
  });
  assert.equal(await page.locator("#messages .message.user").count(), 2, "the click is shown once and the original question is not duplicated");
  assert.equal(await card.isDisabled(), true, "an old card cannot be clicked again");
});

test("refund form sends fixed category, full binding, and one demo application", async (t) => {
  const page = await openPage(t);
  let chatPayload;
  let refundPayload;
  let refundCount = 0;
  await page.route("**/api/v1/chat/stream", async (route) => {
    chatPayload = route.request().postDataJSON();
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: [
        frame("token", { content: "可以提交演示申请。" }),
        frame("refund_form", {
          message_id: 91, request_id: "refund-offer-91", order_id: "DEMO-1001", request_type: "退款",
        }),
        frame("done", { message_id: 91, conversation_id: chatPayload.conversation_id }),
      ].join("\r\n\r\n") + "\r\n\r\n",
    });
  });
  await page.route("**/api/v1/refund-applications", async (route) => {
    refundCount += 1;
    refundPayload = route.request().postDataJSON();
    await route.fulfill({ json: { status: "recorded", message: "演示申请已记录；未执行真实退款。" } });
  });

  await page.locator("#message-input").fill("订单 DEMO-1001 可以退款吗");
  await page.locator("#chat-form button[type=submit]").click();
  const refundForm = page.locator(".refund-form");
  await refundForm.waitFor();
  assert.match(await refundForm.textContent(), /不执行真实退款/);
  await refundForm.locator("select").selectOption("商品质量问题");
  const submit = refundForm.getByRole("button", { name: "提交演示申请" });
  await submit.click();
  await refundForm.getByText("演示申请已记录；未执行真实退款。", { exact: true }).waitFor();
  assert.equal(refundCount, 1);
  const conversationId = await page.evaluate(() => localStorage.getItem("mewhelp-conversation-id"));
  assert.equal(chatPayload.conversation_id, conversationId);
  assert.deepEqual(refundPayload, {
    conversation_id: conversationId,
    user_id: "demo-user",
    message_id: 91,
    request_id: "refund-offer-91",
    order_id: "DEMO-1001",
    request_type: "退款",
    reason: "商品质量问题",
  });
  assert.equal(await submit.isDisabled(), true, "the form cannot be double-submitted");
});
