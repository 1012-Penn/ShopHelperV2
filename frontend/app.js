const state = { conversationId: `web-${Date.now()}`, messages: [], sending: false };
const $ = (selector) => document.querySelector(selector);
const messagesEl = $("#messages");
const inputEl = $("#message-input");
const sendButton = $("#send-button");

function addMessage(role, content, streaming = false) {
  const message = { role, content, streaming };
  state.messages.push(message);
  renderMessage(message);
  return message;
}

function renderMessage(message) {
  const item = document.createElement("article");
  item.className = `message ${message.role}${message.streaming ? " typing" : ""}`;
  const avatar = document.createElement("div"); avatar.className = "bubble-avatar"; avatar.textContent = message.role === "assistant" ? "喵" : "我";
  const wrap = document.createElement("div"); wrap.className = "bubble-wrap";
  const meta = document.createElement("div"); meta.className = "message-meta"; meta.textContent = message.role === "assistant" ? "喵助理 · 刚刚" : "我 · 刚刚";
  const bubble = document.createElement("div"); bubble.className = "bubble"; bubble.textContent = message.content;
  wrap.append(meta, bubble); item.append(avatar, wrap); messagesEl.append(item); messagesEl.scrollTop = messagesEl.scrollHeight; message.element = item;
}

function updateMessage(message) {
  message.element.querySelector(".bubble").textContent = message.content;
  message.element.classList.toggle("typing", message.streaming);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function historyForRequest() {
  return state.messages.filter((message) => !message.streaming && message.content).map(({ role, content }) => ({ role, content }));
}

function showWelcome() {
  state.messages = []; messagesEl.replaceChildren();
  addMessage("assistant", "你好呀，我是喵助理～\n购物、物流、退换货等问题都可以问我。");
  addMessage("assistant", "你可以直接告诉我遇到的情况，我会帮你整理出解决办法。");
}

function parseSseBlock(block, assistantMessage) {
  const eventLine = block.split("\n").find((line) => line.startsWith("event:"));
  const dataLine = block.split("\n").find((line) => line.startsWith("data:"));
  if (!eventLine || !dataLine) return;
  const event = eventLine.slice(6).trim();
  try {
    const data = JSON.parse(dataLine.slice(5).trim());
    if (event === "token") { assistantMessage.content += data.content || ""; updateMessage(assistantMessage); }
    if (event === "error") { assistantMessage.content = "抱歉，刚刚连接有点不稳定，请稍后再试。"; assistantMessage.streaming = false; updateMessage(assistantMessage); }
  } catch (error) { console.warn("无法解析 SSE 数据", error); }
}

async function sendMessage(text) {
  const content = text.trim(); if (!content || state.sending) return;
  state.sending = true; sendButton.disabled = true; addMessage("user", content); inputEl.value = ""; inputEl.style.height = "auto";
  const assistantMessage = addMessage("assistant", "", true);
  try {
    const response = await fetch("/api/v1/chat/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ conversation_id: state.conversationId, message: content, history: historyForRequest().slice(0, -1) }) });
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
    while (true) {
      const { value, done } = await reader.read(); buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const blocks = buffer.split("\n\n"); buffer = blocks.pop() || ""; blocks.forEach((block) => parseSseBlock(block, assistantMessage)); if (done) break;
    }
    if (buffer.trim()) parseSseBlock(buffer, assistantMessage);
  } catch (error) { assistantMessage.content = "暂时连接不上客服服务，请确认后端已启动后再试。"; console.error(error); }
  finally { assistantMessage.streaming = false; updateMessage(assistantMessage); state.sending = false; sendButton.disabled = false; inputEl.focus(); }
}

async function extractAfterSale() {
  const text = $("#after-sale-input").value.trim(); const resultEl = $("#extract-result");
  if (!text) { resultEl.hidden = false; resultEl.textContent = "先描述一下你的售后情况吧～"; return; }
  const button = $("#extract-button"); button.disabled = true; button.innerHTML = "正在识别…";
  try {
    const response = await fetch("/api/v1/after-sale/extract", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`); const data = await response.json();
    const labels = [["订单号", data.order_id], ["诉求类型", data.request_type], ["期望方案", data.expected_solution]];
    resultEl.innerHTML = labels.map(([label, value]) => `<div class="result-row"><span>${label}</span><span>${value || "未提及"}</span></div>`).join(""); resultEl.hidden = false;
  } catch (error) { resultEl.hidden = false; resultEl.textContent = "识别失败，请稍后重试。"; console.error(error); }
  finally { button.disabled = false; button.innerHTML = "智能识别售后信息 <span>✦</span>"; }
}

$("#chat-form").addEventListener("submit", (event) => { event.preventDefault(); sendMessage(inputEl.value); });
inputEl.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(inputEl.value); } });
inputEl.addEventListener("input", () => { inputEl.style.height = "auto"; inputEl.style.height = `${Math.min(inputEl.scrollHeight, 110)}px`; });
$("#extract-button").addEventListener("click", extractAfterSale); $("#clear-chat").addEventListener("click", showWelcome);
$("#search-button").addEventListener("click", () => { const query = $("#global-search").value.trim(); if (query) sendMessage(query); });
document.querySelectorAll("[data-fill]").forEach((button) => button.addEventListener("click", () => { inputEl.value = button.dataset.fill; inputEl.focus(); }));
document.querySelectorAll("[data-target]").forEach((button) => button.addEventListener("click", () => document.getElementById(button.dataset.target)?.scrollIntoView({ behavior: "smooth" })));
document.querySelectorAll("#quick-replies button").forEach((button) => button.addEventListener("click", () => sendMessage(button.textContent)));
showWelcome();
