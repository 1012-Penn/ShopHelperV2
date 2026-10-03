import { useRef, useState } from "react";
import {
  ArrowUpRight,
  Bot,
  Box,
  Check,
  ChevronRight,
  CircleHelp,
  Clock3,
  Command,
  Copy,
  Headphones,
  LayoutDashboard,
  MessageCircle,
  MoreHorizontal,
  PackageCheck,
  Plus,
  RotateCcw,
  Search,
  Send,
  Settings2,
  Sparkles,
  Tag,
  Truck,
  ThumbsUp,
  ThumbsDown,
  UserRound,
  WandSparkles,
  X,
} from "lucide-react";

const suggestions = ["查一下物流进度", "我想申请退货", "订单可以开发票吗", "优惠活动怎么参加"];
const initialMessages = [
  { id: "welcome-1", role: "assistant", content: "你好，我是喵助理。", detail: "今天 10:24", system: true },
  { id: "welcome-2", role: "assistant", content: "购物、物流、退换货，告诉我你遇到的情况，我来帮你梳理。", detail: "已为你准备好服务", system: true },
];

function formatTime() {
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit" }).format(new Date());
}

function parseSseBlock(block, onToken, onError, onCitations = () => {}, onDone = () => {}) {
  const event = block.match(/^event:\s*(.+)$/m)?.[1]?.trim();
  const rawData = block.match(/^data:\s*(.+)$/m)?.[1];
  if (!event || !rawData) return;
  try {
    const data = JSON.parse(rawData);
    if (event === "token") onToken(data.content || "");
    if (event === "error") onError(data.message || "上游模型调用失败");
    if (event === "citations") onCitations(data);
    if (event === "done") onDone(data);
  } catch {
    onError("响应格式暂时无法解析");
  }
}

function NavItem({ icon: Icon, label, active, badge, onClick }) {
  return <button className={`nav-item ${active ? "active" : ""}`} onClick={onClick} type="button"><Icon size={18} strokeWidth={active ? 2.2 : 1.8} /><span>{label}</span>{badge && <em>{badge}</em>}</button>;
}

function MessageBubble({ message, onCitation, onFeedback }) {
  const isUser = message.role === "user";
  return <div className={`message-row ${isUser ? "from-user" : "from-assistant"}`}>
    {!isUser && <div className="mini-bot"><Bot size={16} /></div>}
    <div className="message-stack">
      <div className="message-meta"><span>{isUser ? "你" : "喵助理"}</span><time>{message.detail}</time>{!isUser && <span className="ai-chip">AI 助手</span>}</div>
      <div className="message-bubble">{message.content ? message.content.split(/(\[\d+\])/g).map((part, index) => {
        const match = part.match(/^\[(\d+)\]$/);
        const citation = match && message.citations?.find((item) => item.n === Number(match[1]));
        return citation ? <button type="button" className="citation-number" key={index} onClick={() => onCitation(citation)} aria-label={`查看引用 ${citation.n}`}>{part}</button> : <span key={index}>{part}</span>;
      }) : <span className="typing-dots"><i /><i /><i /></span>}</div>
      {!isUser && !message.system && !message.streaming && !message.error && message.content && <div className="answer-feedback">
        <button type="button" aria-label="满意" aria-pressed={message.feedback === "up"} disabled={Boolean(message.feedback)} className={message.feedback === "up" ? "selected" : ""} onClick={() => onFeedback(message, "up")}><ThumbsUp size={15} /></button>
        <button type="button" aria-label="不满意" aria-pressed={message.feedback === "down"} disabled={Boolean(message.feedback)} className={message.feedback === "down" ? "selected" : ""} onClick={() => onFeedback(message, "down")}><ThumbsDown size={15} /></button>
        {message.feedback && <span role="status">已反馈</span>}
      </div>}
    </div>
    {isUser && <div className="mini-user"><UserRound size={15} /></div>}
  </div>;
}

function App() {
  const [messages, setMessages] = useState(initialMessages);
  const [activeCitation, setActiveCitation] = useState(null);
  const feedbackLocks = useRef(new Set());
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [search, setSearch] = useState("");
  const [afterSaleText, setAfterSaleText] = useState("");
  const [extracting, setExtracting] = useState(false);
  const [extracted, setExtracted] = useState(null);
  const [notice, setNotice] = useState("");
  const textareaRef = useRef(null);
  const conversationId = useRef(`web-${Date.now()}`);

  const recordFeedback = (message, choice) => {
    const key = `${conversationId.current}:${message.id}`;
    if (feedbackLocks.current.has(key) || message.feedback) return;
    feedbackLocks.current.add(key);
    setMessages((current) => current.map((item) => item.id === message.id ? { ...item, feedback: choice } : item));
    const record = { conversation_id: conversationId.current, message_id: message.serverMessageId || message.id, choice, time: new Date().toISOString() };
    try {
      const records = JSON.parse(localStorage.getItem("mewhelp.feedback.v1") || "[]");
      localStorage.setItem("mewhelp.feedback.v1", JSON.stringify([...(Array.isArray(records) ? records : []), record]));
    } catch { /* In-memory selection still locks if browser storage is unavailable. */ }
  };

  const sendMessage = async (value = input) => {
    const content = value.trim();
    if (!content || sending) return;
    const userMessage = { id: crypto.randomUUID(), role: "user", content, detail: formatTime() };
    const assistantId = crypto.randomUUID();
    const assistantMessage = { id: assistantId, role: "assistant", content: "", detail: "正在生成", streaming: true };
    setMessages((current) => [...current, userMessage, assistantMessage]);
    setInput("");
    setSending(true);
    try {
      const response = await fetch("/api/v1/chat/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ conversation_id: conversationId.current, message: content }) });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const updateAssistant = (patch) => setMessages((current) => current.map((message) => {
        if (message.id !== assistantId) return message;
        if (patch.append !== undefined) {
          const { append, ...rest } = patch;
          return { ...message, ...rest, content: `${message.content}${append}` };
        }
        return { ...message, ...patch };
      }));
      while (true) {
        const { value: chunk, done } = await reader.read();
        buffer += decoder.decode(chunk || new Uint8Array(), { stream: !done });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";
        blocks.forEach((block) => parseSseBlock(block, (token) => updateAssistant({ detail: "刚刚", streaming: true, append: token }), (error) => updateAssistant({ content: error, detail: "连接异常", streaming: false, error: true }), (data) => updateAssistant({ citations: data.items || [], serverMessageId: data.message_id }), (data) => updateAssistant({ streaming: false, serverMessageId: data.message_id, refused: data.refused })));
        if (done) break;
      }
      if (buffer.trim()) parseSseBlock(buffer, (token) => updateAssistant({ detail: "刚刚", streaming: true, append: token }), (error) => updateAssistant({ content: error, detail: "连接异常", streaming: false, error: true }), (data) => updateAssistant({ citations: data.items || [], serverMessageId: data.message_id }), (data) => updateAssistant({ streaming: false, serverMessageId: data.message_id, refused: data.refused }));
      setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, streaming: false, detail: "刚刚" } : message));
    } catch (error) {
      setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, content: "暂时连接不上客服服务，请确认后端已启动后再试。", streaming: false, detail: "发送失败", error: true } : message));
      setNotice("连接失败，请确认 FastAPI 服务正在运行");
      console.error(error);
    } finally {
      setSending(false);
      textareaRef.current?.focus();
    }
  };

  const clearChat = () => { conversationId.current = `web-${Date.now()}-${crypto.randomUUID()}`; setMessages(initialMessages); setNotice("已开启一段新的对话"); };

  const extractAfterSale = async () => {
    if (!afterSaleText.trim() || extracting) return;
    setExtracting(true); setExtracted(null);
    try {
      const response = await fetch("/api/v1/after-sale/extract", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: afterSaleText }) });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setExtracted(await response.json());
    } catch (error) { setNotice("识别失败，请稍后重试"); console.error(error); }
    finally { setExtracting(false); }
  };

  const submitSearch = (event) => { event.preventDefault(); if (search.trim()) { setInput(search); setSearch(""); textareaRef.current?.focus(); } };
  const submitChat = (event) => { event.preventDefault(); sendMessage(); };

  return <div className="app-shell">
    {activeCitation && <div className="citation-overlay" onClick={() => setActiveCitation(null)}>
      <section className="citation-panel" role="dialog" aria-modal="true" aria-label="引用来源" onClick={(event) => event.stopPropagation()}>
        <header><strong>引用 [{activeCitation.n}] · 来源原文</strong><button type="button" aria-label="关闭来源" onClick={() => setActiveCitation(null)}><X size={20} /></button></header>
        <div className="citation-path">{activeCitation.section_path?.join(" / ") || "未提供章节"}</div>
        <small>以下为生成本段回答时使用的证据快照</small>
        {activeCitation.question && <h3>{activeCitation.question}</h3>}
        <div className="citation-original">{activeCitation.answer}</div>
        <footer><span>Chunk #{activeCitation.chunk_id}</span>{activeCitation.source_url ? <a href={activeCitation.source_url} target="_blank" rel="noopener noreferrer">跳回原文章节 <ArrowUpRight size={15} /></a> : <span>来源为脱敏问答，无公开原文链接</span>}</footer>
      </section>
    </div>}
    <aside className="app-sidebar">
      <div className="sidebar-logo"><span>喵</span><small>MEOW</small></div>
      <div className="sidebar-group"><span className="sidebar-caption">WORKSPACE</span><NavItem icon={LayoutDashboard} label="概览" /><NavItem icon={MessageCircle} label="智能客服" active badge="在线" /><NavItem icon={PackageCheck} label="售后工单" badge="3" /><NavItem icon={Truck} label="物流追踪" /></div>
      <div className="sidebar-group lower"><span className="sidebar-caption">TOOLS</span><NavItem icon={Tag} label="优惠与活动" /><NavItem icon={CircleHelp} label="帮助中心" /></div>
      <div className="sidebar-bottom"><button className="nav-item" type="button"><Settings2 size={18} /><span>设置</span></button><div className="profile"><div className="profile-avatar">M</div><div><strong>Meow Store</strong><small>旗舰店账号</small></div><MoreHorizontal size={16} /></div></div>
    </aside>

    <main className="main-content">
      <header className="top-header"><div className="crumbs"><span>工作台</span><ChevronRight size={14} /><strong>智能客服</strong></div><form className="global-search" onSubmit={submitSearch}><Search size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索订单、客户或问题" /><kbd>⌘ K</kbd></form><div className="header-actions"><span className="online"><i /> 服务在线</span><button type="button" className="header-icon"><Command size={17} /></button><button type="button" className="header-icon"><CircleHelp size={17} /></button><div className="header-avatar">M</div></div></header>

      <section className="welcome-banner"><div className="banner-copy"><div className="banner-kicker"><Sparkles size={14} /> AI CUSTOMER CARE</div><h1>今天，也要把每个问题<br /><span>温柔地解决好。</span></h1><p>智能理解用户诉求，给出清晰、可靠的下一步。</p><div className="banner-stats"><span><strong>多轮</strong>连续对话</span><span><strong>实时</strong>逐段回复</span><span><strong>知识库</strong>政策查询</span></div></div><div className="banner-art"><div className="orb orb-a" /><div className="orb orb-b" /><div className="art-card"><Bot size={18} /><span>AI Copilot</span><strong>Ready to help</strong><div className="wave"><i /><i /><i /><i /><i /><i /><i /></div></div><div className="art-sparkle">✦</div></div></section>

      <div className="content-grid">
        <section className="conversation-panel panel-card" id="conversation"><div className="panel-head"><div className="panel-title"><div className="title-icon orange"><Headphones size={18} /></div><div><h2>实时对话</h2><p>AI 正在为你提供支持</p></div></div><div className="panel-head-actions"><span className="secure"><Check size={13} /> 安全连接</span><button type="button" onClick={clearChat} className="text-action"><RotateCcw size={14} /> 新对话</button></div></div>
          <div className="conversation-body">{messages.map((message) => <MessageBubble key={message.id} message={message} onCitation={setActiveCitation} onFeedback={recordFeedback} />)}{notice && <div className="notice"><span>{notice}</span><button type="button" onClick={() => setNotice("")}><X size={14} /></button></div>}</div>
          <div className="suggestion-row"><span>快捷提问</span>{suggestions.map((suggestion) => <button type="button" key={suggestion} onClick={() => sendMessage(suggestion)}>{suggestion}</button>)}</div>
          <form className="composer" onSubmit={submitChat}><div className="composer-input"><textarea ref={textareaRef} value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(); } }} placeholder="描述你的问题，喵助理会帮你处理…" rows="1" /><button className="add-attachment" type="button" title="添加附件"><Plus size={18} /></button></div><div className="composer-foot"><span><Clock3 size={13} /> 通常几秒内回复</span><button className="send-button" type="submit" disabled={sending || !input.trim()}>{sending ? "生成中" : "发送"}<Send size={15} /></button></div></form>
        </section>

        <aside className="insight-column"><section className="panel-card ticket-card"><div className="panel-head compact"><div className="panel-title"><div className="title-icon purple"><WandSparkles size={17} /></div><div><h2>售后智能识别</h2><p>把自然语言变成可处理的信息</p></div></div><span className="beta">BETA</span></div><textarea value={afterSaleText} onChange={(event) => setAfterSaleText(event.target.value)} placeholder="例如：订单 TEST-123 的耳机坏了，我想换货…" rows="4" /><button className="primary-action" type="button" onClick={extractAfterSale} disabled={extracting || !afterSaleText.trim()}>{extracting ? "正在识别…" : "开始识别"}<ArrowUpRight size={16} /></button>{extracted && <div className="extracted-result"><div className="result-header"><span>识别结果</span><Check size={14} /></div>{[["订单号", extracted.order_id], ["诉求类型", extracted.request_type], ["期望方案", extracted.expected_solution]].map(([label, value]) => <div className="result-item" key={label}><span>{label}</span><strong>{value || "未提及"}</strong></div>)}</div>}</section>
          <section className="panel-card order-card"><div className="panel-head compact"><div className="panel-title"><div className="title-icon blue"><Box size={17} /></div><div><h2>订单助手</h2><p>快速查看相关服务</p></div></div><MoreHorizontal size={18} className="muted-icon" /></div><div className="order-placeholder"><div className="box-illustration"><Box size={23} /></div><div><strong>还没有关联订单</strong><p>在对话中发送订单号即可关联</p></div></div><button className="secondary-action" type="button" onClick={() => setInput("我的订单还没收到")}>输入订单号 <ArrowUpRight size={14} /></button></section>
          <section className="panel-card help-card"><div className="panel-head compact"><div className="panel-title"><div className="title-icon yellow"><CircleHelp size={17} /></div><div><h2>常见问题</h2><p>也许这里正好有答案</p></div></div><ChevronRight size={17} className="muted-icon" /></div>{["退货需要满足什么条件？", "一般多久可以收到货？", "可以修改收货地址吗？"].map((question) => <button className="help-item" type="button" key={question} onClick={() => setInput(question)}><span>{question}</span><ArrowUpRight size={14} /></button>)}</section></aside>
      </div>
      <footer className="page-footer"><span>喵喵生活旗舰店 · 智能客服工作台</span><span>Powered by MewHelp <span className="footer-dot">●</span> 服务在线</span></footer>
    </main>
  </div>;
}

export default App;
