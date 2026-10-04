import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { test, before, after } from 'node:test';
const require = createRequire(new URL('../../.runtime/browser/package.json', import.meta.url));
const { chromium } = require('playwright');
const html = await readFile(new URL('../../app/static/index.html', import.meta.url),'utf8');
let browser, server, base;
const preview = {conversation_id:'c-ticket',request_id:'preview-1',tool_call_id:'call-1',ticket_type:'售后',description:'耳机无法充电 <script>window.injected=true</script>',message_id:501};
const sse = frames => frames.map(([event,data])=>`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join('');
before(async()=>{
  browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
  server=createServer((req,res)=>{res.writeHead(200,{'content-type':'text/html; charset=utf-8'});res.end(html);});
  await new Promise(r=>server.listen(0,'127.0.0.1',r));base=`http://127.0.0.1:${server.address().port}`;
});
after(async()=>{await browser?.close();await new Promise(r=>server?.close(r));});
async function open(t,{pending=false,failure=false}={}) {
  const context=await browser.newContext();t.after(()=>context.close());const page=await context.newPage();page.setDefaultTimeout(5000);
  await page.addInitScript(()=>localStorage.setItem('mewhelp-conversation-id','c-ticket'));
  await page.route('**/api/conversations?*',r=>r.fulfill({json:{items:pending?[{conversation_id:'c-ticket',preview:'请确认工单',has_summary:false}]:[]}}));
  await page.route('**/api/conversations/*/messages?*',r=>r.fulfill({json:{items:pending?[{id:501,role:'assistant',content:'请确认工单信息',actions:{ticket_preview:preview}}]:[]}}));
  await page.route('**/api/v1/chat/pending-ticket?*',r=>r.fulfill({json:pending?preview:null}));
  await page.route('**/api/v1/chat/stream',r=>r.fulfill({contentType:'text/event-stream',body:sse([['ticket_preview',preview]])}));
  const requests=[];
  await page.route('**/api/v1/chat/ticket-confirmation',async r=>{
    requests.push(r.request().postDataJSON());
    if(failure) return r.abort();
    const approve=requests.at(-1).approve;
    await r.fulfill({contentType:'text/event-stream',body:sse([['token',{content:approve?'工单已创建，工单号：TK-DEMO-1':'已取消建单'}],['done',{message_id:502}]])});
  });
  await page.goto(base);
  if(!pending){await page.locator('#message-input').fill('帮我建工单，耳机无法充电');await page.locator('#chat-form button[type=submit]').click();}
  await page.getByRole('button',{name:'确认提交',exact:true}).waitFor();
  return {page,requests};
}
for(const approve of [true,false]) test(approve?'preview confirmation sends exact identity and displays ticket number':'preview cancellation sends false and locks both buttons',async t=>{
  const {page,requests}=await open(t);
  assert.match(await page.locator('.ticket-preview p').textContent(),/<script>/);
  assert.equal(await page.evaluate(()=>window.injected),undefined);
  assert.equal(await page.getByText('连接中断',{exact:false}).count(),0);
  await page.getByRole('button',{name:approve?'确认提交':'取消',exact:true}).click();
  await page.getByText(approve?'工单已创建，工单号：TK-DEMO-1':'已取消建单',{exact:true}).waitFor();
  assert.deepEqual(requests,[{conversation_id:'c-ticket',user_id:'demo-user',request_id:'preview-1',tool_call_id:'call-1',approve}]);
  assert.equal(await page.getByRole('button',{name:'确认提交',exact:true}).isDisabled(),true);
  assert.equal(await page.getByRole('button',{name:'取消',exact:true}).isDisabled(),true);
});
test('pending preview is restored from persisted history',async t=>{
  const {page,requests}=await open(t,{pending:true});
  assert.equal(await page.locator('.ticket-preview').count(),1);
  await page.getByRole('button',{name:'取消',exact:true}).click();
  await page.getByText(/已取消建单/).waitFor();assert.equal(requests.length,1);
});
test('unknown confirmation outcome does not resubmit',async t=>{
  const {page,requests}=await open(t,{failure:true});
  await page.getByRole('button',{name:'确认提交',exact:true}).click();
  await page.getByText(/请求结果暂时无法确认/).waitFor();
  assert.equal(requests.length,1);assert.equal(await page.getByRole('button',{name:'确认提交',exact:true}).isDisabled(),true);
});
