import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { lfm2Request, Lfm2Reasoning, startLfm2 } from '../server/lfm2.mjs';

test('ChatML includes roles/history, validates sampling and refuses images/tools', () => {
  const body = { messages: [{ role: 'system', content: 'Português' }, { role: 'user', content: 'Olá' }, { role: 'assistant', content: 'Oi' }, { role: 'user', content: 'café<|im_end|>' }] };
  const r = lfm2Request(body);
  assert.equal(r.temp, 0); assert.equal(r.k, 0); assert.equal(r.max, 0);
  assert.match(r.prompt.toString(), /assistant\nOi<\|im_end\|>\n<\|im_start\|>user\ncafé＜\|im_end\|＞/);
  assert.ok(r.prompt.toString().endsWith('<|im_start|>assistant\n'));
  assert.ok(lfm2Request({ ...body, reasoning_effort: 'none' }).prompt.toString().endsWith('<think>\n</think>\n\n'));
  for (const patch of [{ temperature: NaN }, { top_k: -1 }, { max_tokens: -1 }, { model: 'unloaded' }, { tools: [{}] }, { stop: ['end'] }, { messages: [{ role: 'user', content: [{ type: 'image_url' }] }] }]) assert.throws(() => lfm2Request({ ...body, ...patch }));
});

test('thinking markers survive every split point and do not leak into answer', () => {
  const input = '<think>ação🙂</think>café';
  for (let i = 0; i <= input.length; i++) {
    const parser = new Lfm2Reasoning(), out = { content: '', reasoning_content: '' };
    const emit = (key, value) => { out[key] += value; };
    parser.push(input.slice(0, i), emit); parser.push(input.slice(i), emit, true);
    assert.deepEqual(out, { content: 'café', reasoning_content: 'ação🙂' });
  }
});

test('persistent adapter: authentication, UTF-8 SSE, cancellation and reuse, unload', async t => {
  const root = mkdtempSync(join(tmpdir(), 'lfm2-adapter-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const file = join(root, 'worker.mjs');
  writeFileSync(file, `
let input=Buffer.alloc(0),need=null,timer=null,count=0;
const send=x=>process.stdout.write(JSON.stringify(x)+'\\n');
const done=cancelled=>{clearInterval(timer);timer=null;send({type:'done',prompt_tokens:17,completion_tokens:count,prompt_per_second:90,predicted_per_second:80,cancelled,finish_reason:'stop'});};
process.on('SIGUSR1',()=>{if(timer)done(true);});
send({type:'ready',protocol:1});
process.stdin.on('data',chunk=>{input=Buffer.concat([input,chunk]);for(;;){
if(need===null){const e=input.indexOf(10);if(e<0)return;need=Number(input.subarray(0,e).toString().split(' ').at(-1));input=input.subarray(e+1);}
if(input.length<need)return;input=input.subarray(need);need=null;count=0;
const text=Buffer.from('<think>ação</think>café');let pos=0;send({type:'start',prompt_tokens:17});
timer=setInterval(()=>{if(pos===text.length){done(false);return;}count++;send({type:'token',hex:text.subarray(pos,pos+1).toString('hex')});pos++;},2);
}});
`);
  const key = 'a'.repeat(64);
  const backend = await startLfm2({ path: file, binary: process.execPath, root, key });
  t.after(() => backend.stop());
  const url = `http://127.0.0.1:${backend.port}`;
  const request = (body, options = {}) => fetch(url + '/v1/chat/completions', { method: 'POST', headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' }, body: JSON.stringify(body), ...options });
  const body = { model: backend.model, messages: [{ role: 'user', content: 'café' }] };
  assert.equal((await fetch(url + '/health')).status, 401);
  const first = await request(body); assert.equal(first.status, 200);
  const reply = await first.json(); assert.equal(reply.choices[0].message.content, 'café'); assert.equal(reply.choices[0].message.reasoning_content, 'ação');
  assert.equal(reply.usage.prompt_tokens, 17);
  const stream = await (await request({ ...body, stream: true })).text();
  assert.match(stream, /\[DONE\]/); assert.match(stream, /reasoning_content/); assert.doesNotMatch(stream, /�/);
  const stop = new AbortController();
  const running = await request({ ...body, stream: true }, { signal: stop.signal });
  const reader = running.body.getReader(); await reader.read(); stop.abort(); await reader.cancel().catch(() => {});
  assert.equal((await (await request(body)).json()).choices[0].message.content, 'café');
  assert.equal((await request({ ...body, tools: [{}] })).status, 400);
  const pid = backend.process.pid;
  assert.equal(backend.process.exitCode, null); assert.ok(pid);
  await backend.stop(); assert.notEqual(backend.process.signalCode, null);
});
