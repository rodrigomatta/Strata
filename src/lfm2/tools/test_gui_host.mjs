#!/usr/bin/env node
/** Explicit test against a running GUI host (uses its real GPU). */
import assert from 'node:assert/strict';
const base = process.argv[2] || 'http://127.0.0.1:18891';
const native = async () => (await fetch(base + '/api/native')).json();
let instance;
for (let i = 0; i < 100; i++) {
  instance = (await native()).instances.find(x => x.engine === 'lfm2');
  if (instance?.status === 'ready') break;
  if (instance?.status === 'error') throw new Error(instance.error);
  await new Promise(r => setTimeout(r, 100));
}
assert.equal(instance?.status, 'ready'); assert.equal(instance.context, 32768);
console.log('PASS: GUI host has real LFM2 loaded with 32768 context tokens');
const abort = new AbortController();
const response = await fetch(base + '/v1/chat/completions', { method: 'POST', signal: abort.signal,
  headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: instance.apiModelId, stream: true, temperature: 0,
    messages: [{ role: 'user', content: 'Escreva uma história muito longa, com pelo menos 3000 palavras, sobre uma viagem pelo oceano. Desenvolva cada capítulo com muitos detalhes.' }] }),
});
assert.equal(response.status, 200);
const reader = response.body.getReader(), decoder = new TextDecoder();
let buffer = '', count = 0;
while (count < 1100) {
  const { done, value } = await reader.read();
  if (done) throw new Error('Long-generation test ended before 1100 delta frames.');
  buffer += decoder.decode(value, { stream: true });
  let end;
  while ((end = buffer.indexOf('\n')) >= 0) {
    const line = buffer.slice(0, end); buffer = buffer.slice(end + 1);
    if (!line.startsWith('data: {')) continue;
    const event = JSON.parse(line.slice(6));
    if (event.error) throw new Error(event.error.message);
    const d = event.choices?.[0]?.delta;
    if (d?.content || d?.reasoning_content) count++;
  }
}
abort.abort(); await reader.cancel().catch(() => {});
let record;
for (let i = 0; i < 60; i++) {
  const state = await native();
  record = state.instances.find(x => x.id === instance.id)?.metrics?.requests?.find(r => r.cancelled);
  if (record) break;
  await new Promise(r => setTimeout(r, 100));
}
assert.ok(record?.output_tokens > 512, JSON.stringify(record));
console.log('PASS: no max_tokens supplied, exceeded 512, cancelled cleanly', record.output_tokens);
const reply = await fetch(base + '/api/chat/stream', { method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ message: 'Responda apenas com a palavra: café', nativeModelId: instance.id, sessionId: 'lfm2-host-smoke', thinking: 'Off', history: [] }),
});
assert.equal(reply.status, 200);
const frames = (await reply.text()).trim().split('\n').map(line => JSON.parse(line));
assert.equal(frames.some(f => f.type === 'error'), false, JSON.stringify(frames));
assert.match(frames.filter(f => f.type === 'delta').map(f => f.text).join(''), /café/);
assert.ok(frames.some(f => f.contextMax === 32768));
const done = frames.find(f => f.type === 'done');
assert.ok(done.tokensOut > 0); assert.ok(done.tps > 0);
console.log('PASS: GUI chat route, reasoning Off, EOS, measured usage and 32k context', done);
console.log('ALL GUI HOST TESTS PASSED');
