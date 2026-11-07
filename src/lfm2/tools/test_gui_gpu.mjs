#!/usr/bin/env node
/** Explicit hardware test: loads the real HIP GGUF once, then tests its private API. */
import assert from 'node:assert/strict';
import { randomBytes } from 'node:crypto';
import { startLfm2 } from '../../../apps/host/server/lfm2.mjs';

const path = process.argv[2];
if (!path) throw new Error('Usage: node src/lfm2/tools/test_gui_gpu.mjs /absolute/model.gguf');
const key = randomBytes(32).toString('hex');
const backend = await startLfm2({ path, key, context: 32768, log: console.log });
const url = `http://127.0.0.1:${backend.port}`;
const ask = (patch = {}, options = {}) => fetch(url + '/v1/chat/completions', {
  method: 'POST', headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
  body: JSON.stringify({ model: backend.model, temperature: 0,
    messages: [{ role: 'user', content: 'Responda apenas com a palavra: café' }], ...patch }), ...options,
});
const json = async response => { const body = await response.json(); assert.equal(response.status, 200, JSON.stringify(body)); return body; };
try {
  assert.equal((await fetch(url + '/health')).status, 401);
  const first = await json(await ask());
  assert.equal(first.choices[0].message.content.trim(), 'café');
  assert.equal(first.choices[0].finish_reason, 'stop');
  assert.ok(first.choices[0].message.reasoning_content.length);
  const second = await json(await ask());
  assert.deepEqual(second.choices, first.choices);
  assert.deepEqual(second.usage, first.usage);
  console.log('PASS: two greedy requests, identical output, model remains loaded', first.timings);
  const history = await json(await ask({ reasoning_effort: 'none', max_tokens: 128, messages: [
    { role: 'user', content: 'A palavra secreta é azulejo.' },
    { role: 'assistant', content: 'Entendido, vou lembrar a palavra.' },
    { role: 'user', content: 'Qual é a palavra secreta? Responda apenas a palavra.' },
  ] }));
  assert.match(history.choices[0].message.content, /azulejo/i);
  console.log('PASS: conversation history', history.choices[0].message.content);
  const stream = await (await ask({ stream: true })).text();
  const frames = stream.split('\n').filter(x => x.startsWith('data: {')).map(x => JSON.parse(x.slice(6)));
  assert.ok(stream.includes('[DONE]'));
  assert.equal(frames.map(x => x.choices[0]?.delta?.content ?? '').join('').trim(), 'café');
  assert.ok(frames.some(x => x.choices[0]?.delta?.reasoning_content));
  assert.equal(frames.at(-1).usage.completion_tokens, first.usage.completion_tokens);
  console.log('PASS: SSE, reasoning separated, UTF-8, usage and EOS');
  const sampled = await json(await ask({ temperature: 0.7, top_k: 40 }));
  assert.ok(sampled.usage.completion_tokens > 0);
  console.log('PASS: sampling', sampled.timings);
  const tooLong = await ask({ messages: [{ role: 'user', content: 'café '.repeat(40000) }] });
  assert.equal(tooLong.status, 400); await tooLong.text();
  console.log('PASS: context overflow rejected');
  const abort = new AbortController();
  const running = await ask({ stream: true, max_tokens: 3000, messages: [{ role: 'user', content: 'Conte uma história muito longa sobre o oceano.' }] }, { signal: abort.signal });
  const reader = running.body.getReader(), decoder = new TextDecoder();
  for (;;) {
    const part = await reader.read(); if (part.done) throw new Error('Expected active stream to cancel.');
    if (decoder.decode(part.value, { stream: true }).includes('reasoning_content')) break;
  }
  abort.abort(); await reader.cancel().catch(() => {});
  const after = await json(await ask());
  assert.deepEqual(after.choices, first.choices);
  console.log('PASS: cancellation, then clean next request');
  console.log('ALL GPU API TESTS PASSED', { model: backend.model, pid: backend.process.pid });
} finally { await backend.stop(); }
