/** Opt-in LFM2 backend. One persistent HIP worker, one cancellable request queue.
 * The private OpenAI listener uses loopback + an unguessable per-load key.
 * Qwen's engine, server and loading code do not pass through this module.
 */
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { randomUUID, timingSafeEqual } from 'node:crypto';
import { existsSync, realpathSync, statSync } from 'node:fs';
import { basename, isAbsolute, join, resolve } from 'node:path';
import { homedir } from 'node:os';
import { readGGUF } from './gguf.mjs';
import { RequestQueue } from './request-queue.mjs';

export const LFM2_ROOT = resolve(import.meta.dirname, '../../..');
export const lfm2Binary = (root = LFM2_ROOT) => process.env.STRATA_LFM2_BIN || join(root, 'build-lfm2', 'strata-lfm2');

export function previewLfm2(input) {
  if (typeof input !== 'string' || !input.trim()) return null;
  const name = input.trim().replace(/^~(?=$|[\\/])/, homedir());
  if (!isAbsolute(name) || !/\.gguf$/i.test(name)) return null;
  const path = realpathSync(name), gguf = readGGUF(path), m = gguf.metadata;
  if (m['general.architecture'] !== 'lfm2moe') return null;
  if ((m['split.count'] ?? 1) !== 1) throw new Error('LFM2 currently needs a single complete GGUF file.');
  if (!m['tokenizer.ggml.tokens']?.count) throw new Error('The GGUF has no tokenizer.');
  const context = m['lfm2moe.context_length'];
  if (!Number.isInteger(context) || context < 256) throw new Error('The LFM2 context metadata is invalid.');
  return { path, preview: { engine: 'lfm2', model_name: m['general.name'] || basename(path),
    context_max: Math.min(context, 32768), file_count: 1, total_bytes: gguf.size,
    file_set_status: 'plausible', issues: [], messages: ['LFM2-MoE · HIP · one AMD GPU · text only. The weights stay loaded between messages.'],
    candidate_selection: { family: 'lfm2', model: path } } };
}

/** Only text roles supported by ChatML. No Jinja evaluation and no silent image/tool loss. */
export function lfm2Request(body, context = 32768, model = 'lfm2') {
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw new Error('Expected a JSON object.');
  if (body.model != null && body.model !== model) throw new Error('Choose the loaded LFM2 model ID.');
  if (!Array.isArray(body.messages) || !body.messages.length || body.messages.length > 1024) throw new Error('Supply 1–1024 text messages.');
  if (body.tools?.length || body.functions?.length || body.response_format?.type && body.response_format.type !== 'text') throw new Error('LFM2 supports plain text chat here; tools and structured output are not enabled.');
  if (body.stop != null && (!Array.isArray(body.stop) || body.stop.length)) throw new Error('Custom stop strings are not supported by LFM2 yet.');
  if (body.n != null && body.n !== 1) throw new Error('LFM2 generates one response at a time.');
  if (body.stream != null && typeof body.stream !== 'boolean') throw new Error('stream must be true or false.');
  const max = body.max_completion_tokens ?? body.max_tokens ?? 0;
  const temp = body.temperature ?? 0, k = body.top_k ?? 0;
  if (!Number.isInteger(max) || max < 0 || max >= context) throw new Error(`max_tokens must be 0 (until EOS/context) or between 1 and ${context - 1}.`);
  if (!Number.isFinite(temp) || temp < 0 || temp > 2) throw new Error('temperature must be between 0 and 2.');
  if (!Number.isInteger(k) || k < 0 || k > 128000) throw new Error('top_k must be between 0 and 128000.');
  let prompt = '';
  for (const message of body.messages) {
    if (!message || !['system', 'user', 'assistant'].includes(message.role) || typeof message.content !== 'string' || message.tool_calls?.length) throw new Error('LFM2 accepts text system, user and assistant messages only.');
    // Treat literal template controls in a user message as text, not new roles.
    const content = message.content.replaceAll('<|im_start|>', '＜|im_start|＞').replaceAll('<|im_end|>', '＜|im_end|＞').replaceAll('<|startoftext|>', '＜|startoftext|＞').replaceAll('<|endoftext|>', '＜|endoftext|＞');
    prompt += `<|im_start|>${message.role}\n${content}<|im_end|>\n`;
  }
  prompt += '<|im_start|>assistant\n';
  // The non-thinking mode uses the model's explicit empty-thought prefix.
  if (['none', 'off'].includes(body.reasoning_effort)) prompt += '<think>\n</think>\n\n';
  const bytes = Buffer.from(prompt);
  if (bytes.length > 4 * 1024 * 1024) throw new Error('The prompt is too large.');
  return { prompt: bytes, max, temp, k, stream: body.stream === true };
}

/** Recognize thinking tags even when split across token/UTF-8 boundaries. */
export class Lfm2Reasoning {
  pending = ''; thinking = false;
  push(text, emit, finish = false) {
    this.pending += text;
    while (this.pending) {
      const tag = this.thinking ? '</think>' : '<think>';
      const at = this.pending.indexOf(tag);
      if (at >= 0) {
        if (at) emit(this.thinking ? 'reasoning_content' : 'content', this.pending.slice(0, at));
        this.pending = this.pending.slice(at + tag.length); this.thinking = !this.thinking; continue;
      }
      let hold = 0;
      if (!finish) for (let n = 1; n < tag.length && n <= this.pending.length; n++) if (tag.startsWith(this.pending.slice(-n))) hold = n;
      const ready = this.pending.slice(0, this.pending.length - hold);
      if (ready) emit(this.thinking ? 'reasoning_content' : 'content', ready);
      this.pending = hold ? this.pending.slice(-hold) : ''; break;
    }
  }
}

export async function startLfm2({ path, context = 32768, key, root = LFM2_ROOT, binary = lfm2Binary(root),
  spawnProcess = spawn, log = () => {}, onExit = () => {}, signal, port = 0, timeout = 180000 }) {
  if (!Number.isInteger(context) || context < 256 || context > 32768) throw new Error('LFM2 context must be between 256 and 32768.');
  if (!key || key.length < 32) throw new Error('The private LFM2 listener needs a random key.');
  if (!existsSync(binary) || !statSync(binary).isFile()) throw new Error('Compile LFM2 first: cmake -S src/lfm2 -B build-lfm2 -DCMAKE_HIP_ARCHITECTURES=gfx1100 && cmake --build build-lfm2 -j2');
  signal?.throwIfAborted();
  const child = spawnProcess(binary, [path, '--worker'], { cwd: root, stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env, LFM2_CTX: String(context) } });
  const model = `lfm2-${basename(path).replace(/\.gguf$/i, '')}`;
  const queue = new RequestQueue();
  let active = null, closing = false, failed = null, buffer = '', tail = '', server;
  let status = 'loading', latest = null;
  const recent = [];
  let readyResolve, readyReject;
  const ready = new Promise((yes, no) => { readyResolve = yes; readyReject = no; });
  const fail = error => {
    if (failed) return;
    failed = error; readyReject(error); active?.reject(error); onExit(error);
  };
  child.stderr.on('data', chunk => { tail = (tail + chunk.toString()).slice(-12000); log(chunk.toString().trim()); });
  child.on('error', error => fail(error));
  child.stdin.on('error', error => fail(error));
  child.on('exit', (code, sig) => { if (!closing) fail(new Error(`LFM2 stopped (${sig ?? code}). ${tail.slice(-1500)}`)); });
  child.stdout.on('data', chunk => {
    buffer += chunk.toString('utf8');
    if (buffer.length > 2 * 1024 * 1024) { fail(new Error('Invalid LFM2 worker frame.')); child.kill(); return; }
    for (;;) {
      const end = buffer.indexOf('\n'); if (end < 0) break;
      const line = buffer.slice(0, end); buffer = buffer.slice(end + 1);
      try {
        const event = JSON.parse(line);
        if (event.type === 'ready' && event.protocol === 1) { status = 'idle'; readyResolve(); }
        else if (event.type === 'error') active?.reject(new Error(Buffer.from(event.hex, 'hex').toString('utf8')));
        else active?.event(event);
      } catch (error) { fail(new Error(`LFM2 worker protocol failed: ${error.message}`)); child.kill(); }
    }
  });
  const stop = async () => {
    if (closing) return; closing = true;
    const reason = new Error('LFM2 was unloaded.');
    readyReject(reason); active?.reject(reason);
    server?.closeAllConnections(); server?.close();
    await new Promise(resolveStop => {
      if (child.exitCode !== null || child.signalCode !== null) { resolveStop(); return; }
      const force = setTimeout(() => child.kill('SIGKILL'), 3000);
      child.once('exit', () => { clearTimeout(force); resolveStop(); }); child.kill('SIGTERM');
    });
  };
  const abortLoad = () => { readyReject(signal.reason); void stop(); };
  signal?.addEventListener('abort', abortLoad, { once: true });
  const timer = setTimeout(() => readyReject(new Error(`LFM2 load timed out. ${tail.slice(-1000)}`)), timeout);
  try { await ready; }
  catch (error) { await stop(); throw error; }
  finally { clearTimeout(timer); signal?.removeEventListener('abort', abortLoad); }

  function generate(request, requestSignal, event) {
    return queue.run(requestSignal, () => new Promise((resolveRequest, rejectRequest) => {
      if (failed || closing) { rejectRequest(failed ?? new Error('LFM2 is not loaded.')); return; }
      let started = false, cancelled = false, settled = false;
      const began = Date.now();
      const cancel = () => { cancelled = true; if (started && !settled) child.kill('SIGUSR1'); };
      const settle = (error, result) => {
        if (settled) return; settled = true; active = null; status = 'idle';
        requestSignal.removeEventListener('abort', cancel);
        if (error) rejectRequest(error); else resolveRequest(result);
      };
      active = { reject: error => settle(error), event: value => {
        if (value.type === 'start') { started = true; status = 'reading'; if (cancelled) child.kill('SIGUSR1'); }
        if (value.type === 'token') status = 'generating';
        if (value.type === 'done') {
          latest = value;
          recent.unshift({ id: randomUUID(), time: Date.now() / 1000, duration_s: (Date.now() - began) / 1000,
            prompt_tokens: value.prompt_tokens, reused: 0, output_tokens: value.completion_tokens,
            prompt_ms: value.prompt_ms, decode_ms: value.decode_ms, decode_tok_s: value.predicted_per_second,
            finish_reason: value.finish_reason, cancelled: value.cancelled });
          recent.splice(16); settle(null, value);
        }
        else event(value);
      } };
      requestSignal.addEventListener('abort', cancel, { once: true });
      if (requestSignal.aborted) cancel();
      status = 'reading';
      child.stdin.write(`LFM2/1 ${request.max} ${request.k} ${request.temp} ${request.prompt.length}\n`);
      child.stdin.write(request.prompt);
    }));
  }
  const json = (res, code, body) => { if (!res.destroyed) { res.writeHead(code, { 'content-type': 'application/json' }); res.end(JSON.stringify(body)); } };
  const auth = req => {
    const got = Buffer.from(String(req.headers.authorization ?? '')), want = Buffer.from(`Bearer ${key}`);
    return got.length === want.length && timingSafeEqual(got, want);
  };
  server = createServer(async (req, res) => {
    const controller = new AbortController();
    res.on('close', () => { if (!res.writableEnded) controller.abort(new Error('Request cancelled.')); });
    try {
      if (!auth(req)) { json(res, 401, { error: { message: 'Invalid private engine key.' } }); return; }
      const url = new URL(req.url, 'http://127.0.0.1');
      if (req.method === 'GET') {
        if (['/health', '/v1/status'].includes(url.pathname)) return json(res, failed ? 503 : 200, { service: 'strata', engine: 'lfm2', loaded: !failed, model, context: { native: context } });
        if (url.pathname === '/v1/models') return json(res, 200, { object: 'list', data: [{ id: model, object: 'model', owned_by: 'strata', context_length: context, vision: false }] });
        if (url.pathname === '/mcp') return json(res, 200, { servers: [], supported: false });
        if (url.pathname === '/metrics') return json(res, 200, { time: Date.now() / 1000, engine: { name: 'lfm2', model, max_context: context }, live: { state: status, running: active ? 1 : 0, queued: queue.waiting, tok_s: latest?.predicted_per_second ?? 0, prefill_tok_s_mean: latest?.prompt_per_second ?? 0 }, requests: recent });
      }
      if (req.method !== 'POST' || url.pathname !== '/v1/chat/completions') return json(res, 404, { error: { message: 'Unknown LFM2 endpoint.' } });
      let bytes = 0, chunks = [];
      for await (const chunk of req) {
        bytes += chunk.length; if (bytes > 4 * 1024 * 1024) { json(res, 413, { error: { message: 'Request body is too large.' } }); return; }
        chunks.push(chunk);
      }
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      const request = lfm2Request(body, context, model);
      const id = `chatcmpl-${randomUUID()}`, created = Math.floor(Date.now() / 1000), split = new Lfm2Reasoning(), decoder = new TextDecoder();
      let content = '', reasoning = '';
      const startStream = () => {
        if (!res.headersSent && !res.destroyed) res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache, no-transform', connection: 'keep-alive' });
      };
      const chunk = (delta, finish = null, more = {}) => {
        if (!res.destroyed) { startStream(); res.write(`data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta, finish_reason: finish }], ...more })}\n\n`); }
      };
      const emit = (field, text) => {
        if (field === 'content') content += text; else reasoning += text;
        if (request.stream) chunk({ [field]: text });
      };
      const done = await generate(request, controller.signal, event => {
        if (event.type === 'start' && request.stream) chunk({ role: 'assistant' });
        if (event.type === 'token') split.push(decoder.decode(Buffer.from(event.hex, 'hex'), { stream: true }), emit);
      });
      split.push(decoder.decode(), emit, true);
      if (controller.signal.aborted) return;
      const usage = { prompt_tokens: done.prompt_tokens, completion_tokens: done.completion_tokens, total_tokens: done.prompt_tokens + done.completion_tokens };
      const timings = { prompt_per_second: done.prompt_per_second, predicted_per_second: done.predicted_per_second };
      if (request.stream) { chunk({}, done.finish_reason, { usage, timings }); res.end('data: [DONE]\n\n'); }
      else json(res, 200, { id, object: 'chat.completion', created, model, choices: [{ index: 0, message: { role: 'assistant', content, ...(reasoning ? { reasoning_content: reasoning } : {}) }, finish_reason: done.finish_reason }], usage, timings });
    } catch (error) {
      if (res.destroyed) return;
      if (res.headersSent) { res.end(`data: ${JSON.stringify({ error: { message: error.message } })}\n\ndata: [DONE]\n\n`); }
      else json(res, failed ? 503 : 400, { error: { message: error.message } });
    }
  });
  server.requestTimeout = 120000; server.headersTimeout = 30000;
  try { await new Promise((yes, no) => { server.once('error', no); server.listen(port, '127.0.0.1', yes); }); }
  catch (error) { await stop(); throw error; }
  return { stop, port: server.address().port, model, process: child };
}
