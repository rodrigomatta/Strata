#!/usr/bin/env node
/** Build and launch the real GUI with a persistent local LFM2 worker. */
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { homedir } from 'node:os';

const host = resolve(import.meta.dirname, '..'), root = resolve(host, '../..');
const args = process.argv.slice(2);
if (args.includes('--help')) {
  console.log('npm run dev:lfm2 -- --model /absolute/model.gguf [--context 32768]'); process.exit(0);
}
let model = process.env.STRATA_LFM2_MODEL, context = process.env.LFM2_CTX ?? '32768';
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--model' && args[i + 1]) model = args[++i];
  else if (args[i] === '--context' && args[i + 1]) context = args[++i];
  else { console.error(`Unknown or incomplete option: ${args[i]}`); process.exit(1); }
}
model = model?.replace(/^~(?=$|[\\/])/, homedir());
if (!model || !existsSync(model)) { console.error('Supply --model /absolute/path/to/LFM2.gguf'); process.exit(1); }
if (!/^\d+$/.test(context) || Number(context) < 256 || Number(context) > 32768) { console.error('Context must be between 256 and 32768.'); process.exit(1); }
const run = (command, argv, options = {}) => new Promise((yes, no) => {
  const child = spawn(command, argv, { cwd: root, stdio: 'inherit', ...options });
  child.on('error', no); child.on('exit', (code, sig) => code === 0 ? yes() : no(new Error(`${command} stopped (${sig ?? code}).`)));
});
try {
  await run('cmake', ['-S', 'src/lfm2', '-B', 'build-lfm2', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_HIP_ARCHITECTURES=gfx1100']);
  await run('cmake', ['--build', 'build-lfm2', '-j2']);
  console.log('LFM2 GUI: http://localhost:5199 — the GGUF loads on the GPU at startup.');
  await run(process.execPath, [join(host, 'scripts/dev.mjs')], { cwd: host, env: { ...process.env,
    STRATA_LFM2_MODEL: resolve(model), LFM2_CTX: context, STRATA_LFM2_ONLY: '1',
    STRATA_LFM2_BIN: join(root, 'build-lfm2/strata-lfm2'), STRATA_NATIVE_REPO: root, STRATA_REAL_ENGINE: '1' } });
} catch (error) { console.error(error.message); process.exitCode = 1; }
