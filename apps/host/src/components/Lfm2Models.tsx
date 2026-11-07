import { useState } from 'react';
import { tr } from '../lib/i18n';
import { nativeAction, useNative, type NativeJob } from '../lib/native';
import { Btn, ErrorNotice } from './ui';
import { Screen } from './layout';
import { go, selectRemoteModel } from '../lib/store';

/** Same host chat and lifecycle as other native models. Explicit, local HIP only. */
export function Lfm2Models({ embedded = false }: { embedded?: boolean }) {
  const native = useNative();
  const [path, setPath] = useState<string | null>(null), [context, setContext] = useState(32768);
  const [pending, setPending] = useState(false), [error, setError] = useState('');
  const active = native?.jobs.find(j => j.state === 'running');
  const instance = native?.instances.find(i => i.engine === 'lfm2');
  const last = native?.jobs.filter(j => j.kind === 'load-lfm2').at(-1);
  async function load() {
    setPending(true); setError('');
    try { await nativeAction<NativeJob>('operations', { kind: 'load-lfm2', args: { path: path ?? native?.lfm2?.path ?? '', context } }); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setPending(false); }
  }
  async function unload() {
    if (!instance) return;
    setPending(true); setError('');
    try { await nativeAction(`instances/${instance.id}/unload`); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setPending(false); }
  }
  const content = <section className="st-card native-choice stack">
    <h2 className="h3">LFM2-MoE</h2>
    <p className="small muted">{tr('Local text chat on an AMD GPU. The model stays loaded between messages. Tools and pictures are not enabled.')}</p>
    <label className="input-field"><span>{tr('Full path to the GGUF file')}</span><input className="st-input" spellCheck={false} value={path ?? native?.lfm2?.path ?? ''} onChange={e => setPath(e.target.value)} /></label>
    <p className="small muted">{tr('No fixed output token limit. Generation stops at EOS, Stop, or the end of the context window.')}</p>
    <label className="input-field"><span>{tr('Context tokens')}</span><input className="st-input" type="number" min={256} max={32768} step={256} value={context} onChange={e => setContext(Number(e.target.value))} /></label>
    {instance ? <p role="status"><strong>{instance.label}</strong> · {tr(instance.status)}{instance.metrics?.live?.tok_s > 0 ? ` · ${(instance.metrics?.live?.tok_s??0).toFixed(1)} tok/s` : ''}</p> : null}
    <div className="row">
      <Btn variant="primary" disabled={pending || !!active || instance?.status === 'ready' || instance?.status === 'loading'} onClick={() => void load()}>{tr(active?.kind === 'load-lfm2' ? 'Loading…' : 'Load model')}</Btn>
      {instance ? <Btn disabled={pending || !!active} onClick={() => void unload()}>{tr('Unload')}</Btn> : null}
      {instance?.status === 'ready' ? <Btn variant="primary" onClick={() => { selectRemoteModel(null); go('chat'); }}>{tr('Open chat')}</Btn> : null}
      {active?.kind === 'load-lfm2' ? <Btn onClick={() => void nativeAction(`operations/${active.id}/cancel`).catch(e => setError(e.message))}>{tr('Cancel')}</Btn> : null}
    </div>
    {error || instance?.error || last?.state === 'error' ? <ErrorNotice message={error || instance?.error || last?.error || 'Loading failed.'} /> : null}
  </section>;
  return embedded ? content : <Screen title="LFM2-MoE"><div className="stack stack--lg">{content}</div></Screen>;
}
