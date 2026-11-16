// Free-model aggregator for make.fun free-model-aggregator-02.
//
// One server, two wires: it answers OpenAI Chat Completions and Anthropic
// Messages requests (JSON and SSE) with REAL completions from live free
// OpenRouter models, dialled through the grader's gateway on Chat Completions.
//
// Routing: rating-greedy ladder, requirements-filtered (JSON prompts only to
// json-capable models), with per-trial failure memory. Every prompt is
// buffer-then-verify-then-relay: the provider's answer must carry the
// prompt's token before anything is relayed to the client; a token-less
// answer is discarded and the next model in the ladder is tried.
// A 429/5xx/403 refusal fails over to the next model and is reported on the
// x-makefun-upstream-failures header when the prompt is then served.
//
// Determinism: no wall clock and no unseeded randomness anywhere in what is
// sent. Every choice derives from the brief, the request bodies and the
// gateway's answers, so a replay makes exactly the live pass's requests.
import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import {
  readBrief,
  freeModels,
  capabilitiesOf,
  meetsRequirements,
  buildCatalog,
  isRecord,
} from './lib/providers.ts';
import type {
  Json,
  ModelRow,
  Requirement,
  Brief,
  CatalogReport,
} from './lib/providers.ts';
import {
  openAIModelList,
  anthropicModelList,
  anthropicError,
  openAIError,
  anthropicToOpenAI,
  openAIToAnthropic,
  openAIJsonWithModel,
  anthropicEventsFromOpenAI,
} from './lib/wire.ts';

// --- Tunables -----------------------------------------------------------------

const TOKEN_PATTERN = /[A-Z]+-[0-9]{4}/;
const MAX_ATTEMPTS_PER_PROMPT = 4;
const UPSTREAM_TIMEOUT_MS = 25000;
const WORK_POLL_MS = 100;
const REGISTER_RETRIES = 5;
const REGISTER_RETRY_MS = 200;
const DEFAULT_MAX_TOKENS = 512;
const MAX_TOKENS_CAP = 8192;
const FIXED_CREATED = 1767225600;

interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  code: string;
  text: string;
  json: Json | null;
}

interface ParsedStream {
  text: string;
  finishReason: string | null;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null;
  firstId: string;
  firstCreated: number;
  chunkCount: number;
}

// --- Small helpers ------------------------------------------------------------

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

function roughTokens(chars: number): number {
  return Math.max(1, Math.ceil(chars / 4));
}

function asRecord(v: unknown): Json | null {
  return isRecord(v) ? v : null;
}

function asString(v: unknown): string | null {
  return typeof v === 'string' ? v : null;
}

// Collect every text string a request body carries (system prompt, message
// contents, tool descriptions) in document order, for token extraction.
function bodyTexts(body: Json): string[] {
  const out: string[] = [];
  const walk = (v: unknown): void => {
    if (typeof v === 'string') {
      out.push(v);
      return;
    }
    if (Array.isArray(v)) {
      for (const item of v) walk(item);
      return;
    }
    if (isRecord(v)) {
      // Visit content-bearing keys first so the instruction text precedes
      // tool schemas; any order works, the token is unique in the body.
      for (const key of ['system', 'content', 'text', 'messages', 'input']) {
        if (key in v) walk(v[key]);
      }
      for (const key of Object.keys(v)) {
        if (key === 'system' || key === 'content' || key === 'text' || key === 'messages' || key === 'input') continue;
        walk(v[key]);
      }
    }
  };
  walk(body);
  return out;
}

function extractToken(body: Json): string | null {
  for (const text of bodyTexts(body)) {
    const m = TOKEN_PATTERN.exec(text);
    if (m) return m[0];
  }
  return null;
}

function requestChars(body: Json): number {
  let n = 0;
  for (const text of bodyTexts(body)) n += text.length;
  return n;
}

// --- Brief, catalog, ladder ---------------------------------------------------

function loadBrief(): Brief | null {
  const path = process.env.MAKEFUN_PROMPT_FILE;
  if (!path) return null;
  const brief = readBrief(path);
  if (brief.error) return null;
  return brief;
}

function emptyCatalog(): CatalogReport {
  return { providers: [] };
}

function catalogOf(brief: Brief | null): CatalogReport {
  if (!brief) return emptyCatalog();
  return buildCatalog(brief);
}

// Rating-ordered ladder for one prompt's requirements. Models that refused
// earlier in this trial are demoted to the end (still in rating order among
// themselves) so they stay available as fallbacks but are never first choice.
function ladderFor(
  models: ModelRow[],
  requires: Requirement,
  failed: ReadonlySet<string>,
): ModelRow[] {
  const eligible = models.filter((m) => {
    if (!m.free) return false;
    return meetsRequirements(requires, capabilitiesOf(m));
  });
  eligible.sort((a, b) => b.rating_milli - a.rating_milli);
  const fresh = eligible.filter((m) => !failed.has(m.id));
  const demoted = eligible.filter((m) => failed.has(m.id));
  return [...fresh, ...demoted];
}

function requirementsOf(wire: 'openai' | 'anthropic', body: Json): Requirement {
  const req: Requirement = {};
  const fmt = asRecord(body.response_format);
  if (fmt && fmt.type === 'json_object') req.json = true;
  if (Array.isArray(body.tools) && body.tools.length > 0) req.tools = true;
  void wire;
  return req;
}

function isStreamBody(body: Json): boolean {
  return body.stream === true;
}

// --- Upstream -----------------------------------------------------------------

function gatewayBase(): string {
  return process.env.OPENAI_BASE_URL || '';
}

function gatewayKey(): string {
  return process.env.OPENAI_API_KEY || '';
}

// One Chat Completions call to the gateway. Returns the raw text plus the
// parsed JSON when the body is JSON. Never throws on HTTP errors; a network
// failure or timeout comes back as ok:false with status 0.
async function callUpstream(
  upstreamBody: Json,
  stream: boolean,
): Promise<UpstreamResult> {
  const base = gatewayBase().replace(/\/$/, '');
  const url = `${base}/chat/completions`;
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${gatewayKey()}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, code: 'network_error', text: '', json: null };
  }
  const text = await res.text();
  let parsed: Json | null = null;
  if (!stream && text) {
    try {
      const v: unknown = JSON.parse(text);
      parsed = asRecord(v);
    } catch {
      parsed = null;
    }
  }
  if (res.ok) return { ok: true, status: res.status, code: '', text, json: parsed };
  let code = `http_${res.status}`;
  if (parsed) {
    const err = asRecord(parsed.error);
    const c = err ? asString(err.code) : null;
    if (c) code = c;
  } else if (text) {
    try {
      const v: unknown = JSON.parse(text);
      const rec = asRecord(v);
      const err = rec ? asRecord(rec.error) : null;
      const c = err ? asString(err.code) : null;
      if (c) code = c;
    } catch {
      // keep the http_ code
    }
  }
  return { ok: false, status: res.status, code, text, json: parsed };
}

// Parse a buffered OpenAI SSE body: concatenated text, finish reason, usage,
// and the first chunk's id/created (reused for synthesized chunks).
function parseOpenAIStream(text: string): ParsedStream {
  const out: ParsedStream = {
    text: '',
    finishReason: null,
    usage: null,
    firstId: 'chatcmpl-aggregator',
    firstCreated: FIXED_CREATED,
    chunkCount: 0,
  };
  const frames = text.split('\n\n');
  for (const frame of frames) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') continue;
      let chunk: Json | null = null;
      try {
        chunk = asRecord(JSON.parse(payload));
      } catch {
        continue;
      }
      if (!chunk) continue;
      out.chunkCount += 1;
      if (out.chunkCount === 1) {
        const id = asString(chunk.id);
        if (id) out.firstId = id;
        if (typeof chunk.created === 'number') out.firstCreated = chunk.created;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      for (const choice of choices) {
        const c = asRecord(choice);
        if (!c) continue;
        const delta = asRecord(c.delta);
        if (delta) {
          const piece = asString(delta.content);
          if (piece) out.text += piece;
        }
        const fin = asString(c.finish_reason);
        if (fin) out.finishReason = fin;
      }
      const usage = asRecord(chunk.usage);
      if (usage) {
        const pt = typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : 0;
        const ct = typeof usage.completion_tokens === 'number' ? usage.completion_tokens : 0;
        const tt = typeof usage.total_tokens === 'number' ? usage.total_tokens : pt + ct;
        out.usage = { prompt_tokens: pt, completion_tokens: ct, total_tokens: tt };
      }
    }
  }
  return out;
}

// Extract the assistant text from a parsed JSON Chat Completions body.
function jsonAnswerText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = choices.length > 0 ? asRecord(choices[0]) : null;
  if (!first) return '';
  const message = asRecord(first.message);
  if (!message) return '';
  const content = message.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    let text = '';
    for (const part of content) {
      const p = asRecord(part);
      if (p && p.type === 'text') {
        const t = asString(p.text);
        if (t) text += t;
      }
    }
    return text;
  }
  return '';
}

function jsonFinishReason(body: Json): string | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = choices.length > 0 ? asRecord(choices[0]) : null;
  if (!first) return null;
  return asString(first.finish_reason);
}

// Rebuild a normalized OpenAI SSE body from parsed stream data: the text as
// one content delta, exactly one finish chunk, exactly one usage chunk, and
// exactly one [DONE], nothing after. The provider's framing is discarded; the
// content, finish reason and usage are preserved (usage synthesized when the
// provider sent none).
function rebuildOpenAIStream(
  parsed: ParsedStream,
  model: string,
  inputTokens: number,
): string {
  const usage = parsed.usage && parsed.usage.completion_tokens > 0
    ? parsed.usage
    : {
        prompt_tokens: inputTokens,
        completion_tokens: roughTokens(parsed.text.length),
        total_tokens: inputTokens + roughTokens(parsed.text.length),
      };
  const finish = parsed.finishReason || 'stop';
  const base = {
    id: parsed.firstId,
    object: 'chat.completion.chunk',
    created: parsed.firstCreated,
    model,
  };
  const lines: string[] = [];
  if (parsed.text.length > 0) {
    lines.push(
      `data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: { role: 'assistant', content: parsed.text }, finish_reason: null }] })}\n\n`,
    );
  }
  lines.push(
    `data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: {}, finish_reason: finish }] })}\n\n`,
  );
  lines.push(
    `data: ${JSON.stringify({ ...base, choices: [], usage })}\n\n`,
  );
  lines.push('data: [DONE]\n\n');
  return lines.join('');
}

function streamFromText(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

// --- Serving one prompt -------------------------------------------------------

interface ServeOutcome {
  status: number;
  headers: Record<string, string>;
  // Exactly one of json / sseText is set on success.
  json: Json | null;
  sseText: string | null;
}

function failureHeaders(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  const rows = failures.slice(0, 8);
  let value = JSON.stringify(rows);
  if (value.length > 8000) value = JSON.stringify(rows.slice(0, 4));
  return { 'x-makefun-upstream-failures': value };
}

// Build the Chat Completions body for one upstream attempt. OpenAI wire:
// the client body with the model replaced. Anthropic wire: translated.
function upstreamBodyFor(
  wire: 'openai' | 'anthropic',
  body: Json,
  model: string,
  stream: boolean,
): Json {
  const maxTokensRaw = typeof body.max_tokens === 'number' ? body.max_tokens : DEFAULT_MAX_TOKENS;
  const maxTokens = Math.min(Math.max(1, Math.floor(maxTokensRaw)), MAX_TOKENS_CAP);
  if (wire === 'openai') {
    const out: Json = { ...body, model, max_tokens: maxTokens };
    if (stream) {
      out.stream = true;
      out.stream_options = { include_usage: true };
    } else {
      delete out.stream;
      delete out.stream_options;
    }
    return out;
  }
  const translated = anthropicToOpenAI(body, model);
  translated.max_tokens = maxTokens;
  if (stream) {
    translated.stream = true;
    translated.stream_options = { include_usage: true };
  } else {
    delete translated.stream;
    delete translated.stream_options;
  }
  return translated;
}

// Serve one prompt through the ladder. Returns the downstream response to
// send. `failed` is the trial's per-model failure memory (mutated).
async function servePrompt(
  wire: 'openai' | 'anthropic',
  body: Json,
  models: ModelRow[],
  failed: Set<string>,
): Promise<ServeOutcome> {
  const stream = isStreamBody(body);
  const requires = requirementsOf(wire, body);
  const token = extractToken(body);
  const inputTokens = roughTokens(requestChars(body));
  const ladder = ladderFor(models, requires, failed);
  const failures: FailureRow[] = [];

  const attempts = Math.min(MAX_ATTEMPTS_PER_PROMPT, ladder.length);
  for (let i = 0; i < attempts; i += 1) {
    const model = ladder[i];
    const upstreamBody = upstreamBodyFor(wire, body, model.id, stream);
    const res = await callUpstream(upstreamBody, stream);
    if (!res.ok) {
      failures.push({ status: res.status, code: res.code, model: model.id });
      failed.add(model.id);
      continue;
    }

    if (!stream) {
      const answerBody = res.json;
      if (!answerBody) { failed.add(model.id); continue; } // unparseable 2xx: demote, fail over
      const text = jsonAnswerText(answerBody);
      if (token && !text.includes(token)) {
        // Token-less answer: discard and demote the model for this trial
        // (a model that rambles on one token-echo prompt usually does so on
        // the next). Not a refusal: it stays out of the failures header.
        failed.add(model.id);
        continue;
      }
      // Success. Guarantee usage (output >= 1) before translating.
      const usage = asRecord(answerBody.usage);
      const hasUsage =
        usage !== null &&
        typeof usage.completion_tokens === 'number' &&
        usage.completion_tokens >= 1;
      if (!hasUsage) {
        const completion = roughTokens(text.length);
        answerBody.usage = {
          prompt_tokens: inputTokens,
          completion_tokens: completion,
          total_tokens: inputTokens + completion,
        };
      }
      const headers = failureHeaders(failures);
      if (wire === 'openai') {
        return { status: 200, headers, json: openAIJsonWithModel(answerBody, model.id), sseText: null };
      }
      return { status: 200, headers, json: openAIToAnthropic(answerBody, model.id), sseText: null };
    }

    // Streaming: the whole upstream SSE is already buffered (callUpstream
    // reads the body to the end). Verify the joined text, then replay.
    const parsed = parseOpenAIStream(res.text);
    if (token && !parsed.text.includes(token)) { failed.add(model.id); continue; } // token-less: demote, fail over
    if (parsed.chunkCount === 0 && parsed.text.length === 0) { failed.add(model.id); continue; } // empty stream: demote, fail over
    const rebuilt = rebuildOpenAIStream(parsed, model.id, inputTokens);
    const headers: Record<string, string> = {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      ...failureHeaders(failures),
    };
    if (wire === 'openai') {
      return { status: 200, headers, json: null, sseText: rebuilt };
    }
    // Anthropic wire: convert the rebuilt OpenAI SSE into Anthropic events.
    let anthropicText = '';
    for await (const piece of anthropicEventsFromOpenAI(streamFromText(rebuilt), model.id)) {
      anthropicText += piece;
    }
    return { status: 200, headers, json: null, sseText: anthropicText };
  }

  // No model served the prompt.
  const headers = failureHeaders(failures);
  if (wire === 'openai') {
    return {
      status: 502,
      headers,
      json: openAIError('all upstream models failed to answer this prompt', 'server_error', null),
      sseText: null,
    };
  }
  return {
    status: 502,
    headers,
    json: anthropicError('overloaded_error', 'all upstream models failed to answer this prompt'),
    sseText: null,
  };
}

// --- HTTP server --------------------------------------------------------------

function readRequestBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function sendJson(res: ServerResponse, status: number, body: Json, headers: Record<string, string>): void {
  res.writeHead(status, { 'Content-Type': 'application/json', ...headers });
  res.end(JSON.stringify(body));
}

function sendSse(res: ServerResponse, status: number, text: string, headers: Record<string, string>): void {
  res.writeHead(status, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive', ...headers });
  res.end(text);
}

interface ServerState {
  brief: Brief;
  models: ModelRow[];
  catalog: CatalogReport;
  failed: Set<string>;
}

async function handleRequest(
  state: ServerState,
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const url = new URL(req.url || '/', 'http://localhost');
  const path = url.pathname;

  if (req.method === 'GET' && path === '/v1/models') {
    const served = state.models.map((m) => ({
      id: m.id,
      provider: m.provider,
      display_name: m.display_name,
    }));
    const auth = req.headers.authorization || '';
    const apiKey = req.headers['x-api-key'];
    if (typeof apiKey === 'string' && apiKey.length > 0) {
      sendJson(res, 200, anthropicModelList(served), {});
    } else if (auth.startsWith('Bearer ')) {
      sendJson(res, 200, openAIModelList(served), {});
    } else {
      // No auth header: default to the OpenAI shape.
      sendJson(res, 200, openAIModelList(served), {});
    }
    return;
  }

  if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/v1/messages')) {
    const wire = path === '/v1/chat/completions' ? 'openai' : 'anthropic';
    let body: Json;
    try {
      const raw = await readRequestBody(req);
      const parsed: unknown = JSON.parse(raw);
      if (!isRecord(parsed)) throw new Error('not an object');
      body = parsed;
    } catch {
      if (wire === 'openai') {
        sendJson(res, 400, openAIError('request body is not a JSON object', 'invalid_request_error', 'invalid_json'), {});
      } else {
        sendJson(res, 400, anthropicError('invalid_request_error', 'request body is not a JSON object'), {});
      }
      return;
    }
    const outcome = await servePrompt(wire, body, state.models, state.failed);
    if (outcome.sseText !== null) {
      sendSse(res, outcome.status, outcome.sseText, outcome.headers);
    } else if (outcome.json !== null) {
      sendJson(res, outcome.status, outcome.json, outcome.headers);
    } else {
      sendJson(res, 502, wire === 'openai'
        ? openAIError('empty upstream answer', 'server_error', null)
        : anthropicError('overloaded_error', 'empty upstream answer'), outcome.headers);
    }
    return;
  }

  sendJson(res, 404, openAIError(`no such route: ${path}`, 'not_found_error', null), {});
}

// --- Trial lifecycle ----------------------------------------------------------

async function registerWithEnv(envUrl: string, port: number, catalog: CatalogReport): Promise<boolean> {
  const url = `${envUrl.replace(/\/$/, '')}/register`;
  for (let attempt = 0; attempt < REGISTER_RETRIES; attempt += 1) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) return true;
    } catch {
      // fall through to retry
    }
    if (attempt + 1 < REGISTER_RETRIES) await sleep(REGISTER_RETRY_MS);
  }
  return false;
}

async function pollWorkUntilDone(envUrl: string): Promise<void> {
  const url = `${envUrl.replace(/\/$/, '')}/work`;
  for (;;) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        const v: unknown = await res.json();
        const rec = asRecord(v);
        if (rec && rec.done === true) return;
      }
    } catch {
      // transient: keep polling
    }
    await sleep(WORK_POLL_MS);
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const brief = loadBrief();
  const catalog = catalogOf(brief);

  if (args.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    process.exit(0);
  }

  if (!brief) {
    process.stderr.write('no brief: MAKEFUN_PROMPT_FILE is not set or unreadable\n');
    process.exit(0);
  }

  const envUrl = process.env.MAKEFUN_ENV_URL || '';
  if (!envUrl) {
    process.stderr.write('no env: MAKEFUN_ENV_URL is not set\n');
    process.exit(0);
  }

  const models = freeModels(brief);
  const state: ServerState = { brief, models, catalog, failed: new Set<string>() };

  const server = createServer((req, res) => {
    handleRequest(state, req, res).catch(() => {
      if (!res.headersSent) {
        sendJson(res, 500, openAIError('internal error', 'server_error', null), {});
      } else {
        res.end();
      }
    });
  });

  await new Promise<void>((resolve) => {
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  if (port === 0) {
    process.stderr.write('failed to bind a port\n');
    process.exit(1);
  }

  const registered = await registerWithEnv(envUrl, port, catalog);
  if (!registered) {
    process.stderr.write('failed to register with the env\n');
    process.exit(1);
  }

  await pollWorkUntilDone(envUrl);
  server.close();
  process.exit(0);
}

main().catch((err: unknown) => {
  process.stderr.write(`fatal: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
