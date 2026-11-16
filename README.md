# Free-Model Aggregator

One server, two wires: every completion it answers comes from a real free
OpenRouter model, dialled through the grader's gateway. It serves the OpenAI
Chat Completions API and the Anthropic Messages API, in JSON and in streaming,
routing each prompt to the best-rated free model that meets its requirements,
with failover across a rating-ordered ladder and a verify-before-relay check
that the provider's answer carries the prompt's token.

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/fadliferdian28/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/fadliferdian28/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

## Run

Node 22, no dependencies:

```
node main.ts            # serve (reads MAKEFUN_PROMPT_FILE + MAKEFUN_ENV_URL)
node main.ts --list     # print the catalog document and exit 0
```

## Layout

- `main.ts` — the server: routing, failover ladder, buffer-then-verify-then-relay,
  both wires, registration and work polling.
- `lib/providers.ts`, `lib/wire.ts` — the kit starter's brief/catalog and
  wire-translation helpers, copied verbatim.

## License

MIT.
