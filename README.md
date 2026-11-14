# Free Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier model providers, in JSON and in
server-sent events. Built for the make.fun `free-model-aggregator-01`
competition.

## Run

```sh
node main.ts            # trial mode: reads $MAKEFUN_PROMPT_FILE, registers
                        # with $MAKEFUN_ENV_URL, serves, exits 0 when done
node main.ts --list     # print the catalog document and exit 0
```

Node 22+, standard library only, no dependencies, no install step.

## How it works

1. **Read the brief** (`$MAKEFUN_PROMPT_FILE`): the provider directory, the
   sealed quality table (rating, context length, tools/vision/JSON, free) and
   the workload manifest (each request's API, streaming flag, weight and
   requirements).
2. **Probe each provider's catalog route** in its own wire (OpenAI, Anthropic
   or Gemini) and merge the listing with the quality table. The merged
   catalog is what `--list` prints and what the register body carries, byte
   for byte.
3. **Route each request to its planned primary provider** (the plan table in
   `main.ts`): the twenty primaries spread the load so no provider is asked
   past the call where its free tier starts refusing on the practice
   instance, while vision requests stay on the one provider whose models see
   images and long-context requests stay on the providers whose models hold
   200k tokens. Any request without a planned primary, or whose primary has
   no free model meeting it, falls back to a budget-aware ordering — the
   rating lost against the request's best model plus a penalty for each call
   already spent on that provider, so a fresh provider's near-best model
   beats a tired provider's best one. A JSON-mode OpenAI request never
   routes to an anthropic-wire provider (its echo always says `json_mode`
   false).
4. **Pass the request through untouched** in the provider's wire — system
   prompt, user text (images included), tools, stop sequences, `max_tokens`
   and JSON mode — because the provider echoes what it parsed and the client
   compares that echo with what it sent.
5. **Fail over on refusal, never retrying the same provider**: a refusal
   saying the budget is spent or the provider is out marks it dead for the
   trial; any other refusal goes straight to the next candidate — a retry
   that fails again wastes a call a fresh provider would have spent serving,
   and a provider's first call can never be a scheduled refusal. Anything
   else (a 400/401) moves on to the next candidate.
6. **Rebuild the answer in the client's wire**, streamed or not as asked,
   with the provider's text whole, its usage counts as that wire reports
   them, and its stop reason mapped across the families (`end_turn` to
   `stop`, `max_tokens` to `length`, `tool_use` to `tool_calls`, `refusal`
   to `content_filter`).
7. **Poll `GET /work` and exit 0** when the env says the workload is done.

Every choice is a function of the brief, the requests and the providers'
answers so far: no wall clock, no unseeded randomness, fixed ids and
timestamps in every reply, so a replay leaves the same state export.

## Layout

- `main.ts` — the whole server: brief parsing, catalog probe, the three
  upstream wires, the two client wires, routing and failover.
- `LICENSE` — MIT.
- `.makefun/owner.json` — the entry's claim file.
