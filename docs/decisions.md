# Design decisions

The choices made while building the first working version, with the reasoning and what it would take to change each one. Entries are grouped by area; the pull request that introduced each is linked.

## Domain

### Requests without a provider use `model` as a route name ([#3](https://github.com/mustafajarrah/llm-gateway-eval/pull/3))

A vendor model ID such as `gpt-4o` means nothing to another vendor, so fallback cannot reuse it. A request either pins a provider (then `model` is a vendor ID and there is no fallback) or leaves it empty (then `model` names a route whose targets each carry their own vendor ID).

*Alternative:* a separate `route` field. Rejected to keep the request shape small. To change: add the field to `LLMRequest` and `PromptVersion` and adjust `gateway.resolve`.

### Temperature limits are per provider, and clamped on fallback ([#3](https://github.com/mustafajarrah/llm-gateway-eval/pull/3))

Anthropic accepts temperatures up to 1, Mistral up to 1.5, the others up to 2. A pinned request over its provider's limit is rejected with a 400. Along a route the gateway lowers the temperature to each target's limit instead, so one strict target does not break the chain.

*To change:* edit `Provider.MaxTemperature` in `internal/domain/llm.go`.

### Latency is serialised as `latency_ms` ([#3](https://github.com/mustafajarrah/llm-gateway-eval/pull/3))

`time.Duration` encodes as raw nanoseconds by default, which surprises API consumers.

### Repositories assign IDs, timestamps and version numbers ([#3](https://github.com/mustafajarrah/llm-gateway-eval/pull/3))

IDs are 128 random bits in hex. Version numbers are assigned inside the storage transaction, which is the only place that can guarantee they are unique and gap-free.

## Providers

### One adapter for every OpenAI-compatible vendor ([#4](https://github.com/mustafajarrah/llm-gateway-eval/pull/4))

OpenAI, Mistral, Ollama and generic endpoints share a client; they differ in base URL, API key and one field name (OpenAI gets `max_completion_tokens`, which its reasoning models require; the rest get `max_tokens`).

*Limitation:* only one `openai_compatible` endpoint can be configured, because providers are keyed by identifier. Supporting Groq and OpenRouter at the same time needs named provider instances.

### What counts as retryable ([#4](https://github.com/mustafajarrah/llm-gateway-eval/pull/4))

Retryable: HTTP 408, 425, 429 and 5xx, transport errors, and 2xx responses that are malformed or empty. Not retryable: other 4xx and cancelled requests.

### The Anthropic adapter uses the official SDK ([#5](https://github.com/mustafajarrah/llm-gateway-eval/pull/5))

It is the module's first third-party dependency. SDK-level retries are switched off, since the gateway owns retries.

- **Sampling parameters are dropped for current models.** Anthropic removed `temperature` and `top_p` from Opus 4.7 and the Claude 5 family onwards; sending them is a 400. The adapter sends them only to model families known to accept them (`AcceptsSampling`). The consequence to be aware of: a prompt version with `temperature: 0` runs on those models at the model's own default.
- **`max_tokens` defaults to 16000**, because the API requires it and on current models the budget also covers thinking. Configurable with `ANTHROPIC_MAX_TOKENS`.
- **Thinking and effort are left at each model's default.**
- **Anthropic's server-side refusal fallback is not enabled.** The gateway has its own fallback, and that feature can change which model serves a request. A refusal surfaces as finish reason `content_filter`.

### Gemini over plain HTTP ([#6](https://github.com/mustafajarrah/llm-gateway-eval/pull/6))

Chosen over Google's SDK to avoid a large dependency tree for one endpoint. The API key travels in a header, never in the URL. A blocked prompt is returned as an empty `content_filter` response, not as an error.

## Gateway

### Fallback happens on any failure, retries only on transient ones ([#7](https://github.com/mustafajarrah/llm-gateway-eval/pull/7))

A rejected key or unknown model is specific to one provider, so the next target of a route is still worth trying. The cost is that a request invalid everywhere is sent to every target before failing.

*Alternative:* stop the chain on non-retryable errors. To change: return early in `Gateway.Complete` when `!domain.IsRetryable(err)`.

### Defaults: 2 attempts per target, 60 s per attempt, 250 ms to 5 s backoff with jitter ([#7](https://github.com/mustafajarrah/llm-gateway-eval/pull/7))

Attempts and timeout are configurable through the environment; the backoff bounds are not yet.

### Hand-written circuit breakers, per model and per provider

A breaker opens after 5 consecutive unhealthy calls, skips its target for 30 seconds, then lets a single probe through.

- **Per model**, fed by every unhealthy call (timeouts, 429s, 5xx, connection failures). Rate limits and overload are usually specific to a model, so one failing model must not block the vendor's others. This matters most for a route such as `anthropic:opus,anthropic:haiku`, where the second target exists to cover the first.
- **Per provider**, fed only by failures to reach the provider at all. Without it, every model would have to discover a vendor-wide outage on its own.
- **Timeouts count against the model only.** A timeout arrives without an HTTP status, like a connection failure, but it usually means a slow model rather than an unreachable vendor.
- **4xx answers and abandoned calls count against nothing.**

Model breakers exist only for targets with recent failures and are capped at 1024, because model names come from requests and an unbounded map would be a way to exhaust memory. Past the cap a model is covered by its provider's breaker only.

Both levels share one threshold and one cooldown. The implementation is `internal/gateway/breaker.go` rather than a library such as `sony/gobreaker`, to keep the dependency count at two. State is in memory and per process: several instances each learn about an outage on their own.

*To change:* `GATEWAY_BREAKER_THRESHOLD` (0 disables) and `GATEWAY_BREAKER_COOLDOWN`.

### Cost is estimated from configured prices, with none built in

`GATEWAY_PRICES` maps `provider:model` to input and output rates. The gateway attaches `cost_usd` to a completion when the target that served it has a price; results store that cost, and run summaries add it up and count the results that had none (`unpriced`).

- **No default price table.** Vendor prices change, and a hard-coded table goes stale silently. The cost of that choice is that nothing has a cost until you configure it.
- **Unknown is not zero.** An unpriced completion has no `cost_usd`; a free model is priced explicitly at `0/0`.
- **Costs are frozen when recorded**, so history is not rewritten by a later price change. It also means a wrong price stays wrong in past runs.
- **It is an estimate.** All input tokens are billed at the input rate; cached-prompt and batch discounts are ignored, and for Gemini thinking tokens are counted as output.
- **Exact match on the requested model.** A price for `gpt-4o` does not apply to a request for `gpt-4o-2024-08-06`.

*To change:* prices could move into the database with an effective date, which would allow repricing past runs.

## Storage

### SQLite through a pure-Go driver ([#9](https://github.com/mustafajarrah/llm-gateway-eval/pull/9))

No external database to run and no cgo, so the binary is static and the image is small. The store uses a single connection, which serialises all access.

*Trade-off:* fine for one instance; several instances sharing data would need Postgres. The repositories are interfaces and `internal/storage/storagetest` is a conformance suite, so a new backend has a clear target.

*Side effect:* the driver raised the module's minimum Go version from 1.24 to 1.26.

### Deleting a prompt deletes its history ([#8](https://github.com/mustafajarrah/llm-gateway-eval/pull/8))

Versions, test cases, runs and results go with it. Deleting a single test case keeps the results already recorded for it.

*Alternative:* soft deletes, to keep results truly append-only.

### "Newest first" means creation order ([#8](https://github.com/mustafajarrah/llm-gateway-eval/pull/8))

Listings sort by insertion sequence, not by the `created_at` value, so entities created in the same instant still have a stable order.

## Evaluation

### Runs execute in the background by default

`POST /v1/prompts/{id}/runs` answers `202` with the run in the `running` state and executes the suite in a goroutine of the same process; clients poll `GET /v1/runs/{id}`. `?wait=true` keeps the original blocking behaviour, where a client that disconnects cancels the run.

There is no job queue, so the consequences are:

- **A run does not survive the process.** On shutdown, runs in flight are cancelled and recorded as `failed`. After a crash, runs left as `running` are marked `failed` at the next start (`InterruptRuns`). Nothing is resumed.
- **Results appear only at the end.** They are saved in one transaction when the run finishes, so there is no partial progress to read.
- **No limit on simultaneous runs.** Each run bounds its own parallelism (`GATEWAY_EVAL_CONCURRENCY`), but nothing caps how many runs execute at once.

*To change:* a persistent queue with workers would make runs durable and cap concurrency.

### A failing test case never aborts a run ([#10](https://github.com/mustafajarrah/llm-gateway-eval/pull/10))

Template errors and provider failures become errored results, counted separately from cases that ran and did not match.

### Scoring is binary ([#10](https://github.com/mustafajarrah/llm-gateway-eval/pull/10))

The three match strategies give a score of 0 or 1. The `score` field is a float so graded scorers can be added.

## HTTP API and service

### Standard library only ([#11](https://github.com/mustafajarrah/llm-gateway-eval/pull/11))

`net/http` routing patterns cover the API; no web framework.

### Strict request decoding ([#11](https://github.com/mustafajarrah/llm-gateway-eval/pull/11))

Unknown JSON fields are a 400. That catches typos and stops clients from setting server-assigned fields, at the cost of being less forgiving to clients that send extra data.

### Provider failures are a 502 ([#11](https://github.com/mustafajarrah/llm-gateway-eval/pull/11))

Whatever the upstream status was, the client gets 502 with the upstream status in the body. Internal errors return a generic message; the cause goes to the log only.

### Loopback by default, and no unauthenticated exposure ([#12](https://github.com/mustafajarrah/llm-gateway-eval/pull/12))

The default address is `127.0.0.1:8080`. On any other address the service refuses to start without `GATEWAY_API_KEY`, because whoever can reach the API can spend the provider keys. Authentication is a single shared bearer token; there are no per-client keys, quotas or rate limits.

### Configuration through environment variables only ([#12](https://github.com/mustafajarrah/llm-gateway-eval/pull/12))

No config file and no `.env` loading, to avoid a dependency. Routes use a compact string syntax, which gets unwieldy beyond a few routes; a config file would be the next step.

## Not verified

No request has been sent to a real provider: the constraint for this phase was zero cost, and no local Ollama was available. Adapters are tested against local servers that imitate each vendor's documented API. The first run against each real API should be treated as a test.
