# llm-gateway-eval

A provider-agnostic LLM gateway with a built-in prompt evaluation suite, written in Go.

The gateway puts several LLM vendors behind a single request/response contract, with retries and fallback between providers. The evaluation suite versions prompt templates, replays test cases against every version, and records the results so regressions are visible before a prompt change ships.

> **Status: working, not yet battle-tested.** Everything described here is implemented and covered by tests, but the provider adapters have only been exercised against local fake servers, never against the real vendor APIs. See [what is not done yet](#roadmap).

## Why

Prompts are code, but they rarely get the same treatment: no versions, no tests, no record of which template produced which output. Separately, calling a single LLM vendor directly leaves an application exposed to that vendor's outages and rate limits.

This project addresses both in one service:

- **One contract, several vendors.** Callers send one request shape; adapters translate it into each vendor's payload and normalise the response (finish reasons, token usage, latency).
- **Retries and fallback.** Transient failures are retried with backoff; a named route falls back from one provider to the next, each with its own model.
- **Immutable prompt versions.** Every change to a template, system prompt, model or sampling parameter creates a new numbered version.
- **Regression testing for prompts.** Test cases belong to the prompt, not to a version, so the same suite runs against every version and two runs can be compared.
- **Traceable results.** Each result records the exact prompt version, the provider and model that actually served it, token usage, estimated cost and latency.

## Quick start

Requires Go 1.26 or later.

```bash
git clone git@github.com:mustafajarrah/llm-gateway-eval.git
```

The cheapest way to try it is a local [Ollama](https://ollama.com), which needs no API key and costs nothing:

```bash
OLLAMA_BASE_URL=http://localhost:11434/v1 make run
```

The service listens on `127.0.0.1:8080` and stores its data in `data/gateway.db`. Send a completion:

```bash
curl -s http://127.0.0.1:8080/v1/completions -d '{
  "provider": "ollama",
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "Say hello in one word."}]
}'
```

To use a hosted provider instead, set its key (for example `ANTHROPIC_API_KEY`) and name it in `provider`. Those calls are billed by the vendor.

## Providers

| Identifier | Provider | Enabled by |
| --- | --- | --- |
| `openai` | OpenAI | `OPENAI_API_KEY` |
| `anthropic` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini` | Google Gemini | `GEMINI_API_KEY` |
| `mistral` | Mistral | `MISTRAL_API_KEY` |
| `ollama` | Ollama (locally hosted models) | `OLLAMA_BASE_URL` |
| `openai_compatible` | Any endpoint speaking the OpenAI chat completions protocol (Groq, OpenRouter, Together, vLLM, Azure OpenAI, ...) | `OPENAI_COMPATIBLE_BASE_URL` |

Only one `openai_compatible` endpoint can be configured at a time.

## Routing

A request reaches a provider in one of two ways.

**Pinned.** With `provider` set, `model` is that vendor's model ID. The request goes to that provider only; transient failures are retried, and there is no fallback.

**Routed.** Without `provider`, `model` is the name of a route: an ordered list of `provider:model` targets defined in `GATEWAY_ROUTES`.

```bash
GATEWAY_ROUTES="fast=ollama:llama3.2,openai:gpt-4o-mini;smart=anthropic:claude-opus-5-5,openai:gpt-4o"
```

```bash
curl -s http://127.0.0.1:8080/v1/completions -d '{
  "model": "fast",
  "messages": [{"role": "user", "content": "Say hello in one word."}]
}'
```

The gateway tries each target in order:

- A target is retried (2 attempts by default, exponential backoff with jitter) while its failures are transient: timeouts, 429s and 5xx.
- Any other failure, such as a rejected API key or an unknown model, skips the retries and moves straight to the next target.
- The temperature is clamped to each target's limit (Anthropic stops at 1, Mistral at 1.5, the others at 2).
- If the caller disconnects, the gateway stops immediately.

The response says which provider and model served the request.

### Circuit breakers

Circuit breakers stop a failing target from costing every request its retries and backoff. There are two levels, and a call goes through only when both allow it.

| Breaker | Counts | Why |
| --- | --- | --- |
| Per model (`provider:model`) | Every unhealthy call: timeouts, 429s, 5xx, connection failures | Rate limits and overload usually hit one model while the vendor's other models keep answering |
| Per provider | Only failures to reach the provider at all: connection refused, DNS, reset | Those mean the vendor is unreachable whatever the model |

So five 429s on one model stop traffic to that model only, while a vendor that cannot be reached is skipped for all of its models after five failed connections, instead of five per model.

Both levels work the same way:

- After 5 consecutive unhealthy calls the breaker **opens**: for 30 seconds the target is skipped without being called. A route moves straight to its next target; a pinned request fails at once with a 502 whose message says which breaker is open.
- After the cooldown the breaker is **half-open**: one probe request goes through. If it succeeds the breaker closes; if not, it reopens for another cooldown.
- A 4xx answer does not count, since it means the provider is up and rejected that request. A timeout counts against the model but not the provider.

`GET /v1/providers` reports the provider breakers under `circuits` and the model breakers under `model_circuits` (`closed`, `open` or `half_open`). Only models with recent failures are listed; a model that is absent is closed. The state is kept in memory, per process.

## Cost

The gateway estimates what each completion costs, from the token usage the provider reports and a price table you supply. No prices are built in: vendors change them, and a stale table would produce confident wrong numbers.

```bash
GATEWAY_PRICES="anthropic:claude-opus-5-5=4/20;openai:gpt-4o-mini=0.15/0.60;ollama:llama3.2=0/0"
```

Each entry is `provider:model=input/output`, in US dollars per million tokens.

- A completion served by a priced target carries `cost_usd`; one served by an unpriced target has no `cost_usd` field at all, rather than a misleading zero.
- Each evaluation result stores its cost, and a run's summary adds them up in `cost_usd`. `unpriced` counts the results that used tokens but had no price, so a non-zero value means the total understates the run.
- Along a route, the price applied is that of the target that actually served the request.
- Costs are fixed when recorded. Changing a price later does not rewrite past runs.

The figure is an estimate: every input token is billed at the plain input rate, so vendor discounts such as cached-prompt or batch pricing are not reflected. Prices are matched on the exact `provider:model` you request.

## Evaluating prompts

The walkthrough below creates a prompt, a version and two test cases, then runs the suite. It assumes the Ollama setup from the quick start.

Create a prompt and note its `id`:

```bash
curl -s http://127.0.0.1:8080/v1/prompts -d '{"name": "capital", "description": "Capital of a country"}'
```

Add a version. Templates use Go's [`text/template`](https://pkg.go.dev/text/template) syntax; referencing a variable the test case does not provide is an error, not a silent `<no value>`.

```bash
curl -s http://127.0.0.1:8080/v1/prompts/$PROMPT_ID/versions -d '{
  "template": "What is the capital of {{.country}}? Answer with the city name only.",
  "provider": "ollama",
  "model": "llama3.2",
  "parameters": {"temperature": 0},
  "change_log": "first version"
}'
```

Add test cases:

```bash
curl -s http://127.0.0.1:8080/v1/prompts/$PROMPT_ID/test-cases -d '{
  "name": "france",
  "variables": {"country": "France"},
  "expected_output": "paris",
  "match_strategy": "contains"
}'
```

```bash
curl -s http://127.0.0.1:8080/v1/prompts/$PROMPT_ID/test-cases -d '{
  "name": "japan",
  "variables": {"country": "Japan"},
  "expected_output": "^Tokyo\\.?$",
  "match_strategy": "regex"
}'
```

Run the suite against the latest version. With `wait=true` the call blocks until the run is over and returns the run with one result per test case:

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/prompts/$PROMPT_ID/runs?wait=true"
```

Without `wait`, the call answers `202 Accepted` at once with the run in the `running` state, and the suite executes in the background. Poll the run until its status is `completed` or `failed`:

```bash
curl -s http://127.0.0.1:8080/v1/runs/$RUN_ID
```

Results are stored when the run finishes, so a running run has none yet. A run still in the background when the service stops is recorded as `failed`.

After adding a second version, run again and compare the two runs to see what regressed and what improved:

```bash
curl -s "http://127.0.0.1:8080/v1/comparisons?base=$RUN_1&candidate=$RUN_2"
```

### Match strategies

| Strategy | Passes when |
| --- | --- |
| `exact` | The trimmed output equals the trimmed expectation (case-sensitive) |
| `contains` | The output contains the expectation (case-insensitive) |
| `regex` | The output matches the expectation as a regular expression |

A test case that cannot be executed (template error, provider failure) is recorded as an errored result; it does not abort the run.

## HTTP API

| Method and path | Purpose |
| --- | --- |
| `GET /healthz` | Liveness check (never authenticated) |
| `GET /v1/providers` | Configured providers, routes, prices and circuit breaker states |
| `POST /v1/completions` | Complete through the gateway |
| `POST /v1/prompts` · `GET /v1/prompts` | Create and list prompts (`?limit=&offset=`) |
| `GET /v1/prompts/{id}` · `DELETE /v1/prompts/{id}` | Read or delete a prompt; deleting cascades to its versions, test cases, runs and results |
| `POST /v1/prompts/{id}/versions` · `GET /v1/prompts/{id}/versions` | Create and list versions |
| `GET /v1/prompts/{id}/versions/{n}` | Read version `n`, or `latest` |
| `POST /v1/prompts/{id}/test-cases` · `GET /v1/prompts/{id}/test-cases` | Create and list test cases |
| `GET /v1/test-cases/{id}` · `DELETE /v1/test-cases/{id}` | Read or delete a test case |
| `POST /v1/prompts/{id}/runs` | Start a run in the background (202); `?wait=true` blocks and returns the report (201). Optional body `{"version": n}` |
| `GET /v1/prompts/{id}/runs` | List runs, newest first (`?limit=&offset=`) |
| `GET /v1/runs/{id}` | A run, with its results once it has finished |
| `GET /v1/comparisons?base=&candidate=` | Regressions and improvements between two runs |

Collections are returned as `{"data": [...]}`. Request bodies are decoded strictly: an unknown field is a 400.

Errors share one shape:

```json
{"error": {"code": "not_found", "message": "prompt \"abc\": not found"}}
```

| Status | `code` | Meaning |
| --- | --- | --- |
| 400 | `invalid_input` | Validation failed, malformed JSON, unknown route |
| 401 | `unauthorized` | Missing or wrong bearer token |
| 404 | `not_found` | Entity does not exist |
| 409 | `conflict` | Name or ID already taken |
| 413 | `body_too_large` | Body over 1 MB |
| 502 | `provider_error` | The upstream provider failed; includes `provider` and `upstream_status` |
| 503 | `unavailable` | The service is shutting down and takes no new runs |
| 504 | `timeout` | The request timed out |
| 500 | `internal` | Unexpected failure; details are in the server log only |

## Configuration

Everything is configured through environment variables; [.env.example](.env.example) lists them all with comments.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GATEWAY_ADDR` | `127.0.0.1:8080` | Listen address |
| `GATEWAY_API_KEY` | none | Bearer token required from clients |
| `GATEWAY_DB_PATH` | `data/gateway.db` | SQLite file, or `:memory:` |
| `GATEWAY_LOG_LEVEL` | `info` | `debug`, `info`, `warn` or `error` |
| `GATEWAY_ROUTES` | none | Fallback chains, see [Routing](#routing) |
| `GATEWAY_MAX_ATTEMPTS` | `2` | Tries per target |
| `GATEWAY_ATTEMPT_TIMEOUT` | `60s` | Limit of a single provider call |
| `GATEWAY_BREAKER_THRESHOLD` | `5` | Consecutive unhealthy calls that open a circuit breaker; `0` disables them |
| `GATEWAY_BREAKER_COOLDOWN` | `30s` | How long an open breaker skips its target |
| `GATEWAY_PRICES` | none | Per-model prices, see [Cost](#cost) |
| `GATEWAY_EVAL_CONCURRENCY` | `4` | Test cases evaluated in parallel |
| `ANTHROPIC_MAX_TOKENS` | `16000` | Output limit for Anthropic requests that set none |

Each provider also accepts an optional `<PROVIDER>_BASE_URL`.

### Authentication

By default the service listens on loopback only and needs no credentials. With `GATEWAY_API_KEY` set, every route except `/healthz` requires `Authorization: Bearer <key>`.

The service **refuses to start** on a non-loopback address without `GATEWAY_API_KEY`: anyone who could reach it could spend the configured provider keys.

### Docker

```bash
make docker
```

```bash
docker run --rm -p 8080:8080 -v gateway-data:/data \
  -e GATEWAY_API_KEY=change-me \
  -e ANTHROPIC_API_KEY \
  llm-gateway-eval
```

The image listens on `:8080`, so `GATEWAY_API_KEY` is mandatory. Data lives in the `/data` volume.

## Architecture

The project follows Clean Architecture. The domain package is the innermost ring: it depends only on the standard library and defines the ports that outer layers implement.

```mermaid
flowchart LR
    HTTP["httpapi<br/>JSON API"] --> GW["gateway<br/>routing, retries, fallback"]
    HTTP --> EVAL["evaluation<br/>runner, comparison"]
    EVAL --> GW
    GW --> DOM["domain<br/>entities + ports"]
    EVAL --> DOM
    PROV["provider/*<br/>openaicompat, anthropic, gemini"] -. implements LLMProvider .-> DOM
    STORE["storage/*<br/>sqlite, memory"] -. implements repositories .-> DOM
```

```
cmd/gateway/             the service binary
internal/
  domain/                entities, validation and ports; standard library only
  provider/
    upstream/            shared HTTP plumbing and error classification
    openaicompat/        OpenAI, Mistral, Ollama and compatible endpoints
    anthropic/           Anthropic, on the official Go SDK
    gemini/              Google Gemini
  gateway/               routing, retries and provider fallback
  storage/
    sqlite/              durable store (pure-Go driver, no cgo)
    memory/              in-memory store for tests
    storagetest/         conformance suite every store must pass
  evaluation/            evaluation runner and run comparison
  httpapi/               HTTP handlers
  config/                environment-based configuration
  app/                   composition root
```

The reasoning behind the main design choices is in [docs/decisions.md](docs/decisions.md).

## Development

```bash
make check
```

That runs `go vet`, `staticcheck`, the tests with the race detector, and a `gofmt` check: the same steps as CI. `make cover` prints per-package coverage.

No test touches the network or a real provider. Adapters are tested against `httptest` servers that imitate each vendor's API.

## Roadmap

Done:

- [x] Domain entities, validation and ports
- [x] Provider adapters for all six providers
- [x] Gateway routing with retries, provider fallback and circuit breakers
- [x] SQLite and in-memory storage
- [x] Evaluation runner (background or blocking) and run comparison
- [x] Cost estimation from token usage and configured prices
- [x] HTTP API, configuration and service binary
- [x] CI (gofmt, vet, staticcheck, tests, Docker build)

Not done yet:

- [ ] Verification against the real vendor APIs
- [ ] Streaming responses
- [ ] Tool calling and structured output
- [ ] Graded scoring (semantic similarity, LLM-as-judge)
- [ ] Several `openai_compatible` endpoints at once
- [ ] Metrics and tracing

## License

[MIT](LICENSE)
