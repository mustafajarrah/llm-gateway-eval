package gateway

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

func ptr[T any](v T) *T { return &v }

// fakeProvider replays a script of results, one per call, and records the
// requests it received. Once the script is exhausted it keeps returning the
// last entry.
type fakeProvider struct {
	name   domain.Provider
	script []result

	mu    sync.Mutex
	calls []domain.LLMRequest
	// onCall, when set, runs before each result is returned.
	onCall func(ctx context.Context)
}

type result struct {
	resp *domain.LLMResponse
	err  error
}

func (f *fakeProvider) Name() domain.Provider { return f.name }

func (f *fakeProvider) Complete(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
	f.mu.Lock()
	f.calls = append(f.calls, *req)
	i := len(f.calls) - 1
	f.mu.Unlock()
	if f.onCall != nil {
		f.onCall(ctx)
	}
	if i >= len(f.script) {
		i = len(f.script) - 1
	}
	return f.script[i].resp, f.script[i].err
}

func ok(provider domain.Provider) result {
	return result{resp: &domain.LLMResponse{Provider: provider, Content: "from " + string(provider)}}
}

func fail(provider domain.Provider, status int, retryable bool) result {
	return result{err: &domain.ProviderError{Provider: provider, StatusCode: status, Retryable: retryable, Err: errors.New("boom")}}
}

type harness struct {
	*Gateway
	sleeps []time.Duration
}

func newGateway(t *testing.T, cfg Config, providers ...domain.LLMProvider) *harness {
	t.Helper()
	cfg.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	g, err := New(providers, cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	h := &harness{Gateway: g}
	g.jitter = func() float64 { return 1 }
	g.sleep = func(_ context.Context, d time.Duration) error {
		h.sleeps = append(h.sleeps, d)
		return nil
	}
	return h
}

func routeRequest(route string) *domain.LLMRequest {
	return &domain.LLMRequest{Model: route, Messages: []domain.Message{{Role: domain.RoleUser, Content: "hi"}}}
}

var fastRoute = domain.Route{Name: "fast", Targets: []domain.Target{
	{Provider: domain.ProviderOpenAI, Model: "gpt-4o-mini"},
	{Provider: domain.ProviderAnthropic, Model: "claude-haiku-4-5"},
}}

func TestCompletePinnedProvider(t *testing.T) {
	openai := &fakeProvider{name: domain.ProviderOpenAI, script: []result{ok(domain.ProviderOpenAI)}}
	g := newGateway(t, Config{}, openai)

	req := routeRequest("gpt-4o")
	req.Provider = domain.ProviderOpenAI
	req.Parameters.Temperature = ptr(1.7)
	resp, err := g.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if resp.Content != "from openai" {
		t.Errorf("response = %+v", *resp)
	}
	got := openai.calls[0]
	if got.Provider != domain.ProviderOpenAI || got.Model != "gpt-4o" || *got.Parameters.Temperature != 1.7 {
		t.Errorf("provider received %+v", got)
	}
}

func TestCompleteRetriesThenSucceeds(t *testing.T) {
	openai := &fakeProvider{name: domain.ProviderOpenAI, script: []result{
		fail(domain.ProviderOpenAI, 429, true),
		fail(domain.ProviderOpenAI, 503, true),
		ok(domain.ProviderOpenAI),
	}}
	g := newGateway(t, Config{MaxAttempts: 3, BaseBackoff: 100 * time.Millisecond, MaxBackoff: 150 * time.Millisecond}, openai)

	req := routeRequest("gpt-4o")
	req.Provider = domain.ProviderOpenAI
	if _, err := g.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if len(openai.calls) != 3 {
		t.Errorf("provider called %d times, want 3", len(openai.calls))
	}
	want := []time.Duration{100 * time.Millisecond, 150 * time.Millisecond}
	if len(g.sleeps) != 2 || g.sleeps[0] != want[0] || g.sleeps[1] != want[1] {
		t.Errorf("backoffs = %v, want %v (doubling, capped)", g.sleeps, want)
	}
}

func TestCompletePinnedProviderFailure(t *testing.T) {
	openai := &fakeProvider{name: domain.ProviderOpenAI, script: []result{fail(domain.ProviderOpenAI, 400, false)}}
	anthropic := &fakeProvider{name: domain.ProviderAnthropic, script: []result{ok(domain.ProviderAnthropic)}}
	g := newGateway(t, Config{Routes: []domain.Route{fastRoute}}, openai, anthropic)

	req := routeRequest("gpt-4o")
	req.Provider = domain.ProviderOpenAI
	_, err := g.Complete(context.Background(), req)
	var pe *domain.ProviderError
	if !errors.As(err, &pe) || pe.StatusCode != 400 {
		t.Fatalf("error = %v, want the provider's 400", err)
	}
	if len(openai.calls) != 1 || len(anthropic.calls) != 0 || len(g.sleeps) != 0 {
		t.Errorf("calls: openai %d anthropic %d sleeps %d; a pinned, non-retryable failure must not retry or fall back",
			len(openai.calls), len(anthropic.calls), len(g.sleeps))
	}
}

func TestCompleteRouteFallsBack(t *testing.T) {
	tests := []struct {
		name            string
		failure         result
		wantOpenAICalls int
	}{
		{name: "after retryable failures", failure: fail(domain.ProviderOpenAI, 503, true), wantOpenAICalls: 2},
		{name: "after a non-retryable failure", failure: fail(domain.ProviderOpenAI, 401, false), wantOpenAICalls: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			openai := &fakeProvider{name: domain.ProviderOpenAI, script: []result{tt.failure}}
			anthropic := &fakeProvider{name: domain.ProviderAnthropic, script: []result{ok(domain.ProviderAnthropic)}}
			g := newGateway(t, Config{Routes: []domain.Route{fastRoute}}, openai, anthropic)

			req := routeRequest("fast")
			req.Parameters.Temperature = ptr(1.6)
			resp, err := g.Complete(context.Background(), req)
			if err != nil {
				t.Fatalf("Complete() error = %v", err)
			}
			if resp.Provider != domain.ProviderAnthropic {
				t.Errorf("served by %s, want anthropic", resp.Provider)
			}
			if len(openai.calls) != tt.wantOpenAICalls {
				t.Errorf("openai called %d times, want %d", len(openai.calls), tt.wantOpenAICalls)
			}
			first, second := openai.calls[0], anthropic.calls[0]
			if first.Provider != domain.ProviderOpenAI || first.Model != "gpt-4o-mini" || *first.Parameters.Temperature != 1.6 {
				t.Errorf("openai received %+v", first)
			}
			if second.Provider != domain.ProviderAnthropic || second.Model != "claude-haiku-4-5" || *second.Parameters.Temperature != 1 {
				t.Errorf("anthropic received %+v, want its own model and a clamped temperature", second)
			}
			if *req.Parameters.Temperature != 1.6 || req.Model != "fast" || req.Provider != "" {
				t.Errorf("Complete() mutated the caller's request: %+v", req)
			}
		})
	}
}

func TestCompleteRouteAllTargetsFail(t *testing.T) {
	openai := &fakeProvider{name: domain.ProviderOpenAI, script: []result{fail(domain.ProviderOpenAI, 500, true)}}
	anthropic := &fakeProvider{name: domain.ProviderAnthropic, script: []result{fail(domain.ProviderAnthropic, 529, true)}}
	g := newGateway(t, Config{Routes: []domain.Route{fastRoute}}, openai, anthropic)

	_, err := g.Complete(context.Background(), routeRequest("fast"))
	var pe *domain.ProviderError
	if !errors.As(err, &pe) {
		t.Fatalf("error %v does not wrap a *domain.ProviderError", err)
	}
	for _, want := range []string{`route "fast"`, "openai:gpt-4o-mini", "status 500", "anthropic:claude-haiku-4-5", "status 529"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
	if len(openai.calls) != 2 || len(anthropic.calls) != 2 {
		t.Errorf("calls: openai %d anthropic %d, want 2 each", len(openai.calls), len(anthropic.calls))
	}
}

func TestCompleteStopsWhenCallerCancels(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	openai := &fakeProvider{name: domain.ProviderOpenAI, script: []result{fail(domain.ProviderOpenAI, 503, true)}}
	openai.onCall = func(context.Context) { cancel() }
	anthropic := &fakeProvider{name: domain.ProviderAnthropic, script: []result{ok(domain.ProviderAnthropic)}}
	g := newGateway(t, Config{Routes: []domain.Route{fastRoute}}, openai, anthropic)

	_, err := g.Complete(ctx, routeRequest("fast"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if len(openai.calls) != 1 || len(anthropic.calls) != 0 {
		t.Errorf("calls: openai %d anthropic %d; a cancelled request must not retry or fall back",
			len(openai.calls), len(anthropic.calls))
	}
}

func TestCompleteStopsWhenBackoffIsInterrupted(t *testing.T) {
	openai := &fakeProvider{name: domain.ProviderOpenAI, script: []result{fail(domain.ProviderOpenAI, 503, true)}}
	g := newGateway(t, Config{MaxAttempts: 5}, openai)
	g.sleep = func(context.Context, time.Duration) error { return context.Canceled }

	req := routeRequest("gpt-4o")
	req.Provider = domain.ProviderOpenAI
	if _, err := g.Complete(context.Background(), req); err == nil {
		t.Fatal("Complete() succeeded, want the provider error")
	}
	if len(openai.calls) != 1 {
		t.Errorf("provider called %d times after an interrupted backoff, want 1", len(openai.calls))
	}
}

func TestCompleteAttemptTimeout(t *testing.T) {
	slow := &fakeProvider{name: domain.ProviderOpenAI, script: []result{
		{err: &domain.ProviderError{Provider: domain.ProviderOpenAI, Retryable: true, Err: context.DeadlineExceeded}},
		ok(domain.ProviderOpenAI),
	}}
	var deadlines []time.Duration
	slow.onCall = func(ctx context.Context) {
		deadline, hasDeadline := ctx.Deadline()
		if !hasDeadline {
			t.Error("provider called without a deadline")
		}
		deadlines = append(deadlines, time.Until(deadline))
	}
	g := newGateway(t, Config{AttemptTimeout: time.Minute}, slow)

	req := routeRequest("gpt-4o")
	req.Provider = domain.ProviderOpenAI
	if _, err := g.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete() error = %v; a per-attempt timeout must be retried", err)
	}
	for _, d := range deadlines {
		if d <= 0 || d > time.Minute {
			t.Errorf("attempt deadline %v, want within the configured minute", d)
		}
	}
}

func TestCompleteRejectsBadRequests(t *testing.T) {
	openai := &fakeProvider{name: domain.ProviderOpenAI, script: []result{ok(domain.ProviderOpenAI)}}
	g := newGateway(t, Config{}, openai)

	pinnedElsewhere := routeRequest("gemini-2.5-flash")
	pinnedElsewhere.Provider = domain.ProviderGemini
	tests := map[string]*domain.LLMRequest{
		"invalid request":         {Model: "gpt-4o"},
		"unknown route":           routeRequest("nope"),
		"provider not configured": pinnedElsewhere,
	}
	for name, req := range tests {
		t.Run(name, func(t *testing.T) {
			if _, err := g.Complete(context.Background(), req); !errors.Is(err, domain.ErrInvalidInput) {
				t.Errorf("expected ErrInvalidInput, got %v", err)
			}
		})
	}
	if len(openai.calls) != 0 {
		t.Errorf("provider called %d times for rejected requests", len(openai.calls))
	}
}

func TestCircuitBreakerSkipsFailingProvider(t *testing.T) {
	openai := &fakeProvider{name: domain.ProviderOpenAI, script: []result{fail(domain.ProviderOpenAI, 503, true)}}
	anthropic := &fakeProvider{name: domain.ProviderAnthropic, script: []result{ok(domain.ProviderAnthropic)}}
	g := newGateway(t, Config{Routes: []domain.Route{fastRoute}, BreakerThreshold: 3, BreakerCooldown: time.Minute}, openai, anthropic)
	now := time.Unix(1000, 0)
	g.breakers[domain.ProviderOpenAI].now = func() time.Time { return now }

	// Request 1 spends both attempts on openai (2 failures); request 2 fails
	// once more, which opens the breaker and cuts its retries short.
	for i := range 2 {
		resp, err := g.Complete(context.Background(), routeRequest("fast"))
		if err != nil || resp.Provider != domain.ProviderAnthropic {
			t.Fatalf("request %d: resp %+v, err %v; want the anthropic fallback", i+1, resp, err)
		}
	}
	if len(openai.calls) != 3 {
		t.Fatalf("openai called %d times, want 3 (the breaker opens on the third failure)", len(openai.calls))
	}
	if len(g.sleeps) != 1 {
		t.Errorf("%d backoffs, want 1: no backoff after the failure that opens the breaker", len(g.sleeps))
	}
	if got := g.Circuits(); got[domain.ProviderOpenAI] != CircuitOpen || got[domain.ProviderAnthropic] != CircuitClosed {
		t.Errorf("Circuits() = %v", got)
	}

	// While open, openai is not called at all and no backoff is spent on it.
	sleeps := len(g.sleeps)
	for range 3 {
		if _, err := g.Complete(context.Background(), routeRequest("fast")); err != nil {
			t.Fatalf("Complete() error = %v", err)
		}
	}
	if len(openai.calls) != 3 || len(g.sleeps) != sleeps {
		t.Errorf("openai calls %d, new backoffs %d; an open breaker must skip the provider entirely",
			len(openai.calls), len(g.sleeps)-sleeps)
	}

	// A pinned request to the broken provider fails fast with a clear cause.
	pinned := routeRequest("gpt-4o")
	pinned.Provider = domain.ProviderOpenAI
	_, err := g.Complete(context.Background(), pinned)
	var pe *domain.ProviderError
	if !errors.Is(err, ErrCircuitOpen) || !errors.As(err, &pe) || pe.Provider != domain.ProviderOpenAI {
		t.Errorf("pinned request error = %v, want a ProviderError wrapping ErrCircuitOpen", err)
	}
	if len(openai.calls) != 3 {
		t.Errorf("openai called for a pinned request while its breaker was open")
	}

	// After the cooldown one probe goes through; the provider has recovered.
	now = now.Add(time.Minute)
	openai.script = []result{ok(domain.ProviderOpenAI)}
	openai.calls = nil
	resp, err := g.Complete(context.Background(), routeRequest("fast"))
	if err != nil || resp.Provider != domain.ProviderOpenAI {
		t.Fatalf("after the cooldown: resp %+v, err %v; want openai again", resp, err)
	}
	if got := g.Circuits()[domain.ProviderOpenAI]; got != CircuitClosed {
		t.Errorf("circuit = %s after a healthy probe, want closed", got)
	}
}

func TestCircuitBreakerIgnoresRejectedRequests(t *testing.T) {
	openai := &fakeProvider{name: domain.ProviderOpenAI, script: []result{fail(domain.ProviderOpenAI, 400, false)}}
	g := newGateway(t, Config{BreakerThreshold: 2}, openai)

	req := routeRequest("gpt-4o")
	req.Provider = domain.ProviderOpenAI
	for range 5 {
		_, _ = g.Complete(context.Background(), req)
	}
	if len(openai.calls) != 5 || g.Circuits()[domain.ProviderOpenAI] != CircuitClosed {
		t.Errorf("calls %d, circuit %s; 4xx answers must not open the breaker",
			len(openai.calls), g.Circuits()[domain.ProviderOpenAI])
	}
}

func TestCircuitBreakerDisabled(t *testing.T) {
	openai := &fakeProvider{name: domain.ProviderOpenAI, script: []result{fail(domain.ProviderOpenAI, 503, true)}}
	g := newGateway(t, Config{BreakerThreshold: 1, DisableBreaker: true}, openai)

	req := routeRequest("gpt-4o")
	req.Provider = domain.ProviderOpenAI
	for range 3 {
		_, _ = g.Complete(context.Background(), req)
	}
	if len(openai.calls) != 6 {
		t.Errorf("openai called %d times, want all 6 attempts with the breaker disabled", len(openai.calls))
	}
	if got := g.Circuits(); len(got) != 0 {
		t.Errorf("Circuits() = %v, want none when disabled", got)
	}
}

func TestCompleteEstimatesCost(t *testing.T) {
	usage := domain.TokenUsage{InputTokens: 1_000_000, OutputTokens: 500_000}
	answer := func(p domain.Provider) result {
		return result{resp: &domain.LLMResponse{Provider: p, Model: "dated-variant-2026", Usage: usage}}
	}
	openai := &fakeProvider{name: domain.ProviderOpenAI, script: []result{fail(domain.ProviderOpenAI, 503, true)}}
	anthropic := &fakeProvider{name: domain.ProviderAnthropic, script: []result{answer(domain.ProviderAnthropic)}}
	g := newGateway(t, Config{
		Routes: []domain.Route{fastRoute},
		Prices: []domain.ModelPrice{
			{Target: fastRoute.Targets[0], Price: domain.Price{InputPerMTok: 100, OutputPerMTok: 100}},
			{Target: fastRoute.Targets[1], Price: domain.Price{InputPerMTok: 1, OutputPerMTok: 5}},
		},
	}, openai, anthropic)

	// The route falls back, so the cost must use the price of the target
	// that actually served the request.
	resp, err := g.Complete(context.Background(), routeRequest("fast"))
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if resp.CostUSD == nil || *resp.CostUSD != 3.5 {
		t.Errorf("cost = %v, want 3.5 (1 input + 2.5 output at the anthropic price)", resp.CostUSD)
	}
	if anthropic.script[0].resp.CostUSD != nil {
		t.Error("Complete() wrote the cost into the provider's own response")
	}

	// A model without a price yields no cost rather than a zero.
	unpriced := routeRequest("claude-opus-5-5")
	unpriced.Provider = domain.ProviderAnthropic
	resp, err = g.Complete(context.Background(), unpriced)
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if resp.CostUSD != nil {
		t.Errorf("cost = %v for an unpriced model, want none", *resp.CostUSD)
	}

	prices := g.Prices()
	if len(prices) != 2 || prices[0].Provider != domain.ProviderAnthropic || prices[1].Provider != domain.ProviderOpenAI {
		t.Errorf("Prices() = %+v, want both, ordered by target", prices)
	}
}

func TestNew(t *testing.T) {
	openai := &fakeProvider{name: domain.ProviderOpenAI}
	anthropic := &fakeProvider{name: domain.ProviderAnthropic}

	g, err := New([]domain.LLMProvider{openai, anthropic}, Config{Routes: []domain.Route{
		{Name: "smart", Targets: []domain.Target{{Provider: domain.ProviderAnthropic, Model: "claude-opus-5-5"}}},
		fastRoute,
	}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if g.maxAttempts != DefaultMaxAttempts || g.attemptTimeout != DefaultAttemptTimeout ||
		g.baseBackoff != DefaultBaseBackoff || g.maxBackoff != DefaultMaxBackoff || g.log == nil {
		t.Errorf("defaults not applied: %+v", g)
	}
	if b := g.breakers[domain.ProviderOpenAI]; b == nil || b.threshold != DefaultBreakerThreshold || b.cooldown != DefaultBreakerCooldown {
		t.Errorf("breaker defaults not applied: %+v", b)
	}
	if got := g.Providers(); len(got) != 2 || got[0] != domain.ProviderAnthropic || got[1] != domain.ProviderOpenAI {
		t.Errorf("Providers() = %v, want alphabetical order", got)
	}
	if got := g.Routes(); len(got) != 2 || got[0].Name != "fast" || got[1].Name != "smart" {
		t.Errorf("Routes() = %v, want ordered by name", got)
	}

	tests := []struct {
		name      string
		providers []domain.LLMProvider
		cfg       Config
	}{
		{name: "negative limit", cfg: Config{MaxAttempts: -1}},
		{name: "negative breaker threshold", cfg: Config{BreakerThreshold: -1}},
		{name: "unknown provider", providers: []domain.LLMProvider{&fakeProvider{name: "cohere"}}},
		{name: "duplicate provider", providers: []domain.LLMProvider{openai, openai}},
		{name: "invalid route", providers: []domain.LLMProvider{openai}, cfg: Config{Routes: []domain.Route{{Name: "empty"}}}},
		{name: "duplicate route", providers: []domain.LLMProvider{openai, anthropic}, cfg: Config{Routes: []domain.Route{fastRoute, fastRoute}}},
		{name: "route to missing provider", providers: []domain.LLMProvider{openai}, cfg: Config{Routes: []domain.Route{fastRoute}}},
		{name: "negative price", cfg: Config{Prices: []domain.ModelPrice{{Target: fastRoute.Targets[0], Price: domain.Price{InputPerMTok: -1}}}}},
		{name: "price without a model", cfg: Config{Prices: []domain.ModelPrice{{Target: domain.Target{Provider: domain.ProviderOpenAI}}}}},
		{name: "duplicate price", cfg: Config{Prices: []domain.ModelPrice{{Target: fastRoute.Targets[0]}, {Target: fastRoute.Targets[0]}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := New(tt.providers, tt.cfg); !errors.Is(err, domain.ErrInvalidInput) {
				t.Errorf("expected ErrInvalidInput, got %v", err)
			}
		})
	}
}

func TestBackoffJitter(t *testing.T) {
	g := newGateway(t, Config{BaseBackoff: 100 * time.Millisecond, MaxBackoff: time.Second})
	g.jitter = func() float64 { return 0 }
	if got := g.backoff(1); got != 50*time.Millisecond {
		t.Errorf("backoff(1) with minimum jitter = %v, want 50ms", got)
	}
	g.jitter = func() float64 { return 1 }
	if got := g.backoff(10); got != time.Second {
		t.Errorf("backoff(10) = %v, want the 1s cap", got)
	}
}

func TestSleep(t *testing.T) {
	if err := sleep(context.Background(), time.Millisecond); err != nil {
		t.Errorf("sleep() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := sleep(ctx, time.Hour); !errors.Is(err, context.Canceled) {
		t.Errorf("sleep() on a cancelled context = %v, want context.Canceled", err)
	}
}
