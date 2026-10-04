package gateway

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

// clock is a manually advanced time source.
type clock struct{ t time.Time }

func (c *clock) now() time.Time          { return c.t }
func (c *clock) advance(d time.Duration) { c.t = c.t.Add(d) }

func newBreaker(threshold int, cooldown time.Duration) (*breaker, *clock) {
	c := &clock{t: time.Unix(1000, 0)}
	return &breaker{threshold: threshold, cooldown: cooldown, now: c.now}, c
}

func TestBreakerOpensAfterConsecutiveFailures(t *testing.T) {
	b, _ := newBreaker(3, time.Minute)

	for i := range 2 {
		if !b.allow() {
			t.Fatalf("call %d rejected while closed", i+1)
		}
		b.record(outcomeUnhealthy)
	}
	// A healthy call in between resets the count.
	b.allow()
	b.record(outcomeHealthy)
	for range 2 {
		b.allow()
		b.record(outcomeUnhealthy)
	}
	if got := b.state(); got != CircuitClosed {
		t.Fatalf("state = %s after 2 consecutive failures, want closed", got)
	}

	b.allow()
	b.record(outcomeUnhealthy)
	if got := b.state(); got != CircuitOpen {
		t.Fatalf("state = %s after 3 consecutive failures, want open", got)
	}
	if b.allow() {
		t.Error("an open breaker allowed a call")
	}
}

func TestBreakerProbe(t *testing.T) {
	trip := func() (*breaker, *clock) {
		b, c := newBreaker(1, time.Minute)
		b.allow()
		b.record(outcomeUnhealthy)
		return b, c
	}

	t.Run("stays open during the cooldown", func(t *testing.T) {
		b, c := trip()
		c.advance(59 * time.Second)
		if b.allow() || b.state() != CircuitOpen {
			t.Errorf("allowed a call %v into a 1m cooldown (state %s)", 59*time.Second, b.state())
		}
	})

	t.Run("lets exactly one probe through", func(t *testing.T) {
		b, c := trip()
		c.advance(time.Minute)
		if got := b.state(); got != CircuitHalfOpen {
			t.Errorf("state = %s after the cooldown, want half_open", got)
		}
		if !b.allow() {
			t.Fatal("probe rejected after the cooldown")
		}
		if b.allow() {
			t.Error("a second call was allowed while the probe is in flight")
		}
		if got := b.state(); got != CircuitHalfOpen {
			t.Errorf("state = %s while probing, want half_open", got)
		}
	})

	t.Run("healthy probe closes", func(t *testing.T) {
		b, c := trip()
		c.advance(time.Minute)
		b.allow()
		b.record(outcomeHealthy)
		if b.state() != CircuitClosed || !b.allow() {
			t.Errorf("state = %s after a healthy probe, want closed", b.state())
		}
	})

	t.Run("unhealthy probe reopens for a full cooldown", func(t *testing.T) {
		b, c := newBreaker(3, time.Minute)
		for range 3 {
			b.allow()
			b.record(outcomeUnhealthy)
		}
		c.advance(time.Minute)
		b.allow()
		b.record(outcomeUnhealthy)
		c.advance(59 * time.Second)
		if b.allow() {
			t.Error("allowed a call before the second cooldown elapsed")
		}
		c.advance(time.Second)
		if !b.allow() {
			t.Error("no probe allowed after the second cooldown")
		}
	})

	t.Run("abandoned probe can be retried at once", func(t *testing.T) {
		b, c := trip()
		c.advance(time.Minute)
		b.allow()
		b.record(outcomeUnknown)
		if b.state() != CircuitHalfOpen || !b.allow() {
			t.Errorf("state = %s after an abandoned probe, want another probe to be allowed", b.state())
		}
	})
}

func TestBreakerIgnoresUnknownOutcomes(t *testing.T) {
	b, _ := newBreaker(2, time.Minute)
	b.allow()
	b.record(outcomeUnhealthy)
	b.allow()
	b.record(outcomeUnknown)
	if got := b.state(); got != CircuitClosed {
		t.Errorf("state = %s; a cancelled call must not count as a failure", got)
	}
}

func TestClassify(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	live := context.Background()

	tests := []struct {
		name         string
		ctx          context.Context
		err          error
		model, whole outcome
	}{
		{name: "success", ctx: live, model: outcomeHealthy, whole: outcomeHealthy},
		{name: "rate limited", ctx: live, model: outcomeUnhealthy, whole: outcomeHealthy,
			err: &domain.ProviderError{StatusCode: 429, Retryable: true, Err: errors.New("slow down")}},
		{name: "server error", ctx: live, model: outcomeUnhealthy, whole: outcomeHealthy,
			err: &domain.ProviderError{StatusCode: 503, Retryable: true, Err: errors.New("overloaded")}},
		{name: "attempt timed out", ctx: live, model: outcomeUnhealthy, whole: outcomeUnknown,
			err: &domain.ProviderError{Retryable: true, Err: context.DeadlineExceeded}},
		{name: "connection refused", ctx: live, model: outcomeUnhealthy, whole: outcomeUnhealthy,
			err: fmt.Errorf("wrapped: %w", &domain.ProviderError{Retryable: true, Err: errors.New("connection refused")})},
		{name: "request rejected", ctx: live, model: outcomeHealthy, whole: outcomeHealthy,
			err: &domain.ProviderError{StatusCode: 400, Err: errors.New("bad request")}},
		{name: "not a provider error", ctx: live, model: outcomeHealthy, whole: outcomeHealthy,
			err: errors.New("unexpected")},
		{name: "caller gave up", ctx: cancelled, model: outcomeUnknown, whole: outcomeUnknown,
			err: &domain.ProviderError{Retryable: true, Err: errors.New("connection refused")}},
	}
	for _, tt := range tests {
		model, whole := classify(tt.ctx, tt.err)
		if model != tt.model || whole != tt.whole {
			t.Errorf("%s: classify() = (%v, %v), want (%v, %v)", tt.name, model, whole, tt.model, tt.whole)
		}
	}
}

func TestBreakerSetModelBreakersAreTransient(t *testing.T) {
	c := &clock{t: time.Unix(1000, 0)}
	s := newBreakerSet(2, time.Minute, c.now, []domain.Provider{domain.ProviderOpenAI})
	target := domain.Target{Provider: domain.ProviderOpenAI, Model: "gpt-4o"}
	live := context.Background()
	overloaded := &domain.ProviderError{StatusCode: 503, Retryable: true, Err: errors.New("overloaded")}

	// A healthy model never gets an entry.
	if err := s.allow(target); err != nil {
		t.Fatalf("allow() error = %v", err)
	}
	s.record(live, target, nil)
	if got := s.modelStates(); len(got) != 0 {
		t.Errorf("modelStates() = %v after a healthy call, want none", got)
	}

	// One failure creates it; the next healthy call drops it again.
	s.allow(target)
	s.record(live, target, overloaded)
	if got := s.modelStates(); got["openai:gpt-4o"] != CircuitClosed {
		t.Errorf("modelStates() = %v, want a closed breaker with one failure", got)
	}
	s.allow(target)
	s.record(live, target, nil)
	if got := s.modelStates(); len(got) != 0 {
		t.Errorf("modelStates() = %v after recovery, want none", got)
	}
}

func TestBreakerSetIsBounded(t *testing.T) {
	c := &clock{t: time.Unix(1000, 0)}
	s := newBreakerSet(1, time.Minute, c.now, []domain.Provider{domain.ProviderOpenAI})
	live := context.Background()
	overloaded := &domain.ProviderError{StatusCode: 503, Retryable: true, Err: errors.New("overloaded")}

	for i := range maxModelBreakers + 50 {
		target := domain.Target{Provider: domain.ProviderOpenAI, Model: fmt.Sprintf("made-up-%d", i)}
		if err := s.allow(target); err != nil {
			t.Fatalf("allow(%s) error = %v", target, err)
		}
		s.record(live, target, overloaded)
	}
	if got := len(s.modelStates()); got != maxModelBreakers {
		t.Errorf("%d model breakers, want the cap of %d", got, maxModelBreakers)
	}
	// Past the cap a model has no breaker of its own and stays callable.
	beyond := domain.Target{Provider: domain.ProviderOpenAI, Model: fmt.Sprintf("made-up-%d", maxModelBreakers+10)}
	if err := s.allow(beyond); err != nil {
		t.Errorf("allow() past the cap error = %v", err)
	}
	s.record(live, beyond, nil)
}

func TestBreakerSetReleasesModelProbeWhenProviderIsOpen(t *testing.T) {
	c := &clock{t: time.Unix(1000, 0)}
	s := newBreakerSet(1, time.Minute, c.now, []domain.Provider{domain.ProviderOpenAI})
	target := domain.Target{Provider: domain.ProviderOpenAI, Model: "gpt-4o"}
	live := context.Background()
	refused := &domain.ProviderError{Retryable: true, Err: errors.New("connection refused")}

	// A connection failure opens both levels.
	s.allow(target)
	s.record(live, target, refused)
	if !s.open(target) || s.providerStates()[domain.ProviderOpenAI] != CircuitOpen || s.modelStates()["openai:gpt-4o"] != CircuitOpen {
		t.Fatalf("after a connection failure: provider %v, models %v", s.providerStates(), s.modelStates())
	}

	// Make only the model breaker eligible for a probe by reopening the
	// provider's later.
	c.advance(30 * time.Second)
	other := domain.Target{Provider: domain.ProviderOpenAI, Model: "gpt-4o-mini"}
	s.providers[domain.ProviderOpenAI].openedAt = c.now()
	c.advance(30 * time.Second)

	err := s.allow(target)
	if !errors.Is(err, ErrCircuitOpen) || err.Error() != "circuit breaker is open for provider openai" {
		t.Fatalf("allow() error = %v, want the provider breaker to reject", err)
	}
	if got := s.modelStates()["openai:gpt-4o"]; got != CircuitHalfOpen {
		t.Errorf("model breaker = %s, want half_open", got)
	}
	// The model's probe slot was handed back, so once the provider recovers
	// the model can be probed.
	c.advance(30 * time.Second)
	if err := s.allow(target); err != nil {
		t.Errorf("allow() after the provider cooldown error = %v, want the probe to go through", err)
	}
	s.record(live, target, nil)
	if s.open(target) || s.open(other) {
		t.Error("breakers still open after a healthy probe")
	}
}
