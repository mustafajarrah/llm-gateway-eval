package gateway

import (
	"context"
	"errors"
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
	retryable := &domain.ProviderError{Retryable: true, Err: errors.New("503")}
	rejected := &domain.ProviderError{StatusCode: 400, Err: errors.New("bad request")}

	tests := []struct {
		name string
		ctx  context.Context
		err  error
		want outcome
	}{
		{name: "success", ctx: context.Background(), want: outcomeHealthy},
		{name: "retryable failure", ctx: context.Background(), err: retryable, want: outcomeUnhealthy},
		{name: "request rejected", ctx: context.Background(), err: rejected, want: outcomeHealthy},
		{name: "caller gave up", ctx: cancelled, err: retryable, want: outcomeUnknown},
	}
	for _, tt := range tests {
		if got := classify(tt.ctx, tt.err); got != tt.want {
			t.Errorf("%s: classify() = %v, want %v", tt.name, got, tt.want)
		}
	}
}
