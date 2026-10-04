package gateway

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

// ErrCircuitOpen is the cause of a *domain.ProviderError returned without
// calling the provider, because the circuit breaker of the provider or of the
// requested model is open.
var ErrCircuitOpen = errors.New("circuit breaker is open")

// Circuit breaker states, as reported by Gateway.Circuits.
const (
	CircuitClosed   = "closed"
	CircuitOpen     = "open"
	CircuitHalfOpen = "half_open"
)

// outcome is what a call tells a breaker about the thing it guards.
type outcome int

const (
	// outcomeHealthy resets the breaker's failure count and closes it.
	outcomeHealthy outcome = iota
	// outcomeUnhealthy counts towards opening the breaker.
	outcomeUnhealthy
	// outcomeUnknown says nothing either way and leaves the count alone.
	outcomeUnknown
)

// breaker is a consecutive-failure circuit breaker.
//
// Closed, it lets every call through and counts consecutive unhealthy
// outcomes. On reaching the threshold it opens and rejects calls for the
// cooldown. After that it is half-open: a single probe call goes through; a
// healthy probe closes the breaker and an unhealthy one reopens it for
// another cooldown.
type breaker struct {
	threshold int
	cooldown  time.Duration
	now       func() time.Time

	mu       sync.Mutex
	failures int
	open     bool
	openedAt time.Time
	// probing is set while the half-open probe call is in flight.
	probing bool
}

// allow reports whether a call may proceed. A true result must be followed by
// exactly one call to record.
func (b *breaker) allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !b.open {
		return true
	}
	if b.probing || b.now().Sub(b.openedAt) < b.cooldown {
		return false
	}
	b.probing = true
	return true
}

// record feeds the result of an allowed call back into the breaker.
func (b *breaker) record(o outcome) {
	b.mu.Lock()
	defer b.mu.Unlock()

	wasProbe := b.probing
	b.probing = false
	switch o {
	case outcomeHealthy:
		b.failures = 0
		b.open = false
	case outcomeUnhealthy:
		b.failures++
		if wasProbe || b.failures >= b.threshold {
			b.open = true
			b.openedAt = b.now()
		}
	case outcomeUnknown:
		// An abandoned probe leaves the breaker open and immediately
		// eligible for another probe.
	}
}

// state returns the breaker's current state name.
func (b *breaker) state() string {
	b.mu.Lock()
	defer b.mu.Unlock()

	switch {
	case !b.open:
		return CircuitClosed
	case b.probing || b.now().Sub(b.openedAt) >= b.cooldown:
		return CircuitHalfOpen
	default:
		return CircuitOpen
	}
}

// maxModelBreakers bounds the per-model breakers kept at once. Model names
// come from requests, so without a cap a client could grow the map without
// limit by failing against endless made-up names. Past the cap, new models
// are covered by their provider's breaker only.
const maxModelBreakers = 1024

// breakerSet holds the gateway's two levels of circuit breakers.
//
// A provider-level breaker counts only failures to reach the provider at all
// (connection refused, DNS, reset): those say the vendor is unreachable
// whatever the model. A model-level breaker counts every unhealthy call to
// one (provider, model) target, including the 429s and 5xx that typically hit
// a single model while the vendor's other models keep answering.
//
// A call goes through only when both levels allow it.
type breakerSet struct {
	threshold int
	cooldown  time.Duration
	now       func() time.Time

	// providers is fixed at construction, one breaker per provider.
	providers map[domain.Provider]*breaker

	// models holds a breaker only for targets with recent failures: one is
	// created on a target's first unhealthy call and dropped on its next
	// healthy one.
	mu     sync.Mutex
	models map[domain.Target]*breaker
}

func newBreakerSet(threshold int, cooldown time.Duration, now func() time.Time, providers []domain.Provider) *breakerSet {
	s := &breakerSet{
		threshold: threshold,
		cooldown:  cooldown,
		now:       now,
		providers: make(map[domain.Provider]*breaker, len(providers)),
		models:    make(map[domain.Target]*breaker),
	}
	for _, p := range providers {
		s.providers[p] = s.newBreaker()
	}
	return s
}

func (s *breakerSet) newBreaker() *breaker {
	return &breaker{threshold: s.threshold, cooldown: s.cooldown, now: s.now}
}

// model returns the breaker of target, creating it when create is set and
// the cap allows. It returns nil when there is none.
func (s *breakerSet) model(target domain.Target, create bool) *breaker {
	s.mu.Lock()
	defer s.mu.Unlock()

	b := s.models[target]
	if b == nil && create && len(s.models) < maxModelBreakers {
		b = s.newBreaker()
		s.models[target] = b
	}
	return b
}

// allow reports whether a call to target may proceed. When it may not, the
// error says which breaker is open and wraps ErrCircuitOpen. A nil result
// must be followed by exactly one call to record.
func (s *breakerSet) allow(target domain.Target) error {
	model := s.model(target, false)
	if model != nil && !model.allow() {
		return fmt.Errorf("%w for model %s", ErrCircuitOpen, target.Model)
	}
	if !s.providers[target.Provider].allow() {
		if model != nil {
			// Hand back the probe slot the model breaker may have granted.
			model.record(outcomeUnknown)
		}
		return fmt.Errorf("%w for provider %s", ErrCircuitOpen, target.Provider)
	}
	return nil
}

// record feeds the result of an allowed call to both levels.
func (s *breakerSet) record(ctx context.Context, target domain.Target, err error) {
	modelOutcome, providerOutcome := classify(ctx, err)
	s.providers[target.Provider].record(providerOutcome)

	model := s.model(target, modelOutcome == outcomeUnhealthy)
	if model == nil {
		return
	}
	model.record(modelOutcome)
	if modelOutcome == outcomeHealthy {
		s.mu.Lock()
		// Only drop the entry if it is still this breaker; a concurrent
		// failure may have replaced it.
		if s.models[target] == model {
			delete(s.models, target)
		}
		s.mu.Unlock()
	}
}

// open reports whether calls to target are being rejected right now.
func (s *breakerSet) open(target domain.Target) bool {
	if s.providers[target.Provider].state() == CircuitOpen {
		return true
	}
	model := s.model(target, false)
	return model != nil && model.state() == CircuitOpen
}

// providerStates returns the state of every provider-level breaker.
func (s *breakerSet) providerStates() map[domain.Provider]string {
	states := make(map[domain.Provider]string, len(s.providers))
	for name, b := range s.providers {
		states[name] = b.state()
	}
	return states
}

// modelStates returns the state of every model-level breaker, keyed by
// "provider:model". Targets without recent failures have no entry.
func (s *breakerSet) modelStates() map[string]string {
	s.mu.Lock()
	defer s.mu.Unlock()

	states := make(map[string]string, len(s.models))
	for target, b := range s.models {
		states[target.String()] = b.state()
	}
	return states
}

// classify turns the result of a provider call into what it says about the
// health of the model and of the provider as a whole.
func classify(ctx context.Context, err error) (model, provider outcome) {
	switch {
	case err == nil:
		return outcomeHealthy, outcomeHealthy
	case ctx.Err() != nil:
		// The caller gave up first.
		return outcomeUnknown, outcomeUnknown
	case !domain.IsRetryable(err):
		// The provider answered and rejected this particular request.
		return outcomeHealthy, outcomeHealthy
	}

	var providerErr *domain.ProviderError
	errors.As(err, &providerErr) // always succeeds: IsRetryable found one
	switch {
	case providerErr.StatusCode != 0:
		// An HTTP answer (429, 5xx) proves the vendor is reachable.
		return outcomeUnhealthy, outcomeHealthy
	case errors.Is(err, context.DeadlineExceeded):
		// A slow model, which is not evidence about the vendor's other
		// models either way.
		return outcomeUnhealthy, outcomeUnknown
	default:
		// No answer at all: connection refused, DNS failure, reset.
		return outcomeUnhealthy, outcomeUnhealthy
	}
}
