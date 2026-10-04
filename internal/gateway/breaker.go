package gateway

import (
	"errors"
	"sync"
	"time"
)

// ErrCircuitOpen is the cause of a *domain.ProviderError returned without
// calling the provider, because its circuit breaker is open.
var ErrCircuitOpen = errors.New("circuit breaker is open")

// Circuit breaker states, as reported by Gateway.Circuits.
const (
	CircuitClosed   = "closed"
	CircuitOpen     = "open"
	CircuitHalfOpen = "half_open"
)

// outcome is what a provider call tells the breaker about the provider.
type outcome int

const (
	// outcomeHealthy: the provider answered, with a completion or with an
	// error that is the request's fault (4xx).
	outcomeHealthy outcome = iota
	// outcomeUnhealthy: timeout, 429, 5xx or transport failure.
	outcomeUnhealthy
	// outcomeUnknown: the caller gave up first, which says nothing about
	// the provider.
	outcomeUnknown
)

// breaker is a consecutive-failure circuit breaker for one provider.
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
