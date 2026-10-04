// Package gateway routes completion requests to provider adapters, retrying
// transient failures and falling back along configured routes.
package gateway

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"sort"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

// Defaults applied by New when the corresponding Config field is zero.
const (
	DefaultMaxAttempts    = 2
	DefaultAttemptTimeout = 60 * time.Second
	DefaultBaseBackoff    = 250 * time.Millisecond
	DefaultMaxBackoff     = 5 * time.Second
	// DefaultBreakerThreshold is how many consecutive unhealthy calls open a
	// provider's circuit breaker, and DefaultBreakerCooldown how long it then
	// stays open before a probe is let through.
	DefaultBreakerThreshold = 5
	DefaultBreakerCooldown  = 30 * time.Second
)

// Config configures a Gateway.
type Config struct {
	// Routes are the named fallback chains. Every target must name a
	// registered provider.
	Routes []domain.Route
	// Prices are used to estimate the cost of each completion. A target
	// without a price yields responses without a cost.
	Prices []domain.ModelPrice
	// MaxAttempts is how many times one target is tried before moving on.
	MaxAttempts int
	// AttemptTimeout bounds a single call to a provider.
	AttemptTimeout time.Duration
	// BaseBackoff is the wait before the second attempt on a target; it
	// doubles on each further attempt, up to MaxBackoff.
	BaseBackoff time.Duration
	MaxBackoff  time.Duration
	// BreakerThreshold is the number of consecutive unhealthy calls after
	// which a circuit breaker opens and calls fail immediately. There is a
	// breaker per (provider, model) target, fed by every timeout, 429, 5xx
	// and transport failure, and one per provider, fed only by failures to
	// reach the provider at all.
	BreakerThreshold int
	// BreakerCooldown is how long an open breaker rejects calls before
	// letting a single probe through.
	BreakerCooldown time.Duration
	// DisableBreaker turns circuit breaking off.
	DisableBreaker bool
	// Logger defaults to slog.Default().
	Logger *slog.Logger
}

// Gateway dispatches requests to providers. It is safe for concurrent use.
type Gateway struct {
	providers map[domain.Provider]domain.LLMProvider
	// breakers is nil when circuit breaking is disabled.
	breakers       *breakerSet
	routes         map[string]domain.Route
	prices         map[domain.Target]domain.Price
	maxAttempts    int
	attemptTimeout time.Duration
	baseBackoff    time.Duration
	maxBackoff     time.Duration
	log            *slog.Logger

	// sleep and jitter are replaced in tests.
	sleep  func(ctx context.Context, d time.Duration) error
	jitter func() float64
}

// New builds a Gateway over providers.
func New(providers []domain.LLMProvider, cfg Config) (*Gateway, error) {
	g := &Gateway{
		providers:      make(map[domain.Provider]domain.LLMProvider, len(providers)),
		routes:         make(map[string]domain.Route, len(cfg.Routes)),
		prices:         make(map[domain.Target]domain.Price, len(cfg.Prices)),
		maxAttempts:    cfg.MaxAttempts,
		attemptTimeout: cfg.AttemptTimeout,
		baseBackoff:    cfg.BaseBackoff,
		maxBackoff:     cfg.MaxBackoff,
		log:            cfg.Logger,
		sleep:          sleep,
		jitter:         rand.Float64,
	}
	if cfg.MaxAttempts < 0 || cfg.AttemptTimeout < 0 || cfg.BaseBackoff < 0 || cfg.MaxBackoff < 0 ||
		cfg.BreakerThreshold < 0 || cfg.BreakerCooldown < 0 {
		return nil, fmt.Errorf("%w: gateway limits must not be negative", domain.ErrInvalidInput)
	}
	if g.maxAttempts == 0 {
		g.maxAttempts = DefaultMaxAttempts
	}
	if g.attemptTimeout == 0 {
		g.attemptTimeout = DefaultAttemptTimeout
	}
	if g.baseBackoff == 0 {
		g.baseBackoff = DefaultBaseBackoff
	}
	if g.maxBackoff == 0 {
		g.maxBackoff = DefaultMaxBackoff
	}
	if g.log == nil {
		g.log = slog.Default()
	}
	threshold, cooldown := cfg.BreakerThreshold, cfg.BreakerCooldown
	if threshold == 0 {
		threshold = DefaultBreakerThreshold
	}
	if cooldown == 0 {
		cooldown = DefaultBreakerCooldown
	}

	for _, p := range providers {
		name := p.Name()
		if !name.Valid() {
			return nil, fmt.Errorf("%w: unknown provider %q", domain.ErrInvalidInput, name)
		}
		if _, dup := g.providers[name]; dup {
			return nil, fmt.Errorf("%w: provider %s registered twice", domain.ErrInvalidInput, name)
		}
		g.providers[name] = p
	}
	if !cfg.DisableBreaker {
		g.breakers = newBreakerSet(threshold, cooldown, time.Now, g.Providers())
	}
	for _, route := range cfg.Routes {
		if err := route.Validate(); err != nil {
			return nil, err
		}
		if _, dup := g.routes[route.Name]; dup {
			return nil, fmt.Errorf("%w: route %q defined twice", domain.ErrInvalidInput, route.Name)
		}
		for _, target := range route.Targets {
			if _, ok := g.providers[target.Provider]; !ok {
				return nil, fmt.Errorf("%w: route %q uses provider %s, which is not configured",
					domain.ErrInvalidInput, route.Name, target.Provider)
			}
		}
		g.routes[route.Name] = route
	}
	for _, price := range cfg.Prices {
		if err := price.Validate(); err != nil {
			return nil, fmt.Errorf("price for %s: %w", price.Target, err)
		}
		if _, dup := g.prices[price.Target]; dup {
			return nil, fmt.Errorf("%w: price for %s defined twice", domain.ErrInvalidInput, price.Target)
		}
		g.prices[price.Target] = price.Price
	}
	return g, nil
}

// Providers returns the configured providers in alphabetical order.
func (g *Gateway) Providers() []domain.Provider {
	names := make([]domain.Provider, 0, len(g.providers))
	for name := range g.providers {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool { return names[i] < names[j] })
	return names
}

// Prices returns the configured prices ordered by provider, then model.
func (g *Gateway) Prices() []domain.ModelPrice {
	prices := make([]domain.ModelPrice, 0, len(g.prices))
	for target, price := range g.prices {
		prices = append(prices, domain.ModelPrice{Target: target, Price: price})
	}
	sort.Slice(prices, func(i, j int) bool { return prices[i].Target.String() < prices[j].Target.String() })
	return prices
}

// Circuits returns the state of every provider-level circuit breaker:
// CircuitClosed, CircuitOpen or CircuitHalfOpen. It is empty when breaking is
// disabled.
func (g *Gateway) Circuits() map[domain.Provider]string {
	if g.breakers == nil {
		return map[domain.Provider]string{}
	}
	return g.breakers.providerStates()
}

// ModelCircuits returns the state of the model-level circuit breakers, keyed
// by "provider:model". Only targets with recent failures are listed; a target
// that is absent is closed.
func (g *Gateway) ModelCircuits() map[string]string {
	if g.breakers == nil {
		return map[string]string{}
	}
	return g.breakers.modelStates()
}

// Routes returns the configured routes ordered by name.
func (g *Gateway) Routes() []domain.Route {
	routes := make([]domain.Route, 0, len(g.routes))
	for _, route := range g.routes {
		routes = append(routes, route)
	}
	sort.Slice(routes, func(i, j int) bool { return routes[i].Name < routes[j].Name })
	return routes
}

// Complete validates req and sends it to a provider.
//
// A request that pins a provider goes to that provider only. Otherwise
// req.Model names a route, whose targets are tried in order until one
// succeeds. Each target gets up to MaxAttempts tries, with exponential backoff
// between them, as long as its failures are retryable; any other failure moves
// straight to the next target. Along a route the temperature is clamped to
// each target's limit.
//
// Circuit breakers guard each (provider, model) target and each provider.
// After enough consecutive unhealthy calls a breaker opens, and until its
// cooldown passes the target is skipped without being called, so a route
// moves on to its next target at once instead of spending retries and backoff
// on something known to be failing.
//
// A successful response carries an estimated cost when a price is configured
// for the target that served it.
//
// When every target fails, the returned error wraps the *domain.ProviderError
// of each one.
func (g *Gateway) Complete(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	targets, routed, err := g.resolve(req)
	if err != nil {
		return nil, err
	}

	var failures []error
	for _, target := range targets {
		attemptReq := *req
		attemptReq.Provider = target.Provider
		attemptReq.Model = target.Model
		if routed {
			attemptReq.Parameters = req.Parameters.ClampFor(target.Provider)
		}

		resp, err := g.tryTarget(ctx, g.providers[target.Provider], &attemptReq)
		if err == nil {
			return g.withCost(resp, target), nil
		}
		if ctx.Err() != nil {
			// The caller gave up; do not burn the remaining targets.
			return nil, fmt.Errorf("%s: %w", target, context.Cause(ctx))
		}
		failures = append(failures, fmt.Errorf("%s: %w", target, err))
	}
	if !routed {
		return nil, failures[0]
	}
	return nil, fmt.Errorf("route %q: all %d targets failed: %w", req.Model, len(targets), errors.Join(failures...))
}

// withCost returns resp with its estimated cost, when the target that served
// it has a price. The price is looked up by the model that was requested from
// the provider, not the one it reports back, which is often a dated variant
// of the same model.
func (g *Gateway) withCost(resp *domain.LLMResponse, target domain.Target) *domain.LLMResponse {
	price, ok := g.prices[target]
	if !ok {
		return resp
	}
	priced := *resp
	cost := price.Cost(resp.Usage)
	priced.CostUSD = &cost
	return &priced
}

// resolve turns a request into the ordered targets to try. routed reports
// whether they came from a route rather than a pinned provider.
func (g *Gateway) resolve(req *domain.LLMRequest) (targets []domain.Target, routed bool, err error) {
	if req.Provider != "" {
		if _, ok := g.providers[req.Provider]; !ok {
			return nil, false, fmt.Errorf("%w: provider %s is not configured", domain.ErrInvalidInput, req.Provider)
		}
		return []domain.Target{{Provider: req.Provider, Model: req.Model}}, false, nil
	}
	route, ok := g.routes[req.Model]
	if !ok {
		return nil, false, fmt.Errorf("%w: no route named %q; set provider to address a vendor model directly",
			domain.ErrInvalidInput, req.Model)
	}
	return route.Targets, true, nil
}

// tryTarget calls one provider, retrying retryable failures for as long as the
// circuit breakers allow.
func (g *Gateway) tryTarget(ctx context.Context, provider domain.LLMProvider, req *domain.LLMRequest) (*domain.LLMResponse, error) {
	target := domain.Target{Provider: req.Provider, Model: req.Model}
	var lastErr error
	for attempt := 1; attempt <= g.maxAttempts; attempt++ {
		if g.breakers != nil {
			if err := g.breakers.allow(target); err != nil {
				if lastErr == nil {
					lastErr = &domain.ProviderError{Provider: req.Provider, Retryable: true, Err: err}
				}
				g.log.WarnContext(ctx, "target skipped", "provider", req.Provider, "model", req.Model, "reason", err)
				break
			}
		}
		resp, err := g.attempt(ctx, provider, req)
		if g.breakers != nil {
			g.breakers.record(ctx, target, err)
		}
		if err == nil {
			return resp, nil
		}
		lastErr = err
		g.log.WarnContext(ctx, "provider call failed",
			"provider", req.Provider, "model", req.Model, "attempt", attempt,
			"retryable", domain.IsRetryable(err), "error", err)

		if ctx.Err() != nil || !domain.IsRetryable(err) || attempt == g.maxAttempts {
			break
		}
		if g.breakers != nil && g.breakers.open(target) {
			// This failure opened a breaker; waiting to retry is pointless.
			break
		}
		if err := g.sleep(ctx, g.backoff(attempt)); err != nil {
			break
		}
	}
	return nil, lastErr
}

// attempt performs a single provider call under the per-attempt timeout.
func (g *Gateway) attempt(ctx context.Context, provider domain.LLMProvider, req *domain.LLMRequest) (*domain.LLMResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, g.attemptTimeout)
	defer cancel()
	return provider.Complete(ctx, req)
}

// backoff returns the wait after the given failed attempt (1-based):
// exponential, capped at maxBackoff, with jitter in [50%, 100%] so that
// concurrent requests do not retry in lockstep.
func (g *Gateway) backoff(attempt int) time.Duration {
	d := g.baseBackoff
	for i := 1; i < attempt && d < g.maxBackoff; i++ {
		d *= 2
	}
	if d > g.maxBackoff {
		d = g.maxBackoff
	}
	return time.Duration(float64(d) * (0.5 + g.jitter()/2))
}

// sleep waits for d or until ctx is done.
func sleep(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
