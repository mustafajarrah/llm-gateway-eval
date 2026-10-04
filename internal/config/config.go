// Package config reads the service configuration from environment variables.
package config

import (
	"errors"
	"fmt"
	"log/slog"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

// Defaults for settings that are not set in the environment.
const (
	// DefaultAddr listens on loopback only: the API can spend money on the
	// configured provider keys, so exposing it is an explicit decision.
	DefaultAddr   = "127.0.0.1:8080"
	DefaultDBPath = "data/gateway.db"
)

// Provider holds the connection settings of one upstream provider.
type Provider struct {
	Name    domain.Provider
	APIKey  string
	BaseURL string
}

// Config is the full service configuration.
type Config struct {
	// Addr is the listen address (GATEWAY_ADDR).
	Addr string
	// APIKey is the bearer token clients must present (GATEWAY_API_KEY);
	// empty disables authentication.
	APIKey string
	// DBPath is the SQLite file, or ":memory:" (GATEWAY_DB_PATH).
	DBPath string
	// LogLevel is the minimum level logged (GATEWAY_LOG_LEVEL).
	LogLevel slog.Level

	// Providers lists the enabled providers. A provider is enabled by
	// setting its API key or, for the keyless ones, its base URL.
	Providers []Provider
	// AnthropicMaxTokens is the default output limit for Anthropic requests
	// that do not set one (ANTHROPIC_MAX_TOKENS); 0 keeps the adapter default.
	AnthropicMaxTokens int

	// Routes are the fallback chains (GATEWAY_ROUTES).
	Routes []domain.Route
	// MaxAttempts and AttemptTimeout tune retries (GATEWAY_MAX_ATTEMPTS,
	// GATEWAY_ATTEMPT_TIMEOUT); zero keeps the gateway defaults.
	MaxAttempts    int
	AttemptTimeout time.Duration
	// BreakerThreshold is the number of consecutive unhealthy calls that
	// opens a provider's circuit breaker (GATEWAY_BREAKER_THRESHOLD); zero
	// keeps the gateway default. BreakerDisabled is set by a threshold of 0
	// in the environment.
	BreakerThreshold int
	BreakerDisabled  bool
	// BreakerCooldown is how long an open breaker rejects calls
	// (GATEWAY_BREAKER_COOLDOWN); zero keeps the gateway default.
	BreakerCooldown time.Duration
	// EvalConcurrency is the number of test cases run in parallel
	// (GATEWAY_EVAL_CONCURRENCY); zero keeps the runner default.
	EvalConcurrency int
}

// providerEnv describes how one provider is configured from the environment.
type providerEnv struct {
	name    domain.Provider
	keyVar  string
	baseVar string
	// keyless providers are enabled by their base URL instead of a key.
	keyless bool
}

var providerEnvs = []providerEnv{
	{name: domain.ProviderOpenAI, keyVar: "OPENAI_API_KEY", baseVar: "OPENAI_BASE_URL"},
	{name: domain.ProviderAnthropic, keyVar: "ANTHROPIC_API_KEY", baseVar: "ANTHROPIC_BASE_URL"},
	{name: domain.ProviderGemini, keyVar: "GEMINI_API_KEY", baseVar: "GEMINI_BASE_URL"},
	{name: domain.ProviderMistral, keyVar: "MISTRAL_API_KEY", baseVar: "MISTRAL_BASE_URL"},
	{name: domain.ProviderOllama, keyVar: "OLLAMA_API_KEY", baseVar: "OLLAMA_BASE_URL", keyless: true},
	{name: domain.ProviderOpenAICompatible, keyVar: "OPENAI_COMPATIBLE_API_KEY", baseVar: "OPENAI_COMPATIBLE_BASE_URL", keyless: true},
}

// FromEnv builds a Config from getenv (normally os.Getenv). Every problem
// found is reported, not just the first.
func FromEnv(getenv func(string) string) (*Config, error) {
	get := func(key string) string { return strings.TrimSpace(getenv(key)) }
	var errs []error

	cfg := &Config{
		Addr:     get("GATEWAY_ADDR"),
		APIKey:   get("GATEWAY_API_KEY"),
		DBPath:   get("GATEWAY_DB_PATH"),
		LogLevel: slog.LevelInfo,
	}
	if cfg.Addr == "" {
		cfg.Addr = DefaultAddr
	}
	if _, _, err := net.SplitHostPort(cfg.Addr); err != nil {
		errs = append(errs, fmt.Errorf("GATEWAY_ADDR %q is not host:port: %v", cfg.Addr, err))
	}
	if cfg.DBPath == "" {
		cfg.DBPath = DefaultDBPath
	}
	if raw := get("GATEWAY_LOG_LEVEL"); raw != "" {
		if err := cfg.LogLevel.UnmarshalText([]byte(raw)); err != nil {
			errs = append(errs, fmt.Errorf("GATEWAY_LOG_LEVEL %q is not one of debug, info, warn, error", raw))
		}
	}

	for _, env := range providerEnvs {
		p := Provider{Name: env.name, APIKey: get(env.keyVar), BaseURL: get(env.baseVar)}
		enabled := p.APIKey != ""
		if env.keyless {
			enabled = p.BaseURL != ""
			if !enabled && p.APIKey != "" {
				errs = append(errs, fmt.Errorf("%s is set but %s is not; %s is enabled by its base URL", env.keyVar, env.baseVar, env.name))
			}
		}
		if enabled {
			cfg.Providers = append(cfg.Providers, p)
		}
	}

	intVar := func(key string, dst *int) {
		raw := get(key)
		if raw == "" {
			return
		}
		n, err := strconv.Atoi(raw)
		if err != nil || n < 1 {
			errs = append(errs, fmt.Errorf("%s must be a positive integer, got %q", key, raw))
			return
		}
		*dst = n
	}
	intVar("ANTHROPIC_MAX_TOKENS", &cfg.AnthropicMaxTokens)
	intVar("GATEWAY_MAX_ATTEMPTS", &cfg.MaxAttempts)
	intVar("GATEWAY_EVAL_CONCURRENCY", &cfg.EvalConcurrency)

	durationVar := func(key string, dst *time.Duration) {
		raw := get(key)
		if raw == "" {
			return
		}
		d, err := time.ParseDuration(raw)
		if err != nil || d <= 0 {
			errs = append(errs, fmt.Errorf("%s must be a positive duration such as 30s, got %q", key, raw))
			return
		}
		*dst = d
	}
	durationVar("GATEWAY_ATTEMPT_TIMEOUT", &cfg.AttemptTimeout)
	durationVar("GATEWAY_BREAKER_COOLDOWN", &cfg.BreakerCooldown)

	// Unlike the other counts, 0 is meaningful here: it disables the breaker.
	if raw := get("GATEWAY_BREAKER_THRESHOLD"); raw != "" {
		n, err := strconv.Atoi(raw)
		switch {
		case err != nil || n < 0:
			errs = append(errs, fmt.Errorf("GATEWAY_BREAKER_THRESHOLD must be a non-negative integer, got %q", raw))
		case n == 0:
			cfg.BreakerDisabled = true
		default:
			cfg.BreakerThreshold = n
		}
	}

	routes, err := ParseRoutes(get("GATEWAY_ROUTES"))
	if err != nil {
		errs = append(errs, fmt.Errorf("GATEWAY_ROUTES: %v", err))
	}
	cfg.Routes = routes

	if len(errs) > 0 {
		return nil, fmt.Errorf("%w: %w", domain.ErrInvalidInput, errors.Join(errs...))
	}
	return cfg, nil
}

// ParseRoutes parses the GATEWAY_ROUTES syntax: routes separated by ";", each
// "name=provider:model,provider:model,...", targets in fallback order. For
// example:
//
//	fast=ollama:llama3.2,openai:gpt-4o-mini;smart=anthropic:claude-opus-5-5
//
// A model may itself contain ":" (as Ollama tags do); only the first one
// separates the provider.
func ParseRoutes(spec string) ([]domain.Route, error) {
	var routes []domain.Route
	for _, part := range strings.Split(spec, ";") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		name, targets, ok := strings.Cut(part, "=")
		if !ok {
			return nil, fmt.Errorf("route %q does not have the form name=provider:model", part)
		}
		route := domain.Route{Name: strings.TrimSpace(name)}
		for _, raw := range strings.Split(targets, ",") {
			provider, model, ok := strings.Cut(strings.TrimSpace(raw), ":")
			if !ok {
				return nil, fmt.Errorf("route %q: target %q is not provider:model", route.Name, strings.TrimSpace(raw))
			}
			route.Targets = append(route.Targets, domain.Target{
				Provider: domain.Provider(strings.TrimSpace(provider)),
				Model:    strings.TrimSpace(model),
			})
		}
		if err := route.Validate(); err != nil {
			return nil, err
		}
		routes = append(routes, route)
	}
	return routes, nil
}

// LoopbackOnly reports whether Addr accepts connections from this machine
// only.
func (c *Config) LoopbackOnly() bool {
	host, _, err := net.SplitHostPort(c.Addr)
	if err != nil {
		return false
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
