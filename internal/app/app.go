// Package app is the composition root: it turns a Config into a running set
// of adapters, services and an HTTP handler.
package app

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/mustafajarrah/llm-gateway-eval/internal/config"
	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
	"github.com/mustafajarrah/llm-gateway-eval/internal/evaluation"
	"github.com/mustafajarrah/llm-gateway-eval/internal/gateway"
	"github.com/mustafajarrah/llm-gateway-eval/internal/httpapi"
	"github.com/mustafajarrah/llm-gateway-eval/internal/provider/anthropic"
	"github.com/mustafajarrah/llm-gateway-eval/internal/provider/gemini"
	"github.com/mustafajarrah/llm-gateway-eval/internal/provider/openaicompat"
	"github.com/mustafajarrah/llm-gateway-eval/internal/storage/sqlite"
)

// ErrUnprotected is returned when the service would listen beyond loopback
// without an API key. Anyone who could reach it could spend the configured
// provider keys, so that combination is refused rather than warned about.
var ErrUnprotected = errors.New("refusing to listen on a non-loopback address without GATEWAY_API_KEY")

// App is the assembled service.
type App struct {
	// Handler serves the HTTP API.
	Handler http.Handler
	store   *sqlite.Store
}

// New assembles the service described by cfg. The caller must Close it.
func New(ctx context.Context, cfg *config.Config, logger *slog.Logger) (*App, error) {
	if cfg.APIKey == "" && !cfg.LoopbackOnly() {
		return nil, fmt.Errorf("%w (GATEWAY_ADDR is %q)", ErrUnprotected, cfg.Addr)
	}

	providers, err := buildProviders(cfg)
	if err != nil {
		return nil, err
	}
	gw, err := gateway.New(providers, gateway.Config{
		Routes:           cfg.Routes,
		MaxAttempts:      cfg.MaxAttempts,
		AttemptTimeout:   cfg.AttemptTimeout,
		BreakerThreshold: cfg.BreakerThreshold,
		BreakerCooldown:  cfg.BreakerCooldown,
		DisableBreaker:   cfg.BreakerDisabled,
		Logger:           logger,
	})
	if err != nil {
		return nil, err
	}
	if len(providers) == 0 {
		logger.Warn("no provider is configured; completions and evaluation runs will fail until one is")
	}
	if cfg.APIKey == "" {
		logger.Warn("GATEWAY_API_KEY is not set; the API is unauthenticated (loopback only)")
	}

	store, err := sqlite.Open(ctx, cfg.DBPath)
	if err != nil {
		return nil, fmt.Errorf("open storage: %w", err)
	}
	app := &App{store: store}

	runner, err := evaluation.NewRunner(store, store, gw, evaluation.Config{Concurrency: cfg.EvalConcurrency})
	if err == nil {
		app.Handler, err = httpapi.NewHandler(httpapi.Config{
			Gateway:   gw,
			Prompts:   store,
			Evals:     store,
			Evaluator: runner,
			APIKey:    cfg.APIKey,
			Logger:    logger,
		})
	}
	if err != nil {
		store.Close()
		return nil, err
	}

	logger.Info("service assembled", "providers", gw.Providers(), "routes", len(gw.Routes()), "db", cfg.DBPath)
	return app, nil
}

// Close releases the storage.
func (a *App) Close() error { return a.store.Close() }

// buildProviders creates one adapter per enabled provider.
func buildProviders(cfg *config.Config) ([]domain.LLMProvider, error) {
	providers := make([]domain.LLMProvider, 0, len(cfg.Providers))
	for _, p := range cfg.Providers {
		var (
			adapter domain.LLMProvider
			err     error
		)
		switch p.Name {
		case domain.ProviderAnthropic:
			adapter, err = anthropic.New(anthropic.Config{
				APIKey: p.APIKey, BaseURL: p.BaseURL, DefaultMaxTokens: cfg.AnthropicMaxTokens,
			})
		case domain.ProviderGemini:
			adapter, err = gemini.New(gemini.Config{APIKey: p.APIKey, BaseURL: p.BaseURL})
		default:
			// openai, mistral, ollama and openai_compatible share a protocol;
			// the adapter rejects anything else.
			adapter, err = openaicompat.New(openaicompat.Config{Provider: p.Name, APIKey: p.APIKey, BaseURL: p.BaseURL})
		}
		if err != nil {
			return nil, fmt.Errorf("provider %s: %w", p.Name, err)
		}
		providers = append(providers, adapter)
	}
	return providers, nil
}
