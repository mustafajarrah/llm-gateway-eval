package config

import (
	"errors"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

func env(vars map[string]string) func(string) string {
	return func(key string) string { return vars[key] }
}

func TestFromEnvDefaults(t *testing.T) {
	cfg, err := FromEnv(env(nil))
	if err != nil {
		t.Fatalf("FromEnv() error = %v", err)
	}
	if cfg.Addr != DefaultAddr || cfg.DBPath != DefaultDBPath || cfg.APIKey != "" || cfg.LogLevel != slog.LevelInfo ||
		len(cfg.Providers) != 0 || len(cfg.Routes) != 0 || cfg.MaxAttempts != 0 || cfg.AttemptTimeout != 0 ||
		cfg.EvalConcurrency != 0 || cfg.AnthropicMaxTokens != 0 {
		t.Errorf("defaults = %+v", cfg)
	}
	if !cfg.LoopbackOnly() {
		t.Error("the default address must be loopback only")
	}
}

func TestFromEnvFull(t *testing.T) {
	cfg, err := FromEnv(env(map[string]string{
		"GATEWAY_ADDR":               " :9090 ",
		"GATEWAY_API_KEY":            "secret",
		"GATEWAY_DB_PATH":            ":memory:",
		"GATEWAY_LOG_LEVEL":          "debug",
		"GATEWAY_MAX_ATTEMPTS":       "3",
		"GATEWAY_ATTEMPT_TIMEOUT":    "45s",
		"GATEWAY_EVAL_CONCURRENCY":   "8",
		"GATEWAY_BREAKER_THRESHOLD":  "10",
		"GATEWAY_BREAKER_COOLDOWN":   "2m",
		"GATEWAY_PRICES":             " anthropic:claude-opus-5-5 = 4 / 20 ; ollama:llama3.2:latest=0/0 ;",
		"GATEWAY_ROUTES":             "fast = ollama:llama3.2:latest , openai:gpt-4o-mini ; smart=anthropic:claude-opus-5-5;",
		"OPENAI_API_KEY":             "sk-openai",
		"ANTHROPIC_API_KEY":          "sk-ant",
		"ANTHROPIC_BASE_URL":         "https://proxy.example/anthropic",
		"ANTHROPIC_MAX_TOKENS":       "2048",
		"GEMINI_API_KEY":             "g-key",
		"MISTRAL_API_KEY":            "m-key",
		"OLLAMA_BASE_URL":            "http://localhost:11434/v1",
		"OPENAI_COMPATIBLE_BASE_URL": "https://api.groq.com/openai/v1",
		"OPENAI_COMPATIBLE_API_KEY":  "gsk",
	}))
	if err != nil {
		t.Fatalf("FromEnv() error = %v", err)
	}
	if cfg.Addr != ":9090" || cfg.APIKey != "secret" || cfg.DBPath != ":memory:" || cfg.LogLevel != slog.LevelDebug ||
		cfg.MaxAttempts != 3 || cfg.AttemptTimeout != 45*time.Second || cfg.EvalConcurrency != 8 || cfg.AnthropicMaxTokens != 2048 ||
		cfg.BreakerThreshold != 10 || cfg.BreakerCooldown != 2*time.Minute || cfg.BreakerDisabled {
		t.Errorf("config = %+v", cfg)
	}
	if cfg.LoopbackOnly() {
		t.Error(`":9090" listens on every interface`)
	}

	want := []Provider{
		{Name: domain.ProviderOpenAI, APIKey: "sk-openai"},
		{Name: domain.ProviderAnthropic, APIKey: "sk-ant", BaseURL: "https://proxy.example/anthropic"},
		{Name: domain.ProviderGemini, APIKey: "g-key"},
		{Name: domain.ProviderMistral, APIKey: "m-key"},
		{Name: domain.ProviderOllama, BaseURL: "http://localhost:11434/v1"},
		{Name: domain.ProviderOpenAICompatible, APIKey: "gsk", BaseURL: "https://api.groq.com/openai/v1"},
	}
	if len(cfg.Providers) != len(want) {
		t.Fatalf("providers = %+v", cfg.Providers)
	}
	for i, p := range want {
		if cfg.Providers[i] != p {
			t.Errorf("providers[%d] = %+v, want %+v", i, cfg.Providers[i], p)
		}
	}

	if len(cfg.Routes) != 2 {
		t.Fatalf("routes = %+v", cfg.Routes)
	}
	fast := cfg.Routes[0]
	if fast.Name != "fast" || len(fast.Targets) != 2 ||
		fast.Targets[0] != (domain.Target{Provider: domain.ProviderOllama, Model: "llama3.2:latest"}) ||
		fast.Targets[1] != (domain.Target{Provider: domain.ProviderOpenAI, Model: "gpt-4o-mini"}) {
		t.Errorf("fast route = %+v", fast)
	}
	if smart := cfg.Routes[1]; smart.Name != "smart" || smart.Targets[0].Model != "claude-opus-5-5" {
		t.Errorf("smart route = %+v", smart)
	}

	wantPrices := []domain.ModelPrice{
		{Target: domain.Target{Provider: domain.ProviderAnthropic, Model: "claude-opus-5-5"}, Price: domain.Price{InputPerMTok: 4, OutputPerMTok: 20}},
		{Target: domain.Target{Provider: domain.ProviderOllama, Model: "llama3.2:latest"}},
	}
	if len(cfg.Prices) != 2 || cfg.Prices[0] != wantPrices[0] || cfg.Prices[1] != wantPrices[1] {
		t.Errorf("prices = %+v, want %+v", cfg.Prices, wantPrices)
	}
}

func TestFromEnvReportsEveryProblem(t *testing.T) {
	_, err := FromEnv(env(map[string]string{
		"GATEWAY_ADDR":              "8080",
		"GATEWAY_LOG_LEVEL":         "loud",
		"GATEWAY_MAX_ATTEMPTS":      "0",
		"GATEWAY_EVAL_CONCURRENCY":  "many",
		"GATEWAY_ATTEMPT_TIMEOUT":   "-5s",
		"GATEWAY_BREAKER_THRESHOLD": "-1",
		"GATEWAY_BREAKER_COOLDOWN":  "soon",
		"GATEWAY_PRICES":            "openai:gpt-4o=cheap/expensive",
		"GATEWAY_ROUTES":            "fast=cohere:command",
		"OLLAMA_API_KEY":            "key-without-url",
		"OPENAI_COMPATIBLE_API_KEY": "key-without-url",
	}))
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Fatalf("FromEnv() error = %v, want ErrInvalidInput", err)
	}
	for _, want := range []string{
		"GATEWAY_ADDR", "GATEWAY_LOG_LEVEL", "GATEWAY_MAX_ATTEMPTS", "GATEWAY_EVAL_CONCURRENCY",
		"GATEWAY_ATTEMPT_TIMEOUT", "GATEWAY_ROUTES", "OLLAMA_BASE_URL", "OPENAI_COMPATIBLE_BASE_URL",
		"GATEWAY_BREAKER_THRESHOLD", "GATEWAY_BREAKER_COOLDOWN", "GATEWAY_PRICES",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error does not mention %s:\n%v", want, err)
		}
	}
}

func TestFromEnvDisablesBreaker(t *testing.T) {
	cfg, err := FromEnv(env(map[string]string{"GATEWAY_BREAKER_THRESHOLD": "0"}))
	if err != nil {
		t.Fatalf("FromEnv() error = %v", err)
	}
	if !cfg.BreakerDisabled || cfg.BreakerThreshold != 0 {
		t.Errorf("config = %+v, want the breaker disabled", cfg)
	}
}

func TestParseRoutes(t *testing.T) {
	if routes, err := ParseRoutes("  "); err != nil || len(routes) != 0 {
		t.Errorf("ParseRoutes(blank) = %v, %v", routes, err)
	}
	for name, spec := range map[string]string{
		"no equals":        "fast",
		"no colon":         "fast=openai",
		"no targets":       "fast=",
		"no name":          "=openai:gpt-4o",
		"unknown provider": "fast=cohere:command",
		"empty model":      "fast=openai:",
	} {
		if _, err := ParseRoutes(spec); err == nil {
			t.Errorf("%s: ParseRoutes(%q) succeeded, want an error", name, spec)
		}
	}
}

func TestParsePrices(t *testing.T) {
	if prices, err := ParsePrices(" "); err != nil || len(prices) != 0 {
		t.Errorf("ParsePrices(blank) = %v, %v", prices, err)
	}
	for name, spec := range map[string]string{
		"no equals":        "openai:gpt-4o",
		"no colon":         "gpt-4o=1/2",
		"no slash":         "openai:gpt-4o=1",
		"not numbers":      "openai:gpt-4o=a/b",
		"negative":         "openai:gpt-4o=-1/2",
		"unknown provider": "cohere:command=1/2",
		"empty model":      "openai:=1/2",
	} {
		if _, err := ParsePrices(spec); err == nil {
			t.Errorf("%s: ParsePrices(%q) succeeded, want an error", name, spec)
		}
	}
}

func TestLoopbackOnly(t *testing.T) {
	tests := map[string]bool{
		"127.0.0.1:8080": true,
		"localhost:8080": true,
		"[::1]:8080":     true,
		":8080":          false,
		"0.0.0.0:8080":   false,
		"10.0.0.5:8080":  false,
		"example.com:80": false,
		"not-an-address": false,
	}
	for addr, want := range tests {
		if got := (&Config{Addr: addr}).LoopbackOnly(); got != want {
			t.Errorf("LoopbackOnly(%q) = %v, want %v", addr, got, want)
		}
	}
}
