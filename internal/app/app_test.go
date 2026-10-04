package app

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mustafajarrah/llm-gateway-eval/internal/config"
	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
	"github.com/mustafajarrah/llm-gateway-eval/internal/storage/sqlite"
)

var quiet = slog.New(slog.NewTextHandler(io.Discard, nil))

// fakeUpstream speaks the OpenAI chat completions protocol and answers with
// the last message upper-cased. It fails the first call for model "flaky".
func fakeUpstream(t *testing.T) *httptest.Server {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model    string `json:"model"`
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("upstream received a malformed body: %v", err)
		}
		if n := calls.Add(1); req.Model == "flaky" && n == 1 {
			http.Error(w, `{"error":{"message":"try again"}}`, http.StatusServiceUnavailable)
			return
		}
		answer := strings.ToUpper(req.Messages[len(req.Messages)-1].Content)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      "cmpl-1",
			"model":   req.Model,
			"choices": []map[string]any{{"message": map[string]string{"content": answer}, "finish_reason": "stop"}},
			"usage":   map[string]int{"prompt_tokens": 7, "completion_tokens": 3},
		})
	}))
	t.Cleanup(srv.Close)
	return srv
}

func call(t *testing.T, h http.Handler, method, path, body string, wantStatus int) map[string]any {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(method, path, strings.NewReader(body)))
	if rec.Code != wantStatus {
		t.Fatalf("%s %s = %d, want %d; body: %s", method, path, rec.Code, wantStatus, rec.Body)
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("%s %s: decode %q: %v", method, path, rec.Body, err)
	}
	return out
}

// TestEndToEnd drives the assembled service through its HTTP API: a real
// adapter talking to a fake upstream, the gateway, the evaluation runner and
// SQLite on disk.
func TestEndToEnd(t *testing.T) {
	upstream := fakeUpstream(t)
	cfg := &config.Config{
		Addr:   config.DefaultAddr,
		DBPath: filepath.Join(t.TempDir(), "gateway.db"),
		Providers: []config.Provider{
			{Name: domain.ProviderOllama, BaseURL: upstream.URL},
			{Name: domain.ProviderOpenAICompatible, BaseURL: upstream.URL, APIKey: "k"},
		},
		Routes: []domain.Route{{Name: "local", Targets: []domain.Target{
			{Provider: domain.ProviderOllama, Model: "flaky"},
		}}},
		MaxAttempts: 2,
	}
	app, err := New(context.Background(), cfg, quiet)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer app.Close()
	h := app.Handler

	providers := call(t, h, "GET", "/v1/providers", "", http.StatusOK)
	if got := providers["providers"].([]any); len(got) != 2 || got[0] != "ollama" || got[1] != "openai_compatible" {
		t.Errorf("providers = %v", got)
	}

	// The route's only target fails once with a 503 and succeeds on retry.
	completion := call(t, h, "POST", "/v1/completions",
		`{"model":"local","messages":[{"role":"user","content":"hello"}]}`, http.StatusOK)
	if completion["content"] != "HELLO" || completion["provider"] != "ollama" || completion["model"] != "flaky" {
		t.Errorf("completion = %v", completion)
	}

	prompt := call(t, h, "POST", "/v1/prompts", `{"name":"shout"}`, http.StatusCreated)
	base := "/v1/prompts/" + prompt["id"].(string)
	call(t, h, "POST", base+"/versions",
		`{"template":"{{.word}}","provider":"openai_compatible","model":"any"}`, http.StatusCreated)
	call(t, h, "POST", base+"/test-cases",
		`{"name":"upper","variables":{"word":"go"},"expected_output":"GO","match_strategy":"exact"}`, http.StatusCreated)
	call(t, h, "POST", base+"/test-cases",
		`{"name":"lower","variables":{"word":"go"},"expected_output":"go","match_strategy":"exact"}`, http.StatusCreated)

	report := call(t, h, "POST", base+"/runs?wait=true", "", http.StatusCreated)
	run := report["run"].(map[string]any)
	summary := run["summary"].(map[string]any)
	if run["status"] != "completed" || summary["total"] != 2.0 || summary["passed"] != 1.0 || summary["failed"] != 1.0 {
		t.Errorf("run = %v", run)
	}
	if usage := summary["usage"].(map[string]any); usage["input_tokens"] != 14.0 || usage["output_tokens"] != 6.0 {
		t.Errorf("usage = %v, want the two calls summed", usage)
	}

	stored := call(t, h, "GET", "/v1/runs/"+run["id"].(string), "", http.StatusOK)
	if results := stored["results"].([]any); len(results) != 2 || results[0].(map[string]any)["actual_output"] != "GO" {
		t.Errorf("stored results = %v", stored["results"])
	}
}

// TestBackgroundRunsAcrossRestarts covers the lifecycle of asynchronous runs:
// Close cancels a run in flight, and a run left "running" in the database by
// a process that died is marked as failed at the next start.
func TestBackgroundRunsAcrossRestarts(t *testing.T) {
	entered := make(chan struct{}, 1)
	// An upstream that never answers until the client goes away.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The server only notices a client disconnect once the request body
		// has been read.
		_, _ = io.Copy(io.Discard, r.Body)
		entered <- struct{}{}
		<-r.Context().Done()
	}))
	t.Cleanup(upstream.Close)

	cfg := &config.Config{
		Addr:      config.DefaultAddr,
		DBPath:    filepath.Join(t.TempDir(), "gateway.db"),
		Providers: []config.Provider{{Name: domain.ProviderOllama, BaseURL: upstream.URL}},
	}
	first, err := New(context.Background(), cfg, quiet)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	prompt := call(t, first.Handler, "POST", "/v1/prompts", `{"name":"slow"}`, http.StatusCreated)
	base := "/v1/prompts/" + prompt["id"].(string)
	call(t, first.Handler, "POST", base+"/versions", `{"template":"{{.w}}","provider":"ollama","model":"m"}`, http.StatusCreated)
	call(t, first.Handler, "POST", base+"/test-cases",
		`{"name":"c","variables":{"w":"x"},"expected_output":"x","match_strategy":"exact"}`, http.StatusCreated)

	started := call(t, first.Handler, "POST", base+"/runs", "", http.StatusAccepted)
	if started["status"] != "running" {
		t.Fatalf("started run = %v", started)
	}
	<-entered
	if err := first.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	// Simulate a crash: put a run back into "running" behind the app's back.
	store, err := sqlite.Open(context.Background(), cfg.DBPath)
	if err != nil {
		t.Fatal(err)
	}
	cancelled, err := store.GetRun(context.Background(), started["id"].(string))
	if err != nil {
		t.Fatal(err)
	}
	if cancelled.Status != domain.RunStatusFailed || !strings.Contains(cancelled.Error, "run cancelled") {
		t.Errorf("run after Close = %+v, want it recorded as failed", cancelled)
	}
	orphan := &domain.EvaluationRun{PromptID: cancelled.PromptID, PromptVersionID: cancelled.PromptVersionID, Status: domain.RunStatusRunning}
	if err := store.CreateRun(context.Background(), orphan); err != nil {
		t.Fatal(err)
	}
	store.Close()

	second, err := New(context.Background(), cfg, quiet)
	if err != nil {
		t.Fatalf("New() after restart error = %v", err)
	}
	defer second.Close()
	recovered := call(t, second.Handler, "GET", "/v1/runs/"+orphan.ID, "", http.StatusOK)["run"].(map[string]any)
	if recovered["status"] != "failed" || !strings.Contains(recovered["error"].(string), "interrupted") {
		t.Errorf("orphaned run after restart = %v, want it marked as failed", recovered)
	}
}

func TestNewRefusesUnprotectedExposure(t *testing.T) {
	cfg := &config.Config{Addr: ":8080", DBPath: ":memory:"}
	if _, err := New(context.Background(), cfg, quiet); !errors.Is(err, ErrUnprotected) {
		t.Fatalf("New() error = %v, want ErrUnprotected", err)
	}

	cfg.APIKey = "secret"
	app, err := New(context.Background(), cfg, quiet)
	if err != nil {
		t.Fatalf("New() with an API key error = %v", err)
	}
	defer app.Close()

	rec := httptest.NewRecorder()
	app.Handler.ServeHTTP(rec, httptest.NewRequest("GET", "/v1/providers", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("unauthenticated request = %d, want 401", rec.Code)
	}
}

func TestNewErrors(t *testing.T) {
	tests := map[string]*config.Config{
		"provider without its key": {
			Addr: config.DefaultAddr, DBPath: ":memory:",
			Providers: []config.Provider{{Name: domain.ProviderAnthropic}},
		},
		"gemini without its key": {
			Addr: config.DefaultAddr, DBPath: ":memory:",
			Providers: []config.Provider{{Name: domain.ProviderGemini}},
		},
		"generic provider without a base URL": {
			Addr: config.DefaultAddr, DBPath: ":memory:",
			Providers: []config.Provider{{Name: domain.ProviderOpenAICompatible}},
		},
		"route to a provider that is not enabled": {
			Addr: config.DefaultAddr, DBPath: ":memory:",
			Routes: []domain.Route{{Name: "fast", Targets: []domain.Target{{Provider: domain.ProviderOpenAI, Model: "gpt-4o"}}}},
		},
		"bad eval concurrency": {Addr: config.DefaultAddr, DBPath: ":memory:", EvalConcurrency: -1},
		"no database path":     {Addr: config.DefaultAddr},
	}
	for name, cfg := range tests {
		t.Run(name, func(t *testing.T) {
			if _, err := New(context.Background(), cfg, quiet); !errors.Is(err, domain.ErrInvalidInput) {
				t.Errorf("New() error = %v, want ErrInvalidInput", err)
			}
		})
	}
}

func TestNewBuildsEveryAdapter(t *testing.T) {
	cfg := &config.Config{
		Addr: config.DefaultAddr, DBPath: ":memory:",
		Providers: []config.Provider{
			{Name: domain.ProviderOpenAI, APIKey: "k"},
			{Name: domain.ProviderAnthropic, APIKey: "k"},
			{Name: domain.ProviderGemini, APIKey: "k"},
			{Name: domain.ProviderMistral, APIKey: "k"},
			{Name: domain.ProviderOllama, BaseURL: "http://localhost:11434/v1"},
			{Name: domain.ProviderOpenAICompatible, BaseURL: "http://localhost:9999/v1"},
		},
	}
	app, err := New(context.Background(), cfg, quiet)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer app.Close()
	providers := call(t, app.Handler, "GET", "/v1/providers", "", http.StatusOK)["providers"].([]any)
	if len(providers) != 6 {
		t.Errorf("providers = %v, want all six", providers)
	}
}
