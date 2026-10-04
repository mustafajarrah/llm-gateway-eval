package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
	"github.com/mustafajarrah/llm-gateway-eval/internal/evaluation"
	"github.com/mustafajarrah/llm-gateway-eval/internal/storage/memory"
)

// fakeGateway echoes the last user message back, or fails with err.
type fakeGateway struct {
	err error

	// mu guards last: evaluation runs call Complete from several goroutines.
	mu   sync.Mutex
	last *domain.LLMRequest
}

func (g *fakeGateway) Complete(_ context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	g.mu.Lock()
	g.last = req
	g.mu.Unlock()
	if g.err != nil {
		return nil, g.err
	}
	return &domain.LLMResponse{
		ID:           "resp-1",
		Provider:     domain.ProviderOpenAI,
		Model:        "gpt-4o",
		Content:      req.Messages[len(req.Messages)-1].Content,
		FinishReason: domain.FinishReasonStop,
		Usage:        domain.TokenUsage{InputTokens: 5, OutputTokens: 2},
		Latency:      120 * time.Millisecond,
	}, nil
}

func (g *fakeGateway) Providers() []domain.Provider {
	return []domain.Provider{domain.ProviderAnthropic, domain.ProviderOpenAI}
}

func (g *fakeGateway) Circuits() map[domain.Provider]string {
	return map[domain.Provider]string{domain.ProviderAnthropic: "closed", domain.ProviderOpenAI: "open"}
}

func (g *fakeGateway) Routes() []domain.Route {
	return []domain.Route{{Name: "fast", Targets: []domain.Target{{Provider: domain.ProviderOpenAI, Model: "gpt-4o-mini"}}}}
}

type server struct {
	t       *testing.T
	handler http.Handler
	gateway *fakeGateway
	store   *memory.Store
	token   string
}

func newServer(t *testing.T, mutate func(*Config)) *server {
	t.Helper()
	store := memory.New()
	gateway := &fakeGateway{}
	runner, err := evaluation.NewRunner(store, store, gateway, evaluation.Config{})
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		Gateway: gateway, Prompts: store, Evals: store, Evaluator: runner,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	if mutate != nil {
		mutate(&cfg)
	}
	handler, err := NewHandler(cfg)
	if err != nil {
		t.Fatalf("NewHandler() error = %v", err)
	}
	return &server{t: t, handler: handler, gateway: gateway, store: store, token: cfg.APIKey}
}

// do sends a request and decodes the JSON response into out (when non-nil).
func (s *server) do(method, path, body string, wantStatus int, out any) *httptest.ResponseRecorder {
	s.t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if s.token != "" {
		req.Header.Set("Authorization", "Bearer "+s.token)
	}
	rec := httptest.NewRecorder()
	s.handler.ServeHTTP(rec, req)
	if rec.Code != wantStatus {
		s.t.Fatalf("%s %s = %d, want %d; body: %s", method, path, rec.Code, wantStatus, rec.Body)
	}
	if out != nil {
		if got := rec.Header().Get("Content-Type"); got != "application/json" {
			s.t.Errorf("%s %s Content-Type = %q", method, path, got)
		}
		if err := json.Unmarshal(rec.Body.Bytes(), out); err != nil {
			s.t.Fatalf("%s %s: decode %q: %v", method, path, rec.Body, err)
		}
	}
	return rec
}

// wantError asserts the response is an error body with the given code.
func (s *server) wantError(method, path, body string, wantStatus int, wantCode string) errorDetail {
	s.t.Helper()
	var resp errorBody
	s.do(method, path, body, wantStatus, &resp)
	if resp.Error.Code != wantCode || resp.Error.Message == "" {
		s.t.Errorf("%s %s error = %+v, want code %q with a message", method, path, resp.Error, wantCode)
	}
	return resp.Error
}

func (s *server) createPrompt(name string) domain.Prompt {
	s.t.Helper()
	var p domain.Prompt
	s.do("POST", "/v1/prompts", `{"name":"`+name+`"}`, http.StatusCreated, &p)
	return p
}

func TestHealthz(t *testing.T) {
	s := newServer(t, func(c *Config) { c.APIKey = "secret" })
	s.token = "" // health checks carry no credentials
	var body map[string]string
	s.do("GET", "/healthz", "", http.StatusOK, &body)
	if body["status"] != "ok" {
		t.Errorf("body = %v", body)
	}
}

func TestAuthentication(t *testing.T) {
	s := newServer(t, func(c *Config) { c.APIKey = "secret" })

	var ok providersResponse
	s.do("GET", "/v1/providers", "", http.StatusOK, &ok)

	for name, header := range map[string]string{
		"no header":    "",
		"wrong token":  "Bearer nope",
		"wrong scheme": "Basic secret",
		"bare token":   "secret",
		"prefix only":  "Bearer secre",
	} {
		t.Run(name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/v1/providers", nil)
			if header != "" {
				req.Header.Set("Authorization", header)
			}
			rec := httptest.NewRecorder()
			s.handler.ServeHTTP(rec, req)
			if rec.Code != http.StatusUnauthorized {
				t.Fatalf("status = %d, want 401", rec.Code)
			}
			if !strings.Contains(rec.Header().Get("WWW-Authenticate"), "Bearer") {
				t.Errorf("WWW-Authenticate = %q", rec.Header().Get("WWW-Authenticate"))
			}
			var body errorBody
			if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil || body.Error.Code != "unauthorized" {
				t.Errorf("body = %s", rec.Body)
			}
		})
	}
}

func TestProviders(t *testing.T) {
	s := newServer(t, nil)
	var resp providersResponse
	s.do("GET", "/v1/providers", "", http.StatusOK, &resp)
	if len(resp.Providers) != 2 || resp.Providers[0] != domain.ProviderAnthropic ||
		len(resp.Routes) != 1 || resp.Routes[0].Name != "fast" || resp.Routes[0].Targets[0].Model != "gpt-4o-mini" ||
		resp.Circuits[domain.ProviderOpenAI] != "open" || resp.Circuits[domain.ProviderAnthropic] != "closed" {
		t.Errorf("response = %+v", resp)
	}
}

func TestCompletions(t *testing.T) {
	s := newServer(t, nil)

	var raw map[string]any
	s.do("POST", "/v1/completions", `{
		"provider": "openai",
		"model": "gpt-4o",
		"system_prompt": "Be brief.",
		"messages": [{"role": "user", "content": "ping"}],
		"parameters": {"temperature": 0, "max_tokens": 16},
		"metadata": {"trace": "abc"}
	}`, http.StatusOK, &raw)
	if raw["content"] != "ping" || raw["provider"] != "openai" || raw["finish_reason"] != "stop" || raw["latency_ms"] != 120.0 {
		t.Errorf("response = %v", raw)
	}
	sent := s.gateway.last
	if sent.SystemPrompt != "Be brief." || *sent.Parameters.Temperature != 0 || sent.Parameters.MaxTokens != 16 || sent.Metadata["trace"] != "abc" {
		t.Errorf("gateway received %+v", sent)
	}

	s.wantError("POST", "/v1/completions", `{"model":"gpt-4o"}`, http.StatusBadRequest, "invalid_input")
	s.wantError("POST", "/v1/completions", ``, http.StatusBadRequest, "invalid_input")
	s.wantError("POST", "/v1/completions", `{"model":`, http.StatusBadRequest, "invalid_input")
	s.wantError("POST", "/v1/completions", `{"modle":"gpt-4o"}`, http.StatusBadRequest, "invalid_input")
	s.wantError("POST", "/v1/completions", `{"model":"a","messages":[{"role":"user","content":"x"}]} {"again":1}`,
		http.StatusBadRequest, "invalid_input")
}

func TestCompletionFailures(t *testing.T) {
	const validBody = `{"provider":"openai","model":"gpt-4o","messages":[{"role":"user","content":"ping"}]}`
	tests := []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
	}{
		{name: "provider error", wantStatus: http.StatusBadGateway, wantCode: "provider_error",
			err: &domain.ProviderError{Provider: domain.ProviderOpenAI, StatusCode: 429, Retryable: true, Err: errors.New("slow down")}},
		{name: "timeout", err: context.DeadlineExceeded, wantStatus: http.StatusGatewayTimeout, wantCode: "timeout"},
		{name: "client went away", err: context.Canceled, wantStatus: statusClientClosedRequest, wantCode: "cancelled"},
		{name: "unexpected", err: errors.New("pq: secret internal detail"), wantStatus: http.StatusInternalServerError, wantCode: "internal"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := newServer(t, nil)
			s.gateway.err = tt.err
			detail := s.wantError("POST", "/v1/completions", validBody, tt.wantStatus, tt.wantCode)
			switch tt.wantCode {
			case "provider_error":
				if detail.Provider != domain.ProviderOpenAI || detail.UpstreamStatus != 429 || !strings.Contains(detail.Message, "slow down") {
					t.Errorf("detail = %+v, want the provider and upstream status", detail)
				}
			case "internal":
				if strings.Contains(detail.Message, "secret") {
					t.Errorf("internal error leaked its cause: %q", detail.Message)
				}
			}
		})
	}
}

func TestBodyLimit(t *testing.T) {
	s := newServer(t, func(c *Config) { c.MaxBodyBytes = 64 })
	big := `{"name":"` + strings.Repeat("x", 200) + `"}`
	s.wantError("POST", "/v1/prompts", big, http.StatusRequestEntityTooLarge, "body_too_large")
}

func TestPrompts(t *testing.T) {
	s := newServer(t, nil)

	var empty list[domain.Prompt]
	rec := s.do("GET", "/v1/prompts", "", http.StatusOK, &empty)
	if !bytes.Contains(rec.Body.Bytes(), []byte(`"data":[]`)) {
		t.Errorf("empty list body = %s, want an empty array", rec.Body)
	}

	var created domain.Prompt
	s.do("POST", "/v1/prompts", `{"name":"  summariser ","description":"sums up","tags":["a","b"]}`, http.StatusCreated, &created)
	if created.ID == "" || created.Name != "summariser" || created.Description != "sums up" || len(created.Tags) != 2 || created.CreatedAt.IsZero() {
		t.Errorf("created = %+v", created)
	}
	s.createPrompt("second")

	var got domain.Prompt
	s.do("GET", "/v1/prompts/"+created.ID, "", http.StatusOK, &got)
	if got.ID != created.ID || got.Name != "summariser" {
		t.Errorf("got = %+v", got)
	}

	var page list[domain.Prompt]
	s.do("GET", "/v1/prompts?limit=1&offset=1", "", http.StatusOK, &page)
	if len(page.Data) != 1 || page.Data[0].ID != created.ID {
		t.Errorf("page = %+v, want the older prompt", page.Data)
	}

	s.wantError("POST", "/v1/prompts", `{"name":"summariser"}`, http.StatusConflict, "conflict")
	s.wantError("POST", "/v1/prompts", `{"name":" "}`, http.StatusBadRequest, "invalid_input")
	s.wantError("POST", "/v1/prompts", `{"id":"mine","name":"x"}`, http.StatusBadRequest, "invalid_input")
	s.wantError("GET", "/v1/prompts/missing", "", http.StatusNotFound, "not_found")
	s.wantError("GET", "/v1/prompts?limit=abc", "", http.StatusBadRequest, "invalid_input")
	s.wantError("GET", "/v1/prompts?offset=-1", "", http.StatusBadRequest, "invalid_input")

	if rec := s.do("DELETE", "/v1/prompts/"+created.ID, "", http.StatusNoContent, nil); rec.Body.Len() != 0 {
		t.Errorf("DELETE body = %q, want none", rec.Body)
	}
	s.wantError("DELETE", "/v1/prompts/"+created.ID, "", http.StatusNotFound, "not_found")
}

func TestVersions(t *testing.T) {
	s := newServer(t, nil)
	p := s.createPrompt("summariser")
	base := "/v1/prompts/" + p.ID + "/versions"

	s.wantError("GET", base+"/latest", "", http.StatusNotFound, "not_found")

	var v1 domain.PromptVersion
	s.do("POST", base, `{
		"template": "Summarise: {{.text}}",
		"system_prompt": "Be brief.",
		"provider": "anthropic",
		"model": "claude-haiku-4-5",
		"parameters": {"temperature": 0.2, "max_tokens": 256, "stop": ["END"]},
		"change_log": "first"
	}`, http.StatusCreated, &v1)
	if v1.ID == "" || v1.PromptID != p.ID || v1.Version != 1 || v1.Provider != domain.ProviderAnthropic ||
		*v1.Parameters.Temperature != 0.2 || v1.Parameters.Stop[0] != "END" || v1.ChangeLog != "first" {
		t.Errorf("created = %+v", v1)
	}
	var v2 domain.PromptVersion
	s.do("POST", base, `{"template":"TL;DR: {{.text}}","model":"fast"}`, http.StatusCreated, &v2)
	if v2.Version != 2 || v2.Provider != "" || v2.Model != "fast" {
		t.Errorf("second version = %+v", v2)
	}

	var all list[domain.PromptVersion]
	s.do("GET", base, "", http.StatusOK, &all)
	if len(all.Data) != 2 || all.Data[0].Version != 1 || all.Data[1].Version != 2 {
		t.Errorf("versions = %+v", all.Data)
	}
	var got domain.PromptVersion
	s.do("GET", base+"/1", "", http.StatusOK, &got)
	if got.ID != v1.ID {
		t.Errorf("version 1 = %+v", got)
	}
	s.do("GET", base+"/latest", "", http.StatusOK, &got)
	if got.ID != v2.ID {
		t.Errorf("latest = %+v", got)
	}

	s.wantError("GET", base+"/9", "", http.StatusNotFound, "not_found")
	s.wantError("GET", base+"/0", "", http.StatusBadRequest, "invalid_input")
	s.wantError("GET", base+"/newest", "", http.StatusBadRequest, "invalid_input")
	s.wantError("POST", base, `{"template":"{{.broken","model":"fast"}`, http.StatusBadRequest, "invalid_input")
	s.wantError("POST", base, `{"template":"x","model":"m","provider":"anthropic","parameters":{"temperature":1.5}}`,
		http.StatusBadRequest, "invalid_input")
	s.wantError("POST", base, `{"template":"x","model":"m","version":7}`, http.StatusBadRequest, "invalid_input")
	s.wantError("POST", "/v1/prompts/missing/versions", `{"template":"x","model":"m"}`, http.StatusNotFound, "not_found")
	s.wantError("GET", "/v1/prompts/missing/versions", "", http.StatusNotFound, "not_found")
}

func TestTestCases(t *testing.T) {
	s := newServer(t, nil)
	p := s.createPrompt("summariser")
	base := "/v1/prompts/" + p.ID + "/test-cases"

	var tc domain.EvaluationTestCase
	s.do("POST", base, `{"name":" mentions go ","variables":{"text":"Go"},"expected_output":"go","match_strategy":"contains"}`,
		http.StatusCreated, &tc)
	if tc.ID == "" || tc.PromptID != p.ID || tc.Name != "mentions go" || tc.Variables["text"] != "Go" || tc.MatchStrategy != domain.MatchContains {
		t.Errorf("created = %+v", tc)
	}

	var got domain.EvaluationTestCase
	s.do("GET", "/v1/test-cases/"+tc.ID, "", http.StatusOK, &got)
	if got.ID != tc.ID {
		t.Errorf("got = %+v", got)
	}
	var all list[domain.EvaluationTestCase]
	s.do("GET", base, "", http.StatusOK, &all)
	if len(all.Data) != 1 || all.Data[0].ID != tc.ID {
		t.Errorf("list = %+v", all.Data)
	}

	s.wantError("POST", base, `{"name":"x","expected_output":"(","match_strategy":"regex"}`, http.StatusBadRequest, "invalid_input")
	s.wantError("POST", base, `{"name":"x","expected_output":"x","match_strategy":"fuzzy"}`, http.StatusBadRequest, "invalid_input")
	s.wantError("POST", base, `not json`, http.StatusBadRequest, "invalid_input")
	s.wantError("POST", "/v1/prompts/missing/test-cases", `{"name":"x","expected_output":"x","match_strategy":"exact"}`,
		http.StatusNotFound, "not_found")
	s.wantError("GET", "/v1/prompts/missing/test-cases", "", http.StatusNotFound, "not_found")
	s.wantError("GET", "/v1/test-cases/missing", "", http.StatusNotFound, "not_found")

	s.do("DELETE", "/v1/test-cases/"+tc.ID, "", http.StatusNoContent, nil)
	s.wantError("DELETE", "/v1/test-cases/"+tc.ID, "", http.StatusNotFound, "not_found")
}

func TestRunsAndComparison(t *testing.T) {
	s := newServer(t, nil)
	p := s.createPrompt("echo")
	base := "/v1/prompts/" + p.ID

	s.do("POST", base+"/versions", `{"template":"{{.in}}","provider":"openai","model":"gpt-4o"}`, http.StatusCreated, nil)
	s.wantError("POST", base+"/runs", ``, http.StatusBadRequest, "invalid_input") // no test cases yet

	var pass, flip domain.EvaluationTestCase
	s.do("POST", base+"/test-cases", `{"name":"pass","variables":{"in":"yes"},"expected_output":"yes","match_strategy":"contains"}`,
		http.StatusCreated, &pass)
	s.do("POST", base+"/test-cases", `{"name":"flip","variables":{"in":"yes"},"expected_output":"yes","match_strategy":"exact"}`,
		http.StatusCreated, &flip)

	var first evaluation.Report
	rec := s.do("POST", base+"/runs", ``, http.StatusCreated, &first)
	if first.Run.Status != domain.RunStatusCompleted || first.Run.Version != 1 || first.Run.Summary.Passed != 2 || len(first.Results) != 2 {
		t.Errorf("first run = %+v", first)
	}
	if !bytes.Contains(rec.Body.Bytes(), []byte(`"latency_ms":120`)) {
		t.Errorf("run body = %s, want results with latency_ms", rec.Body)
	}

	s.do("POST", base+"/versions", `{"template":"v2 {{.in}}","provider":"openai","model":"gpt-4o"}`, http.StatusCreated, nil)
	var second evaluation.Report
	s.do("POST", base+"/runs", `{"version":2}`, http.StatusCreated, &second)
	if second.Run.Version != 2 || second.Run.Summary.Passed != 1 || second.Run.Summary.Failed != 1 {
		t.Errorf("second run = %+v", second.Run)
	}

	var stored evaluation.Report
	s.do("GET", "/v1/runs/"+first.Run.ID, "", http.StatusOK, &stored)
	if stored.Run.ID != first.Run.ID || len(stored.Results) != 2 {
		t.Errorf("stored run = %+v", stored)
	}
	var runs list[domain.EvaluationRun]
	s.do("GET", base+"/runs?limit=5", "", http.StatusOK, &runs)
	if len(runs.Data) != 2 || runs.Data[0].ID != second.Run.ID {
		t.Errorf("runs = %+v, want newest first", runs.Data)
	}

	var cmp evaluation.Comparison
	s.do("GET", "/v1/comparisons?base="+first.Run.ID+"&candidate="+second.Run.ID, "", http.StatusOK, &cmp)
	if len(cmp.Regressions) != 1 || cmp.Regressions[0].TestCaseID != flip.ID || cmp.Unchanged != 1 || len(cmp.Improvements) != 0 {
		t.Errorf("comparison = %+v", cmp)
	}

	s.wantError("POST", base+"/runs", `{"version":9}`, http.StatusNotFound, "not_found")
	s.wantError("POST", base+"/runs", `{"version":-1}`, http.StatusBadRequest, "invalid_input")
	s.wantError("POST", base+"/runs", `{"version":"two"}`, http.StatusBadRequest, "invalid_input")
	s.wantError("POST", "/v1/prompts/missing/runs", ``, http.StatusNotFound, "not_found")
	s.wantError("GET", "/v1/prompts/missing/runs", "", http.StatusNotFound, "not_found")
	s.wantError("GET", base+"/runs?limit=x", "", http.StatusBadRequest, "invalid_input")
	s.wantError("GET", "/v1/runs/missing", "", http.StatusNotFound, "not_found")
	s.wantError("GET", "/v1/comparisons?base="+first.Run.ID, "", http.StatusBadRequest, "invalid_input")
	s.wantError("GET", "/v1/comparisons?base="+first.Run.ID+"&candidate=missing", "", http.StatusNotFound, "not_found")
}

func TestUnknownRoutes(t *testing.T) {
	s := newServer(t, nil)
	s.do("GET", "/v1/nope", "", http.StatusNotFound, nil)
	s.do("PUT", "/v1/prompts", "", http.StatusMethodNotAllowed, nil)
}

func TestPanicBecomes500(t *testing.T) {
	s := newServer(t, func(c *Config) { c.Evaluator = panickingEvaluator{} })
	s.wantError("GET", "/v1/runs/any", "", http.StatusInternalServerError, "internal")
}

type panickingEvaluator struct{ Evaluator }

func (panickingEvaluator) Report(context.Context, string) (*evaluation.Report, error) {
	panic("boom")
}

func TestStorageFailuresAre500(t *testing.T) {
	boom := errors.New("disk full")
	s := newServer(t, func(c *Config) {
		c.Prompts = failingPrompts{PromptRepository: c.Prompts, err: boom}
	})
	s.wantError("GET", "/v1/prompts", "", http.StatusInternalServerError, "internal")
}

type failingPrompts struct {
	domain.PromptRepository
	err error
}

func (f failingPrompts) ListPrompts(context.Context, domain.ListOptions) ([]domain.Prompt, error) {
	return nil, f.err
}

func TestNewHandler(t *testing.T) {
	store := memory.New()
	gateway := &fakeGateway{}
	runner, _ := evaluation.NewRunner(store, store, gateway, evaluation.Config{})
	full := Config{Gateway: gateway, Prompts: store, Evals: store, Evaluator: runner}

	if _, err := NewHandler(full); err != nil {
		t.Errorf("NewHandler() with defaults error = %v", err)
	}
	for name, mutate := range map[string]func(*Config){
		"no gateway":    func(c *Config) { c.Gateway = nil },
		"no prompts":    func(c *Config) { c.Prompts = nil },
		"no evals":      func(c *Config) { c.Evals = nil },
		"no evaluator":  func(c *Config) { c.Evaluator = nil },
		"negative body": func(c *Config) { c.MaxBodyBytes = -1 },
	} {
		cfg := full
		mutate(&cfg)
		if _, err := NewHandler(cfg); !errors.Is(err, domain.ErrInvalidInput) {
			t.Errorf("%s: error = %v, want ErrInvalidInput", name, err)
		}
	}
}
