package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

func ptr[T any](v T) *T { return &v }

const okBody = `{
	"id": "chatcmpl-1",
	"model": "gpt-4o-2024-08-06",
	"choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
	"usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
}`

// fakeServer answers every request with status and body, and records what it
// received.
type fakeServer struct {
	*httptest.Server
	path   string
	header http.Header
	body   map[string]any
}

func newFakeServer(t *testing.T, status int, body string) *fakeServer {
	t.Helper()
	f := &fakeServer{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.path = r.URL.Path
		f.header = r.Header.Clone()
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &f.body); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(f.Close)
	return f
}

func newClient(t *testing.T, provider domain.Provider, srv *fakeServer, apiKey string) *Client {
	t.Helper()
	c, err := New(Config{Provider: provider, BaseURL: srv.URL + "/v1/", APIKey: apiKey, HTTPClient: srv.Client()})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return c
}

func request() *domain.LLMRequest {
	return &domain.LLMRequest{
		Model:        "gpt-4o",
		SystemPrompt: "Be brief.",
		Messages: []domain.Message{
			{Role: domain.RoleUser, Content: "Hi"},
			{Role: domain.RoleAssistant, Content: "Hello"},
			{Role: domain.RoleUser, Content: "Again"},
		},
		Parameters: domain.ModelParameters{
			Temperature: ptr(0.0),
			TopP:        ptr(0.9),
			MaxTokens:   64,
			Stop:        []string{"END"},
		},
		Metadata: map[string]string{"trace": "abc"},
	}
}

func TestCompleteOpenAI(t *testing.T) {
	srv := newFakeServer(t, http.StatusOK, okBody)
	c := newClient(t, domain.ProviderOpenAI, srv, "sk-test")
	ticks := []time.Time{time.Unix(100, 0), time.Unix(100, int64(250*time.Millisecond))}
	c.now = func() time.Time { tick := ticks[0]; ticks = ticks[1:]; return tick }

	resp, err := c.Complete(context.Background(), request())
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}

	if srv.path != "/v1/chat/completions" {
		t.Errorf("path = %q", srv.path)
	}
	if got := srv.header.Get("Authorization"); got != "Bearer sk-test" {
		t.Errorf("Authorization = %q", got)
	}
	wantBody := map[string]any{
		"model": "gpt-4o",
		"messages": []any{
			map[string]any{"role": "system", "content": "Be brief."},
			map[string]any{"role": "user", "content": "Hi"},
			map[string]any{"role": "assistant", "content": "Hello"},
			map[string]any{"role": "user", "content": "Again"},
		},
		"temperature":           0.0,
		"top_p":                 0.9,
		"max_completion_tokens": 64.0,
		"stop":                  []any{"END"},
	}
	if got, want := mustJSON(t, srv.body), mustJSON(t, wantBody); got != want {
		t.Errorf("request body =\n%s\nwant\n%s", got, want)
	}

	want := domain.LLMResponse{
		ID:           "chatcmpl-1",
		Provider:     domain.ProviderOpenAI,
		Model:        "gpt-4o-2024-08-06",
		Content:      "Hello!",
		FinishReason: domain.FinishReasonStop,
		Usage:        domain.TokenUsage{InputTokens: 12, OutputTokens: 3},
		Latency:      250 * time.Millisecond,
		CreatedAt:    time.Unix(100, int64(250*time.Millisecond)).UTC(),
	}
	if *resp != want {
		t.Errorf("response = %+v, want %+v", *resp, want)
	}
}

func TestCompleteCompatibleEndpoint(t *testing.T) {
	srv := newFakeServer(t, http.StatusOK, `{"choices":[{"message":{"content":"ok"},"finish_reason":"length"}]}`)
	c := newClient(t, domain.ProviderOllama, srv, "")

	resp, err := c.Complete(context.Background(), &domain.LLMRequest{
		Model:      "llama3.2",
		Messages:   []domain.Message{{Role: domain.RoleUser, Content: "Hi"}},
		Parameters: domain.ModelParameters{MaxTokens: 32},
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if _, ok := srv.header["Authorization"]; ok {
		t.Error("Authorization header sent without an API key")
	}
	if srv.body["max_tokens"] != 32.0 {
		t.Errorf("max_tokens = %v, want 32", srv.body["max_tokens"])
	}
	for _, absent := range []string{"max_completion_tokens", "temperature", "top_p", "stop"} {
		if _, ok := srv.body[absent]; ok {
			t.Errorf("request body unexpectedly contains %q", absent)
		}
	}
	if len(srv.body["messages"].([]any)) != 1 {
		t.Errorf("messages = %v, want no system message", srv.body["messages"])
	}
	if resp.Provider != domain.ProviderOllama || resp.Model != "llama3.2" || resp.FinishReason != domain.FinishReasonLength {
		t.Errorf("response = %+v", *resp)
	}
}

func TestCompleteErrors(t *testing.T) {
	tests := []struct {
		name      string
		status    int
		body      string
		retryable bool
	}{
		{name: "rate limited", status: 429, body: `{"error":{"message":"slow down"}}`, retryable: true},
		{name: "bad request", status: 400, body: `{"error":{"message":"unknown model"}}`},
		{name: "unauthorized", status: 401, body: `{"error":{"message":"bad key"}}`},
		{name: "server error", status: 503, body: `unavailable`, retryable: true},
		{name: "no choices", status: 200, body: `{"id":"x","choices":[]}`, retryable: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, tt.status, tt.body)
			c := newClient(t, domain.ProviderMistral, srv, "key")

			resp, err := c.Complete(context.Background(), request())
			if resp != nil {
				t.Errorf("response = %+v, want nil", resp)
			}
			var pe *domain.ProviderError
			if !errors.As(err, &pe) {
				t.Fatalf("error %v is not a *domain.ProviderError", err)
			}
			if pe.Provider != domain.ProviderMistral || pe.StatusCode != tt.status || pe.Retryable != tt.retryable {
				t.Errorf("got %+v, want status %d retryable %v", pe, tt.status, tt.retryable)
			}
			if domain.IsRetryable(err) != tt.retryable {
				t.Errorf("IsRetryable() = %v, want %v", domain.IsRetryable(err), tt.retryable)
			}
		})
	}
}

func TestFinishReason(t *testing.T) {
	tests := map[string]domain.FinishReason{
		"stop":           domain.FinishReasonStop,
		"length":         domain.FinishReasonLength,
		"model_length":   domain.FinishReasonLength,
		"content_filter": domain.FinishReasonContentFilter,
		"tool_calls":     domain.FinishReasonUnknown,
		"":               domain.FinishReasonUnknown,
	}
	for in, want := range tests {
		if got := finishReason(in); got != want {
			t.Errorf("finishReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestNew(t *testing.T) {
	defaults := map[domain.Provider]string{
		domain.ProviderOpenAI:  "https://api.openai.com/v1/chat/completions",
		domain.ProviderMistral: "https://api.mistral.ai/v1/chat/completions",
		domain.ProviderOllama:  "http://localhost:11434/v1/chat/completions",
	}
	for provider, endpoint := range defaults {
		c, err := New(Config{Provider: provider})
		if err != nil {
			t.Fatalf("New(%s) error = %v", provider, err)
		}
		if c.endpoint != endpoint || c.Name() != provider {
			t.Errorf("New(%s): endpoint %q, name %q", provider, c.endpoint, c.Name())
		}
	}

	c, err := New(Config{Provider: domain.ProviderOpenAICompatible, BaseURL: " https://api.groq.com/openai/v1/ "})
	if err != nil {
		t.Fatalf("New(openai_compatible) error = %v", err)
	}
	if c.endpoint != "https://api.groq.com/openai/v1/chat/completions" {
		t.Errorf("endpoint = %q", c.endpoint)
	}

	if _, err := New(Config{Provider: domain.ProviderOpenAICompatible}); !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("missing base URL: expected ErrInvalidInput, got %v", err)
	}
	if _, err := New(Config{Provider: domain.ProviderAnthropic}); !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("wrong protocol: expected ErrInvalidInput, got %v", err)
	}
	if got := DefaultBaseURL(domain.ProviderGemini); got != "" {
		t.Errorf("DefaultBaseURL(gemini) = %q, want empty", got)
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(data)
}
