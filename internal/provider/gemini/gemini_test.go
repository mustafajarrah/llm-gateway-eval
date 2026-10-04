package gemini

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
	"responseId": "resp-1",
	"modelVersion": "gemini-2.5-flash",
	"candidates": [{
		"content": {"role": "model", "parts": [
			{"text": "let me think", "thought": true},
			{"text": "Hello"},
			{"text": ", world"}
		]},
		"finishReason": "STOP"
	}],
	"usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4, "thoughtsTokenCount": 6, "totalTokenCount": 19}
}`

type fakeServer struct {
	*httptest.Server
	path   string
	query  string
	header http.Header
	body   map[string]any
}

func newFakeServer(t *testing.T, status int, body string) *fakeServer {
	t.Helper()
	f := &fakeServer{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.path = r.URL.EscapedPath()
		f.query = r.URL.RawQuery
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

func newClient(t *testing.T, srv *fakeServer) *Client {
	t.Helper()
	c, err := New(Config{APIKey: "g-key", BaseURL: srv.URL + "/v1beta/", HTTPClient: srv.Client()})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return c
}

func request() *domain.LLMRequest {
	return &domain.LLMRequest{
		Model:        "gemini-2.5-flash",
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
	}
}

func TestComplete(t *testing.T) {
	srv := newFakeServer(t, http.StatusOK, okBody)
	c := newClient(t, srv)
	ticks := []time.Time{time.Unix(100, 0), time.Unix(100, int64(400*time.Millisecond))}
	c.now = func() time.Time { tick := ticks[0]; ticks = ticks[1:]; return tick }

	resp, err := c.Complete(context.Background(), request())
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}

	if srv.path != "/v1beta/models/gemini-2.5-flash:generateContent" {
		t.Errorf("path = %q", srv.path)
	}
	if srv.query != "" {
		t.Errorf("query = %q, the API key must not travel in the URL", srv.query)
	}
	if got := srv.header.Get("X-Goog-Api-Key"); got != "g-key" {
		t.Errorf("X-Goog-Api-Key = %q", got)
	}
	wantBody := map[string]any{
		"systemInstruction": map[string]any{"parts": []any{map[string]any{"text": "Be brief."}}},
		"contents": []any{
			map[string]any{"role": "user", "parts": []any{map[string]any{"text": "Hi"}}},
			map[string]any{"role": "model", "parts": []any{map[string]any{"text": "Hello"}}},
			map[string]any{"role": "user", "parts": []any{map[string]any{"text": "Again"}}},
		},
		"generationConfig": map[string]any{
			"temperature":     0.0,
			"topP":            0.9,
			"maxOutputTokens": 64.0,
			"stopSequences":   []any{"END"},
		},
	}
	if got, want := mustJSON(t, srv.body), mustJSON(t, wantBody); got != want {
		t.Errorf("request body =\n%s\nwant\n%s", got, want)
	}

	want := domain.LLMResponse{
		ID:           "resp-1",
		Provider:     domain.ProviderGemini,
		Model:        "gemini-2.5-flash",
		Content:      "Hello, world",
		FinishReason: domain.FinishReasonStop,
		Usage:        domain.TokenUsage{InputTokens: 9, OutputTokens: 10},
		Latency:      400 * time.Millisecond,
		CreatedAt:    time.Unix(100, int64(400*time.Millisecond)).UTC(),
	}
	if *resp != want {
		t.Errorf("response = %+v, want %+v", *resp, want)
	}
}

func TestCompleteMinimalRequest(t *testing.T) {
	srv := newFakeServer(t, http.StatusOK, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"MAX_TOKENS"}]}`)
	resp, err := newClient(t, srv).Complete(context.Background(), &domain.LLMRequest{
		Model:    "models/tuned/my model",
		Messages: []domain.Message{{Role: domain.RoleUser, Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if srv.path != "/v1beta/models/tuned%2Fmy%20model:generateContent" {
		t.Errorf("path = %q, want the model escaped and the models/ prefix stripped", srv.path)
	}
	for _, absent := range []string{"systemInstruction", "generationConfig"} {
		if _, ok := srv.body[absent]; ok {
			t.Errorf("request body unexpectedly contains %q", absent)
		}
	}
	if resp.Model != "tuned/my model" || resp.Content != "ok" || resp.FinishReason != domain.FinishReasonLength {
		t.Errorf("response = %+v", *resp)
	}
}

func TestCompleteBlockedPrompt(t *testing.T) {
	srv := newFakeServer(t, http.StatusOK, `{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":5}}`)
	resp, err := newClient(t, srv).Complete(context.Background(), request())
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if resp.FinishReason != domain.FinishReasonContentFilter || resp.Content != "" || resp.Usage.InputTokens != 5 {
		t.Errorf("response = %+v, want an empty content_filter result", *resp)
	}
}

func TestCompleteErrors(t *testing.T) {
	tests := []struct {
		name      string
		status    int
		body      string
		retryable bool
		message   string
	}{
		{name: "invalid argument", status: 400, body: `{"error":{"code":400,"message":"bad request","status":"INVALID_ARGUMENT"}}`, message: "bad request"},
		{name: "permission denied", status: 403, body: `{"error":{"code":403,"message":"bad key","status":"PERMISSION_DENIED"}}`, message: "bad key"},
		{name: "quota", status: 429, body: `{"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}`, retryable: true, message: "quota exceeded"},
		{name: "unavailable", status: 503, body: `{"error":{"code":503,"message":"overloaded","status":"UNAVAILABLE"}}`, retryable: true, message: "overloaded"},
		{name: "no candidates", status: 200, body: `{}`, retryable: true, message: "response contained no candidates"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, tt.status, tt.body)
			resp, err := newClient(t, srv).Complete(context.Background(), request())
			if resp != nil {
				t.Errorf("response = %+v, want nil", resp)
			}
			var pe *domain.ProviderError
			if !errors.As(err, &pe) {
				t.Fatalf("error %v is not a *domain.ProviderError", err)
			}
			if pe.Provider != domain.ProviderGemini || pe.StatusCode != tt.status || pe.Retryable != tt.retryable {
				t.Errorf("got %+v, want status %d retryable %v", pe, tt.status, tt.retryable)
			}
			if pe.Err.Error() != tt.message {
				t.Errorf("message = %q, want %q", pe.Err.Error(), tt.message)
			}
		})
	}
}

func TestFinishReason(t *testing.T) {
	tests := map[string]domain.FinishReason{
		"STOP":               domain.FinishReasonStop,
		"MAX_TOKENS":         domain.FinishReasonLength,
		"SAFETY":             domain.FinishReasonContentFilter,
		"RECITATION":         domain.FinishReasonContentFilter,
		"BLOCKLIST":          domain.FinishReasonContentFilter,
		"PROHIBITED_CONTENT": domain.FinishReasonContentFilter,
		"SPII":               domain.FinishReasonContentFilter,
		"IMAGE_SAFETY":       domain.FinishReasonContentFilter,
		"OTHER":              domain.FinishReasonUnknown,
		"":                   domain.FinishReasonUnknown,
	}
	for in, want := range tests {
		if got := finishReason(in); got != want {
			t.Errorf("finishReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestNew(t *testing.T) {
	c, err := New(Config{APIKey: "key"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if c.baseURL != DefaultBaseURL || c.Name() != domain.ProviderGemini {
		t.Errorf("client = %+v", c)
	}
	if _, err := New(Config{}); !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("missing key: expected ErrInvalidInput, got %v", err)
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
