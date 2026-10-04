package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

func ptr[T any](v T) *T { return &v }

const okBody = `{
	"id": "msg_01",
	"type": "message",
	"role": "assistant",
	"model": "claude-opus-5-5",
	"content": [
		{"type": "thinking", "thinking": "", "signature": "sig"},
		{"type": "text", "text": "Hello"},
		{"type": "text", "text": ", world"}
	],
	"stop_reason": "end_turn",
	"stop_sequence": null,
	"usage": {"input_tokens": 10, "cache_creation_input_tokens": 2, "cache_read_input_tokens": 3, "output_tokens": 7}
}`

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
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(f.Close)
	return f
}

func newClient(t *testing.T, srv *fakeServer) *Client {
	t.Helper()
	c, err := New(Config{APIKey: "sk-ant-test", BaseURL: srv.URL, HTTPClient: srv.Client()})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return c
}

func request(model string) *domain.LLMRequest {
	return &domain.LLMRequest{
		Provider:     domain.ProviderAnthropic,
		Model:        model,
		SystemPrompt: "Be brief.",
		Messages: []domain.Message{
			{Role: domain.RoleUser, Content: "Hi"},
			{Role: domain.RoleAssistant, Content: "Hello"},
			{Role: domain.RoleUser, Content: "Again"},
		},
		Parameters: domain.ModelParameters{
			Temperature: ptr(0.5),
			TopP:        ptr(0.9),
			MaxTokens:   64,
			Stop:        []string{"END"},
		},
	}
}

func TestCompleteCurrentModel(t *testing.T) {
	srv := newFakeServer(t, http.StatusOK, okBody)
	c := newClient(t, srv)
	ticks := []time.Time{time.Unix(100, 0), time.Unix(101, 0)}
	c.now = func() time.Time { tick := ticks[0]; ticks = ticks[1:]; return tick }

	resp, err := c.Complete(context.Background(), request("claude-opus-5-5"))
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}

	if srv.path != "/v1/messages" {
		t.Errorf("path = %q", srv.path)
	}
	if got := srv.header.Get("X-Api-Key"); got != "sk-ant-test" {
		t.Errorf("X-Api-Key = %q", got)
	}
	if got := srv.header.Get("Anthropic-Version"); got == "" {
		t.Error("Anthropic-Version header missing")
	}
	wantBody := map[string]any{
		"model":      "claude-opus-5-5",
		"max_tokens": 64.0,
		"system":     []any{map[string]any{"type": "text", "text": "Be brief."}},
		"messages": []any{
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "Hi"}}},
			map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text", "text": "Hello"}}},
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "Again"}}},
		},
		"stop_sequences": []any{"END"},
	}
	if got, want := mustJSON(t, srv.body), mustJSON(t, wantBody); got != want {
		t.Errorf("request body =\n%s\nwant (no sampling parameters)\n%s", got, want)
	}

	want := domain.LLMResponse{
		ID:           "msg_01",
		Provider:     domain.ProviderAnthropic,
		Model:        "claude-opus-5-5",
		Content:      "Hello, world",
		FinishReason: domain.FinishReasonStop,
		Usage:        domain.TokenUsage{InputTokens: 15, OutputTokens: 7},
		Latency:      time.Second,
		CreatedAt:    time.Unix(101, 0).UTC(),
	}
	if *resp != want {
		t.Errorf("response = %+v, want %+v", *resp, want)
	}
}

func TestCompleteSamplingParameters(t *testing.T) {
	tests := []struct {
		name            string
		mutate          func(r *domain.LLMRequest)
		wantTemperature any
		wantTopP        any
	}{
		{name: "temperature wins over top_p", mutate: func(*domain.LLMRequest) {}, wantTemperature: 0.5},
		{name: "top_p alone", mutate: func(r *domain.LLMRequest) { r.Parameters.Temperature = nil }, wantTopP: 0.9},
		{name: "neither", mutate: func(r *domain.LLMRequest) { r.Parameters.Temperature, r.Parameters.TopP = nil, nil }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, http.StatusOK, okBody)
			req := request("claude-haiku-4-5")
			tt.mutate(req)
			if _, err := newClient(t, srv).Complete(context.Background(), req); err != nil {
				t.Fatalf("Complete() error = %v", err)
			}
			if got := srv.body["temperature"]; got != tt.wantTemperature {
				t.Errorf("temperature = %v, want %v", got, tt.wantTemperature)
			}
			if got := srv.body["top_p"]; got != tt.wantTopP {
				t.Errorf("top_p = %v, want %v", got, tt.wantTopP)
			}
		})
	}
}

func TestCompleteDefaults(t *testing.T) {
	srv := newFakeServer(t, http.StatusOK, `{"id":"msg_02","content":[],"stop_reason":"max_tokens","usage":{"input_tokens":1,"output_tokens":0}}`)
	resp, err := newClient(t, srv).Complete(context.Background(), &domain.LLMRequest{
		Model:    "claude-sonnet-5-5",
		Messages: []domain.Message{{Role: domain.RoleUser, Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if srv.body["max_tokens"] != float64(DefaultMaxTokens) {
		t.Errorf("max_tokens = %v, want the default %d", srv.body["max_tokens"], DefaultMaxTokens)
	}
	for _, absent := range []string{"system", "stop_sequences", "temperature", "top_p"} {
		if _, ok := srv.body[absent]; ok {
			t.Errorf("request body unexpectedly contains %q", absent)
		}
	}
	if resp.Model != "claude-sonnet-5-5" || resp.Content != "" || resp.FinishReason != domain.FinishReasonLength {
		t.Errorf("response = %+v", *resp)
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
		{name: "invalid request", status: 400, body: `{"type":"error","error":{"type":"invalid_request_error","message":"bad model"}}`, message: "bad model"},
		{name: "unauthorized", status: 401, body: `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`, message: "invalid x-api-key"},
		{name: "rate limited", status: 429, body: `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`, retryable: true, message: "slow down"},
		{name: "overloaded", status: 529, body: `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`, retryable: true, message: "Overloaded"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newFakeServer(t, tt.status, tt.body)
			resp, err := newClient(t, srv).Complete(context.Background(), request("claude-opus-5-5"))
			if resp != nil {
				t.Errorf("response = %+v, want nil", resp)
			}
			var pe *domain.ProviderError
			if !errors.As(err, &pe) {
				t.Fatalf("error %v is not a *domain.ProviderError", err)
			}
			if pe.Provider != domain.ProviderAnthropic || pe.StatusCode != tt.status || pe.Retryable != tt.retryable {
				t.Errorf("got %+v, want status %d retryable %v", pe, tt.status, tt.retryable)
			}
			if pe.Err.Error() != tt.message {
				t.Errorf("message = %q, want %q", pe.Err.Error(), tt.message)
			}
		})
	}
}

func TestCompleteTransportErrors(t *testing.T) {
	srv := newFakeServer(t, http.StatusOK, okBody)
	c := newClient(t, srv)
	srv.Close()

	_, err := c.Complete(context.Background(), request("claude-opus-5-5"))
	var pe *domain.ProviderError
	if !errors.As(err, &pe) {
		t.Fatalf("error %v is not a *domain.ProviderError", err)
	}
	if !pe.Retryable || pe.StatusCode != 0 {
		t.Errorf("connection failure: got %+v, want retryable with status 0", pe)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = c.Complete(ctx, request("claude-opus-5-5"))
	if !errors.As(err, &pe) {
		t.Fatalf("error %v is not a *domain.ProviderError", err)
	}
	if pe.Retryable || !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled request: got %+v, want a non-retryable context.Canceled", pe)
	}
}

func TestAcceptsSampling(t *testing.T) {
	tests := map[string]bool{
		"claude-3-5-haiku-20241022":  true,
		"claude-haiku-4-5":           true,
		"claude-haiku-4-5-20251001":  true,
		"claude-sonnet-4-6":          true,
		"claude-sonnet-4-5-20250929": true,
		"claude-opus-4-6":            true,
		"claude-opus-4-20250514":     true,
		"claude-opus-4-7":            false,
		"claude-opus-4-8":            false,
		"claude-opus-5":              false,
		"claude-opus-5-5":            false,
		"claude-sonnet-5":            false,
		"claude-sonnet-5-5":          false,
		"claude-fable-5-1":           false,
		"some-future-model":          false,
	}
	for model, want := range tests {
		if got := AcceptsSampling(model); got != want {
			t.Errorf("AcceptsSampling(%q) = %v, want %v", model, got, want)
		}
	}
}

func TestFinishReason(t *testing.T) {
	tests := map[sdk.StopReason]domain.FinishReason{
		sdk.StopReasonEndTurn:                    domain.FinishReasonStop,
		sdk.StopReasonStopSequence:               domain.FinishReasonStop,
		sdk.StopReasonMaxTokens:                  domain.FinishReasonLength,
		sdk.StopReasonModelContextWindowExceeded: domain.FinishReasonLength,
		sdk.StopReasonRefusal:                    domain.FinishReasonContentFilter,
		sdk.StopReasonToolUse:                    domain.FinishReasonUnknown,
		sdk.StopReasonPauseTurn:                  domain.FinishReasonUnknown,
	}
	for in, want := range tests {
		if got := finishReason(in); got != want {
			t.Errorf("finishReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestNew(t *testing.T) {
	c, err := New(Config{APIKey: "key", DefaultMaxTokens: 512})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if c.Name() != domain.ProviderAnthropic || c.defaultMaxTokens != 512 {
		t.Errorf("client = %+v", c)
	}
	if _, err := New(Config{APIKey: "  "}); !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("missing key: expected ErrInvalidInput, got %v", err)
	}
	if _, err := New(Config{APIKey: "key", DefaultMaxTokens: -1}); !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("negative default: expected ErrInvalidInput, got %v", err)
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
