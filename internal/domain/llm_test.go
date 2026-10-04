package domain

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

func ptr[T any](v T) *T { return &v }

func validRequest() *LLMRequest {
	return &LLMRequest{
		Model:    "gpt-4o",
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
	}
}

func TestLLMRequestValidate(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(r *LLMRequest)
		wantErr bool
	}{
		{name: "valid", mutate: func(*LLMRequest) {}},
		{name: "valid with pinned provider", mutate: func(r *LLMRequest) { r.Provider = ProviderAnthropic }},
		{name: "unknown provider", mutate: func(r *LLMRequest) { r.Provider = "cohere" }, wantErr: true},
		{name: "missing model", mutate: func(r *LLMRequest) { r.Model = "" }, wantErr: true},
		{name: "no messages", mutate: func(r *LLMRequest) { r.Messages = nil }, wantErr: true},
		{name: "invalid role", mutate: func(r *LLMRequest) { r.Messages[0].Role = "system" }, wantErr: true},
		{name: "empty content", mutate: func(r *LLMRequest) { r.Messages[0].Content = "" }, wantErr: true},
		{name: "temperature too high", mutate: func(r *LLMRequest) { r.Parameters.Temperature = ptr(2.5) }, wantErr: true},
		{name: "temperature above pinned provider limit", mutate: func(r *LLMRequest) {
			r.Provider, r.Parameters.Temperature = ProviderAnthropic, ptr(1.2)
		}, wantErr: true},
		{name: "same temperature within another provider's limit", mutate: func(r *LLMRequest) {
			r.Provider, r.Parameters.Temperature = ProviderOpenAI, ptr(1.2)
		}},
		{name: "zero temperature allowed", mutate: func(r *LLMRequest) { r.Parameters.Temperature = ptr(0.0) }},
		{name: "top_p out of range", mutate: func(r *LLMRequest) { r.Parameters.TopP = ptr(-0.1) }, wantErr: true},
		{name: "negative max tokens", mutate: func(r *LLMRequest) { r.Parameters.MaxTokens = -1 }, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := validRequest()
			tt.mutate(r)
			err := r.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil && !errors.Is(err, ErrInvalidInput) {
				t.Errorf("Validate() error %v does not wrap ErrInvalidInput", err)
			}
		})
	}
}

func TestTokenUsage(t *testing.T) {
	u := TokenUsage{InputTokens: 10, OutputTokens: 5}.Add(TokenUsage{InputTokens: 1, OutputTokens: 2})
	if u.InputTokens != 11 || u.OutputTokens != 7 || u.Total() != 18 {
		t.Errorf("unexpected usage %+v (total %d)", u, u.Total())
	}
}

func TestProviderError(t *testing.T) {
	cause := errors.New("upstream boom")
	withStatus := &ProviderError{Provider: ProviderOpenAI, StatusCode: 503, Retryable: true, Err: cause}
	if got, want := withStatus.Error(), "provider openai: status 503: upstream boom"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
	transport := &ProviderError{Provider: ProviderAnthropic, Err: cause}
	if got, want := transport.Error(), "provider anthropic: upstream boom"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
	if !errors.Is(withStatus, cause) {
		t.Error("ProviderError should unwrap to its cause")
	}
}

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "nil", err: nil, want: false},
		{name: "plain error", err: errors.New("x"), want: false},
		{name: "retryable", err: &ProviderError{Retryable: true, Err: errors.New("429")}, want: true},
		{name: "non-retryable", err: &ProviderError{Retryable: false, Err: errors.New("400")}, want: false},
		{name: "wrapped retryable", err: fmt.Errorf("call: %w", &ProviderError{Retryable: true, Err: errors.New("503")}), want: true},
		{name: "per-attempt timeout", err: &ProviderError{Retryable: true, Err: context.DeadlineExceeded}, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsRetryable(tt.err); got != tt.want {
				t.Errorf("IsRetryable() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestModelParametersForProvider(t *testing.T) {
	limits := map[Provider]float64{
		ProviderOpenAI: 2, ProviderAnthropic: 1, ProviderGemini: 2,
		ProviderMistral: 1.5, ProviderOllama: 2, ProviderOpenAICompatible: 2,
	}
	for provider, limit := range limits {
		if got := provider.MaxTemperature(); got != limit {
			t.Errorf("%s.MaxTemperature() = %g, want %g", provider, got, limit)
		}
	}

	hot := ModelParameters{Temperature: ptr(1.8), MaxTokens: 10}
	if err := hot.ValidateFor(ProviderOpenAI); err != nil {
		t.Errorf("ValidateFor(openai) error = %v", err)
	}
	if err := hot.ValidateFor(ProviderMistral); err == nil {
		t.Error("ValidateFor(mistral) accepted temperature 1.8")
	}
	if err := (ModelParameters{MaxTokens: -1}).ValidateFor(ProviderOpenAI); err == nil {
		t.Error("ValidateFor() skipped the generic checks")
	}

	clamped := hot.ClampFor(ProviderAnthropic)
	if *clamped.Temperature != 1 || clamped.MaxTokens != 10 {
		t.Errorf("ClampFor(anthropic) = %+v", clamped)
	}
	if *hot.Temperature != 1.8 {
		t.Error("ClampFor() mutated its receiver")
	}
	if same := hot.ClampFor(ProviderOpenAI); *same.Temperature != 1.8 {
		t.Errorf("ClampFor(openai) changed temperature to %g", *same.Temperature)
	}
	if unset := (ModelParameters{}).ClampFor(ProviderAnthropic); unset.Temperature != nil {
		t.Error("ClampFor() set a temperature that was not set")
	}
}

func TestTargetAndRoute(t *testing.T) {
	target := Target{Provider: ProviderOpenAI, Model: "gpt-4o"}
	if err := target.Validate(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if got := target.String(); got != "openai:gpt-4o" {
		t.Errorf("String() = %q", got)
	}
	if err := (Target{Provider: "x", Model: " "}).Validate(); !errors.Is(err, ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}

	route := Route{Name: "fast", Targets: []Target{target, {Provider: ProviderAnthropic, Model: "claude-haiku-4-5"}}}
	if err := route.Validate(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	tests := []struct {
		name  string
		route Route
	}{
		{name: "no name", route: Route{Targets: []Target{target}}},
		{name: "no targets", route: Route{Name: "fast"}},
		{name: "bad target", route: Route{Name: "fast", Targets: []Target{{Provider: ProviderOpenAI}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.route.Validate(); !errors.Is(err, ErrInvalidInput) {
				t.Errorf("expected ErrInvalidInput, got %v", err)
			}
		})
	}
}

func TestLLMResponseJSON(t *testing.T) {
	in := LLMResponse{
		ID: "resp_1", Provider: ProviderOpenAI, Model: "gpt-4o", Content: "hi",
		FinishReason: FinishReasonStop,
		Usage:        TokenUsage{InputTokens: 3, OutputTokens: 1},
		Latency:      250 * time.Millisecond,
		CreatedAt:    time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC),
	}
	data, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if !strings.Contains(string(data), `"latency_ms":250`) || strings.Contains(string(data), `"latency":`) {
		t.Errorf("Marshal() = %s, want latency_ms only", data)
	}
	var out LLMResponse
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if out != in {
		t.Errorf("round trip = %+v, want %+v", out, in)
	}
	if err := json.Unmarshal([]byte(`{"usage":"x"}`), &out); err == nil {
		t.Error("Unmarshal() accepted a malformed response")
	}
}
