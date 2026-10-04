package domain

import (
	"context"
	"errors"
	"fmt"
	"testing"
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
