package domain

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Provider identifies an upstream LLM vendor.
type Provider string

// Supported providers.
const (
	ProviderOpenAI    Provider = "openai"
	ProviderAnthropic Provider = "anthropic"
	ProviderGemini    Provider = "gemini"
	ProviderMistral   Provider = "mistral"
	// ProviderOllama serves locally hosted models; it needs no API key.
	ProviderOllama Provider = "ollama"
	// ProviderOpenAICompatible is any endpoint speaking the OpenAI chat
	// completions protocol at a configurable base URL (Groq, OpenRouter,
	// Together, vLLM, Azure OpenAI, ...).
	ProviderOpenAICompatible Provider = "openai_compatible"
)

// Valid reports whether p is a provider known to the gateway.
func (p Provider) Valid() bool {
	switch p {
	case ProviderOpenAI, ProviderAnthropic, ProviderGemini, ProviderMistral,
		ProviderOllama, ProviderOpenAICompatible:
		return true
	default:
		return false
	}
}

// String implements fmt.Stringer.
func (p Provider) String() string { return string(p) }

// Role is the author of a chat message.
type Role string

// Conversation roles. System instructions are carried separately in
// LLMRequest.SystemPrompt because providers disagree on how to transmit them
// (OpenAI-style APIs use a message, Anthropic and Gemini a top-level field).
const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Valid reports whether r is a supported conversation role.
func (r Role) Valid() bool {
	return r == RoleUser || r == RoleAssistant
}

// Message is a single turn in a conversation.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

// ModelParameters are the sampling knobs common to all providers. Pointer
// fields distinguish "not set" (use the provider default) from an explicit
// zero value.
type ModelParameters struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

// Validate checks that the parameters are within provider-agnostic bounds.
func (p ModelParameters) Validate() error {
	var errs []error
	if p.Temperature != nil && (*p.Temperature < 0 || *p.Temperature > 2) {
		errs = append(errs, fmt.Errorf("temperature must be in [0, 2], got %g", *p.Temperature))
	}
	if p.TopP != nil && (*p.TopP < 0 || *p.TopP > 1) {
		errs = append(errs, fmt.Errorf("top_p must be in [0, 1], got %g", *p.TopP))
	}
	if p.MaxTokens < 0 {
		errs = append(errs, fmt.Errorf("max_tokens must be non-negative, got %d", p.MaxTokens))
	}
	return errors.Join(errs...)
}

// LLMRequest is the provider-agnostic completion request accepted by the
// gateway. Provider adapters translate it into vendor-specific payloads.
type LLMRequest struct {
	// Provider optionally pins the request to a single vendor. When empty the
	// gateway routes according to its configured fallback chain.
	Provider Provider `json:"provider,omitempty"`
	// Model is the vendor model identifier, e.g. "gpt-4o" or "claude-sonnet-4-5".
	Model        string          `json:"model"`
	SystemPrompt string          `json:"system_prompt,omitempty"`
	Messages     []Message       `json:"messages"`
	Parameters   ModelParameters `json:"parameters"`
	// Metadata is opaque caller data (trace IDs, tenant IDs) propagated to
	// logs and persisted results; it is never sent to the provider.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// Validate checks the request for structural errors. The returned error wraps
// ErrInvalidInput so transports can map it to a 400 response.
func (r *LLMRequest) Validate() error {
	var errs []error
	if r.Provider != "" && !r.Provider.Valid() {
		errs = append(errs, fmt.Errorf("unknown provider %q", r.Provider))
	}
	if r.Model == "" {
		errs = append(errs, errors.New("model is required"))
	}
	if len(r.Messages) == 0 {
		errs = append(errs, errors.New("at least one message is required"))
	}
	for i, m := range r.Messages {
		if !m.Role.Valid() {
			errs = append(errs, fmt.Errorf("messages[%d]: invalid role %q", i, m.Role))
		}
		if m.Content == "" {
			errs = append(errs, fmt.Errorf("messages[%d]: content is required", i))
		}
	}
	if err := r.Parameters.Validate(); err != nil {
		errs = append(errs, err)
	}
	if len(errs) > 0 {
		return fmt.Errorf("%w: %w", ErrInvalidInput, errors.Join(errs...))
	}
	return nil
}

// FinishReason explains why the model stopped generating.
type FinishReason string

// Normalised finish reasons. Adapters map vendor values (e.g. OpenAI "length",
// Anthropic "max_tokens") onto these.
const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonContentFilter FinishReason = "content_filter"
	FinishReasonUnknown       FinishReason = "unknown"
)

// TokenUsage reports token consumption for a single completion.
type TokenUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Total returns the sum of input and output tokens.
func (u TokenUsage) Total() int { return u.InputTokens + u.OutputTokens }

// Add returns the element-wise sum of u and other, useful for aggregating
// usage across an evaluation run.
func (u TokenUsage) Add(other TokenUsage) TokenUsage {
	return TokenUsage{
		InputTokens:  u.InputTokens + other.InputTokens,
		OutputTokens: u.OutputTokens + other.OutputTokens,
	}
}

// LLMResponse is the provider-agnostic completion result.
type LLMResponse struct {
	// ID is the vendor-assigned response identifier, kept for traceability.
	ID           string       `json:"id"`
	Provider     Provider     `json:"provider"`
	Model        string       `json:"model"`
	Content      string       `json:"content"`
	FinishReason FinishReason `json:"finish_reason"`
	Usage        TokenUsage   `json:"usage"`
	// Latency is the wall-clock time spent waiting on the provider.
	Latency   time.Duration `json:"latency"`
	CreatedAt time.Time     `json:"created_at"`
}

// LLMProvider is the port every vendor adapter implements. Implementations
// must be safe for concurrent use, honour ctx cancellation, and return a
// *ProviderError for upstream failures so the fallback layer can decide
// whether to try the next provider.
type LLMProvider interface {
	// Name returns the provider identifier.
	Name() Provider
	// Complete performs a single, non-streaming completion.
	Complete(ctx context.Context, req *LLMRequest) (*LLMResponse, error)
}

// ProviderError describes a failure returned by an upstream provider.
type ProviderError struct {
	Provider Provider
	// StatusCode is the upstream HTTP status, or 0 for transport-level errors.
	StatusCode int
	// Retryable reports whether the same request may succeed if retried or
	// routed to another provider (timeouts, 429s, 5xx).
	Retryable bool
	Err       error
}

// Error implements the error interface.
func (e *ProviderError) Error() string {
	if e.StatusCode != 0 {
		return fmt.Sprintf("provider %s: status %d: %v", e.Provider, e.StatusCode, e.Err)
	}
	return fmt.Sprintf("provider %s: %v", e.Provider, e.Err)
}

// Unwrap exposes the underlying cause to errors.Is / errors.As.
func (e *ProviderError) Unwrap() error { return e.Err }

// IsRetryable reports whether err (or any error it wraps) is a retryable
// *ProviderError. It deliberately ignores the caller's context: a per-attempt
// timeout is retryable, whereas a cancelled parent context is not, and only
// the fallback layer (which owns both) can tell them apart via ctx.Err().
func IsRetryable(err error) bool {
	var pe *ProviderError
	return errors.As(err, &pe) && pe.Retryable
}
