package domain

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
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

// MaxTemperature returns the highest sampling temperature the provider
// accepts. Vendors disagree: Anthropic stops at 1 and Mistral at 1.5, while
// OpenAI-style APIs and Gemini go up to 2.
func (p Provider) MaxTemperature() float64 {
	switch p {
	case ProviderAnthropic:
		return 1
	case ProviderMistral:
		return 1.5
	default:
		return 2
	}
}

// Target is a concrete (provider, model) pair a request can be sent to.
type Target struct {
	Provider Provider `json:"provider"`
	// Model is the vendor model identifier, e.g. "gpt-4o".
	Model string `json:"model"`
}

// Validate checks that the target names a known provider and a model.
func (t Target) Validate() error {
	var errs []error
	if !t.Provider.Valid() {
		errs = append(errs, fmt.Errorf("unknown provider %q", t.Provider))
	}
	if strings.TrimSpace(t.Model) == "" {
		errs = append(errs, errors.New("model is required"))
	}
	if len(errs) > 0 {
		return fmt.Errorf("%w: %w", ErrInvalidInput, errors.Join(errs...))
	}
	return nil
}

// String implements fmt.Stringer, e.g. "openai:gpt-4o".
func (t Target) String() string { return string(t.Provider) + ":" + t.Model }

// Route is a named, ordered fallback chain. A request that does not pin a
// provider uses its Model field as a route name; the gateway then tries each
// target in order, substituting the target's own vendor model identifier. This
// is what makes cross-provider fallback possible: a vendor model ID such as
// "gpt-4o" means nothing to another vendor.
type Route struct {
	Name    string   `json:"name"`
	Targets []Target `json:"targets"`
}

// Validate checks that the route has a name and at least one valid target.
func (r Route) Validate() error {
	var errs []error
	if strings.TrimSpace(r.Name) == "" {
		errs = append(errs, errors.New("route name is required"))
	}
	if len(r.Targets) == 0 {
		errs = append(errs, errors.New("route needs at least one target"))
	}
	for i, t := range r.Targets {
		if err := t.Validate(); err != nil {
			errs = append(errs, fmt.Errorf("targets[%d]: %w", i, err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("%w: route %q: %w", ErrInvalidInput, r.Name, errors.Join(errs...))
	}
	return nil
}

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

// ValidateFor is Validate plus the limits specific to provider.
func (p ModelParameters) ValidateFor(provider Provider) error {
	if err := p.Validate(); err != nil {
		return err
	}
	if limit := provider.MaxTemperature(); p.Temperature != nil && *p.Temperature > limit {
		return fmt.Errorf("temperature must be in [0, %g] for provider %s, got %g", limit, provider, *p.Temperature)
	}
	return nil
}

// ClampFor returns a copy of p with Temperature lowered to the provider's
// maximum when it exceeds it. The gateway uses it when falling back along a
// route, so that a temperature valid for the first target does not make every
// stricter target fail.
func (p ModelParameters) ClampFor(provider Provider) ModelParameters {
	if limit := provider.MaxTemperature(); p.Temperature != nil && *p.Temperature > limit {
		p.Temperature = &limit
	}
	return p
}

// LLMRequest is the provider-agnostic completion request accepted by the
// gateway. Provider adapters translate it into vendor-specific payloads.
type LLMRequest struct {
	// Provider optionally pins the request to a single vendor, with no
	// fallback. When empty, Model names a Route and the gateway walks that
	// route's fallback chain.
	Provider Provider `json:"provider,omitempty"`
	// Model is the vendor model identifier (e.g. "gpt-4o") when Provider is
	// set, and a route name (e.g. "fast") when it is not.
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
	if err := r.Parameters.ValidateFor(r.Provider); err != nil {
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

// Price is what a provider charges for a model, in US dollars per million
// tokens.
type Price struct {
	InputPerMTok  float64 `json:"input_per_mtok"`
	OutputPerMTok float64 `json:"output_per_mtok"`
}

// Validate checks that neither rate is negative.
func (p Price) Validate() error {
	if p.InputPerMTok < 0 || p.OutputPerMTok < 0 {
		return fmt.Errorf("%w: prices must not be negative, got %g/%g", ErrInvalidInput, p.InputPerMTok, p.OutputPerMTok)
	}
	return nil
}

// Cost returns the price of usage in US dollars. It is an estimate: it bills
// every input token at the plain input rate, ignoring vendor discounts such as
// cached-prompt or batch pricing.
func (p Price) Cost(usage TokenUsage) float64 {
	return (float64(usage.InputTokens)*p.InputPerMTok + float64(usage.OutputTokens)*p.OutputPerMTok) / 1e6
}

// ModelPrice is the price of one target.
type ModelPrice struct {
	Target
	Price
}

// Validate checks the target and the price.
func (m ModelPrice) Validate() error {
	if err := m.Target.Validate(); err != nil {
		return err
	}
	return m.Price.Validate()
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
	// CostUSD is the estimated cost of the completion, set by the gateway
	// when a price is configured for the target that served it and nil
	// otherwise.
	CostUSD *float64 `json:"cost_usd,omitempty"`
	// Latency is the wall-clock time spent waiting on the provider. It is
	// serialised as integer milliseconds under "latency_ms".
	Latency   time.Duration `json:"-"`
	CreatedAt time.Time     `json:"created_at"`
}

// MarshalJSON encodes Latency as "latency_ms" instead of time.Duration's
// default of raw nanoseconds.
func (r LLMResponse) MarshalJSON() ([]byte, error) {
	type alias LLMResponse
	return json.Marshal(struct {
		alias
		LatencyMS int64 `json:"latency_ms"`
	}{alias(r), r.Latency.Milliseconds()})
}

// UnmarshalJSON is the inverse of MarshalJSON.
func (r *LLMResponse) UnmarshalJSON(data []byte) error {
	type alias LLMResponse
	aux := struct {
		*alias
		LatencyMS int64 `json:"latency_ms"`
	}{alias: (*alias)(r)}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	r.Latency = time.Duration(aux.LatencyMS) * time.Millisecond
	return nil
}

// LLMProvider is the port every vendor adapter implements. Implementations
// must be safe for concurrent use, honour ctx cancellation, and return a
// *ProviderError for upstream failures so the gateway can decide whether to
// retry.
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
	// Retryable reports whether the same request may succeed if sent to the
	// same provider again (timeouts, 429s, 5xx). It does not govern fallback:
	// a non-retryable failure such as a rejected API key or an unknown model
	// is specific to one provider, so the gateway still tries the next target
	// of a route.
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
