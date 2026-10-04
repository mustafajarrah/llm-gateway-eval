// Package anthropic implements domain.LLMProvider on top of the official
// Anthropic Go SDK (Messages API).
package anthropic

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
	"github.com/mustafajarrah/llm-gateway-eval/internal/provider/upstream"
)

// DefaultMaxTokens is the output limit sent when a request does not set one.
// The Messages API requires max_tokens, and on current models the budget also
// covers the model's thinking, so a small default would truncate answers.
const DefaultMaxTokens = 16000

// Config configures a Client.
type Config struct {
	// APIKey is required.
	APIKey string
	// BaseURL overrides the API root; it defaults to the SDK's
	// https://api.anthropic.com.
	BaseURL string
	// DefaultMaxTokens overrides DefaultMaxTokens.
	DefaultMaxTokens int
	// HTTPClient defaults to the SDK's client.
	HTTPClient *http.Client
}

// Client is a domain.LLMProvider backed by the Anthropic Messages API. It is
// safe for concurrent use.
type Client struct {
	sdk              sdk.Client
	defaultMaxTokens int
	now              func() time.Time
}

var _ domain.LLMProvider = (*Client)(nil)

// New builds a Client from cfg.
func New(cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.APIKey) == "" {
		return nil, fmt.Errorf("%w: anthropic requires an API key", domain.ErrInvalidInput)
	}
	if cfg.DefaultMaxTokens < 0 {
		return nil, fmt.Errorf("%w: default max tokens must not be negative", domain.ErrInvalidInput)
	}
	opts := []option.RequestOption{
		option.WithAPIKey(cfg.APIKey),
		// Retries and fallback belong to the gateway, which sees every
		// provider; SDK-level retries would multiply with them.
		option.WithMaxRetries(0),
	}
	if baseURL := strings.TrimSpace(cfg.BaseURL); baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
	}
	maxTokens := cfg.DefaultMaxTokens
	if maxTokens == 0 {
		maxTokens = DefaultMaxTokens
	}
	return &Client{sdk: sdk.NewClient(opts...), defaultMaxTokens: maxTokens, now: time.Now}, nil
}

// Name returns the provider identifier.
func (c *Client) Name() domain.Provider { return domain.ProviderAnthropic }

// Complete performs a single, non-streaming completion.
func (c *Client) Complete(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
	params := sdk.MessageNewParams{
		Model:         sdk.Model(req.Model),
		MaxTokens:     int64(c.defaultMaxTokens),
		Messages:      make([]sdk.MessageParam, 0, len(req.Messages)),
		StopSequences: req.Parameters.Stop,
	}
	if req.Parameters.MaxTokens > 0 {
		params.MaxTokens = int64(req.Parameters.MaxTokens)
	}
	if req.SystemPrompt != "" {
		params.System = []sdk.TextBlockParam{{Text: req.SystemPrompt}}
	}
	for _, m := range req.Messages {
		block := sdk.NewTextBlock(m.Content)
		if m.Role == domain.RoleAssistant {
			params.Messages = append(params.Messages, sdk.NewAssistantMessage(block))
		} else {
			params.Messages = append(params.Messages, sdk.NewUserMessage(block))
		}
	}
	if AcceptsSampling(req.Model) {
		// The models that still take sampling parameters reject a request
		// carrying both, so temperature wins when both are set.
		switch {
		case req.Parameters.Temperature != nil:
			params.Temperature = sdk.Float(*req.Parameters.Temperature)
		case req.Parameters.TopP != nil:
			params.TopP = sdk.Float(*req.Parameters.TopP)
		}
	}

	start := c.now()
	msg, err := c.sdk.Messages.New(ctx, params)
	if err != nil {
		return nil, providerError(err)
	}
	finished := c.now()

	// Thinking blocks precede the answer on current models; only text blocks
	// are part of the completion.
	var content strings.Builder
	for _, block := range msg.Content {
		if block.Type == "text" {
			content.WriteString(block.Text)
		}
	}
	model := string(msg.Model)
	if model == "" {
		model = req.Model
	}
	return &domain.LLMResponse{
		ID:           msg.ID,
		Provider:     domain.ProviderAnthropic,
		Model:        model,
		Content:      content.String(),
		FinishReason: finishReason(msg.StopReason),
		Usage: domain.TokenUsage{
			// input_tokens excludes cached tokens; add them back so the
			// figure is the full prompt size.
			InputTokens:  int(msg.Usage.InputTokens + msg.Usage.CacheCreationInputTokens + msg.Usage.CacheReadInputTokens),
			OutputTokens: int(msg.Usage.OutputTokens),
		},
		Latency:   finished.Sub(start),
		CreatedAt: finished.UTC(),
	}, nil
}

// samplingModelPrefixes lists the model families that still accept the
// temperature and top_p parameters.
var samplingModelPrefixes = []string{
	"claude-3",
	"claude-haiku-4",
	"claude-sonnet-4",
	"claude-opus-4-0",
	"claude-opus-4-1",
	"claude-opus-4-5",
	"claude-opus-4-6",
	"claude-opus-4-2025",
}

// AcceptsSampling reports whether model takes temperature and top_p. Anthropic
// removed them starting with Opus 4.7 and the Claude 5 family, where sending
// them is a 400; the adapter omits them there instead of failing the request.
// Unknown models are treated as current ones, since removal is the direction
// of travel.
func AcceptsSampling(model string) bool {
	for _, prefix := range samplingModelPrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

func finishReason(reason sdk.StopReason) domain.FinishReason {
	switch reason {
	case sdk.StopReasonEndTurn, sdk.StopReasonStopSequence:
		return domain.FinishReasonStop
	case sdk.StopReasonMaxTokens, sdk.StopReasonModelContextWindowExceeded:
		return domain.FinishReasonLength
	case sdk.StopReasonRefusal:
		return domain.FinishReasonContentFilter
	default:
		return domain.FinishReasonUnknown
	}
}

// providerError converts an SDK error into a *domain.ProviderError.
func providerError(err error) error {
	var apiErr *sdk.Error
	if errors.As(err, &apiErr) {
		return &domain.ProviderError{
			Provider:   domain.ProviderAnthropic,
			StatusCode: apiErr.StatusCode,
			Retryable:  upstream.RetryableStatus(apiErr.StatusCode),
			Err:        errors.New(upstream.ErrorMessage([]byte(apiErr.RawJSON()))),
		}
	}
	// No HTTP response: a cancelled request must not be retried; any other
	// transport failure may succeed next time.
	return &domain.ProviderError{
		Provider:  domain.ProviderAnthropic,
		Retryable: !errors.Is(err, context.Canceled),
		Err:       err,
	}
}
