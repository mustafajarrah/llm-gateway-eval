// Package openaicompat implements domain.LLMProvider for every vendor that
// speaks the OpenAI chat completions protocol: OpenAI itself, Mistral, Ollama
// and any other compatible endpoint (Groq, OpenRouter, Together, vLLM, Azure
// OpenAI, ...). The vendors differ only in base URL, API key and a couple of
// field names.
package openaicompat

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
	"github.com/mustafajarrah/llm-gateway-eval/internal/provider/upstream"
)

// Default base URLs of the vendors with a well-known endpoint.
const (
	OpenAIBaseURL  = "https://api.openai.com/v1"
	MistralBaseURL = "https://api.mistral.ai/v1"
	OllamaBaseURL  = "http://localhost:11434/v1"
)

// DefaultBaseURL returns the well-known base URL for provider, or "" when the
// provider has none and the caller must supply one.
func DefaultBaseURL(provider domain.Provider) string {
	switch provider {
	case domain.ProviderOpenAI:
		return OpenAIBaseURL
	case domain.ProviderMistral:
		return MistralBaseURL
	case domain.ProviderOllama:
		return OllamaBaseURL
	default:
		return ""
	}
}

// Config configures a Client.
type Config struct {
	// Provider is the identifier the client reports; it must be one of
	// openai, mistral, ollama or openai_compatible.
	Provider domain.Provider
	// BaseURL is the API root, up to and including the version segment (e.g.
	// "https://api.openai.com/v1"). It defaults to DefaultBaseURL(Provider)
	// and is required for openai_compatible.
	BaseURL string
	// APIKey is sent as a bearer token. It may be empty for endpoints that
	// need no authentication, such as a local Ollama.
	APIKey string
	// HTTPClient defaults to a client without a timeout; the gateway bounds
	// each attempt through the request context.
	HTTPClient *http.Client
}

// Client is a domain.LLMProvider backed by an OpenAI-compatible endpoint. It
// is safe for concurrent use.
type Client struct {
	provider domain.Provider
	endpoint string
	apiKey   string
	http     *http.Client
	now      func() time.Time
}

var _ domain.LLMProvider = (*Client)(nil)

// New builds a Client from cfg.
func New(cfg Config) (*Client, error) {
	switch cfg.Provider {
	case domain.ProviderOpenAI, domain.ProviderMistral, domain.ProviderOllama, domain.ProviderOpenAICompatible:
	default:
		return nil, fmt.Errorf("%w: provider %q does not speak the OpenAI protocol", domain.ErrInvalidInput, cfg.Provider)
	}
	baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL(cfg.Provider)
	}
	if baseURL == "" {
		return nil, fmt.Errorf("%w: provider %s requires a base URL", domain.ErrInvalidInput, cfg.Provider)
	}
	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	return &Client{
		provider: cfg.Provider,
		endpoint: baseURL + "/chat/completions",
		apiKey:   cfg.APIKey,
		http:     client,
		now:      time.Now,
	}, nil
}

// Name returns the provider identifier.
func (c *Client) Name() domain.Provider { return c.provider }

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature *float64      `json:"temperature,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
	// MaxTokens is the limit field understood by compatible endpoints;
	// OpenAI itself deprecated it in favour of MaxCompletionTokens, which is
	// the only one its reasoning models accept.
	MaxTokens           int      `json:"max_tokens,omitempty"`
	MaxCompletionTokens int      `json:"max_completion_tokens,omitempty"`
	Stop                []string `json:"stop,omitempty"`
}

type chatResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// Complete performs a single, non-streaming chat completion.
func (c *Client) Complete(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
	payload := chatRequest{
		Model:       req.Model,
		Messages:    make([]chatMessage, 0, len(req.Messages)+1),
		Temperature: req.Parameters.Temperature,
		TopP:        req.Parameters.TopP,
		Stop:        req.Parameters.Stop,
	}
	if c.provider == domain.ProviderOpenAI {
		payload.MaxCompletionTokens = req.Parameters.MaxTokens
	} else {
		payload.MaxTokens = req.Parameters.MaxTokens
	}
	if req.SystemPrompt != "" {
		payload.Messages = append(payload.Messages, chatMessage{Role: "system", Content: req.SystemPrompt})
	}
	for _, m := range req.Messages {
		payload.Messages = append(payload.Messages, chatMessage{Role: string(m.Role), Content: m.Content})
	}

	header := http.Header{}
	if c.apiKey != "" {
		header.Set("Authorization", "Bearer "+c.apiKey)
	}

	var out chatResponse
	start := c.now()
	if err := upstream.PostJSON(ctx, c.http, c.provider, c.endpoint, header, payload, &out); err != nil {
		return nil, err
	}
	finished := c.now()

	if len(out.Choices) == 0 {
		return nil, &domain.ProviderError{
			Provider:   c.provider,
			StatusCode: http.StatusOK,
			Retryable:  true,
			Err:        errors.New("response contained no choices"),
		}
	}
	model := out.Model
	if model == "" {
		model = req.Model
	}
	return &domain.LLMResponse{
		ID:           out.ID,
		Provider:     c.provider,
		Model:        model,
		Content:      out.Choices[0].Message.Content,
		FinishReason: finishReason(out.Choices[0].FinishReason),
		Usage: domain.TokenUsage{
			InputTokens:  out.Usage.PromptTokens,
			OutputTokens: out.Usage.CompletionTokens,
		},
		Latency:   finished.Sub(start),
		CreatedAt: finished.UTC(),
	}, nil
}

func finishReason(reason string) domain.FinishReason {
	switch reason {
	case "stop":
		return domain.FinishReasonStop
	case "length", "model_length":
		return domain.FinishReasonLength
	case "content_filter":
		return domain.FinishReasonContentFilter
	default:
		return domain.FinishReasonUnknown
	}
}
