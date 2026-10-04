// Package gemini implements domain.LLMProvider for the Google Gemini API
// (generateContent).
package gemini

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
	"github.com/mustafajarrah/llm-gateway-eval/internal/provider/upstream"
)

// DefaultBaseURL is the Gemini API root, including the API version.
const DefaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// Config configures a Client.
type Config struct {
	// APIKey is required. It is sent in the x-goog-api-key header, never in
	// the URL, so it cannot leak through logged URLs or transport errors.
	APIKey string
	// BaseURL defaults to DefaultBaseURL.
	BaseURL string
	// HTTPClient defaults to a client without a timeout; the gateway bounds
	// each attempt through the request context.
	HTTPClient *http.Client
}

// Client is a domain.LLMProvider backed by the Gemini API. It is safe for
// concurrent use.
type Client struct {
	baseURL string
	apiKey  string
	http    *http.Client
	now     func() time.Time
}

var _ domain.LLMProvider = (*Client)(nil)

// New builds a Client from cfg.
func New(cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.APIKey) == "" {
		return nil, fmt.Errorf("%w: gemini requires an API key", domain.ErrInvalidInput)
	}
	baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	return &Client{baseURL: baseURL, apiKey: cfg.APIKey, http: client, now: time.Now}, nil
}

// Name returns the provider identifier.
func (c *Client) Name() domain.Provider { return domain.ProviderGemini }

type part struct {
	Text string `json:"text"`
	// Thought marks a part holding the model's reasoning summary rather than
	// its answer.
	Thought bool `json:"thought,omitempty"`
}

type content struct {
	Role  string `json:"role,omitempty"`
	Parts []part `json:"parts"`
}

type generationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type generateRequest struct {
	SystemInstruction *content          `json:"systemInstruction,omitempty"`
	Contents          []content         `json:"contents"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
}

type generateResponse struct {
	ResponseID   string `json:"responseId"`
	ModelVersion string `json:"modelVersion"`
	Candidates   []struct {
		Content      content `json:"content"`
		FinishReason string  `json:"finishReason"`
	} `json:"candidates"`
	PromptFeedback struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		ThoughtsTokenCount   int `json:"thoughtsTokenCount"`
	} `json:"usageMetadata"`
}

// Complete performs a single, non-streaming completion.
func (c *Client) Complete(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
	payload := generateRequest{Contents: make([]content, 0, len(req.Messages))}
	if req.SystemPrompt != "" {
		payload.SystemInstruction = &content{Parts: []part{{Text: req.SystemPrompt}}}
	}
	for _, m := range req.Messages {
		role := "user"
		if m.Role == domain.RoleAssistant {
			role = "model"
		}
		payload.Contents = append(payload.Contents, content{Role: role, Parts: []part{{Text: m.Content}}})
	}
	p := req.Parameters
	if p.Temperature != nil || p.TopP != nil || p.MaxTokens > 0 || len(p.Stop) > 0 {
		payload.GenerationConfig = &generationConfig{
			Temperature:     p.Temperature,
			TopP:            p.TopP,
			MaxOutputTokens: p.MaxTokens,
			StopSequences:   p.Stop,
		}
	}

	// The API addresses models as "models/<id>"; accept either spelling.
	model := strings.TrimPrefix(req.Model, "models/")
	endpoint := c.baseURL + "/models/" + url.PathEscape(model) + ":generateContent"
	header := http.Header{"X-Goog-Api-Key": []string{c.apiKey}}

	var out generateResponse
	start := c.now()
	if err := upstream.PostJSON(ctx, c.http, domain.ProviderGemini, endpoint, header, payload, &out); err != nil {
		return nil, err
	}
	finished := c.now()

	resp := &domain.LLMResponse{
		ID:       out.ResponseID,
		Provider: domain.ProviderGemini,
		Model:    out.ModelVersion,
		Usage: domain.TokenUsage{
			InputTokens: out.UsageMetadata.PromptTokenCount,
			// Thinking tokens are billed as output.
			OutputTokens: out.UsageMetadata.CandidatesTokenCount + out.UsageMetadata.ThoughtsTokenCount,
		},
		Latency:   finished.Sub(start),
		CreatedAt: finished.UTC(),
	}
	if resp.Model == "" {
		resp.Model = model
	}

	if len(out.Candidates) == 0 {
		// A blocked prompt is a successful response without candidates.
		if out.PromptFeedback.BlockReason != "" {
			resp.FinishReason = domain.FinishReasonContentFilter
			return resp, nil
		}
		return nil, &domain.ProviderError{
			Provider:   domain.ProviderGemini,
			StatusCode: http.StatusOK,
			Retryable:  true,
			Err:        errors.New("response contained no candidates"),
		}
	}

	var text strings.Builder
	for _, p := range out.Candidates[0].Content.Parts {
		if !p.Thought {
			text.WriteString(p.Text)
		}
	}
	resp.Content = text.String()
	resp.FinishReason = finishReason(out.Candidates[0].FinishReason)
	return resp, nil
}

func finishReason(reason string) domain.FinishReason {
	switch reason {
	case "STOP":
		return domain.FinishReasonStop
	case "MAX_TOKENS":
		return domain.FinishReasonLength
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY":
		return domain.FinishReasonContentFilter
	default:
		return domain.FinishReasonUnknown
	}
}
