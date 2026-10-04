// Package upstream holds the HTTP plumbing shared by the provider adapters:
// sending a JSON request, bounding the response size and turning every kind of
// failure into a *domain.ProviderError with the right Retryable flag.
package upstream

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

// maxResponseBytes bounds how much of an upstream response is read, so a
// misbehaving endpoint cannot exhaust the gateway's memory.
const maxResponseBytes = 10 << 20

// maxErrorMessageLen bounds the upstream error text copied into an error.
const maxErrorMessageLen = 500

// RetryableStatus reports whether an upstream HTTP status is worth retrying or
// routing to another provider: timeouts, rate limits and server-side failures.
func RetryableStatus(code int) bool {
	switch code {
	case http.StatusRequestTimeout, http.StatusTooEarly, http.StatusTooManyRequests:
		return true
	default:
		return code >= 500
	}
}

// PostJSON sends in as a JSON POST to url and decodes a 2xx response body into
// out. Every failure is returned as a *domain.ProviderError.
func PostJSON(ctx context.Context, client *http.Client, provider domain.Provider, url string, header http.Header, in, out any) error {
	body, err := json.Marshal(in)
	if err != nil {
		return &domain.ProviderError{Provider: provider, Err: fmt.Errorf("encode request: %w", err)}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return &domain.ProviderError{Provider: provider, Err: fmt.Errorf("build request: %w", err)}
	}
	for key, values := range header {
		req.Header[key] = values
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		// A cancelled request must not be retried; anything else at the
		// transport level (timeouts, resets, DNS) may succeed next time.
		return &domain.ProviderError{Provider: provider, Retryable: !errors.Is(err, context.Canceled), Err: err}
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return &domain.ProviderError{
			Provider:   provider,
			StatusCode: resp.StatusCode,
			Retryable:  !errors.Is(err, context.Canceled),
			Err:        fmt.Errorf("read response: %w", err),
		}
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return &domain.ProviderError{
			Provider:   provider,
			StatusCode: resp.StatusCode,
			Retryable:  RetryableStatus(resp.StatusCode),
			Err:        errors.New(ErrorMessage(data)),
		}
	}
	if err := json.Unmarshal(data, out); err != nil {
		return &domain.ProviderError{
			Provider:   provider,
			StatusCode: resp.StatusCode,
			Retryable:  true,
			Err:        fmt.Errorf("decode response: %w", err),
		}
	}
	return nil
}

// ErrorMessage extracts a human-readable message from an upstream error body.
// It understands the shapes used by the supported vendors, namely
// {"error":{"message":"..."}}, {"error":"..."}, {"message":"..."} and
// {"detail":"..."}, and falls back to the raw body, truncated.
func ErrorMessage(body []byte) string {
	var payload struct {
		Error   json.RawMessage `json:"error"`
		Message string          `json:"message"`
		Detail  string          `json:"detail"`
	}
	if json.Unmarshal(body, &payload) == nil {
		var nested struct {
			Message string `json:"message"`
		}
		var text string
		switch {
		case json.Unmarshal(payload.Error, &nested) == nil && nested.Message != "":
			return truncate(nested.Message)
		case json.Unmarshal(payload.Error, &text) == nil && text != "":
			return truncate(text)
		case payload.Message != "":
			return truncate(payload.Message)
		case payload.Detail != "":
			return truncate(payload.Detail)
		}
	}
	if text := strings.TrimSpace(string(body)); text != "" {
		return truncate(text)
	}
	return "empty error response"
}

func truncate(s string) string {
	if len(s) <= maxErrorMessageLen {
		return s
	}
	return s[:maxErrorMessageLen] + "..."
}
