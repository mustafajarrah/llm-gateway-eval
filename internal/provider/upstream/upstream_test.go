package upstream

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

type payload struct {
	Value string `json:"value"`
}

func providerError(t *testing.T, err error) *domain.ProviderError {
	t.Helper()
	var pe *domain.ProviderError
	if !errors.As(err, &pe) {
		t.Fatalf("error %v is not a *domain.ProviderError", err)
	}
	return pe
}

func TestPostJSONSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q", got)
		}
		if got := r.Header.Get("X-Test"); got != "yes" {
			t.Errorf("X-Test = %q, custom header was not forwarded", got)
		}
		body, _ := io.ReadAll(r.Body)
		if string(body) != `{"value":"ping"}` {
			t.Errorf("body = %s", body)
		}
		_, _ = w.Write([]byte(`{"value":"pong"}`))
	}))
	defer srv.Close()

	var out payload
	header := http.Header{"X-Test": []string{"yes"}}
	err := PostJSON(context.Background(), srv.Client(), domain.ProviderOpenAI, srv.URL, header, payload{Value: "ping"}, &out)
	if err != nil {
		t.Fatalf("PostJSON() error = %v", err)
	}
	if out.Value != "pong" {
		t.Errorf("decoded %+v", out)
	}
}

func TestPostJSONStatusErrors(t *testing.T) {
	tests := []struct {
		status    int
		body      string
		retryable bool
		message   string
	}{
		{status: 400, body: `{"error":{"message":"bad model"}}`, message: "bad model"},
		{status: 401, body: `{"error":"no key"}`, message: "no key"},
		{status: 404, body: `{"message":"not here"}`, message: "not here"},
		{status: 408, body: `{"detail":"slow"}`, retryable: true, message: "slow"},
		{status: 425, body: ``, retryable: true, message: "empty error response"},
		{status: 429, body: `{"error":{"message":"rate limited"}}`, retryable: true, message: "rate limited"},
		{status: 500, body: `<html>oops</html>`, retryable: true, message: "<html>oops</html>"},
		{status: 529, body: `{"error":{"type":"overloaded_error","message":"overloaded"}}`, retryable: true, message: "overloaded"},
	}
	for _, tt := range tests {
		t.Run(http.StatusText(tt.status)+"/"+tt.message, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer srv.Close()

			var out payload
			err := PostJSON(context.Background(), srv.Client(), domain.ProviderMistral, srv.URL, nil, payload{}, &out)
			pe := providerError(t, err)
			if pe.Provider != domain.ProviderMistral || pe.StatusCode != tt.status || pe.Retryable != tt.retryable {
				t.Errorf("got %+v, want status %d retryable %v", pe, tt.status, tt.retryable)
			}
			if pe.Err.Error() != tt.message {
				t.Errorf("message = %q, want %q", pe.Err.Error(), tt.message)
			}
		})
	}
}

func TestPostJSONMalformedResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"value":`))
	}))
	defer srv.Close()

	var out payload
	pe := providerError(t, PostJSON(context.Background(), srv.Client(), domain.ProviderGemini, srv.URL, nil, payload{}, &out))
	if !pe.Retryable || pe.StatusCode != 200 {
		t.Errorf("got %+v, want a retryable error with status 200", pe)
	}
}

func TestPostJSONTransportErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url, client := srv.URL, srv.Client()
	srv.Close()

	var out payload
	pe := providerError(t, PostJSON(context.Background(), client, domain.ProviderOllama, url, nil, payload{}, &out))
	if !pe.Retryable || pe.StatusCode != 0 {
		t.Errorf("connection failure: got %+v, want retryable with status 0", pe)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	pe = providerError(t, PostJSON(ctx, client, domain.ProviderOllama, url, nil, payload{}, &out))
	if pe.Retryable {
		t.Errorf("cancelled request must not be retryable: %+v", pe)
	}
	if !errors.Is(pe, context.Canceled) {
		t.Errorf("error %v does not wrap context.Canceled", pe)
	}
}

func TestPostJSONRequestBuildErrors(t *testing.T) {
	var out payload
	pe := providerError(t, PostJSON(context.Background(), http.DefaultClient, domain.ProviderOpenAI, "http://example.invalid", nil, make(chan int), &out))
	if pe.Retryable || !strings.Contains(pe.Error(), "encode request") {
		t.Errorf("unencodable body: got %v", pe)
	}
	pe = providerError(t, PostJSON(context.Background(), http.DefaultClient, domain.ProviderOpenAI, "://bad", nil, payload{}, &out))
	if pe.Retryable || !strings.Contains(pe.Error(), "build request") {
		t.Errorf("bad URL: got %v", pe)
	}
}

func TestErrorMessageTruncates(t *testing.T) {
	long := strings.Repeat("x", maxErrorMessageLen+50)
	got := ErrorMessage([]byte(long))
	if len(got) != maxErrorMessageLen+3 || !strings.HasSuffix(got, "...") {
		t.Errorf("ErrorMessage() returned %d bytes", len(got))
	}
	if got := ErrorMessage([]byte(`{"error":{"code":1}}`)); got != `{"error":{"code":1}}` {
		t.Errorf("unknown shape: got %q, want the raw body", got)
	}
}
