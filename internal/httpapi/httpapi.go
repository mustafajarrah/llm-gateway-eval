// Package httpapi exposes the gateway and the evaluation suite over HTTP as a
// JSON API.
package httpapi

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
	"github.com/mustafajarrah/llm-gateway-eval/internal/evaluation"
)

// DefaultMaxBodyBytes bounds request bodies when Config does not say
// otherwise.
const DefaultMaxBodyBytes = 1 << 20

// statusClientClosedRequest is the de facto status (from nginx) for a request
// the client abandoned; it only ever reaches the access log.
const statusClientClosedRequest = 499

// Gateway is the completion service the API fronts.
type Gateway interface {
	Complete(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error)
	Providers() []domain.Provider
	Routes() []domain.Route
	// Circuits reports each provider's circuit breaker state.
	Circuits() map[domain.Provider]string
	Prices() []domain.ModelPrice
}

// Evaluator runs and compares evaluation runs.
type Evaluator interface {
	Run(ctx context.Context, promptID string, version int) (*evaluation.Report, error)
	Start(ctx context.Context, promptID string, version int) (*domain.EvaluationRun, error)
	Report(ctx context.Context, runID string) (*evaluation.Report, error)
	Compare(ctx context.Context, baseRunID, candidateRunID string) (*evaluation.Comparison, error)
}

// Config wires the API to its collaborators.
type Config struct {
	Gateway   Gateway
	Prompts   domain.PromptRepository
	Evals     domain.EvaluationRepository
	Evaluator Evaluator
	// APIKey, when set, must be presented by clients as a bearer token on
	// every route except /healthz. When empty the API is unauthenticated.
	APIKey string
	// MaxBodyBytes defaults to DefaultMaxBodyBytes.
	MaxBodyBytes int64
	// Logger defaults to slog.Default().
	Logger *slog.Logger
}

type api struct {
	gateway   Gateway
	prompts   domain.PromptRepository
	evals     domain.EvaluationRepository
	evaluator Evaluator
	maxBody   int64
	log       *slog.Logger
}

// NewHandler builds the HTTP handler.
func NewHandler(cfg Config) (http.Handler, error) {
	if cfg.Gateway == nil || cfg.Prompts == nil || cfg.Evals == nil || cfg.Evaluator == nil {
		return nil, fmt.Errorf("%w: httpapi needs a gateway, both repositories and an evaluator", domain.ErrInvalidInput)
	}
	if cfg.MaxBodyBytes < 0 {
		return nil, fmt.Errorf("%w: max body bytes must not be negative", domain.ErrInvalidInput)
	}
	a := &api{
		gateway:   cfg.Gateway,
		prompts:   cfg.Prompts,
		evals:     cfg.Evals,
		evaluator: cfg.Evaluator,
		maxBody:   cfg.MaxBodyBytes,
		log:       cfg.Logger,
	}
	if a.maxBody == 0 {
		a.maxBody = DefaultMaxBodyBytes
	}
	if a.log == nil {
		a.log = slog.Default()
	}

	v1 := http.NewServeMux()
	v1.HandleFunc("GET /v1/providers", a.listProviders)
	v1.HandleFunc("POST /v1/completions", a.complete)

	v1.HandleFunc("POST /v1/prompts", a.createPrompt)
	v1.HandleFunc("GET /v1/prompts", a.listPrompts)
	v1.HandleFunc("GET /v1/prompts/{id}", a.getPrompt)
	v1.HandleFunc("DELETE /v1/prompts/{id}", a.deletePrompt)

	v1.HandleFunc("POST /v1/prompts/{id}/versions", a.createVersion)
	v1.HandleFunc("GET /v1/prompts/{id}/versions", a.listVersions)
	v1.HandleFunc("GET /v1/prompts/{id}/versions/{version}", a.getVersion)

	v1.HandleFunc("POST /v1/prompts/{id}/test-cases", a.createTestCase)
	v1.HandleFunc("GET /v1/prompts/{id}/test-cases", a.listTestCases)
	v1.HandleFunc("GET /v1/test-cases/{id}", a.getTestCase)
	v1.HandleFunc("DELETE /v1/test-cases/{id}", a.deleteTestCase)

	v1.HandleFunc("POST /v1/prompts/{id}/runs", a.createRun)
	v1.HandleFunc("GET /v1/prompts/{id}/runs", a.listRuns)
	v1.HandleFunc("GET /v1/runs/{id}", a.getRun)
	v1.HandleFunc("GET /v1/comparisons", a.compareRuns)

	root := http.NewServeMux()
	root.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		a.writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	root.Handle("/", authenticate(cfg.APIKey, a, v1))

	return a.logRequests(root), nil
}

// ---- middleware ----

// statusRecorder remembers the status code a handler wrote.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// logRequests logs one line per request and turns a handler panic into a 500
// instead of a dropped connection.
func (a *api) logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		defer func() {
			if p := recover(); p != nil {
				a.log.ErrorContext(r.Context(), "handler panic", "panic", p, "method", r.Method, "path", r.URL.Path)
				a.writeError(rec, r, errors.New("panic in handler"))
			}
			a.log.InfoContext(r.Context(), "request",
				"method", r.Method, "path", r.URL.Path, "status", rec.status,
				"duration_ms", time.Since(start).Milliseconds())
		}()
		next.ServeHTTP(rec, r)
	})
}

// authenticate requires "Authorization: Bearer <apiKey>" when apiKey is set.
func authenticate(apiKey string, a *api, next http.Handler) http.Handler {
	if apiKey == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		// Constant-time comparison, so response timing does not leak how
		// much of a guessed key is right.
		if !ok || subtle.ConstantTimeCompare([]byte(token), []byte(apiKey)) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="llm-gateway"`)
			a.writeJSON(w, http.StatusUnauthorized, errorBody{Error: errorDetail{
				Code: "unauthorized", Message: "missing or invalid bearer token",
			}})
			return
		}
		next.ServeHTTP(w, r)
	})
}

// ---- encoding ----

type errorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	// Provider and UpstreamStatus are set for provider_error.
	Provider       domain.Provider `json:"provider,omitempty"`
	UpstreamStatus int             `json:"upstream_status,omitempty"`
}

type errorBody struct {
	Error errorDetail `json:"error"`
}

// list is the envelope of every collection response.
type list[T any] struct {
	Data []T `json:"data"`
}

func (a *api) writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		a.log.Error("write response", "error", err)
	}
}

// writeError maps an error to a status code and a JSON error body.
func (a *api) writeError(w http.ResponseWriter, r *http.Request, err error) {
	status, detail := classify(err)
	if status == http.StatusInternalServerError {
		// The cause may hold internals (SQL, file paths); log it and keep
		// the response generic.
		a.log.ErrorContext(r.Context(), "internal error", "method", r.Method, "path", r.URL.Path, "error", err)
	}
	a.writeJSON(w, status, errorBody{Error: detail})
}

func classify(err error) (int, errorDetail) {
	var (
		providerErr *domain.ProviderError
		tooLarge    *http.MaxBytesError
	)
	switch {
	case errors.As(err, &tooLarge):
		return http.StatusRequestEntityTooLarge, errorDetail{
			Code: "body_too_large", Message: fmt.Sprintf("request body exceeds %d bytes", tooLarge.Limit),
		}
	case errors.Is(err, domain.ErrInvalidInput):
		return http.StatusBadRequest, errorDetail{Code: "invalid_input", Message: err.Error()}
	case errors.Is(err, domain.ErrNotFound):
		return http.StatusNotFound, errorDetail{Code: "not_found", Message: err.Error()}
	case errors.Is(err, domain.ErrConflict):
		return http.StatusConflict, errorDetail{Code: "conflict", Message: err.Error()}
	case errors.Is(err, evaluation.ErrShuttingDown):
		return http.StatusServiceUnavailable, errorDetail{Code: "unavailable", Message: "the service is shutting down"}
	case errors.Is(err, context.Canceled):
		return statusClientClosedRequest, errorDetail{Code: "cancelled", Message: "request cancelled"}
	case errors.Is(err, context.DeadlineExceeded):
		return http.StatusGatewayTimeout, errorDetail{Code: "timeout", Message: "request timed out"}
	case errors.As(err, &providerErr):
		return http.StatusBadGateway, errorDetail{
			Code:           "provider_error",
			Message:        err.Error(),
			Provider:       providerErr.Provider,
			UpstreamStatus: providerErr.StatusCode,
		}
	default:
		return http.StatusInternalServerError, errorDetail{Code: "internal", Message: "internal server error"}
	}
}

// decode reads a single JSON object from the request body into v. Unknown
// fields are rejected so that a typo, or an attempt to set a server-assigned
// field such as "id", fails loudly. With optional set, an empty body is
// accepted and leaves v untouched.
func (a *api) decode(w http.ResponseWriter, r *http.Request, v any, optional bool) error {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, a.maxBody))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		var tooLarge *http.MaxBytesError
		switch {
		case errors.As(err, &tooLarge):
			return err
		case errors.Is(err, io.EOF) && optional:
			return nil
		case errors.Is(err, io.EOF):
			return fmt.Errorf("%w: request body is empty", domain.ErrInvalidInput)
		default:
			return fmt.Errorf("%w: malformed JSON body: %v", domain.ErrInvalidInput, err)
		}
	}
	if dec.More() {
		return fmt.Errorf("%w: request body must hold a single JSON object", domain.ErrInvalidInput)
	}
	return nil
}

// listOptions parses the limit and offset query parameters.
func listOptions(r *http.Request) (domain.ListOptions, error) {
	var opts domain.ListOptions
	for name, dst := range map[string]*int{"limit": &opts.Limit, "offset": &opts.Offset} {
		raw := r.URL.Query().Get(name)
		if raw == "" {
			continue
		}
		n, err := strconv.Atoi(raw)
		if err != nil || n < 0 {
			return opts, fmt.Errorf("%w: %s must be a non-negative integer, got %q", domain.ErrInvalidInput, name, raw)
		}
		*dst = n
	}
	return opts, nil
}

// ---- gateway ----

type providersResponse struct {
	Providers []domain.Provider `json:"providers"`
	Routes    []domain.Route    `json:"routes"`
	// Circuits maps each provider to "closed", "open" or "half_open". It is
	// empty when circuit breaking is disabled.
	Circuits map[domain.Provider]string `json:"circuits"`
	// Prices are the configured per-model prices, in US dollars per million
	// tokens.
	Prices []domain.ModelPrice `json:"prices"`
}

func (a *api) listProviders(w http.ResponseWriter, _ *http.Request) {
	a.writeJSON(w, http.StatusOK, providersResponse{
		Providers: a.gateway.Providers(),
		Routes:    a.gateway.Routes(),
		Circuits:  a.gateway.Circuits(),
		Prices:    a.gateway.Prices(),
	})
}

func (a *api) complete(w http.ResponseWriter, r *http.Request) {
	var req domain.LLMRequest
	if err := a.decode(w, r, &req, false); err != nil {
		a.writeError(w, r, err)
		return
	}
	resp, err := a.gateway.Complete(r.Context(), &req)
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusOK, resp)
}

// ---- prompts ----

type createPromptRequest struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Tags        []string `json:"tags"`
}

func (a *api) createPrompt(w http.ResponseWriter, r *http.Request) {
	var body createPromptRequest
	if err := a.decode(w, r, &body, false); err != nil {
		a.writeError(w, r, err)
		return
	}
	p := &domain.Prompt{Name: strings.TrimSpace(body.Name), Description: body.Description, Tags: body.Tags}
	if err := a.prompts.CreatePrompt(r.Context(), p); err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusCreated, p)
}

func (a *api) listPrompts(w http.ResponseWriter, r *http.Request) {
	opts, err := listOptions(r)
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	prompts, err := a.prompts.ListPrompts(r.Context(), opts)
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusOK, list[domain.Prompt]{Data: prompts})
}

func (a *api) getPrompt(w http.ResponseWriter, r *http.Request) {
	p, err := a.prompts.GetPrompt(r.Context(), r.PathValue("id"))
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusOK, p)
}

func (a *api) deletePrompt(w http.ResponseWriter, r *http.Request) {
	if err := a.prompts.DeletePrompt(r.Context(), r.PathValue("id")); err != nil {
		a.writeError(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// ---- prompt versions ----

type createVersionRequest struct {
	Template     string                 `json:"template"`
	SystemPrompt string                 `json:"system_prompt"`
	Provider     domain.Provider        `json:"provider"`
	Model        string                 `json:"model"`
	Parameters   domain.ModelParameters `json:"parameters"`
	ChangeLog    string                 `json:"change_log"`
}

func (a *api) createVersion(w http.ResponseWriter, r *http.Request) {
	var body createVersionRequest
	if err := a.decode(w, r, &body, false); err != nil {
		a.writeError(w, r, err)
		return
	}
	v := &domain.PromptVersion{
		PromptID:     r.PathValue("id"),
		Template:     body.Template,
		SystemPrompt: body.SystemPrompt,
		Provider:     body.Provider,
		Model:        body.Model,
		Parameters:   body.Parameters,
		ChangeLog:    body.ChangeLog,
	}
	if err := a.prompts.CreateVersion(r.Context(), v); err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusCreated, v)
}

func (a *api) listVersions(w http.ResponseWriter, r *http.Request) {
	versions, err := a.prompts.ListVersions(r.Context(), r.PathValue("id"))
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusOK, list[domain.PromptVersion]{Data: versions})
}

func (a *api) getVersion(w http.ResponseWriter, r *http.Request) {
	var (
		v   *domain.PromptVersion
		err error
	)
	if raw := r.PathValue("version"); raw == "latest" {
		v, err = a.prompts.GetLatestVersion(r.Context(), r.PathValue("id"))
	} else {
		var number int
		number, err = strconv.Atoi(raw)
		if err != nil || number < 1 {
			err = fmt.Errorf("%w: version must be a positive integer or \"latest\", got %q", domain.ErrInvalidInput, raw)
		} else {
			v, err = a.prompts.GetVersion(r.Context(), r.PathValue("id"), number)
		}
	}
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusOK, v)
}

// ---- test cases ----

type createTestCaseRequest struct {
	Name           string               `json:"name"`
	Variables      map[string]string    `json:"variables"`
	ExpectedOutput string               `json:"expected_output"`
	MatchStrategy  domain.MatchStrategy `json:"match_strategy"`
}

func (a *api) createTestCase(w http.ResponseWriter, r *http.Request) {
	var body createTestCaseRequest
	if err := a.decode(w, r, &body, false); err != nil {
		a.writeError(w, r, err)
		return
	}
	tc := &domain.EvaluationTestCase{
		PromptID:       r.PathValue("id"),
		Name:           strings.TrimSpace(body.Name),
		Variables:      body.Variables,
		ExpectedOutput: body.ExpectedOutput,
		MatchStrategy:  body.MatchStrategy,
	}
	if err := a.evals.CreateTestCase(r.Context(), tc); err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusCreated, tc)
}

func (a *api) listTestCases(w http.ResponseWriter, r *http.Request) {
	cases, err := a.evals.ListTestCases(r.Context(), r.PathValue("id"))
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusOK, list[domain.EvaluationTestCase]{Data: cases})
}

func (a *api) getTestCase(w http.ResponseWriter, r *http.Request) {
	tc, err := a.evals.GetTestCase(r.Context(), r.PathValue("id"))
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusOK, tc)
}

func (a *api) deleteTestCase(w http.ResponseWriter, r *http.Request) {
	if err := a.evals.DeleteTestCase(r.Context(), r.PathValue("id")); err != nil {
		a.writeError(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// ---- runs ----

type createRunRequest struct {
	// Version selects the prompt version to evaluate; 0 or absent means the
	// latest.
	Version int `json:"version"`
}

// createRun starts an evaluation run. By default it answers 202 with the run
// in the "running" state and the suite executes in the background; the client
// polls GET /v1/runs/{id}. With ?wait=true it blocks until the run is over
// and answers 201 with the full report.
func (a *api) createRun(w http.ResponseWriter, r *http.Request) {
	wait := false
	if raw := r.URL.Query().Get("wait"); raw != "" {
		parsed, err := strconv.ParseBool(raw)
		if err != nil {
			a.writeError(w, r, fmt.Errorf("%w: wait must be true or false, got %q", domain.ErrInvalidInput, raw))
			return
		}
		wait = parsed
	}
	var body createRunRequest
	if err := a.decode(w, r, &body, true); err != nil {
		a.writeError(w, r, err)
		return
	}

	if wait {
		report, err := a.evaluator.Run(r.Context(), r.PathValue("id"), body.Version)
		if err != nil {
			a.writeError(w, r, err)
			return
		}
		a.writeJSON(w, http.StatusCreated, report)
		return
	}
	run, err := a.evaluator.Start(r.Context(), r.PathValue("id"), body.Version)
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	w.Header().Set("Location", "/v1/runs/"+run.ID)
	a.writeJSON(w, http.StatusAccepted, run)
}

func (a *api) listRuns(w http.ResponseWriter, r *http.Request) {
	opts, err := listOptions(r)
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	runs, err := a.evals.ListRuns(r.Context(), r.PathValue("id"), opts)
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusOK, list[domain.EvaluationRun]{Data: runs})
}

func (a *api) getRun(w http.ResponseWriter, r *http.Request) {
	report, err := a.evaluator.Report(r.Context(), r.PathValue("id"))
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusOK, report)
}

func (a *api) compareRuns(w http.ResponseWriter, r *http.Request) {
	base, candidate := r.URL.Query().Get("base"), r.URL.Query().Get("candidate")
	if base == "" || candidate == "" {
		a.writeError(w, r, fmt.Errorf("%w: the base and candidate query parameters are required", domain.ErrInvalidInput))
		return
	}
	cmp, err := a.evaluator.Compare(r.Context(), base, candidate)
	if err != nil {
		a.writeError(w, r, err)
		return
	}
	a.writeJSON(w, http.StatusOK, cmp)
}
