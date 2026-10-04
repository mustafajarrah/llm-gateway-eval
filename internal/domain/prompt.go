package domain

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"text/template"
	"time"
)

// Prompt is a named, versioned prompt template. The Prompt itself carries only
// identity and descriptive metadata; the actual text lives in immutable
// PromptVersion records so that every evaluation result can be traced back to
// the exact template that produced it.
type Prompt struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Description string    `json:"description,omitempty"`
	Tags        []string  `json:"tags,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// Validate checks the prompt's invariants.
func (p *Prompt) Validate() error {
	if strings.TrimSpace(p.Name) == "" {
		return fmt.Errorf("%w: prompt name is required", ErrInvalidInput)
	}
	return nil
}

// PromptVersion is an immutable snapshot of a prompt template together with
// the model configuration it is meant to run against. Versions are numbered
// sequentially per prompt starting at 1.
type PromptVersion struct {
	ID       string `json:"id"`
	PromptID string `json:"prompt_id"`
	// Version is assigned by the repository on creation; 0 means "not yet
	// assigned".
	Version int `json:"version"`
	// Template is a Go text/template rendered with the test case (or caller)
	// variables, e.g. "Summarise the following text:\n{{.text}}".
	Template     string `json:"template"`
	SystemPrompt string `json:"system_prompt,omitempty"`
	// Provider and Model follow the LLMRequest convention: with a Provider,
	// Model is a vendor model identifier; without one, Model is a route name.
	Provider   Provider        `json:"provider,omitempty"`
	Model      string          `json:"model"`
	Parameters ModelParameters `json:"parameters"`
	// ChangeLog is a short human-readable note describing what changed.
	ChangeLog string    `json:"change_log,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

// Validate checks the version's invariants, including that Template parses.
func (v *PromptVersion) Validate() error {
	var errs []error
	if v.PromptID == "" {
		errs = append(errs, errors.New("prompt_id is required"))
	}
	if v.Version < 0 {
		errs = append(errs, fmt.Errorf("version must not be negative, got %d", v.Version))
	}
	if strings.TrimSpace(v.Template) == "" {
		errs = append(errs, errors.New("template is required"))
	} else if _, err := v.parse(); err != nil {
		errs = append(errs, fmt.Errorf("template: %w", err))
	}
	if v.Provider != "" && !v.Provider.Valid() {
		errs = append(errs, fmt.Errorf("unknown provider %q", v.Provider))
	}
	if v.Model == "" {
		errs = append(errs, errors.New("model is required"))
	}
	if err := v.Parameters.ValidateFor(v.Provider); err != nil {
		errs = append(errs, err)
	}
	if len(errs) > 0 {
		return fmt.Errorf("%w: %w", ErrInvalidInput, errors.Join(errs...))
	}
	return nil
}

// Render executes the template with vars. Referencing a variable that is not
// present in vars is an error rather than a silent "<no value>", so a typo in
// a template fails loudly instead of producing a misleading evaluation.
func (v *PromptVersion) Render(vars map[string]string) (string, error) {
	tmpl, err := v.parse()
	if err != nil {
		return "", fmt.Errorf("%w: parse template: %w", ErrInvalidInput, err)
	}
	if vars == nil {
		vars = map[string]string{}
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, vars); err != nil {
		return "", fmt.Errorf("%w: render template: %w", ErrInvalidInput, err)
	}
	return buf.String(), nil
}

// BuildRequest renders the template with vars and wraps the result in an
// LLMRequest ready to be sent through the gateway.
func (v *PromptVersion) BuildRequest(vars map[string]string) (*LLMRequest, error) {
	content, err := v.Render(vars)
	if err != nil {
		return nil, err
	}
	return &LLMRequest{
		Provider:     v.Provider,
		Model:        v.Model,
		SystemPrompt: v.SystemPrompt,
		Messages:     []Message{{Role: RoleUser, Content: content}},
		Parameters:   v.Parameters,
	}, nil
}

func (v *PromptVersion) parse() (*template.Template, error) {
	return template.New("prompt").Option("missingkey=error").Parse(v.Template)
}

// MatchStrategy determines how an actual LLM output is compared with the
// expected output of a test case.
type MatchStrategy string

// Supported match strategies.
const (
	// MatchExact passes when the trimmed output equals the expectation.
	MatchExact MatchStrategy = "exact"
	// MatchContains passes when the output contains the expectation
	// (case-insensitive).
	MatchContains MatchStrategy = "contains"
	// MatchRegex passes when the output matches the expectation as a regular
	// expression.
	MatchRegex MatchStrategy = "regex"
)

// Valid reports whether s is a supported strategy.
func (s MatchStrategy) Valid() bool {
	switch s {
	case MatchExact, MatchContains, MatchRegex:
		return true
	default:
		return false
	}
}

// EvaluationTestCase is a single input/expectation pair attached to a prompt.
// Test cases belong to the Prompt rather than a specific version so the same
// suite can be replayed against every version for regression comparison.
type EvaluationTestCase struct {
	ID       string `json:"id"`
	PromptID string `json:"prompt_id"`
	Name     string `json:"name"`
	// Variables are substituted into the PromptVersion template.
	Variables      map[string]string `json:"variables"`
	ExpectedOutput string            `json:"expected_output"`
	MatchStrategy  MatchStrategy     `json:"match_strategy"`
	CreatedAt      time.Time         `json:"created_at"`
}

// Validate checks the test case's invariants. For MatchRegex it also verifies
// that ExpectedOutput compiles.
func (tc *EvaluationTestCase) Validate() error {
	var errs []error
	if tc.PromptID == "" {
		errs = append(errs, errors.New("prompt_id is required"))
	}
	if strings.TrimSpace(tc.Name) == "" {
		errs = append(errs, errors.New("name is required"))
	}
	if !tc.MatchStrategy.Valid() {
		errs = append(errs, fmt.Errorf("unknown match strategy %q", tc.MatchStrategy))
	}
	if tc.ExpectedOutput == "" {
		errs = append(errs, errors.New("expected_output is required"))
	} else if tc.MatchStrategy == MatchRegex {
		if _, err := regexp.Compile(tc.ExpectedOutput); err != nil {
			errs = append(errs, fmt.Errorf("expected_output: %w", err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("%w: %w", ErrInvalidInput, errors.Join(errs...))
	}
	return nil
}

// Matches reports whether actual satisfies the test case's expectation
// according to its MatchStrategy.
func (tc *EvaluationTestCase) Matches(actual string) (bool, error) {
	switch tc.MatchStrategy {
	case MatchExact:
		return strings.TrimSpace(actual) == strings.TrimSpace(tc.ExpectedOutput), nil
	case MatchContains:
		return strings.Contains(strings.ToLower(actual), strings.ToLower(tc.ExpectedOutput)), nil
	case MatchRegex:
		re, err := regexp.Compile(tc.ExpectedOutput)
		if err != nil {
			return false, fmt.Errorf("%w: compile expected_output: %w", ErrInvalidInput, err)
		}
		return re.MatchString(actual), nil
	default:
		return false, fmt.Errorf("%w: unknown match strategy %q", ErrInvalidInput, tc.MatchStrategy)
	}
}

// EvaluationResult records the outcome of running one test case against one
// prompt version. Results are append-only; re-running a suite produces new
// rows grouped under a new RunID.
type EvaluationResult struct {
	ID              string `json:"id"`
	RunID           string `json:"run_id"`
	TestCaseID      string `json:"test_case_id"`
	PromptVersionID string `json:"prompt_version_id"`
	// Provider and Model record what actually served the request, which may
	// differ from the version's configuration if the gateway fell back.
	Provider     Provider `json:"provider"`
	Model        string   `json:"model"`
	ActualOutput string   `json:"actual_output"`
	Passed       bool     `json:"passed"`
	// Score is a normalised quality score in [0, 1]. Binary strategies use
	// 0 or 1; graded scorers (e.g. LLM-as-judge) may use the full range.
	Score float64    `json:"score"`
	Usage TokenUsage `json:"usage"`
	// Latency is serialised as integer milliseconds under "latency_ms".
	Latency time.Duration `json:"-"`
	// Error is non-empty when the test case could not be executed (render
	// failure, provider outage). Such results always have Passed == false.
	Error     string    `json:"error,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

// MarshalJSON encodes Latency as "latency_ms" instead of time.Duration's
// default of raw nanoseconds.
func (r EvaluationResult) MarshalJSON() ([]byte, error) {
	type alias EvaluationResult
	return json.Marshal(struct {
		alias
		LatencyMS int64 `json:"latency_ms"`
	}{alias(r), r.Latency.Milliseconds()})
}

// UnmarshalJSON is the inverse of MarshalJSON.
func (r *EvaluationResult) UnmarshalJSON(data []byte) error {
	type alias EvaluationResult
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

// RunStatus is the lifecycle state of an EvaluationRun.
type RunStatus string

// Run lifecycle states.
const (
	// RunStatusRunning means test cases are still being executed.
	RunStatusRunning RunStatus = "running"
	// RunStatusCompleted means every test case produced a result, whether it
	// passed, failed or errored.
	RunStatusCompleted RunStatus = "completed"
	// RunStatusFailed means the run was aborted before producing a full set
	// of results (cancellation, storage failure).
	RunStatusFailed RunStatus = "failed"
)

// Valid reports whether s is a known run status.
func (s RunStatus) Valid() bool {
	switch s {
	case RunStatusRunning, RunStatusCompleted, RunStatusFailed:
		return true
	default:
		return false
	}
}

// RunSummary aggregates the results of an evaluation run.
type RunSummary struct {
	Total  int `json:"total"`
	Passed int `json:"passed"`
	// Failed counts test cases that ran but did not match the expectation.
	Failed int `json:"failed"`
	// Errored counts test cases that could not be executed at all.
	Errored int        `json:"errored"`
	Usage   TokenUsage `json:"usage"`
	// TotalLatencyMS is the sum of the per-result provider latencies.
	TotalLatencyMS int64 `json:"total_latency_ms"`
}

// PassRate returns Passed / Total in [0, 1], or 0 for an empty run.
func (s RunSummary) PassRate() float64 {
	if s.Total == 0 {
		return 0
	}
	return float64(s.Passed) / float64(s.Total)
}

// Summarize computes the aggregate of results.
func Summarize(results []EvaluationResult) RunSummary {
	var s RunSummary
	for _, r := range results {
		s.Total++
		switch {
		case r.Error != "":
			s.Errored++
		case r.Passed:
			s.Passed++
		default:
			s.Failed++
		}
		s.Usage = s.Usage.Add(r.Usage)
		s.TotalLatencyMS += r.Latency.Milliseconds()
	}
	return s
}

// EvaluationRun is one execution of a prompt's test suite against one prompt
// version. It groups the EvaluationResult rows sharing its ID as RunID.
type EvaluationRun struct {
	ID              string     `json:"id"`
	PromptID        string     `json:"prompt_id"`
	PromptVersionID string     `json:"prompt_version_id"`
	Version         int        `json:"version"`
	Status          RunStatus  `json:"status"`
	Summary         RunSummary `json:"summary"`
	// Error explains why a run ended in RunStatusFailed.
	Error      string    `json:"error,omitempty"`
	StartedAt  time.Time `json:"started_at"`
	FinishedAt time.Time `json:"finished_at,omitzero"`
}

// Validate checks the run's invariants.
func (r *EvaluationRun) Validate() error {
	var errs []error
	if r.PromptID == "" {
		errs = append(errs, errors.New("prompt_id is required"))
	}
	if r.PromptVersionID == "" {
		errs = append(errs, errors.New("prompt_version_id is required"))
	}
	if !r.Status.Valid() {
		errs = append(errs, fmt.Errorf("unknown run status %q", r.Status))
	}
	if len(errs) > 0 {
		return fmt.Errorf("%w: %w", ErrInvalidInput, errors.Join(errs...))
	}
	return nil
}

// ListOptions controls pagination for list queries.
type ListOptions struct {
	Limit  int
	Offset int
}

// DefaultListLimit and MaxListLimit bound page sizes for list queries.
const (
	DefaultListLimit = 50
	MaxListLimit     = 500
)

// Normalize returns a copy of o with Limit clamped to (0, MaxListLimit] and a
// non-negative Offset.
func (o ListOptions) Normalize() ListOptions {
	if o.Limit <= 0 {
		o.Limit = DefaultListLimit
	}
	if o.Limit > MaxListLimit {
		o.Limit = MaxListLimit
	}
	if o.Offset < 0 {
		o.Offset = 0
	}
	return o
}

// PromptRepository persists prompts and their versions. Implementations
// return errors wrapping ErrNotFound or ErrConflict where applicable. On
// creation they assign an ID when it is empty and CreatedAt when it is zero.
type PromptRepository interface {
	// CreatePrompt stores a new prompt. It returns ErrConflict if the name is
	// already taken.
	CreatePrompt(ctx context.Context, p *Prompt) error
	// GetPrompt returns the prompt with the given ID.
	GetPrompt(ctx context.Context, id string) (*Prompt, error)
	// ListPrompts returns prompts ordered by creation time, newest first.
	ListPrompts(ctx context.Context, opts ListOptions) ([]Prompt, error)
	// DeletePrompt removes a prompt and, transitively, its versions, test
	// cases and results.
	DeletePrompt(ctx context.Context, id string) error

	// CreateVersion stores a new version. Implementations assign the next
	// sequential Version number atomically, overwriting v.Version.
	CreateVersion(ctx context.Context, v *PromptVersion) error
	// GetVersion returns a specific version of a prompt.
	GetVersion(ctx context.Context, promptID string, version int) (*PromptVersion, error)
	// GetLatestVersion returns the highest-numbered version of a prompt.
	GetLatestVersion(ctx context.Context, promptID string) (*PromptVersion, error)
	// ListVersions returns all versions of a prompt in ascending order.
	ListVersions(ctx context.Context, promptID string) ([]PromptVersion, error)
}

// EvaluationRepository persists test cases, runs and evaluation results. Like
// PromptRepository, it assigns missing IDs and timestamps on creation.
type EvaluationRepository interface {
	// CreateTestCase stores a new test case.
	CreateTestCase(ctx context.Context, tc *EvaluationTestCase) error
	// GetTestCase returns the test case with the given ID.
	GetTestCase(ctx context.Context, id string) (*EvaluationTestCase, error)
	// ListTestCases returns every test case attached to a prompt.
	ListTestCases(ctx context.Context, promptID string) ([]EvaluationTestCase, error)
	// DeleteTestCase removes a test case.
	DeleteTestCase(ctx context.Context, id string) error

	// CreateRun stores a new run.
	CreateRun(ctx context.Context, run *EvaluationRun) error
	// UpdateRun overwrites the status, summary, error and finish time of an
	// existing run.
	UpdateRun(ctx context.Context, run *EvaluationRun) error
	// InterruptRuns marks every run still in RunStatusRunning as failed, with
	// reason as its error and at as its finish time, and returns how many it
	// changed. It is called at startup: a run left running by a previous
	// process will never finish.
	InterruptRuns(ctx context.Context, reason string, at time.Time) (int, error)
	// GetRun returns the run with the given ID.
	GetRun(ctx context.Context, id string) (*EvaluationRun, error)
	// ListRuns returns the runs of a prompt, newest first.
	ListRuns(ctx context.Context, promptID string, opts ListOptions) ([]EvaluationRun, error)

	// SaveResults stores a batch of results atomically: either all rows are
	// written or none are.
	SaveResults(ctx context.Context, results []EvaluationResult) error
	// ListResultsByRun returns all results produced by a single run.
	ListResultsByRun(ctx context.Context, runID string) ([]EvaluationResult, error)
	// ListResultsByVersion returns results for a prompt version, newest first.
	ListResultsByVersion(ctx context.Context, promptVersionID string, opts ListOptions) ([]EvaluationResult, error)
}
