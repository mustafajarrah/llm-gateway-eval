package domain

import (
	"bytes"
	"context"
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
	Version  int    `json:"version"`
	// Template is a Go text/template rendered with the test case (or caller)
	// variables, e.g. "Summarise the following text:\n{{.text}}".
	Template     string          `json:"template"`
	SystemPrompt string          `json:"system_prompt,omitempty"`
	Provider     Provider        `json:"provider"`
	Model        string          `json:"model"`
	Parameters   ModelParameters `json:"parameters"`
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
	if v.Version < 1 {
		errs = append(errs, fmt.Errorf("version must be >= 1, got %d", v.Version))
	}
	if strings.TrimSpace(v.Template) == "" {
		errs = append(errs, errors.New("template is required"))
	} else if _, err := v.parse(); err != nil {
		errs = append(errs, fmt.Errorf("template: %w", err))
	}
	if !v.Provider.Valid() {
		errs = append(errs, fmt.Errorf("unknown provider %q", v.Provider))
	}
	if v.Model == "" {
		errs = append(errs, errors.New("model is required"))
	}
	if err := v.Parameters.Validate(); err != nil {
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
	Score   float64       `json:"score"`
	Usage   TokenUsage    `json:"usage"`
	Latency time.Duration `json:"latency"`
	// Error is non-empty when the test case could not be executed (render
	// failure, provider outage). Such results always have Passed == false.
	Error     string    `json:"error,omitempty"`
	CreatedAt time.Time `json:"created_at"`
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
// return errors wrapping ErrNotFound or ErrConflict where applicable.
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

// EvaluationRepository persists test cases and evaluation results.
type EvaluationRepository interface {
	// CreateTestCase stores a new test case.
	CreateTestCase(ctx context.Context, tc *EvaluationTestCase) error
	// GetTestCase returns the test case with the given ID.
	GetTestCase(ctx context.Context, id string) (*EvaluationTestCase, error)
	// ListTestCases returns every test case attached to a prompt.
	ListTestCases(ctx context.Context, promptID string) ([]EvaluationTestCase, error)
	// DeleteTestCase removes a test case.
	DeleteTestCase(ctx context.Context, id string) error

	// SaveResults stores a batch of results atomically: either all rows are
	// written or none are.
	SaveResults(ctx context.Context, results []EvaluationResult) error
	// ListResultsByRun returns all results produced by a single run.
	ListResultsByRun(ctx context.Context, runID string) ([]EvaluationResult, error)
	// ListResultsByVersion returns results for a prompt version, newest first.
	ListResultsByVersion(ctx context.Context, promptVersionID string, opts ListOptions) ([]EvaluationResult, error)
}
