// Package evaluation runs a prompt's test suite against a prompt version and
// compares runs with each other.
package evaluation

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

// DefaultConcurrency is how many test cases run at once when Config does not
// say otherwise.
const DefaultConcurrency = 4

// finalizeTimeout bounds the bookkeeping writes made after a run was
// cancelled, when the caller's context can no longer be used.
const finalizeTimeout = 10 * time.Second

// Completer is the part of the gateway the runner needs.
type Completer interface {
	Complete(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error)
}

// ErrShuttingDown is returned by Start once Shutdown has been called.
var ErrShuttingDown = errors.New("evaluation runner is shutting down")

// Config configures a Runner.
type Config struct {
	// Concurrency is the number of test cases executed in parallel within
	// one run.
	Concurrency int
	// Logger defaults to slog.Default().
	Logger *slog.Logger
}

// Runner executes evaluation runs. It is safe for concurrent use.
type Runner struct {
	prompts     domain.PromptRepository
	evals       domain.EvaluationRepository
	llm         Completer
	concurrency int
	log         *slog.Logger
	now         func() time.Time

	// background is the context of runs launched by Start; Shutdown cancels
	// it. mu guards closed and makes Start and Shutdown mutually exclusive,
	// so no run is launched after Shutdown began waiting.
	background context.Context
	cancel     context.CancelFunc
	mu         sync.Mutex
	closed     bool
	wg         sync.WaitGroup
}

// NewRunner builds a Runner.
func NewRunner(prompts domain.PromptRepository, evals domain.EvaluationRepository, llm Completer, cfg Config) (*Runner, error) {
	if cfg.Concurrency < 0 {
		return nil, fmt.Errorf("%w: concurrency must not be negative", domain.ErrInvalidInput)
	}
	concurrency := cfg.Concurrency
	if concurrency == 0 {
		concurrency = DefaultConcurrency
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	background, cancel := context.WithCancel(context.Background())
	return &Runner{
		prompts: prompts, evals: evals, llm: llm, concurrency: concurrency, log: logger, now: time.Now,
		background: background, cancel: cancel,
	}, nil
}

// job is a run that has been recorded as running and is ready to execute.
type job struct {
	run     domain.EvaluationRun
	version *domain.PromptVersion
	cases   []domain.EvaluationTestCase
}

// Report is a run together with its results, in test case order.
type Report struct {
	Run     domain.EvaluationRun      `json:"run"`
	Results []domain.EvaluationResult `json:"results"`
}

// Run executes every test case of the prompt against the given version (0
// means the latest) and persists the outcome. It blocks until the run is
// over.
//
// A test case that cannot be executed (template error, provider failure) does
// not abort the run: it is recorded as an errored result. The run itself ends
// as failed only when ctx is cancelled or the results cannot be stored; in
// both cases the error is returned alongside the report built so far.
func (r *Runner) Run(ctx context.Context, promptID string, version int) (*Report, error) {
	j, err := r.prepare(ctx, promptID, version)
	if err != nil {
		return nil, err
	}
	return r.finish(ctx, j)
}

// Start records a new run and executes it in the background, returning the
// run in RunStatusRunning as soon as it is recorded. The run does not depend
// on ctx once Start returns; poll Report for its outcome. Problems that can
// be detected up front (unknown prompt or version, no test cases) are
// returned here and leave no run behind.
func (r *Runner) Start(ctx context.Context, promptID string, version int) (*domain.EvaluationRun, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, ErrShuttingDown
	}
	j, err := r.prepare(ctx, promptID, version)
	if err != nil {
		return nil, err
	}
	started := j.run

	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		if _, err := r.finish(r.background, j); err != nil {
			r.log.Error("background evaluation run failed", "run_id", started.ID, "prompt_id", promptID, "error", err)
		}
	}()
	return &started, nil
}

// Shutdown stops accepting background runs, cancels the ones in flight (they
// are recorded as failed) and waits for them, or for ctx.
func (r *Runner) Shutdown(ctx context.Context) error {
	r.mu.Lock()
	r.closed = true
	r.mu.Unlock()
	r.cancel()

	done := make(chan struct{})
	go func() {
		r.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// prepare validates the request and records the run as running.
func (r *Runner) prepare(ctx context.Context, promptID string, version int) (*job, error) {
	v, err := r.version(ctx, promptID, version)
	if err != nil {
		return nil, err
	}
	cases, err := r.evals.ListTestCases(ctx, promptID)
	if err != nil {
		return nil, err
	}
	if len(cases) == 0 {
		return nil, fmt.Errorf("%w: prompt %q has no test cases to run", domain.ErrInvalidInput, promptID)
	}

	j := &job{version: v, cases: cases, run: domain.EvaluationRun{
		PromptID:        promptID,
		PromptVersionID: v.ID,
		Version:         v.Version,
		Status:          domain.RunStatusRunning,
		StartedAt:       r.now().UTC(),
	}}
	if err := r.evals.CreateRun(ctx, &j.run); err != nil {
		return nil, err
	}
	return j, nil
}

// finish executes a prepared run and records its outcome.
func (r *Runner) finish(ctx context.Context, j *job) (*Report, error) {
	run := j.run
	results := r.execute(ctx, &run, j.version, j.cases)

	// From here on the outcome must be recorded even if the caller is gone.
	finalCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), finalizeTimeout)
	defer cancel()

	runErr := context.Cause(ctx)
	if runErr == nil {
		runErr = r.evals.SaveResults(finalCtx, results)
	} else {
		runErr = fmt.Errorf("run cancelled: %w", runErr)
		results = nil
	}

	run.FinishedAt = r.now().UTC()
	if runErr != nil {
		run.Status = domain.RunStatusFailed
		run.Error = runErr.Error()
	} else {
		run.Status = domain.RunStatusCompleted
		run.Summary = domain.Summarize(results)
	}
	if err := r.evals.UpdateRun(finalCtx, &run); err != nil {
		runErr = errors.Join(runErr, fmt.Errorf("record run outcome: %w", err))
	}
	if results == nil {
		results = []domain.EvaluationResult{}
	}
	return &Report{Run: run, Results: results}, runErr
}

func (r *Runner) version(ctx context.Context, promptID string, version int) (*domain.PromptVersion, error) {
	if version < 0 {
		return nil, fmt.Errorf("%w: version must not be negative", domain.ErrInvalidInput)
	}
	if version == 0 {
		return r.prompts.GetLatestVersion(ctx, promptID)
	}
	return r.prompts.GetVersion(ctx, promptID, version)
}

// execute runs the test cases with bounded parallelism and returns one result
// per case, in case order.
func (r *Runner) execute(ctx context.Context, run *domain.EvaluationRun, v *domain.PromptVersion, cases []domain.EvaluationTestCase) []domain.EvaluationResult {
	results := make([]domain.EvaluationResult, len(cases))
	slots := make(chan struct{}, r.concurrency)
	var wg sync.WaitGroup
	for i := range cases {
		wg.Add(1)
		slots <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-slots }()
			results[i] = r.executeCase(ctx, run, v, &cases[i])
		}()
	}
	wg.Wait()
	return results
}

func (r *Runner) executeCase(ctx context.Context, run *domain.EvaluationRun, v *domain.PromptVersion, tc *domain.EvaluationTestCase) domain.EvaluationResult {
	result := domain.EvaluationResult{
		RunID:           run.ID,
		TestCaseID:      tc.ID,
		PromptVersionID: v.ID,
		// Overwritten below with what actually served the request.
		Provider: v.Provider,
		Model:    v.Model,
	}
	fail := func(stage string, err error) domain.EvaluationResult {
		result.Error = fmt.Sprintf("%s: %v", stage, err)
		result.CreatedAt = r.now().UTC()
		return result
	}

	req, err := v.BuildRequest(tc.Variables)
	if err != nil {
		return fail("render template", err)
	}
	req.Metadata = map[string]string{
		"prompt_id":    run.PromptID,
		"run_id":       run.ID,
		"test_case_id": tc.ID,
	}
	resp, err := r.llm.Complete(ctx, req)
	if err != nil {
		return fail("completion", err)
	}
	result.Provider = resp.Provider
	result.Model = resp.Model
	result.ActualOutput = resp.Content
	result.Usage = resp.Usage
	result.CostUSD = resp.CostUSD
	result.Latency = resp.Latency

	passed, err := tc.Matches(resp.Content)
	if err != nil {
		return fail("match", err)
	}
	result.Passed = passed
	if passed {
		result.Score = 1
	}
	result.CreatedAt = r.now().UTC()
	return result
}

// Report loads a stored run with its results.
func (r *Runner) Report(ctx context.Context, runID string) (*Report, error) {
	run, err := r.evals.GetRun(ctx, runID)
	if err != nil {
		return nil, err
	}
	results, err := r.evals.ListResultsByRun(ctx, runID)
	if err != nil {
		return nil, err
	}
	return &Report{Run: *run, Results: results}, nil
}

// CaseDiff is the pair of results one test case produced in two runs.
type CaseDiff struct {
	TestCaseID string                  `json:"test_case_id"`
	Base       domain.EvaluationResult `json:"base"`
	Candidate  domain.EvaluationResult `json:"candidate"`
}

// Comparison is the difference between two runs of the same prompt, usually
// of two different versions.
type Comparison struct {
	Base      domain.EvaluationRun `json:"base"`
	Candidate domain.EvaluationRun `json:"candidate"`
	// Regressions passed in the base run and no longer pass.
	Regressions []CaseDiff `json:"regressions"`
	// Improvements did not pass in the base run and now do.
	Improvements []CaseDiff `json:"improvements"`
	// Unchanged counts the test cases with the same pass/fail outcome.
	Unchanged int `json:"unchanged"`
	// OnlyInBase and OnlyInCandidate list test case IDs present in a single
	// run, because cases were added or removed in between.
	OnlyInBase      []string `json:"only_in_base"`
	OnlyInCandidate []string `json:"only_in_candidate"`
}

// Compare reports which test cases changed outcome between two runs.
func (r *Runner) Compare(ctx context.Context, baseRunID, candidateRunID string) (*Comparison, error) {
	base, err := r.Report(ctx, baseRunID)
	if err != nil {
		return nil, fmt.Errorf("base run: %w", err)
	}
	candidate, err := r.Report(ctx, candidateRunID)
	if err != nil {
		return nil, fmt.Errorf("candidate run: %w", err)
	}
	if base.Run.PromptID != candidate.Run.PromptID {
		return nil, fmt.Errorf("%w: runs %q and %q belong to different prompts", domain.ErrInvalidInput, baseRunID, candidateRunID)
	}

	cmp := &Comparison{
		Base:            base.Run,
		Candidate:       candidate.Run,
		Regressions:     []CaseDiff{},
		Improvements:    []CaseDiff{},
		OnlyInBase:      []string{},
		OnlyInCandidate: []string{},
	}
	candidates := make(map[string]domain.EvaluationResult, len(candidate.Results))
	for _, result := range candidate.Results {
		candidates[result.TestCaseID] = result
	}
	seen := make(map[string]bool, len(base.Results))
	for _, before := range base.Results {
		seen[before.TestCaseID] = true
		after, ok := candidates[before.TestCaseID]
		if !ok {
			cmp.OnlyInBase = append(cmp.OnlyInBase, before.TestCaseID)
			continue
		}
		diff := CaseDiff{TestCaseID: before.TestCaseID, Base: before, Candidate: after}
		switch {
		case before.Passed && !after.Passed:
			cmp.Regressions = append(cmp.Regressions, diff)
		case !before.Passed && after.Passed:
			cmp.Improvements = append(cmp.Improvements, diff)
		default:
			cmp.Unchanged++
		}
	}
	for _, result := range candidate.Results {
		if !seen[result.TestCaseID] {
			cmp.OnlyInCandidate = append(cmp.OnlyInCandidate, result.TestCaseID)
		}
	}
	return cmp, nil
}
