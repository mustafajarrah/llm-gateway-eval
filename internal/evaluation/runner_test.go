package evaluation

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
	"github.com/mustafajarrah/llm-gateway-eval/internal/storage/memory"
)

var ctx = context.Background()

// completerFunc adapts a function to the Completer interface.
type completerFunc func(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error)

func (f completerFunc) Complete(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
	return f(ctx, req)
}

// echo answers with the user message, so each test case controls its own
// output through its variables.
func echo(_ context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
	content := req.Messages[0].Content
	if content == "fail" {
		return nil, &domain.ProviderError{Provider: domain.ProviderOpenAI, StatusCode: 503, Retryable: true, Err: errors.New("down")}
	}
	return &domain.LLMResponse{
		Provider: domain.ProviderAnthropic,
		Model:    "claude-haiku-4-5",
		Content:  content,
		Usage:    domain.TokenUsage{InputTokens: 10, OutputTokens: 2},
		Latency:  100 * time.Millisecond,
	}, nil
}

type fixture struct {
	store  *memory.Store
	prompt *domain.Prompt
	runner *Runner
}

func newFixture(t *testing.T, llm Completer, template string) *fixture {
	t.Helper()
	store := memory.New()
	prompt := &domain.Prompt{Name: "echo"}
	if err := store.CreatePrompt(ctx, prompt); err != nil {
		t.Fatal(err)
	}
	f := &fixture{store: store, prompt: prompt}
	f.addVersion(t, template)
	runner, err := NewRunner(store, store, llm, Config{Concurrency: 2})
	if err != nil {
		t.Fatalf("NewRunner() error = %v", err)
	}
	f.runner = runner
	return f
}

func (f *fixture) addVersion(t *testing.T, template string) *domain.PromptVersion {
	t.Helper()
	v := &domain.PromptVersion{PromptID: f.prompt.ID, Template: template, Provider: domain.ProviderOpenAI, Model: "gpt-4o"}
	if err := f.store.CreateVersion(ctx, v); err != nil {
		t.Fatal(err)
	}
	return v
}

func (f *fixture) addCase(t *testing.T, name, input, expected string, strategy domain.MatchStrategy) *domain.EvaluationTestCase {
	t.Helper()
	tc := &domain.EvaluationTestCase{
		PromptID: f.prompt.ID, Name: name, ExpectedOutput: expected, MatchStrategy: strategy,
	}
	if input != "" {
		tc.Variables = map[string]string{"in": input}
	}
	if err := f.store.CreateTestCase(ctx, tc); err != nil {
		t.Fatal(err)
	}
	return tc
}

func TestRun(t *testing.T) {
	var seen sync.Map
	llm := completerFunc(func(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
		seen.Store(req.Metadata["test_case_id"], *req)
		return echo(ctx, req)
	})
	f := newFixture(t, llm, "{{.in}}")
	pass := f.addCase(t, "passes", "Paris", "paris", domain.MatchContains)
	miss := f.addCase(t, "misses", "Berlin", "Paris", domain.MatchExact)
	down := f.addCase(t, "provider down", "fail", "x", domain.MatchExact)
	noVar := f.addCase(t, "missing variable", "", "x", domain.MatchExact)

	report, err := f.runner.Run(ctx, f.prompt.ID, 0)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	run := report.Run
	if run.ID == "" || run.PromptID != f.prompt.ID || run.Version != 1 || run.Status != domain.RunStatusCompleted ||
		run.Error != "" || run.StartedAt.IsZero() || run.FinishedAt.Before(run.StartedAt) {
		t.Errorf("run = %+v", run)
	}
	wantSummary := domain.RunSummary{
		Total: 4, Passed: 1, Failed: 1, Errored: 2,
		Usage:          domain.TokenUsage{InputTokens: 20, OutputTokens: 4},
		TotalLatencyMS: 200,
	}
	if run.Summary != wantSummary {
		t.Errorf("summary = %+v, want %+v", run.Summary, wantSummary)
	}

	if len(report.Results) != 4 {
		t.Fatalf("got %d results, want 4", len(report.Results))
	}
	for i, tc := range []*domain.EvaluationTestCase{pass, miss, down, noVar} {
		r := report.Results[i]
		if r.TestCaseID != tc.ID || r.RunID != run.ID || r.PromptVersionID != run.PromptVersionID || r.ID == "" || r.CreatedAt.IsZero() {
			t.Errorf("results[%d] = %+v, want the result of case %q in order", i, r, tc.Name)
		}
	}
	got := report.Results
	if !got[0].Passed || got[0].Score != 1 || got[0].ActualOutput != "Paris" ||
		got[0].Provider != domain.ProviderAnthropic || got[0].Model != "claude-haiku-4-5" || got[0].Latency != 100*time.Millisecond {
		t.Errorf("passing result = %+v, want it to record what actually served the request", got[0])
	}
	if got[1].Passed || got[1].Score != 0 || got[1].Error != "" || got[1].ActualOutput != "Berlin" {
		t.Errorf("failing result = %+v", got[1])
	}
	if got[2].Passed || !strings.HasPrefix(got[2].Error, "completion: ") || !strings.Contains(got[2].Error, "status 503") ||
		got[2].Provider != domain.ProviderOpenAI || got[2].Model != "gpt-4o" {
		t.Errorf("provider-failure result = %+v, want an error and the version's own target", got[2])
	}
	if got[3].Passed || !strings.HasPrefix(got[3].Error, "render template: ") {
		t.Errorf("render-failure result = %+v", got[3])
	}

	sent, _ := seen.Load(pass.ID)
	req := sent.(domain.LLMRequest)
	if req.Provider != domain.ProviderOpenAI || req.Model != "gpt-4o" || req.Metadata["run_id"] != run.ID || req.Metadata["prompt_id"] != f.prompt.ID {
		t.Errorf("request = %+v, want the version's target and tracing metadata", req)
	}
	if _, called := seen.Load(noVar.ID); called {
		t.Error("the provider was called for a case whose template failed to render")
	}

	stored, err := f.runner.Report(ctx, run.ID)
	if err != nil {
		t.Fatalf("Report() error = %v", err)
	}
	if stored.Run.Status != domain.RunStatusCompleted || stored.Run.Summary != wantSummary || len(stored.Results) != 4 ||
		stored.Results[0].ID != got[0].ID {
		t.Errorf("stored report = %+v, want what Run() returned", stored)
	}
}

func TestRunSpecificVersion(t *testing.T) {
	f := newFixture(t, completerFunc(echo), "v1 {{.in}}")
	f.addVersion(t, "v2 {{.in}}")
	f.addCase(t, "case", "x", "v1 x", domain.MatchExact)

	first, err := f.runner.Run(ctx, f.prompt.ID, 1)
	if err != nil {
		t.Fatalf("Run(version 1) error = %v", err)
	}
	if first.Run.Version != 1 || first.Run.Summary.Passed != 1 {
		t.Errorf("run of version 1 = %+v", first.Run)
	}
	latest, err := f.runner.Run(ctx, f.prompt.ID, 0)
	if err != nil {
		t.Fatalf("Run(latest) error = %v", err)
	}
	if latest.Run.Version != 2 || latest.Run.Summary.Failed != 1 {
		t.Errorf("run of the latest version = %+v", latest.Run)
	}
}

func TestRunMatchError(t *testing.T) {
	f := newFixture(t, completerFunc(echo), "{{.in}}")
	tc := f.addCase(t, "case", "x", "x", domain.MatchExact)
	// A strategy that became invalid after the case was stored.
	broken := &brokenCases{EvaluationRepository: f.store, strategy: "fuzzy", id: tc.ID}
	runner, _ := NewRunner(f.store, broken, completerFunc(echo), Config{})

	report, err := runner.Run(ctx, f.prompt.ID, 0)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if r := report.Results[0]; r.Passed || !strings.HasPrefix(r.Error, "match: ") || r.ActualOutput != "x" {
		t.Errorf("result = %+v, want a match error that keeps the output", r)
	}
}

func TestRunConcurrencyIsBounded(t *testing.T) {
	var inFlight, peak atomic.Int32
	llm := completerFunc(func(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
		n := inFlight.Add(1)
		for {
			old := peak.Load()
			if n <= old || peak.CompareAndSwap(old, n) {
				break
			}
		}
		time.Sleep(5 * time.Millisecond)
		inFlight.Add(-1)
		return echo(ctx, req)
	})
	f := newFixture(t, llm, "{{.in}}")
	for i := range 12 {
		f.addCase(t, "case", string(rune('a'+i)), "x", domain.MatchExact)
	}
	if _, err := f.runner.Run(ctx, f.prompt.ID, 0); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got := peak.Load(); got > 2 {
		t.Errorf("peak concurrency = %d, want at most the configured 2", got)
	}
}

func TestRunRejections(t *testing.T) {
	f := newFixture(t, completerFunc(echo), "{{.in}}")

	if _, err := f.runner.Run(ctx, f.prompt.ID, 0); !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("no test cases: error = %v, want ErrInvalidInput", err)
	}
	f.addCase(t, "case", "x", "x", domain.MatchExact)
	if _, err := f.runner.Run(ctx, f.prompt.ID, -1); !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("negative version: error = %v, want ErrInvalidInput", err)
	}
	if _, err := f.runner.Run(ctx, f.prompt.ID, 7); !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("unknown version: error = %v, want ErrNotFound", err)
	}
	if _, err := f.runner.Run(ctx, "missing", 0); !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("unknown prompt: error = %v, want ErrNotFound", err)
	}
	runs, err := f.store.ListRuns(ctx, f.prompt.ID, domain.ListOptions{})
	if err != nil || len(runs) != 0 {
		t.Errorf("rejected runs left %d run records behind (%v)", len(runs), err)
	}
}

func TestRunCancelled(t *testing.T) {
	runCtx, cancel := context.WithCancel(ctx)
	llm := completerFunc(func(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
		cancel()
		return nil, ctx.Err()
	})
	f := newFixture(t, llm, "{{.in}}")
	f.addCase(t, "case", "x", "x", domain.MatchExact)

	report, err := f.runner.Run(runCtx, f.prompt.ID, 0)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Run() error = %v, want context.Canceled", err)
	}
	if report.Run.Status != domain.RunStatusFailed || !strings.Contains(report.Run.Error, "run cancelled") ||
		report.Results == nil || len(report.Results) != 0 {
		t.Errorf("report = %+v, want a failed run without results", report)
	}
	stored, err := f.store.GetRun(ctx, report.Run.ID)
	if err != nil {
		t.Fatalf("GetRun() error = %v", err)
	}
	if stored.Status != domain.RunStatusFailed || stored.FinishedAt.IsZero() {
		t.Errorf("stored run = %+v, want the failure recorded despite the cancelled context", stored)
	}
}

func TestRunStorageFailures(t *testing.T) {
	boom := errors.New("disk full")
	tests := []struct {
		name       string
		failOn     string
		wantReport bool
	}{
		{name: "listing test cases", failOn: "ListTestCases"},
		{name: "creating the run", failOn: "CreateRun"},
		{name: "saving results", failOn: "SaveResults", wantReport: true},
		{name: "recording the outcome", failOn: "UpdateRun", wantReport: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFixture(t, completerFunc(echo), "{{.in}}")
			f.addCase(t, "case", "x", "x", domain.MatchExact)
			failing := &failingEvals{EvaluationRepository: f.store, failOn: tt.failOn, err: boom}
			runner, _ := NewRunner(f.store, failing, completerFunc(echo), Config{})

			report, err := runner.Run(ctx, f.prompt.ID, 0)
			if !errors.Is(err, boom) {
				t.Fatalf("Run() error = %v, want the storage error", err)
			}
			if (report != nil) != tt.wantReport {
				t.Fatalf("report = %+v, wantReport %v", report, tt.wantReport)
			}
			if tt.failOn == "SaveResults" {
				stored, err := f.store.GetRun(ctx, report.Run.ID)
				if err != nil {
					t.Fatalf("GetRun() error = %v", err)
				}
				if stored.Status != domain.RunStatusFailed || !strings.Contains(stored.Error, "disk full") {
					t.Errorf("stored run = %+v, want it marked failed with the cause", stored)
				}
			}
		})
	}
}

// waitForRun polls until the run leaves the running state.
func waitForRun(t *testing.T, f *fixture, runID string) *domain.EvaluationRun {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		run, err := f.store.GetRun(ctx, runID)
		if err != nil {
			t.Fatalf("GetRun() error = %v", err)
		}
		if run.Status != domain.RunStatusRunning {
			return run
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("run %s is still running", runID)
	return nil
}

func TestStartRunsInTheBackground(t *testing.T) {
	release := make(chan struct{})
	llm := completerFunc(func(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
		<-release
		return echo(ctx, req)
	})
	f := newFixture(t, llm, "{{.in}}")
	f.addCase(t, "case", "x", "x", domain.MatchExact)

	// The request context ends as soon as Start returns, as an HTTP
	// request's would; the run must not depend on it.
	reqCtx, cancel := context.WithCancel(ctx)
	started, err := f.runner.Start(reqCtx, f.prompt.ID, 0)
	cancel()
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	if started.ID == "" || started.Status != domain.RunStatusRunning || started.Version != 1 || !started.FinishedAt.IsZero() {
		t.Errorf("Start() = %+v, want a running run", started)
	}
	if stored, _ := f.store.GetRun(ctx, started.ID); stored == nil || stored.Status != domain.RunStatusRunning {
		t.Errorf("stored run = %+v, want it visible as running while in flight", stored)
	}

	close(release)
	finished := waitForRun(t, f, started.ID)
	if finished.Status != domain.RunStatusCompleted || finished.Summary.Passed != 1 || finished.FinishedAt.IsZero() {
		t.Errorf("finished run = %+v", finished)
	}
	report, err := f.runner.Report(ctx, started.ID)
	if err != nil || len(report.Results) != 1 || !report.Results[0].Passed {
		t.Errorf("report = %+v, %v", report, err)
	}
	if err := f.runner.Shutdown(ctx); err != nil {
		t.Errorf("Shutdown() with nothing in flight error = %v", err)
	}
}

func TestStartRejections(t *testing.T) {
	f := newFixture(t, completerFunc(echo), "{{.in}}")
	if _, err := f.runner.Start(ctx, f.prompt.ID, 0); !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("no test cases: error = %v, want ErrInvalidInput", err)
	}
	if _, err := f.runner.Start(ctx, "missing", 0); !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("unknown prompt: error = %v, want ErrNotFound", err)
	}
	if runs, _ := f.store.ListRuns(ctx, f.prompt.ID, domain.ListOptions{}); len(runs) != 0 {
		t.Errorf("rejected starts left %d runs behind", len(runs))
	}
}

func TestShutdownCancelsBackgroundRuns(t *testing.T) {
	entered := make(chan struct{})
	llm := completerFunc(func(ctx context.Context, _ *domain.LLMRequest) (*domain.LLMResponse, error) {
		close(entered)
		<-ctx.Done()
		return nil, ctx.Err()
	})
	f := newFixture(t, llm, "{{.in}}")
	f.addCase(t, "case", "x", "x", domain.MatchExact)

	started, err := f.runner.Start(ctx, f.prompt.ID, 0)
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	<-entered
	if err := f.runner.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown() error = %v", err)
	}

	// Shutdown waited, so the outcome is already recorded.
	stored, err := f.store.GetRun(ctx, started.ID)
	if err != nil {
		t.Fatalf("GetRun() error = %v", err)
	}
	if stored.Status != domain.RunStatusFailed || !strings.Contains(stored.Error, "run cancelled") || stored.FinishedAt.IsZero() {
		t.Errorf("run after Shutdown = %+v, want it recorded as failed", stored)
	}
	if _, err := f.runner.Start(ctx, f.prompt.ID, 0); !errors.Is(err, ErrShuttingDown) {
		t.Errorf("Start() after Shutdown error = %v, want ErrShuttingDown", err)
	}
}

func TestShutdownGivesUpWhenItsContextEnds(t *testing.T) {
	release := make(chan struct{})
	entered := make(chan struct{})
	// A completer that ignores cancellation.
	llm := completerFunc(func(ctx context.Context, req *domain.LLMRequest) (*domain.LLMResponse, error) {
		close(entered)
		<-release
		return echo(ctx, req)
	})
	f := newFixture(t, llm, "{{.in}}")
	f.addCase(t, "case", "x", "x", domain.MatchExact)
	started, err := f.runner.Start(ctx, f.prompt.ID, 0)
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	<-entered

	shutdownCtx, cancel := context.WithTimeout(ctx, 20*time.Millisecond)
	defer cancel()
	if err := f.runner.Shutdown(shutdownCtx); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Shutdown() error = %v, want context.DeadlineExceeded", err)
	}
	close(release)
	waitForRun(t, f, started.ID)
}

func TestReportErrors(t *testing.T) {
	f := newFixture(t, completerFunc(echo), "{{.in}}")
	if _, err := f.runner.Report(ctx, "missing"); !errors.Is(err, domain.ErrNotFound) {
		t.Errorf("Report(missing) error = %v, want ErrNotFound", err)
	}

	f.addCase(t, "case", "x", "x", domain.MatchExact)
	report, err := f.runner.Run(ctx, f.prompt.ID, 0)
	if err != nil {
		t.Fatal(err)
	}
	boom := errors.New("read error")
	failing := &failingEvals{EvaluationRepository: f.store, failOn: "ListResultsByRun", err: boom}
	runner, _ := NewRunner(f.store, failing, completerFunc(echo), Config{})
	if _, err := runner.Report(ctx, report.Run.ID); !errors.Is(err, boom) {
		t.Errorf("Report() error = %v, want the storage error", err)
	}
}

func TestCompare(t *testing.T) {
	// Version 1 echoes the input and version 2 prefixes it with "v2 ", which
	// changes which expectations hold.
	f := newFixture(t, completerFunc(echo), "{{.in}}")
	regress := f.addCase(t, "regresses", "yes", "yes", domain.MatchExact)
	improve := f.addCase(t, "improves", "yes", "v2 yes", domain.MatchExact)
	same := f.addCase(t, "stays green", "ok", "ok", domain.MatchContains)
	removed := f.addCase(t, "removed later", "gone", "gone", domain.MatchExact)

	base, err := f.runner.Run(ctx, f.prompt.ID, 0)
	if err != nil {
		t.Fatal(err)
	}

	f.addVersion(t, "v2 {{.in}}")
	if err := f.store.DeleteTestCase(ctx, removed.ID); err != nil {
		t.Fatal(err)
	}
	added := f.addCase(t, "added later", "new", "new", domain.MatchContains)
	candidate, err := f.runner.Run(ctx, f.prompt.ID, 0)
	if err != nil {
		t.Fatal(err)
	}

	cmp, err := f.runner.Compare(ctx, base.Run.ID, candidate.Run.ID)
	if err != nil {
		t.Fatalf("Compare() error = %v", err)
	}
	if cmp.Base.Version != 1 || cmp.Candidate.Version != 2 {
		t.Errorf("compared versions %d and %d, want 1 and 2", cmp.Base.Version, cmp.Candidate.Version)
	}
	if len(cmp.Regressions) != 1 || cmp.Regressions[0].TestCaseID != regress.ID ||
		!cmp.Regressions[0].Base.Passed || cmp.Regressions[0].Candidate.Passed ||
		cmp.Regressions[0].Candidate.ActualOutput != "v2 yes" {
		t.Errorf("regressions = %+v", cmp.Regressions)
	}
	if len(cmp.Improvements) != 1 || cmp.Improvements[0].TestCaseID != improve.ID {
		t.Errorf("improvements = %+v", cmp.Improvements)
	}
	if cmp.Unchanged != 1 {
		t.Errorf("unchanged = %d, want 1 (%s)", cmp.Unchanged, same.Name)
	}
	if len(cmp.OnlyInBase) != 1 || cmp.OnlyInBase[0] != removed.ID {
		t.Errorf("only in base = %v", cmp.OnlyInBase)
	}
	if len(cmp.OnlyInCandidate) != 1 || cmp.OnlyInCandidate[0] != added.ID {
		t.Errorf("only in candidate = %v", cmp.OnlyInCandidate)
	}

	identical, err := f.runner.Compare(ctx, base.Run.ID, base.Run.ID)
	if err != nil {
		t.Fatalf("Compare() of a run with itself error = %v", err)
	}
	if identical.Unchanged != 4 || identical.Regressions == nil || len(identical.Regressions) != 0 ||
		identical.OnlyInBase == nil || identical.OnlyInCandidate == nil || identical.Improvements == nil {
		t.Errorf("self comparison = %+v, want 4 unchanged and empty, non-nil lists", identical)
	}
}

func TestCompareErrors(t *testing.T) {
	f := newFixture(t, completerFunc(echo), "{{.in}}")
	f.addCase(t, "case", "x", "x", domain.MatchExact)
	run, err := f.runner.Run(ctx, f.prompt.ID, 0)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := f.runner.Compare(ctx, "missing", run.Run.ID); !errors.Is(err, domain.ErrNotFound) || !strings.Contains(err.Error(), "base run") {
		t.Errorf("missing base: error = %v", err)
	}
	if _, err := f.runner.Compare(ctx, run.Run.ID, "missing"); !errors.Is(err, domain.ErrNotFound) || !strings.Contains(err.Error(), "candidate run") {
		t.Errorf("missing candidate: error = %v", err)
	}

	other := &domain.Prompt{Name: "other"}
	if err := f.store.CreatePrompt(ctx, other); err != nil {
		t.Fatal(err)
	}
	foreign := &domain.EvaluationRun{PromptID: other.ID, PromptVersionID: "v", Status: domain.RunStatusCompleted}
	if err := f.store.CreateRun(ctx, foreign); err != nil {
		t.Fatal(err)
	}
	if _, err := f.runner.Compare(ctx, run.Run.ID, foreign.ID); !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("runs of different prompts: error = %v, want ErrInvalidInput", err)
	}
}

func TestNewRunner(t *testing.T) {
	store := memory.New()
	r, err := NewRunner(store, store, completerFunc(echo), Config{})
	if err != nil {
		t.Fatalf("NewRunner() error = %v", err)
	}
	if r.concurrency != DefaultConcurrency {
		t.Errorf("concurrency = %d, want the default %d", r.concurrency, DefaultConcurrency)
	}
	if _, err := NewRunner(store, store, completerFunc(echo), Config{Concurrency: -1}); !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("negative concurrency: error = %v, want ErrInvalidInput", err)
	}
}

// failingEvals makes one repository method fail.
type failingEvals struct {
	domain.EvaluationRepository
	failOn string
	err    error
}

func (f *failingEvals) ListTestCases(ctx context.Context, promptID string) ([]domain.EvaluationTestCase, error) {
	if f.failOn == "ListTestCases" {
		return nil, f.err
	}
	return f.EvaluationRepository.ListTestCases(ctx, promptID)
}

func (f *failingEvals) CreateRun(ctx context.Context, run *domain.EvaluationRun) error {
	if f.failOn == "CreateRun" {
		return f.err
	}
	return f.EvaluationRepository.CreateRun(ctx, run)
}

func (f *failingEvals) SaveResults(ctx context.Context, results []domain.EvaluationResult) error {
	if f.failOn == "SaveResults" {
		return f.err
	}
	return f.EvaluationRepository.SaveResults(ctx, results)
}

func (f *failingEvals) UpdateRun(ctx context.Context, run *domain.EvaluationRun) error {
	if f.failOn == "UpdateRun" {
		return f.err
	}
	return f.EvaluationRepository.UpdateRun(ctx, run)
}

func (f *failingEvals) ListResultsByRun(ctx context.Context, runID string) ([]domain.EvaluationResult, error) {
	if f.failOn == "ListResultsByRun" {
		return nil, f.err
	}
	return f.EvaluationRepository.ListResultsByRun(ctx, runID)
}

// brokenCases corrupts the match strategy of one listed test case.
type brokenCases struct {
	domain.EvaluationRepository
	strategy domain.MatchStrategy
	id       string
}

func (b *brokenCases) ListTestCases(ctx context.Context, promptID string) ([]domain.EvaluationTestCase, error) {
	cases, err := b.EvaluationRepository.ListTestCases(ctx, promptID)
	for i := range cases {
		if cases[i].ID == b.id {
			cases[i].MatchStrategy = b.strategy
		}
	}
	return cases, err
}
