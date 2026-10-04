// Package storagetest is the conformance suite every storage backend must
// pass. It pins down the behaviour the domain repository interfaces leave to
// their doc comments: ID and timestamp assignment, ordering, error kinds,
// cascading deletes, atomic batches and isolation from caller mutation.
package storagetest

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

// Store is the full set of repositories a backend provides.
type Store interface {
	domain.PromptRepository
	domain.EvaluationRepository
}

// Run executes the suite. newStore must return a fresh, empty store on every
// call and register its own cleanup with t.Cleanup.
func Run(t *testing.T, newStore func(t *testing.T) Store) {
	tests := map[string]func(*testing.T, Store){
		"Prompts":             testPrompts,
		"PromptListing":       testPromptListing,
		"Versions":            testVersions,
		"ConcurrentVersions":  testConcurrentVersions,
		"TestCases":           testTestCases,
		"Runs":                testRuns,
		"InterruptRuns":       testInterruptRuns,
		"Results":             testResults,
		"ResultsAreAtomic":    testResultsAreAtomic,
		"DeletePromptCascade": testDeletePromptCascade,
		"Isolation":           testIsolation,
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) { test(t, newStore(t)) })
	}
}

var ctx = context.Background()

func ptr[T any](v T) *T { return &v }

func wantErr(t *testing.T, err, target error, what string) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Errorf("%s: error = %v, want %v", what, err, target)
	}
}

func mustCreatePrompt(t *testing.T, s Store, name string) *domain.Prompt {
	t.Helper()
	p := &domain.Prompt{Name: name, Description: "about " + name, Tags: []string{"a", "b"}}
	if err := s.CreatePrompt(ctx, p); err != nil {
		t.Fatalf("CreatePrompt(%q) error = %v", name, err)
	}
	return p
}

func newVersion(promptID string) *domain.PromptVersion {
	return &domain.PromptVersion{
		PromptID:     promptID,
		Template:     "Summarise: {{.text}}",
		SystemPrompt: "Be brief.",
		Provider:     domain.ProviderOpenAI,
		Model:        "gpt-4o",
		Parameters: domain.ModelParameters{
			Temperature: ptr(0.0),
			TopP:        ptr(0.9),
			MaxTokens:   128,
			Stop:        []string{"END", "STOP"},
		},
		ChangeLog: "first draft",
	}
}

func mustCreateVersion(t *testing.T, s Store, promptID string) *domain.PromptVersion {
	t.Helper()
	v := newVersion(promptID)
	if err := s.CreateVersion(ctx, v); err != nil {
		t.Fatalf("CreateVersion() error = %v", err)
	}
	return v
}

func mustCreateRun(t *testing.T, s Store, v *domain.PromptVersion) *domain.EvaluationRun {
	t.Helper()
	run := &domain.EvaluationRun{
		PromptID:        v.PromptID,
		PromptVersionID: v.ID,
		Version:         v.Version,
		Status:          domain.RunStatusRunning,
	}
	if err := s.CreateRun(ctx, run); err != nil {
		t.Fatalf("CreateRun() error = %v", err)
	}
	return run
}

func testPrompts(t *testing.T, s Store) {
	p := mustCreatePrompt(t, s, "summariser")
	if len(p.ID) == 0 || p.CreatedAt.IsZero() || p.UpdatedAt.IsZero() {
		t.Fatalf("CreatePrompt() did not assign ID and timestamps: %+v", p)
	}

	got, err := s.GetPrompt(ctx, p.ID)
	if err != nil {
		t.Fatalf("GetPrompt() error = %v", err)
	}
	if got.ID != p.ID || got.Name != "summariser" || got.Description != "about summariser" ||
		len(got.Tags) != 2 || got.Tags[0] != "a" || got.Tags[1] != "b" ||
		!got.CreatedAt.Equal(p.CreatedAt) || !got.UpdatedAt.Equal(p.UpdatedAt) {
		t.Errorf("GetPrompt() = %+v, want %+v", got, p)
	}

	explicit := &domain.Prompt{
		ID:        "fixed-id",
		Name:      "explicit",
		CreatedAt: time.Date(2026, 1, 2, 3, 4, 5, 6000, time.UTC),
	}
	if err := s.CreatePrompt(ctx, explicit); err != nil {
		t.Fatalf("CreatePrompt() with explicit ID error = %v", err)
	}
	got, err = s.GetPrompt(ctx, "fixed-id")
	if err != nil {
		t.Fatalf("GetPrompt(fixed-id) error = %v", err)
	}
	if !got.CreatedAt.Equal(explicit.CreatedAt) || !got.UpdatedAt.Equal(explicit.CreatedAt) {
		t.Errorf("explicit timestamps not preserved: %+v", got)
	}

	wantErr(t, s.CreatePrompt(ctx, &domain.Prompt{Name: "summariser"}), domain.ErrConflict, "duplicate name")
	wantErr(t, s.CreatePrompt(ctx, &domain.Prompt{ID: "fixed-id", Name: "other"}), domain.ErrConflict, "duplicate ID")
	wantErr(t, s.CreatePrompt(ctx, &domain.Prompt{Name: " "}), domain.ErrInvalidInput, "blank name")

	_, err = s.GetPrompt(ctx, "missing")
	wantErr(t, err, domain.ErrNotFound, "GetPrompt(missing)")
	wantErr(t, s.DeletePrompt(ctx, "missing"), domain.ErrNotFound, "DeletePrompt(missing)")

	if err := s.DeletePrompt(ctx, p.ID); err != nil {
		t.Fatalf("DeletePrompt() error = %v", err)
	}
	_, err = s.GetPrompt(ctx, p.ID)
	wantErr(t, err, domain.ErrNotFound, "GetPrompt after delete")
	if err := s.CreatePrompt(ctx, &domain.Prompt{Name: "summariser"}); err != nil {
		t.Errorf("the name of a deleted prompt must be reusable: %v", err)
	}
}

func testPromptListing(t *testing.T, s Store) {
	empty, err := s.ListPrompts(ctx, domain.ListOptions{})
	if err != nil || empty == nil || len(empty) != 0 {
		t.Fatalf("ListPrompts() on an empty store = %v, %v; want an empty, non-nil slice", empty, err)
	}

	for i := range 5 {
		mustCreatePrompt(t, s, fmt.Sprintf("p%d", i))
	}
	names := func(opts domain.ListOptions) string {
		t.Helper()
		prompts, err := s.ListPrompts(ctx, opts)
		if err != nil {
			t.Fatalf("ListPrompts(%+v) error = %v", opts, err)
		}
		var out string
		for _, p := range prompts {
			out += p.Name + " "
		}
		return out
	}
	if got := names(domain.ListOptions{}); got != "p4 p3 p2 p1 p0 " {
		t.Errorf("ListPrompts() = %q, want newest first", got)
	}
	if got := names(domain.ListOptions{Limit: 2, Offset: 1}); got != "p3 p2 " {
		t.Errorf("ListPrompts(limit 2, offset 1) = %q", got)
	}
	if got := names(domain.ListOptions{Limit: -1, Offset: -1}); got != "p4 p3 p2 p1 p0 " {
		t.Errorf("ListPrompts() with out-of-range options = %q, want them normalised", got)
	}
	if got := names(domain.ListOptions{Offset: 10}); got != "" {
		t.Errorf("ListPrompts(offset past the end) = %q, want nothing", got)
	}
}

func testVersions(t *testing.T, s Store) {
	p := mustCreatePrompt(t, s, "summariser")
	other := mustCreatePrompt(t, s, "other")

	_, err := s.GetLatestVersion(ctx, p.ID)
	wantErr(t, err, domain.ErrNotFound, "GetLatestVersion with no versions")
	none, err := s.ListVersions(ctx, p.ID)
	if err != nil || none == nil || len(none) != 0 {
		t.Fatalf("ListVersions() with no versions = %v, %v; want an empty, non-nil slice", none, err)
	}

	v1 := newVersion(p.ID)
	v1.Version = 42 // must be overwritten
	if err := s.CreateVersion(ctx, v1); err != nil {
		t.Fatalf("CreateVersion() error = %v", err)
	}
	if v1.Version != 1 || v1.ID == "" || v1.CreatedAt.IsZero() {
		t.Fatalf("CreateVersion() did not assign version 1, ID and timestamp: %+v", v1)
	}
	v2 := newVersion(p.ID)
	v2.Provider, v2.Model, v2.Parameters, v2.SystemPrompt = "", "fast", domain.ModelParameters{}, ""
	if err := s.CreateVersion(ctx, v2); err != nil {
		t.Fatalf("CreateVersion() error = %v", err)
	}
	if v2.Version != 2 {
		t.Errorf("second version numbered %d, want 2", v2.Version)
	}
	if first := mustCreateVersion(t, s, other.ID); first.Version != 1 {
		t.Errorf("numbering is not per prompt: first version of another prompt is %d", first.Version)
	}

	got, err := s.GetVersion(ctx, p.ID, 1)
	if err != nil {
		t.Fatalf("GetVersion() error = %v", err)
	}
	if got.ID != v1.ID || got.PromptID != p.ID || got.Version != 1 || got.Template != v1.Template ||
		got.SystemPrompt != "Be brief." || got.Provider != domain.ProviderOpenAI || got.Model != "gpt-4o" ||
		got.ChangeLog != "first draft" || !got.CreatedAt.Equal(v1.CreatedAt) {
		t.Errorf("GetVersion() = %+v, want %+v", got, v1)
	}
	params := got.Parameters
	if params.Temperature == nil || *params.Temperature != 0 || params.TopP == nil || *params.TopP != 0.9 ||
		params.MaxTokens != 128 || len(params.Stop) != 2 || params.Stop[1] != "STOP" {
		t.Errorf("parameters did not round-trip: %+v", params)
	}

	latest, err := s.GetLatestVersion(ctx, p.ID)
	if err != nil {
		t.Fatalf("GetLatestVersion() error = %v", err)
	}
	if latest.Version != 2 || latest.Provider != "" || latest.Model != "fast" ||
		latest.Parameters.Temperature != nil || latest.Parameters.TopP != nil || len(latest.Parameters.Stop) != 0 {
		t.Errorf("GetLatestVersion() = %+v, want version 2 with unset parameters", latest)
	}

	all, err := s.ListVersions(ctx, p.ID)
	if err != nil {
		t.Fatalf("ListVersions() error = %v", err)
	}
	if len(all) != 2 || all[0].Version != 1 || all[1].Version != 2 {
		t.Errorf("ListVersions() = %+v, want versions 1 and 2 in ascending order", all)
	}

	refreshed, err := s.GetPrompt(ctx, p.ID)
	if err != nil {
		t.Fatalf("GetPrompt() error = %v", err)
	}
	if refreshed.UpdatedAt.Before(p.UpdatedAt) {
		t.Errorf("UpdatedAt went backwards: %v -> %v", p.UpdatedAt, refreshed.UpdatedAt)
	}

	_, err = s.GetVersion(ctx, p.ID, 3)
	wantErr(t, err, domain.ErrNotFound, "GetVersion(missing)")
	_, err = s.ListVersions(ctx, "missing")
	wantErr(t, err, domain.ErrNotFound, "ListVersions(missing prompt)")
	wantErr(t, s.CreateVersion(ctx, newVersion("missing")), domain.ErrNotFound, "CreateVersion(missing prompt)")
	dup := newVersion(p.ID)
	dup.ID = v1.ID
	wantErr(t, s.CreateVersion(ctx, dup), domain.ErrConflict, "duplicate version ID")
	bad := newVersion(p.ID)
	bad.Template = "{{.broken"
	wantErr(t, s.CreateVersion(ctx, bad), domain.ErrInvalidInput, "invalid template")
}

func testConcurrentVersions(t *testing.T, s Store) {
	p := mustCreatePrompt(t, s, "contended")

	const writers = 8
	var wg sync.WaitGroup
	errs := make(chan error, writers)
	for range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- s.CreateVersion(ctx, newVersion(p.ID))
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Errorf("concurrent CreateVersion() error = %v", err)
		}
	}

	versions, err := s.ListVersions(ctx, p.ID)
	if err != nil {
		t.Fatalf("ListVersions() error = %v", err)
	}
	if len(versions) != writers {
		t.Fatalf("got %d versions, want %d", len(versions), writers)
	}
	for i, v := range versions {
		if v.Version != i+1 {
			t.Errorf("versions[%d].Version = %d, want %d: numbering must be gap-free and unique", i, v.Version, i+1)
		}
	}
}

func testTestCases(t *testing.T, s Store) {
	p := mustCreatePrompt(t, s, "summariser")

	none, err := s.ListTestCases(ctx, p.ID)
	if err != nil || none == nil || len(none) != 0 {
		t.Fatalf("ListTestCases() with no cases = %v, %v; want an empty, non-nil slice", none, err)
	}

	first := &domain.EvaluationTestCase{
		PromptID:       p.ID,
		Name:           "mentions go",
		Variables:      map[string]string{"text": "Go is great", "lang": "en"},
		ExpectedOutput: "go",
		MatchStrategy:  domain.MatchContains,
	}
	second := &domain.EvaluationTestCase{
		PromptID:       p.ID,
		Name:           "no variables",
		ExpectedOutput: `^\d+$`,
		MatchStrategy:  domain.MatchRegex,
	}
	for _, tc := range []*domain.EvaluationTestCase{first, second} {
		if err := s.CreateTestCase(ctx, tc); err != nil {
			t.Fatalf("CreateTestCase() error = %v", err)
		}
		if tc.ID == "" || tc.CreatedAt.IsZero() {
			t.Fatalf("CreateTestCase() did not assign ID and timestamp: %+v", tc)
		}
	}

	got, err := s.GetTestCase(ctx, first.ID)
	if err != nil {
		t.Fatalf("GetTestCase() error = %v", err)
	}
	if got.PromptID != p.ID || got.Name != "mentions go" || got.ExpectedOutput != "go" ||
		got.MatchStrategy != domain.MatchContains || len(got.Variables) != 2 ||
		got.Variables["text"] != "Go is great" || !got.CreatedAt.Equal(first.CreatedAt) {
		t.Errorf("GetTestCase() = %+v, want %+v", got, first)
	}

	all, err := s.ListTestCases(ctx, p.ID)
	if err != nil {
		t.Fatalf("ListTestCases() error = %v", err)
	}
	if len(all) != 2 || all[0].ID != first.ID || all[1].ID != second.ID {
		t.Errorf("ListTestCases() = %+v, want both cases, oldest first", all)
	}

	dup := *first
	wantErr(t, s.CreateTestCase(ctx, &dup), domain.ErrConflict, "duplicate test case ID")
	wantErr(t, s.CreateTestCase(ctx, &domain.EvaluationTestCase{
		PromptID: "missing", Name: "x", ExpectedOutput: "x", MatchStrategy: domain.MatchExact,
	}), domain.ErrNotFound, "CreateTestCase(missing prompt)")
	wantErr(t, s.CreateTestCase(ctx, &domain.EvaluationTestCase{PromptID: p.ID}), domain.ErrInvalidInput, "invalid test case")
	_, err = s.GetTestCase(ctx, "missing")
	wantErr(t, err, domain.ErrNotFound, "GetTestCase(missing)")
	_, err = s.ListTestCases(ctx, "missing")
	wantErr(t, err, domain.ErrNotFound, "ListTestCases(missing prompt)")
	wantErr(t, s.DeleteTestCase(ctx, "missing"), domain.ErrNotFound, "DeleteTestCase(missing)")

	if err := s.DeleteTestCase(ctx, first.ID); err != nil {
		t.Fatalf("DeleteTestCase() error = %v", err)
	}
	all, err = s.ListTestCases(ctx, p.ID)
	if err != nil || len(all) != 1 || all[0].ID != second.ID {
		t.Errorf("ListTestCases() after delete = %+v, %v", all, err)
	}
}

func testRuns(t *testing.T, s Store) {
	p := mustCreatePrompt(t, s, "summariser")
	v := mustCreateVersion(t, s, p.ID)

	none, err := s.ListRuns(ctx, p.ID, domain.ListOptions{})
	if err != nil || none == nil || len(none) != 0 {
		t.Fatalf("ListRuns() with no runs = %v, %v; want an empty, non-nil slice", none, err)
	}

	run := mustCreateRun(t, s, v)
	if run.ID == "" || run.StartedAt.IsZero() {
		t.Fatalf("CreateRun() did not assign ID and start time: %+v", run)
	}
	got, err := s.GetRun(ctx, run.ID)
	if err != nil {
		t.Fatalf("GetRun() error = %v", err)
	}
	if got.PromptID != p.ID || got.PromptVersionID != v.ID || got.Version != 1 ||
		got.Status != domain.RunStatusRunning || !got.FinishedAt.IsZero() || !got.StartedAt.Equal(run.StartedAt) {
		t.Errorf("GetRun() = %+v, want %+v", got, run)
	}

	finished := *run
	finished.Status = domain.RunStatusCompleted
	finished.Summary = domain.RunSummary{
		Total: 3, Passed: 1, Failed: 1, Errored: 1,
		Usage:          domain.TokenUsage{InputTokens: 30, OutputTokens: 9},
		TotalLatencyMS: 1234,
	}
	finished.Error = "one case errored"
	finished.FinishedAt = run.StartedAt.Add(2 * time.Second)
	// Identity fields are not updatable.
	finished.PromptID, finished.PromptVersionID, finished.Version = "tampered", "tampered", 99
	if err := s.UpdateRun(ctx, &finished); err != nil {
		t.Fatalf("UpdateRun() error = %v", err)
	}
	got, err = s.GetRun(ctx, run.ID)
	if err != nil {
		t.Fatalf("GetRun() error = %v", err)
	}
	if got.Status != domain.RunStatusCompleted || got.Summary != finished.Summary ||
		got.Error != "one case errored" || !got.FinishedAt.Equal(finished.FinishedAt) {
		t.Errorf("UpdateRun() did not persist the outcome: %+v", got)
	}
	if got.PromptID != p.ID || got.PromptVersionID != v.ID || got.Version != 1 || !got.StartedAt.Equal(run.StartedAt) {
		t.Errorf("UpdateRun() changed identity fields: %+v", got)
	}

	second := mustCreateRun(t, s, v)
	third := mustCreateRun(t, s, v)
	runs, err := s.ListRuns(ctx, p.ID, domain.ListOptions{})
	if err != nil {
		t.Fatalf("ListRuns() error = %v", err)
	}
	if len(runs) != 3 || runs[0].ID != third.ID || runs[1].ID != second.ID || runs[2].ID != run.ID {
		t.Errorf("ListRuns() = %+v, want newest first", runs)
	}
	paged, err := s.ListRuns(ctx, p.ID, domain.ListOptions{Limit: 1, Offset: 1})
	if err != nil || len(paged) != 1 || paged[0].ID != second.ID {
		t.Errorf("ListRuns(limit 1, offset 1) = %+v, %v", paged, err)
	}

	dup := *run
	wantErr(t, s.CreateRun(ctx, &dup), domain.ErrConflict, "duplicate run ID")
	wantErr(t, s.CreateRun(ctx, &domain.EvaluationRun{PromptID: "missing", PromptVersionID: "v", Status: domain.RunStatusRunning}),
		domain.ErrNotFound, "CreateRun(missing prompt)")
	wantErr(t, s.CreateRun(ctx, &domain.EvaluationRun{PromptID: p.ID}), domain.ErrInvalidInput, "invalid run")
	missing := finished
	missing.ID, missing.PromptID, missing.PromptVersionID = "missing", p.ID, v.ID
	wantErr(t, s.UpdateRun(ctx, &missing), domain.ErrNotFound, "UpdateRun(missing)")
	invalid := *run
	invalid.Status = "paused"
	wantErr(t, s.UpdateRun(ctx, &invalid), domain.ErrInvalidInput, "UpdateRun(invalid status)")
	_, err = s.GetRun(ctx, "missing")
	wantErr(t, err, domain.ErrNotFound, "GetRun(missing)")
	_, err = s.ListRuns(ctx, "missing", domain.ListOptions{})
	wantErr(t, err, domain.ErrNotFound, "ListRuns(missing prompt)")
}

func testInterruptRuns(t *testing.T, s Store) {
	if n, err := s.InterruptRuns(ctx, "restart", time.Now()); err != nil || n != 0 {
		t.Fatalf("InterruptRuns() on an empty store = %d, %v; want 0", n, err)
	}

	p := mustCreatePrompt(t, s, "summariser")
	v := mustCreateVersion(t, s, p.ID)
	stuck := mustCreateRun(t, s, v)
	alsoStuck := mustCreateRun(t, s, v)
	done := mustCreateRun(t, s, v)
	finished := *done
	finished.Status = domain.RunStatusCompleted
	finished.Summary = domain.RunSummary{Total: 1, Passed: 1}
	finished.FinishedAt = done.StartedAt.Add(time.Second)
	if err := s.UpdateRun(ctx, &finished); err != nil {
		t.Fatalf("UpdateRun() error = %v", err)
	}

	at := time.Date(2026, 5, 6, 7, 8, 9, 0, time.UTC)
	n, err := s.InterruptRuns(ctx, "interrupted by restart", at)
	if err != nil || n != 2 {
		t.Fatalf("InterruptRuns() = %d, %v; want 2", n, err)
	}
	for _, id := range []string{stuck.ID, alsoStuck.ID} {
		got, err := s.GetRun(ctx, id)
		if err != nil {
			t.Fatalf("GetRun() error = %v", err)
		}
		if got.Status != domain.RunStatusFailed || got.Error != "interrupted by restart" || !got.FinishedAt.Equal(at) {
			t.Errorf("interrupted run = %+v", got)
		}
	}
	got, err := s.GetRun(ctx, done.ID)
	if err != nil {
		t.Fatalf("GetRun() error = %v", err)
	}
	if got.Status != domain.RunStatusCompleted || got.Error != "" || got.Summary.Passed != 1 || !got.FinishedAt.Equal(finished.FinishedAt) {
		t.Errorf("a completed run was touched: %+v", got)
	}
	if n, err := s.InterruptRuns(ctx, "again", at); err != nil || n != 0 {
		t.Errorf("second InterruptRuns() = %d, %v; want 0", n, err)
	}
}

func testResults(t *testing.T, s Store) {
	p := mustCreatePrompt(t, s, "summariser")
	v := mustCreateVersion(t, s, p.ID)
	run := mustCreateRun(t, s, v)
	otherRun := mustCreateRun(t, s, v)

	none, err := s.ListResultsByRun(ctx, run.ID)
	if err != nil || none == nil || len(none) != 0 {
		t.Fatalf("ListResultsByRun() with no results = %v, %v; want an empty, non-nil slice", none, err)
	}
	if err := s.SaveResults(ctx, nil); err != nil {
		t.Errorf("SaveResults(nil) error = %v", err)
	}

	batch := []domain.EvaluationResult{
		{
			RunID: run.ID, TestCaseID: "tc1", PromptVersionID: v.ID,
			Provider: domain.ProviderAnthropic, Model: "claude-haiku-4-5",
			ActualOutput: "Go is great", Passed: true, Score: 1,
			Usage:   domain.TokenUsage{InputTokens: 12, OutputTokens: 4},
			Latency: 345 * time.Millisecond,
		},
		{RunID: run.ID, TestCaseID: "tc2", PromptVersionID: v.ID, Score: 0.25, Error: "provider down"},
	}
	if err := s.SaveResults(ctx, batch); err != nil {
		t.Fatalf("SaveResults() error = %v", err)
	}
	for i, r := range batch {
		if r.ID == "" || r.CreatedAt.IsZero() {
			t.Errorf("SaveResults() did not report the ID and timestamp of results[%d]: %+v", i, r)
		}
	}
	if err := s.SaveResults(ctx, []domain.EvaluationResult{{RunID: otherRun.ID, TestCaseID: "tc1", PromptVersionID: v.ID}}); err != nil {
		t.Fatalf("SaveResults() error = %v", err)
	}

	got, err := s.ListResultsByRun(ctx, run.ID)
	if err != nil {
		t.Fatalf("ListResultsByRun() error = %v", err)
	}
	if len(got) != 2 || got[0].TestCaseID != "tc1" || got[1].TestCaseID != "tc2" {
		t.Fatalf("ListResultsByRun() = %+v, want the run's two results in saved order", got)
	}
	first := got[0]
	if first.ID != batch[0].ID || first.RunID != run.ID || first.PromptVersionID != v.ID ||
		first.Provider != domain.ProviderAnthropic || first.Model != "claude-haiku-4-5" ||
		first.ActualOutput != "Go is great" || !first.Passed || first.Score != 1 ||
		first.Usage != (domain.TokenUsage{InputTokens: 12, OutputTokens: 4}) ||
		first.Latency != 345*time.Millisecond || first.Error != "" || !first.CreatedAt.Equal(batch[0].CreatedAt) {
		t.Errorf("result did not round-trip: %+v, want %+v", first, batch[0])
	}
	if got[1].Passed || got[1].Score != 0.25 || got[1].Error != "provider down" {
		t.Errorf("result did not round-trip: %+v", got[1])
	}

	byVersion, err := s.ListResultsByVersion(ctx, v.ID, domain.ListOptions{})
	if err != nil {
		t.Fatalf("ListResultsByVersion() error = %v", err)
	}
	if len(byVersion) != 3 || byVersion[0].RunID != otherRun.ID || byVersion[1].TestCaseID != "tc2" || byVersion[2].TestCaseID != "tc1" {
		t.Errorf("ListResultsByVersion() = %+v, want all three results, newest first", byVersion)
	}
	paged, err := s.ListResultsByVersion(ctx, v.ID, domain.ListOptions{Limit: 1, Offset: 2})
	if err != nil || len(paged) != 1 || paged[0].TestCaseID != "tc1" {
		t.Errorf("ListResultsByVersion(limit 1, offset 2) = %+v, %v", paged, err)
	}
	empty, err := s.ListResultsByVersion(ctx, "missing", domain.ListOptions{})
	if err != nil || empty == nil || len(empty) != 0 {
		t.Errorf("ListResultsByVersion(missing) = %v, %v; want an empty, non-nil slice", empty, err)
	}
}

func testResultsAreAtomic(t *testing.T, s Store) {
	p := mustCreatePrompt(t, s, "summariser")
	v := mustCreateVersion(t, s, p.ID)
	run := mustCreateRun(t, s, v)

	if err := s.SaveResults(ctx, []domain.EvaluationResult{{ID: "taken", RunID: run.ID, PromptVersionID: v.ID}}); err != nil {
		t.Fatalf("SaveResults() error = %v", err)
	}

	batches := map[string]struct {
		results []domain.EvaluationResult
		want    error
	}{
		"unknown run": {
			results: []domain.EvaluationResult{{RunID: run.ID, PromptVersionID: v.ID}, {RunID: "missing", PromptVersionID: v.ID}},
			want:    domain.ErrNotFound,
		},
		"ID already stored": {
			results: []domain.EvaluationResult{{RunID: run.ID, PromptVersionID: v.ID}, {ID: "taken", RunID: run.ID, PromptVersionID: v.ID}},
			want:    domain.ErrConflict,
		},
		"ID repeated within the batch": {
			results: []domain.EvaluationResult{{ID: "twice", RunID: run.ID, PromptVersionID: v.ID}, {ID: "twice", RunID: run.ID, PromptVersionID: v.ID}},
			want:    domain.ErrConflict,
		},
	}
	for name, batch := range batches {
		wantErr(t, s.SaveResults(ctx, batch.results), batch.want, name)
		stored, err := s.ListResultsByRun(ctx, run.ID)
		if err != nil {
			t.Fatalf("ListResultsByRun() error = %v", err)
		}
		if len(stored) != 1 {
			t.Fatalf("%s: %d results stored, want 1: a failed batch must write nothing", name, len(stored))
		}
	}
}

func testDeletePromptCascade(t *testing.T, s Store) {
	doomed := mustCreatePrompt(t, s, "doomed")
	kept := mustCreatePrompt(t, s, "kept")

	seed := func(p *domain.Prompt) (*domain.PromptVersion, *domain.EvaluationTestCase, *domain.EvaluationRun) {
		v := mustCreateVersion(t, s, p.ID)
		tc := &domain.EvaluationTestCase{PromptID: p.ID, Name: "case", ExpectedOutput: "x", MatchStrategy: domain.MatchExact}
		if err := s.CreateTestCase(ctx, tc); err != nil {
			t.Fatalf("CreateTestCase() error = %v", err)
		}
		run := mustCreateRun(t, s, v)
		if err := s.SaveResults(ctx, []domain.EvaluationResult{{RunID: run.ID, TestCaseID: tc.ID, PromptVersionID: v.ID}}); err != nil {
			t.Fatalf("SaveResults() error = %v", err)
		}
		return v, tc, run
	}
	dv, dtc, drun := seed(doomed)
	kv, ktc, krun := seed(kept)

	if err := s.DeletePrompt(ctx, doomed.ID); err != nil {
		t.Fatalf("DeletePrompt() error = %v", err)
	}

	_, err := s.GetVersion(ctx, doomed.ID, 1)
	wantErr(t, err, domain.ErrNotFound, "version of deleted prompt")
	_, err = s.GetTestCase(ctx, dtc.ID)
	wantErr(t, err, domain.ErrNotFound, "test case of deleted prompt")
	_, err = s.GetRun(ctx, drun.ID)
	wantErr(t, err, domain.ErrNotFound, "run of deleted prompt")
	if results, err := s.ListResultsByRun(ctx, drun.ID); err != nil || len(results) != 0 {
		t.Errorf("results of deleted prompt by run = %+v, %v; want none", results, err)
	}
	if results, err := s.ListResultsByVersion(ctx, dv.ID, domain.ListOptions{}); err != nil || len(results) != 0 {
		t.Errorf("results of deleted prompt by version = %+v, %v; want none", results, err)
	}

	if _, err := s.GetVersion(ctx, kept.ID, 1); err != nil {
		t.Errorf("version of the other prompt: %v", err)
	}
	if _, err := s.GetTestCase(ctx, ktc.ID); err != nil {
		t.Errorf("test case of the other prompt: %v", err)
	}
	if _, err := s.GetRun(ctx, krun.ID); err != nil {
		t.Errorf("run of the other prompt: %v", err)
	}
	if results, err := s.ListResultsByVersion(ctx, kv.ID, domain.ListOptions{}); err != nil || len(results) != 1 {
		t.Errorf("results of the other prompt = %+v, %v; want 1", results, err)
	}
}

// testIsolation checks that mutating an entity after storing it, or after
// reading it, never changes what the store holds.
func testIsolation(t *testing.T, s Store) {
	p := mustCreatePrompt(t, s, "summariser")
	p.Tags[0] = "mutated"
	got, err := s.GetPrompt(ctx, p.ID)
	if err != nil {
		t.Fatalf("GetPrompt() error = %v", err)
	}
	if got.Tags[0] != "a" {
		t.Error("mutating a prompt after CreatePrompt changed the stored tags")
	}
	got.Tags[1] = "mutated"
	if again, _ := s.GetPrompt(ctx, p.ID); again.Tags[1] != "b" {
		t.Error("mutating a prompt returned by GetPrompt changed the stored tags")
	}

	v := mustCreateVersion(t, s, p.ID)
	*v.Parameters.Temperature = 2
	v.Parameters.Stop[0] = "mutated"
	stored, err := s.GetVersion(ctx, p.ID, 1)
	if err != nil {
		t.Fatalf("GetVersion() error = %v", err)
	}
	if *stored.Parameters.Temperature != 0 || stored.Parameters.Stop[0] != "END" {
		t.Error("mutating a version after CreateVersion changed the stored parameters")
	}
	*stored.Parameters.TopP = 0.1
	if again, _ := s.GetVersion(ctx, p.ID, 1); *again.Parameters.TopP != 0.9 {
		t.Error("mutating a version returned by GetVersion changed the stored parameters")
	}

	tc := &domain.EvaluationTestCase{
		PromptID: p.ID, Name: "case", Variables: map[string]string{"text": "original"},
		ExpectedOutput: "x", MatchStrategy: domain.MatchExact,
	}
	if err := s.CreateTestCase(ctx, tc); err != nil {
		t.Fatalf("CreateTestCase() error = %v", err)
	}
	tc.Variables["text"] = "mutated"
	storedCase, err := s.GetTestCase(ctx, tc.ID)
	if err != nil {
		t.Fatalf("GetTestCase() error = %v", err)
	}
	if storedCase.Variables["text"] != "original" {
		t.Error("mutating a test case after CreateTestCase changed the stored variables")
	}
	storedCase.Variables["text"] = "mutated"
	if again, _ := s.GetTestCase(ctx, tc.ID); again.Variables["text"] != "original" {
		t.Error("mutating a test case returned by GetTestCase changed the stored variables")
	}
}
