// Package memory is an in-memory implementation of the domain repositories.
// Data lives only as long as the process; it is meant for tests, local
// experiments and as the reference behaviour for other storage backends.
package memory

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

// Store implements domain.PromptRepository and domain.EvaluationRepository.
// It is safe for concurrent use. Entities are copied on the way in and out, so
// callers can never alias the stored state.
type Store struct {
	mu  sync.RWMutex
	now func() time.Time

	// Slices keep insertion order, which doubles as the tie-breaker for
	// "newest first" listings of entities created at the same instant.
	prompts   []domain.Prompt
	versions  []domain.PromptVersion
	testCases []domain.EvaluationTestCase
	runs      []domain.EvaluationRun
	results   []domain.EvaluationResult
}

var (
	_ domain.PromptRepository     = (*Store)(nil)
	_ domain.EvaluationRepository = (*Store)(nil)
)

// New returns an empty Store.
func New() *Store {
	return &Store{now: time.Now}
}

func notFound(kind, id string) error {
	return fmt.Errorf("%s %q: %w", kind, id, domain.ErrNotFound)
}

func (s *Store) promptIndex(id string) int {
	return slices.IndexFunc(s.prompts, func(p domain.Prompt) bool { return p.ID == id })
}

func (s *Store) runIndex(id string) int {
	return slices.IndexFunc(s.runs, func(r domain.EvaluationRun) bool { return r.ID == id })
}

// page returns the opts window of items after reversing them, i.e. newest
// first for a slice kept in insertion order.
func page[T any](items []T, opts domain.ListOptions, clone func(T) T) []T {
	opts = opts.Normalize()
	out := make([]T, 0, min(len(items), opts.Limit))
	for i := len(items) - 1 - opts.Offset; i >= 0 && len(out) < opts.Limit; i-- {
		out = append(out, clone(items[i]))
	}
	return out
}

func clonePrompt(p domain.Prompt) domain.Prompt {
	p.Tags = slices.Clone(p.Tags)
	return p
}

func cloneParameters(p domain.ModelParameters) domain.ModelParameters {
	if p.Temperature != nil {
		v := *p.Temperature
		p.Temperature = &v
	}
	if p.TopP != nil {
		v := *p.TopP
		p.TopP = &v
	}
	p.Stop = slices.Clone(p.Stop)
	return p
}

func cloneVersion(v domain.PromptVersion) domain.PromptVersion {
	v.Parameters = cloneParameters(v.Parameters)
	return v
}

func cloneTestCase(tc domain.EvaluationTestCase) domain.EvaluationTestCase {
	if tc.Variables != nil {
		vars := make(map[string]string, len(tc.Variables))
		for k, v := range tc.Variables {
			vars[k] = v
		}
		tc.Variables = vars
	}
	return tc
}

func cloneResult(r domain.EvaluationResult) domain.EvaluationResult {
	if r.CostUSD != nil {
		v := *r.CostUSD
		r.CostUSD = &v
	}
	return r
}

func identity[T any](v T) T { return v }

// CreatePrompt stores a new prompt.
func (s *Store) CreatePrompt(_ context.Context, p *domain.Prompt) error {
	if err := p.Validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, existing := range s.prompts {
		if existing.Name == p.Name {
			return fmt.Errorf("prompt name %q: %w", p.Name, domain.ErrConflict)
		}
		if p.ID != "" && existing.ID == p.ID {
			return fmt.Errorf("prompt %q: %w", p.ID, domain.ErrConflict)
		}
	}
	if p.ID == "" {
		p.ID = domain.NewID()
	}
	now := s.now().UTC()
	if p.CreatedAt.IsZero() {
		p.CreatedAt = now
	}
	if p.UpdatedAt.IsZero() {
		p.UpdatedAt = p.CreatedAt
	}
	s.prompts = append(s.prompts, clonePrompt(*p))
	return nil
}

// GetPrompt returns the prompt with the given ID.
func (s *Store) GetPrompt(_ context.Context, id string) (*domain.Prompt, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	i := s.promptIndex(id)
	if i < 0 {
		return nil, notFound("prompt", id)
	}
	p := clonePrompt(s.prompts[i])
	return &p, nil
}

// ListPrompts returns prompts, newest first.
func (s *Store) ListPrompts(_ context.Context, opts domain.ListOptions) ([]domain.Prompt, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return page(s.prompts, opts, clonePrompt), nil
}

// DeletePrompt removes a prompt with its versions, test cases, runs and
// results.
func (s *Store) DeletePrompt(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	i := s.promptIndex(id)
	if i < 0 {
		return notFound("prompt", id)
	}
	s.prompts = slices.Delete(s.prompts, i, i+1)
	s.versions = slices.DeleteFunc(s.versions, func(v domain.PromptVersion) bool { return v.PromptID == id })
	s.testCases = slices.DeleteFunc(s.testCases, func(tc domain.EvaluationTestCase) bool { return tc.PromptID == id })

	removedRuns := make(map[string]bool)
	s.runs = slices.DeleteFunc(s.runs, func(r domain.EvaluationRun) bool {
		if r.PromptID == id {
			removedRuns[r.ID] = true
			return true
		}
		return false
	})
	s.results = slices.DeleteFunc(s.results, func(r domain.EvaluationResult) bool { return removedRuns[r.RunID] })
	return nil
}

// CreateVersion stores a new version, numbered one above the prompt's latest.
func (s *Store) CreateVersion(_ context.Context, v *domain.PromptVersion) error {
	if err := v.Validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	i := s.promptIndex(v.PromptID)
	if i < 0 {
		return notFound("prompt", v.PromptID)
	}
	latest := 0
	for _, existing := range s.versions {
		if v.ID != "" && existing.ID == v.ID {
			return fmt.Errorf("prompt version %q: %w", v.ID, domain.ErrConflict)
		}
		if existing.PromptID == v.PromptID {
			latest = max(latest, existing.Version)
		}
	}
	v.Version = latest + 1
	if v.ID == "" {
		v.ID = domain.NewID()
	}
	now := s.now().UTC()
	if v.CreatedAt.IsZero() {
		v.CreatedAt = now
	}
	s.versions = append(s.versions, cloneVersion(*v))
	s.prompts[i].UpdatedAt = now
	return nil
}

// GetVersion returns a specific version of a prompt.
func (s *Store) GetVersion(_ context.Context, promptID string, version int) (*domain.PromptVersion, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, v := range s.versions {
		if v.PromptID == promptID && v.Version == version {
			v = cloneVersion(v)
			return &v, nil
		}
	}
	return nil, notFound("prompt version", fmt.Sprintf("%s/%d", promptID, version))
}

// GetLatestVersion returns the highest-numbered version of a prompt.
func (s *Store) GetLatestVersion(_ context.Context, promptID string) (*domain.PromptVersion, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var latest *domain.PromptVersion
	for i := range s.versions {
		if v := &s.versions[i]; v.PromptID == promptID && (latest == nil || v.Version > latest.Version) {
			latest = v
		}
	}
	if latest == nil {
		return nil, notFound("prompt version", promptID+"/latest")
	}
	v := cloneVersion(*latest)
	return &v, nil
}

// ListVersions returns all versions of a prompt in ascending order.
func (s *Store) ListVersions(_ context.Context, promptID string) ([]domain.PromptVersion, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.promptIndex(promptID) < 0 {
		return nil, notFound("prompt", promptID)
	}
	out := []domain.PromptVersion{}
	for _, v := range s.versions {
		if v.PromptID == promptID {
			out = append(out, cloneVersion(v))
		}
	}
	return out, nil
}

// CreateTestCase stores a new test case.
func (s *Store) CreateTestCase(_ context.Context, tc *domain.EvaluationTestCase) error {
	if err := tc.Validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.promptIndex(tc.PromptID) < 0 {
		return notFound("prompt", tc.PromptID)
	}
	if tc.ID == "" {
		tc.ID = domain.NewID()
	} else if slices.ContainsFunc(s.testCases, func(e domain.EvaluationTestCase) bool { return e.ID == tc.ID }) {
		return fmt.Errorf("test case %q: %w", tc.ID, domain.ErrConflict)
	}
	if tc.CreatedAt.IsZero() {
		tc.CreatedAt = s.now().UTC()
	}
	s.testCases = append(s.testCases, cloneTestCase(*tc))
	return nil
}

// GetTestCase returns the test case with the given ID.
func (s *Store) GetTestCase(_ context.Context, id string) (*domain.EvaluationTestCase, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, tc := range s.testCases {
		if tc.ID == id {
			tc = cloneTestCase(tc)
			return &tc, nil
		}
	}
	return nil, notFound("test case", id)
}

// ListTestCases returns every test case of a prompt, oldest first.
func (s *Store) ListTestCases(_ context.Context, promptID string) ([]domain.EvaluationTestCase, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.promptIndex(promptID) < 0 {
		return nil, notFound("prompt", promptID)
	}
	out := []domain.EvaluationTestCase{}
	for _, tc := range s.testCases {
		if tc.PromptID == promptID {
			out = append(out, cloneTestCase(tc))
		}
	}
	return out, nil
}

// DeleteTestCase removes a test case. Results already recorded for it are
// kept.
func (s *Store) DeleteTestCase(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	i := slices.IndexFunc(s.testCases, func(tc domain.EvaluationTestCase) bool { return tc.ID == id })
	if i < 0 {
		return notFound("test case", id)
	}
	s.testCases = slices.Delete(s.testCases, i, i+1)
	return nil
}

// CreateRun stores a new run.
func (s *Store) CreateRun(_ context.Context, run *domain.EvaluationRun) error {
	if err := run.Validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.promptIndex(run.PromptID) < 0 {
		return notFound("prompt", run.PromptID)
	}
	if run.ID == "" {
		run.ID = domain.NewID()
	} else if s.runIndex(run.ID) >= 0 {
		return fmt.Errorf("run %q: %w", run.ID, domain.ErrConflict)
	}
	if run.StartedAt.IsZero() {
		run.StartedAt = s.now().UTC()
	}
	s.runs = append(s.runs, *run)
	return nil
}

// UpdateRun overwrites the status, summary, error and finish time of a run.
func (s *Store) UpdateRun(_ context.Context, run *domain.EvaluationRun) error {
	if err := run.Validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	i := s.runIndex(run.ID)
	if i < 0 {
		return notFound("run", run.ID)
	}
	stored := &s.runs[i]
	stored.Status = run.Status
	stored.Summary = run.Summary
	stored.Error = run.Error
	stored.FinishedAt = run.FinishedAt
	return nil
}

// InterruptRuns marks every running run as failed.
func (s *Store) InterruptRuns(_ context.Context, reason string, at time.Time) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	n := 0
	for i := range s.runs {
		if run := &s.runs[i]; run.Status == domain.RunStatusRunning {
			run.Status, run.Error, run.FinishedAt = domain.RunStatusFailed, reason, at
			n++
		}
	}
	return n, nil
}

// GetRun returns the run with the given ID.
func (s *Store) GetRun(_ context.Context, id string) (*domain.EvaluationRun, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	i := s.runIndex(id)
	if i < 0 {
		return nil, notFound("run", id)
	}
	run := s.runs[i]
	return &run, nil
}

// ListRuns returns the runs of a prompt, newest first.
func (s *Store) ListRuns(_ context.Context, promptID string, opts domain.ListOptions) ([]domain.EvaluationRun, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.promptIndex(promptID) < 0 {
		return nil, notFound("prompt", promptID)
	}
	var runs []domain.EvaluationRun
	for _, run := range s.runs {
		if run.PromptID == promptID {
			runs = append(runs, run)
		}
	}
	return page(runs, opts, identity), nil
}

// SaveResults stores a batch of results; either all are written or none.
func (s *Store) SaveResults(_ context.Context, results []domain.EvaluationResult) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := s.now().UTC()
	batch := make([]domain.EvaluationResult, 0, len(results))
	seen := make(map[string]bool, len(results))
	for i, r := range results {
		if s.runIndex(r.RunID) < 0 {
			return fmt.Errorf("results[%d]: %w", i, notFound("run", r.RunID))
		}
		if r.ID == "" {
			r.ID = domain.NewID()
		}
		if seen[r.ID] || slices.ContainsFunc(s.results, func(e domain.EvaluationResult) bool { return e.ID == r.ID }) {
			return fmt.Errorf("results[%d]: result %q: %w", i, r.ID, domain.ErrConflict)
		}
		seen[r.ID] = true
		if r.CreatedAt.IsZero() {
			r.CreatedAt = now
		}
		batch = append(batch, r)
	}
	// Report the assigned IDs and timestamps back only once the whole batch
	// is known to be valid.
	copy(results, batch)
	for _, r := range batch {
		s.results = append(s.results, cloneResult(r))
	}
	return nil
}

// ListResultsByRun returns the results of a run in the order they were saved.
func (s *Store) ListResultsByRun(_ context.Context, runID string) ([]domain.EvaluationResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := []domain.EvaluationResult{}
	for _, r := range s.results {
		if r.RunID == runID {
			out = append(out, cloneResult(r))
		}
	}
	return out, nil
}

// ListResultsByVersion returns results for a prompt version, newest first.
func (s *Store) ListResultsByVersion(_ context.Context, promptVersionID string, opts domain.ListOptions) ([]domain.EvaluationResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var matching []domain.EvaluationResult
	for _, r := range s.results {
		if r.PromptVersionID == promptVersionID {
			matching = append(matching, r)
		}
	}
	return page(matching, opts, cloneResult), nil
}
