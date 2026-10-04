// Package sqlite is the durable implementation of the domain repositories,
// backed by a single SQLite database file through the pure-Go
// modernc.org/sqlite driver (no cgo).
package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"modernc.org/sqlite"
	sqlite3 "modernc.org/sqlite/lib"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

// InMemory is the path that opens a private, non-persistent database.
const InMemory = ":memory:"

// Store implements domain.PromptRepository and domain.EvaluationRepository.
// It is safe for concurrent use.
type Store struct {
	db  *sql.DB
	now func() time.Time
}

var (
	_ domain.PromptRepository     = (*Store)(nil)
	_ domain.EvaluationRepository = (*Store)(nil)
)

// Open opens the database at path, creating the file and its parent
// directory when missing, and brings the schema up to date.
func Open(ctx context.Context, path string) (*Store, error) {
	if path == "" {
		return nil, fmt.Errorf("%w: database path is required", domain.ErrInvalidInput)
	}
	dsn := path
	if path != InMemory {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, fmt.Errorf("create database directory: %w", err)
		}
		dsn = "file:" + path
	}
	// Foreign keys are off by default in SQLite and are what implements the
	// cascading deletes.
	dsn += "?_pragma=foreign_keys(1)&_pragma=busy_timeout(5000)"
	if path != InMemory {
		dsn += "&_pragma=journal_mode(WAL)"
	}

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}
	// SQLite allows one writer at a time, and every connection to an
	// in-memory database would see its own empty database. A single
	// connection sidesteps both and makes each transaction serialisable.
	db.SetMaxOpenConns(1)

	s := &Store{db: db, now: time.Now}
	if err := s.migrate(ctx); err != nil {
		db.Close()
		return nil, err
	}
	return s, nil
}

// Close releases the database.
func (s *Store) Close() error { return s.db.Close() }

func (s *Store) migrate(ctx context.Context) error {
	var version int
	if err := s.db.QueryRowContext(ctx, "PRAGMA user_version").Scan(&version); err != nil {
		return fmt.Errorf("read schema version: %w", err)
	}
	if version > len(migrations) {
		return fmt.Errorf("database schema version %d is newer than this build supports (%d)", version, len(migrations))
	}
	for i := version; i < len(migrations); i++ {
		err := s.tx(ctx, func(tx *sql.Tx) error {
			if _, err := tx.ExecContext(ctx, migrations[i]); err != nil {
				return err
			}
			// PRAGMA does not accept bound parameters.
			_, err := tx.ExecContext(ctx, fmt.Sprintf("PRAGMA user_version = %d", i+1))
			return err
		})
		if err != nil {
			return fmt.Errorf("apply migration %d: %w", i+1, err)
		}
	}
	return nil
}

// tx runs fn in a transaction, committing when it returns nil.
func (s *Store) tx(ctx context.Context, fn func(tx *sql.Tx) error) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	if err := fn(tx); err != nil {
		_ = tx.Rollback()
		return err
	}
	return tx.Commit()
}

// querier is satisfied by both *sql.DB and *sql.Tx.
type querier interface {
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

func notFound(kind, id string) error {
	return fmt.Errorf("%s %q: %w", kind, id, domain.ErrNotFound)
}

// isUnique reports whether err is a UNIQUE or PRIMARY KEY violation.
func isUnique(err error) bool {
	var sqliteErr *sqlite.Error
	if !errors.As(err, &sqliteErr) {
		return false
	}
	code := sqliteErr.Code()
	return code == sqlite3.SQLITE_CONSTRAINT_UNIQUE || code == sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY
}

func toUnix(t time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	return t.UnixNano()
}

func fromUnix(n int64) time.Time {
	if n == 0 {
		return time.Time{}
	}
	return time.Unix(0, n).UTC()
}

func toJSON(v any) string {
	// Only plain maps, slices and structs of basic types are passed in, which
	// cannot fail to encode.
	data, _ := json.Marshal(v)
	return string(data)
}

func exists(ctx context.Context, q querier, table, id string) (bool, error) {
	var one int
	err := q.QueryRowContext(ctx, "SELECT 1 FROM "+table+" WHERE id = ?", id).Scan(&one)
	if errors.Is(err, sql.ErrNoRows) {
		return false, nil
	}
	return err == nil, err
}

func (s *Store) requirePrompt(ctx context.Context, q querier, id string) error {
	ok, err := exists(ctx, q, "prompts", id)
	if err != nil {
		return err
	}
	if !ok {
		return notFound("prompt", id)
	}
	return nil
}

// ---- prompts ----

// CreatePrompt stores a new prompt.
func (s *Store) CreatePrompt(ctx context.Context, p *domain.Prompt) error {
	if err := p.Validate(); err != nil {
		return err
	}
	created := *p
	if created.ID == "" {
		created.ID = domain.NewID()
	}
	if created.CreatedAt.IsZero() {
		created.CreatedAt = s.now().UTC()
	}
	if created.UpdatedAt.IsZero() {
		created.UpdatedAt = created.CreatedAt
	}
	tags := created.Tags
	if tags == nil {
		tags = []string{}
	}
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO prompts (id, name, description, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)`,
		created.ID, created.Name, created.Description, toJSON(tags), toUnix(created.CreatedAt), toUnix(created.UpdatedAt))
	if isUnique(err) {
		return fmt.Errorf("prompt %q (id %q): %w", p.Name, created.ID, domain.ErrConflict)
	}
	if err != nil {
		return fmt.Errorf("insert prompt: %w", err)
	}
	*p = created
	return nil
}

const promptColumns = `id, name, description, tags, created_at, updated_at`

func scanPrompt(row interface{ Scan(...any) error }) (domain.Prompt, error) {
	var (
		p                domain.Prompt
		tags             string
		created, updated int64
	)
	if err := row.Scan(&p.ID, &p.Name, &p.Description, &tags, &created, &updated); err != nil {
		return p, err
	}
	if err := json.Unmarshal([]byte(tags), &p.Tags); err != nil {
		return p, fmt.Errorf("decode tags of prompt %q: %w", p.ID, err)
	}
	if len(p.Tags) == 0 {
		p.Tags = nil
	}
	p.CreatedAt, p.UpdatedAt = fromUnix(created), fromUnix(updated)
	return p, nil
}

// GetPrompt returns the prompt with the given ID.
func (s *Store) GetPrompt(ctx context.Context, id string) (*domain.Prompt, error) {
	p, err := scanPrompt(s.db.QueryRowContext(ctx, `SELECT `+promptColumns+` FROM prompts WHERE id = ?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, notFound("prompt", id)
	}
	if err != nil {
		return nil, fmt.Errorf("get prompt: %w", err)
	}
	return &p, nil
}

// ListPrompts returns prompts, newest first.
func (s *Store) ListPrompts(ctx context.Context, opts domain.ListOptions) ([]domain.Prompt, error) {
	opts = opts.Normalize()
	return queryAll(ctx, s.db, scanPrompt,
		`SELECT `+promptColumns+` FROM prompts ORDER BY seq DESC LIMIT ? OFFSET ?`, opts.Limit, opts.Offset)
}

// DeletePrompt removes a prompt; foreign keys cascade to its versions, test
// cases, runs and results.
func (s *Store) DeletePrompt(ctx context.Context, id string) error {
	return s.deleteByID(ctx, "prompts", "prompt", id)
}

func (s *Store) deleteByID(ctx context.Context, table, kind, id string) error {
	res, err := s.db.ExecContext(ctx, "DELETE FROM "+table+" WHERE id = ?", id)
	if err != nil {
		return fmt.Errorf("delete %s: %w", kind, err)
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return notFound(kind, id)
	}
	return nil
}

// queryAll runs a query and scans every row. The result is never nil.
func queryAll[T any](ctx context.Context, db *sql.DB, scan func(interface{ Scan(...any) error }) (T, error), query string, args ...any) ([]T, error) {
	rows, err := db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	out := []T{}
	for rows.Next() {
		item, err := scan(rows)
		if err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}
		out = append(out, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	return out, nil
}

// ---- prompt versions ----

// CreateVersion stores a new version, numbered one above the prompt's latest.
// The read of the latest number and the insert share a transaction, so
// concurrent writers cannot be handed the same number.
func (s *Store) CreateVersion(ctx context.Context, v *domain.PromptVersion) error {
	if err := v.Validate(); err != nil {
		return err
	}
	created := *v
	if created.ID == "" {
		created.ID = domain.NewID()
	}
	now := s.now().UTC()
	if created.CreatedAt.IsZero() {
		created.CreatedAt = now
	}

	err := s.tx(ctx, func(tx *sql.Tx) error {
		if err := s.requirePrompt(ctx, tx, v.PromptID); err != nil {
			return err
		}
		err := tx.QueryRowContext(ctx,
			`SELECT COALESCE(MAX(version), 0) + 1 FROM prompt_versions WHERE prompt_id = ?`, v.PromptID).Scan(&created.Version)
		if err != nil {
			return fmt.Errorf("next version number: %w", err)
		}
		_, err = tx.ExecContext(ctx,
			`INSERT INTO prompt_versions
				(id, prompt_id, version, template, system_prompt, provider, model, parameters, change_log, created_at)
			 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			created.ID, created.PromptID, created.Version, created.Template, created.SystemPrompt,
			string(created.Provider), created.Model, toJSON(created.Parameters), created.ChangeLog, toUnix(created.CreatedAt))
		if isUnique(err) {
			return fmt.Errorf("prompt version %q: %w", created.ID, domain.ErrConflict)
		}
		if err != nil {
			return fmt.Errorf("insert prompt version: %w", err)
		}
		_, err = tx.ExecContext(ctx, `UPDATE prompts SET updated_at = ? WHERE id = ?`, toUnix(now), v.PromptID)
		return err
	})
	if err != nil {
		return err
	}
	*v = created
	return nil
}

const versionColumns = `id, prompt_id, version, template, system_prompt, provider, model, parameters, change_log, created_at`

func scanVersion(row interface{ Scan(...any) error }) (domain.PromptVersion, error) {
	var (
		v          domain.PromptVersion
		provider   string
		parameters string
		created    int64
	)
	err := row.Scan(&v.ID, &v.PromptID, &v.Version, &v.Template, &v.SystemPrompt, &provider, &v.Model,
		&parameters, &v.ChangeLog, &created)
	if err != nil {
		return v, err
	}
	if err := json.Unmarshal([]byte(parameters), &v.Parameters); err != nil {
		return v, fmt.Errorf("decode parameters of prompt version %q: %w", v.ID, err)
	}
	v.Provider = domain.Provider(provider)
	v.CreatedAt = fromUnix(created)
	return v, nil
}

func (s *Store) getVersion(ctx context.Context, label, where string, args ...any) (*domain.PromptVersion, error) {
	v, err := scanVersion(s.db.QueryRowContext(ctx, `SELECT `+versionColumns+` FROM prompt_versions `+where, args...))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, notFound("prompt version", label)
	}
	if err != nil {
		return nil, fmt.Errorf("get prompt version: %w", err)
	}
	return &v, nil
}

// GetVersion returns a specific version of a prompt.
func (s *Store) GetVersion(ctx context.Context, promptID string, version int) (*domain.PromptVersion, error) {
	return s.getVersion(ctx, fmt.Sprintf("%s/%d", promptID, version),
		`WHERE prompt_id = ? AND version = ?`, promptID, version)
}

// GetLatestVersion returns the highest-numbered version of a prompt.
func (s *Store) GetLatestVersion(ctx context.Context, promptID string) (*domain.PromptVersion, error) {
	return s.getVersion(ctx, promptID+"/latest",
		`WHERE prompt_id = ? ORDER BY version DESC LIMIT 1`, promptID)
}

// ListVersions returns all versions of a prompt in ascending order.
func (s *Store) ListVersions(ctx context.Context, promptID string) ([]domain.PromptVersion, error) {
	if err := s.requirePrompt(ctx, s.db, promptID); err != nil {
		return nil, err
	}
	return queryAll(ctx, s.db, scanVersion,
		`SELECT `+versionColumns+` FROM prompt_versions WHERE prompt_id = ? ORDER BY version`, promptID)
}

// ---- test cases ----

// CreateTestCase stores a new test case.
func (s *Store) CreateTestCase(ctx context.Context, tc *domain.EvaluationTestCase) error {
	if err := tc.Validate(); err != nil {
		return err
	}
	created := *tc
	if created.ID == "" {
		created.ID = domain.NewID()
	}
	if created.CreatedAt.IsZero() {
		created.CreatedAt = s.now().UTC()
	}
	variables := created.Variables
	if variables == nil {
		variables = map[string]string{}
	}
	err := s.tx(ctx, func(tx *sql.Tx) error {
		if err := s.requirePrompt(ctx, tx, tc.PromptID); err != nil {
			return err
		}
		_, err := tx.ExecContext(ctx,
			`INSERT INTO test_cases (id, prompt_id, name, variables, expected_output, match_strategy, created_at)
			 VALUES (?, ?, ?, ?, ?, ?, ?)`,
			created.ID, created.PromptID, created.Name, toJSON(variables), created.ExpectedOutput,
			string(created.MatchStrategy), toUnix(created.CreatedAt))
		if isUnique(err) {
			return fmt.Errorf("test case %q: %w", created.ID, domain.ErrConflict)
		}
		if err != nil {
			return fmt.Errorf("insert test case: %w", err)
		}
		return nil
	})
	if err != nil {
		return err
	}
	*tc = created
	return nil
}

const testCaseColumns = `id, prompt_id, name, variables, expected_output, match_strategy, created_at`

func scanTestCase(row interface{ Scan(...any) error }) (domain.EvaluationTestCase, error) {
	var (
		tc        domain.EvaluationTestCase
		variables string
		strategy  string
		created   int64
	)
	if err := row.Scan(&tc.ID, &tc.PromptID, &tc.Name, &variables, &tc.ExpectedOutput, &strategy, &created); err != nil {
		return tc, err
	}
	if err := json.Unmarshal([]byte(variables), &tc.Variables); err != nil {
		return tc, fmt.Errorf("decode variables of test case %q: %w", tc.ID, err)
	}
	if len(tc.Variables) == 0 {
		tc.Variables = nil
	}
	tc.MatchStrategy = domain.MatchStrategy(strategy)
	tc.CreatedAt = fromUnix(created)
	return tc, nil
}

// GetTestCase returns the test case with the given ID.
func (s *Store) GetTestCase(ctx context.Context, id string) (*domain.EvaluationTestCase, error) {
	tc, err := scanTestCase(s.db.QueryRowContext(ctx, `SELECT `+testCaseColumns+` FROM test_cases WHERE id = ?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, notFound("test case", id)
	}
	if err != nil {
		return nil, fmt.Errorf("get test case: %w", err)
	}
	return &tc, nil
}

// ListTestCases returns every test case of a prompt, oldest first.
func (s *Store) ListTestCases(ctx context.Context, promptID string) ([]domain.EvaluationTestCase, error) {
	if err := s.requirePrompt(ctx, s.db, promptID); err != nil {
		return nil, err
	}
	return queryAll(ctx, s.db, scanTestCase,
		`SELECT `+testCaseColumns+` FROM test_cases WHERE prompt_id = ? ORDER BY seq`, promptID)
}

// DeleteTestCase removes a test case. Results already recorded for it are
// kept.
func (s *Store) DeleteTestCase(ctx context.Context, id string) error {
	return s.deleteByID(ctx, "test_cases", "test case", id)
}

// ---- runs ----

// CreateRun stores a new run.
func (s *Store) CreateRun(ctx context.Context, run *domain.EvaluationRun) error {
	if err := run.Validate(); err != nil {
		return err
	}
	created := *run
	if created.ID == "" {
		created.ID = domain.NewID()
	}
	if created.StartedAt.IsZero() {
		created.StartedAt = s.now().UTC()
	}
	err := s.tx(ctx, func(tx *sql.Tx) error {
		if err := s.requirePrompt(ctx, tx, run.PromptID); err != nil {
			return err
		}
		_, err := tx.ExecContext(ctx,
			`INSERT INTO runs (id, prompt_id, prompt_version_id, version, status, summary, error, started_at, finished_at)
			 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			created.ID, created.PromptID, created.PromptVersionID, created.Version, string(created.Status),
			toJSON(created.Summary), created.Error, toUnix(created.StartedAt), toUnix(created.FinishedAt))
		if isUnique(err) {
			return fmt.Errorf("run %q: %w", created.ID, domain.ErrConflict)
		}
		if err != nil {
			return fmt.Errorf("insert run: %w", err)
		}
		return nil
	})
	if err != nil {
		return err
	}
	*run = created
	return nil
}

// UpdateRun overwrites the status, summary, error and finish time of a run.
func (s *Store) UpdateRun(ctx context.Context, run *domain.EvaluationRun) error {
	if err := run.Validate(); err != nil {
		return err
	}
	res, err := s.db.ExecContext(ctx,
		`UPDATE runs SET status = ?, summary = ?, error = ?, finished_at = ? WHERE id = ?`,
		string(run.Status), toJSON(run.Summary), run.Error, toUnix(run.FinishedAt), run.ID)
	if err != nil {
		return fmt.Errorf("update run: %w", err)
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return notFound("run", run.ID)
	}
	return nil
}

// InterruptRuns marks every running run as failed.
func (s *Store) InterruptRuns(ctx context.Context, reason string, at time.Time) (int, error) {
	res, err := s.db.ExecContext(ctx,
		`UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE status = ?`,
		string(domain.RunStatusFailed), reason, toUnix(at), string(domain.RunStatusRunning))
	if err != nil {
		return 0, fmt.Errorf("interrupt runs: %w", err)
	}
	n, _ := res.RowsAffected()
	return int(n), nil
}

const runColumns = `id, prompt_id, prompt_version_id, version, status, summary, error, started_at, finished_at`

func scanRun(row interface{ Scan(...any) error }) (domain.EvaluationRun, error) {
	var (
		run               domain.EvaluationRun
		status, summary   string
		started, finished int64
	)
	err := row.Scan(&run.ID, &run.PromptID, &run.PromptVersionID, &run.Version, &status, &summary, &run.Error,
		&started, &finished)
	if err != nil {
		return run, err
	}
	if err := json.Unmarshal([]byte(summary), &run.Summary); err != nil {
		return run, fmt.Errorf("decode summary of run %q: %w", run.ID, err)
	}
	run.Status = domain.RunStatus(status)
	run.StartedAt, run.FinishedAt = fromUnix(started), fromUnix(finished)
	return run, nil
}

// GetRun returns the run with the given ID.
func (s *Store) GetRun(ctx context.Context, id string) (*domain.EvaluationRun, error) {
	run, err := scanRun(s.db.QueryRowContext(ctx, `SELECT `+runColumns+` FROM runs WHERE id = ?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, notFound("run", id)
	}
	if err != nil {
		return nil, fmt.Errorf("get run: %w", err)
	}
	return &run, nil
}

// ListRuns returns the runs of a prompt, newest first.
func (s *Store) ListRuns(ctx context.Context, promptID string, opts domain.ListOptions) ([]domain.EvaluationRun, error) {
	if err := s.requirePrompt(ctx, s.db, promptID); err != nil {
		return nil, err
	}
	opts = opts.Normalize()
	return queryAll(ctx, s.db, scanRun,
		`SELECT `+runColumns+` FROM runs WHERE prompt_id = ? ORDER BY seq DESC LIMIT ? OFFSET ?`,
		promptID, opts.Limit, opts.Offset)
}

// ---- results ----

// SaveResults stores a batch of results in one transaction; either all are
// written or none.
func (s *Store) SaveResults(ctx context.Context, results []domain.EvaluationResult) error {
	now := s.now().UTC()
	batch := make([]domain.EvaluationResult, len(results))
	copy(batch, results)

	err := s.tx(ctx, func(tx *sql.Tx) error {
		for i := range batch {
			r := &batch[i]
			ok, err := exists(ctx, tx, "runs", r.RunID)
			if err != nil {
				return err
			}
			if !ok {
				return fmt.Errorf("results[%d]: %w", i, notFound("run", r.RunID))
			}
			if r.ID == "" {
				r.ID = domain.NewID()
			}
			if r.CreatedAt.IsZero() {
				r.CreatedAt = now
			}
			var cost sql.NullFloat64
			if r.CostUSD != nil {
				cost = sql.NullFloat64{Float64: *r.CostUSD, Valid: true}
			}
			_, err = tx.ExecContext(ctx,
				`INSERT INTO results
					(id, run_id, test_case_id, prompt_version_id, provider, model, actual_output, passed, score,
					 input_tokens, output_tokens, latency_ns, error, created_at, cost_usd)
				 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
				r.ID, r.RunID, r.TestCaseID, r.PromptVersionID, string(r.Provider), r.Model, r.ActualOutput,
				r.Passed, r.Score, r.Usage.InputTokens, r.Usage.OutputTokens, int64(r.Latency), r.Error,
				toUnix(r.CreatedAt), cost)
			if isUnique(err) {
				return fmt.Errorf("results[%d]: result %q: %w", i, r.ID, domain.ErrConflict)
			}
			if err != nil {
				return fmt.Errorf("results[%d]: insert: %w", i, err)
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	copy(results, batch)
	return nil
}

const resultColumns = `id, run_id, test_case_id, prompt_version_id, provider, model, actual_output, passed, score,
	input_tokens, output_tokens, latency_ns, error, created_at, cost_usd`

func scanResult(row interface{ Scan(...any) error }) (domain.EvaluationResult, error) {
	var (
		r        domain.EvaluationResult
		provider string
		latency  int64
		created  int64
		cost     sql.NullFloat64
	)
	err := row.Scan(&r.ID, &r.RunID, &r.TestCaseID, &r.PromptVersionID, &provider, &r.Model, &r.ActualOutput,
		&r.Passed, &r.Score, &r.Usage.InputTokens, &r.Usage.OutputTokens, &latency, &r.Error, &created, &cost)
	if err != nil {
		return r, err
	}
	if cost.Valid {
		r.CostUSD = &cost.Float64
	}
	r.Provider = domain.Provider(provider)
	r.Latency = time.Duration(latency)
	r.CreatedAt = fromUnix(created)
	return r, nil
}

// ListResultsByRun returns the results of a run in the order they were saved.
func (s *Store) ListResultsByRun(ctx context.Context, runID string) ([]domain.EvaluationResult, error) {
	return queryAll(ctx, s.db, scanResult,
		`SELECT `+resultColumns+` FROM results WHERE run_id = ? ORDER BY seq`, runID)
}

// ListResultsByVersion returns results for a prompt version, newest first.
func (s *Store) ListResultsByVersion(ctx context.Context, promptVersionID string, opts domain.ListOptions) ([]domain.EvaluationResult, error) {
	opts = opts.Normalize()
	return queryAll(ctx, s.db, scanResult,
		`SELECT `+resultColumns+` FROM results WHERE prompt_version_id = ? ORDER BY seq DESC LIMIT ? OFFSET ?`,
		promptVersionID, opts.Limit, opts.Offset)
}
