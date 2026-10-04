package sqlite

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

// TestMigratesAnOlderDatabase builds a database at schema version 1, with
// data, and checks that Open upgrades it in place without losing anything.
func TestMigratesAnOlderDatabase(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "old.db")

	old, err := sql.Open("sqlite", "file:"+path+"?_pragma=foreign_keys(1)")
	if err != nil {
		t.Fatal(err)
	}
	for _, stmt := range []string{
		migrations[0],
		`PRAGMA user_version = 1`,
		`INSERT INTO prompts (id, name, description, tags, created_at, updated_at) VALUES ('p1', 'old', '', '[]', 1, 1)`,
		`INSERT INTO runs (id, prompt_id, prompt_version_id, version, status, summary, error, started_at, finished_at)
		 VALUES ('r1', 'p1', 'v1', 1, 'completed', '{"total":1,"passed":1}', '', 1, 2)`,
		`INSERT INTO results (id, run_id, test_case_id, prompt_version_id, provider, model, actual_output, passed, score,
		 input_tokens, output_tokens, latency_ns, error, created_at)
		 VALUES ('res1', 'r1', 'tc1', 'v1', 'openai', 'gpt-4o', 'hello', 1, 1, 10, 2, 5000000, '', 3)`,
	} {
		if _, err := old.ExecContext(ctx, stmt); err != nil {
			t.Fatalf("seed old database: %v", err)
		}
	}
	if err := old.Close(); err != nil {
		t.Fatal(err)
	}

	s, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open() of a version 1 database error = %v", err)
	}
	defer s.Close()

	var version int
	if err := s.db.QueryRowContext(ctx, "PRAGMA user_version").Scan(&version); err != nil || version != len(migrations) {
		t.Errorf("schema version = %d (%v), want %d", version, err, len(migrations))
	}
	results, err := s.ListResultsByRun(ctx, "r1")
	if err != nil || len(results) != 1 {
		t.Fatalf("ListResultsByRun() = %+v, %v", results, err)
	}
	if r := results[0]; r.ActualOutput != "hello" || !r.Passed || r.Usage.InputTokens != 10 || r.CostUSD != nil {
		t.Errorf("migrated result = %+v, want the old data with no cost", r)
	}
	run, err := s.GetRun(ctx, "r1")
	if err != nil || run.Summary.Passed != 1 || run.Summary.CostUSD != 0 || run.Summary.Unpriced != 0 {
		t.Errorf("migrated run = %+v, %v", run, err)
	}

	cost := 0.5
	if err := s.SaveResults(ctx, []domain.EvaluationResult{{RunID: "r1", PromptVersionID: "v1", CostUSD: &cost}}); err != nil {
		t.Errorf("SaveResults() after migration error = %v", err)
	}
}

func TestRejectsANewerDatabase(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "future.db")
	future, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := future.ExecContext(ctx, `PRAGMA user_version = 999`); err != nil {
		t.Fatal(err)
	}
	future.Close()

	if _, err := Open(ctx, path); err == nil {
		t.Error("Open() of a database from a newer build succeeded, want an error")
	}
}
