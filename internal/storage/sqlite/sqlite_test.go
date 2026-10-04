package sqlite_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
	"github.com/mustafajarrah/llm-gateway-eval/internal/storage/sqlite"
	"github.com/mustafajarrah/llm-gateway-eval/internal/storage/storagetest"
)

func open(t *testing.T, path string) *sqlite.Store {
	t.Helper()
	s, err := sqlite.Open(context.Background(), path)
	if err != nil {
		t.Fatalf("Open(%q) error = %v", path, err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestConformanceInMemory(t *testing.T) {
	storagetest.Run(t, func(t *testing.T) storagetest.Store { return open(t, sqlite.InMemory) })
}

func TestConformanceOnDisk(t *testing.T) {
	storagetest.Run(t, func(t *testing.T) storagetest.Store {
		return open(t, filepath.Join(t.TempDir(), "gateway.db"))
	})
}

func TestDataSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	// The parent directory does not exist yet; Open must create it.
	path := filepath.Join(t.TempDir(), "nested", "dir", "gateway.db")

	first, err := sqlite.Open(ctx, path)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	p := &domain.Prompt{Name: "durable", Tags: []string{"x"}}
	if err := first.CreatePrompt(ctx, p); err != nil {
		t.Fatalf("CreatePrompt() error = %v", err)
	}
	v := &domain.PromptVersion{PromptID: p.ID, Template: "{{.q}}", Model: "fast"}
	if err := first.CreateVersion(ctx, v); err != nil {
		t.Fatalf("CreateVersion() error = %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	second := open(t, path)
	got, err := second.GetPrompt(ctx, p.ID)
	if err != nil {
		t.Fatalf("GetPrompt() after reopen error = %v", err)
	}
	if got.Name != "durable" || len(got.Tags) != 1 || !got.CreatedAt.Equal(p.CreatedAt) {
		t.Errorf("GetPrompt() after reopen = %+v, want %+v", got, p)
	}
	// Reopening must not re-run migrations, and numbering must continue.
	next := &domain.PromptVersion{PromptID: p.ID, Template: "{{.q}}", Model: "fast"}
	if err := second.CreateVersion(ctx, next); err != nil {
		t.Fatalf("CreateVersion() after reopen error = %v", err)
	}
	if next.Version != 2 {
		t.Errorf("version after reopen = %d, want 2", next.Version)
	}
}

func TestOpenErrors(t *testing.T) {
	ctx := context.Background()
	if _, err := sqlite.Open(ctx, ""); !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("Open(\"\") error = %v, want ErrInvalidInput", err)
	}

	dir := t.TempDir()
	blocker := filepath.Join(dir, "file")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := sqlite.Open(ctx, filepath.Join(blocker, "gateway.db")); err == nil {
		t.Error("Open() under a regular file succeeded, want an error")
	}

	notADatabase := filepath.Join(dir, "garbage.db")
	if err := os.WriteFile(notADatabase, []byte("this is not a sqlite database, not even close"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := sqlite.Open(ctx, notADatabase); err == nil {
		t.Error("Open() on a non-database file succeeded, want an error")
	}
}
