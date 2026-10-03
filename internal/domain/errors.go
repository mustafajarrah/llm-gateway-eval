package domain

import "errors"

// Sentinel errors shared across layers. Adapters wrap them with additional
// context (fmt.Errorf("...: %w", ErrNotFound)) so callers can branch with
// errors.Is while still getting a descriptive message.
var (
	// ErrNotFound is returned when a requested entity does not exist.
	ErrNotFound = errors.New("not found")
	// ErrConflict is returned when an entity violates a uniqueness constraint,
	// e.g. a prompt name or a (prompt, version) pair that already exists.
	ErrConflict = errors.New("conflict")
	// ErrInvalidInput is returned when an entity or request fails validation.
	ErrInvalidInput = errors.New("invalid input")
)
