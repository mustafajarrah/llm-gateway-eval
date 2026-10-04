package domain

import (
	"crypto/rand"
	"encoding/hex"
)

// NewID returns a random 128-bit identifier as 32 lowercase hex characters.
// Repositories use it to assign IDs to entities created without one.
func NewID() string {
	var b [16]byte
	// crypto/rand.Read never returns an error on supported platforms; it
	// aborts the program instead.
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
