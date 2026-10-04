package memory_test

import (
	"testing"

	"github.com/mustafajarrah/llm-gateway-eval/internal/storage/memory"
	"github.com/mustafajarrah/llm-gateway-eval/internal/storage/storagetest"
)

func TestConformance(t *testing.T) {
	storagetest.Run(t, func(*testing.T) storagetest.Store { return memory.New() })
}
