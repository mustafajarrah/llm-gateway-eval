package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/app"
	"github.com/mustafajarrah/llm-gateway-eval/internal/domain"
)

func env(vars map[string]string) func(string) string {
	return func(key string) string { return vars[key] }
}

// syncBuffer is a bytes.Buffer safe for the server's concurrent log writes.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func TestRunServesUntilCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var logs syncBuffer
	addrCh := make(chan string, 1)
	done := make(chan error, 1)
	go func() {
		done <- run(ctx, env(map[string]string{
			"GATEWAY_ADDR":    "127.0.0.1:0",
			"GATEWAY_DB_PATH": ":memory:",
		}), &logs, func(addr string) { addrCh <- addr })
	}()

	var addr string
	select {
	case addr = <-addrCh:
	case err := <-done:
		t.Fatalf("run() returned before becoming ready: %v", err)
	case <-time.After(10 * time.Second):
		t.Fatal("run() did not become ready")
	}

	resp, err := http.Get("http://" + addr + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz error = %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(body), `"ok"`) {
		t.Errorf("GET /healthz = %d %s", resp.StatusCode, body)
	}

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("run() error = %v, want a clean shutdown", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("run() did not stop after cancellation")
	}
	for _, want := range []string{`"msg":"listening"`, `"msg":"shutting down"`, `"path":"/healthz"`} {
		if !strings.Contains(logs.String(), want) {
			t.Errorf("logs do not contain %s:\n%s", want, logs.String())
		}
	}
}

func TestRunErrors(t *testing.T) {
	ctx := context.Background()

	err := run(ctx, env(map[string]string{"GATEWAY_ADDR": "nonsense"}), io.Discard, nil)
	if !errors.Is(err, domain.ErrInvalidInput) {
		t.Errorf("bad config: error = %v, want ErrInvalidInput", err)
	}

	err = run(ctx, env(map[string]string{"GATEWAY_ADDR": ":0", "GATEWAY_DB_PATH": ":memory:"}), io.Discard, nil)
	if !errors.Is(err, app.ErrUnprotected) {
		t.Errorf("exposed without a key: error = %v, want ErrUnprotected", err)
	}

	// Occupy a port so that listening on it fails.
	taken, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer taken.Close()
	err = run(ctx, env(map[string]string{"GATEWAY_ADDR": taken.Addr().String(), "GATEWAY_DB_PATH": ":memory:"}), io.Discard, nil)
	if err == nil || !strings.Contains(err.Error(), "listen on") {
		t.Errorf("port in use: error = %v, want a listen error", err)
	}
}
