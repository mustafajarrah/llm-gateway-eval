// Command gateway runs the LLM gateway and prompt evaluation service.
//
// It is configured through environment variables; see internal/config.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/mustafajarrah/llm-gateway-eval/internal/app"
	"github.com/mustafajarrah/llm-gateway-eval/internal/config"
)

// shutdownTimeout is how long in-flight requests get to finish after a
// termination signal.
const shutdownTimeout = 15 * time.Second

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, os.Getenv, os.Stderr, nil); err != nil {
		fmt.Fprintln(os.Stderr, "gateway:", err)
		os.Exit(1)
	}
}

// run starts the service and blocks until ctx is cancelled or the server
// fails. ready, when non-nil, is called with the bound address once the
// listener is up.
func run(ctx context.Context, getenv func(string) string, logOutput io.Writer, ready func(addr string)) error {
	cfg, err := config.FromEnv(getenv)
	if err != nil {
		return err
	}
	logger := slog.New(slog.NewJSONHandler(logOutput, &slog.HandlerOptions{Level: cfg.LogLevel}))

	service, err := app.New(ctx, cfg, logger)
	if err != nil {
		return err
	}
	defer service.Close()

	listener, err := net.Listen("tcp", cfg.Addr)
	if err != nil {
		return fmt.Errorf("listen on %s: %w", cfg.Addr, err)
	}
	server := &http.Server{
		Handler:           service.Handler,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		IdleTimeout:       2 * time.Minute,
		// No WriteTimeout: evaluation runs are synchronous and can
		// legitimately take minutes. Upstream calls are bounded by the
		// gateway's per-attempt timeout instead.
	}

	serveErr := make(chan error, 1)
	go func() { serveErr <- server.Serve(listener) }()
	logger.Info("listening", "addr", listener.Addr().String())
	if ready != nil {
		ready(listener.Addr().String())
	}

	select {
	case err := <-serveErr:
		return fmt.Errorf("serve: %w", err)
	case <-ctx.Done():
	}

	logger.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("shutdown: %w", err)
	}
	if err := <-serveErr; !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("serve: %w", err)
	}
	return nil
}
