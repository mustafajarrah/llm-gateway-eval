.PHONY: build run test cover fmt vet lint check docker clean

BIN := bin/gateway
IMAGE := llm-gateway-eval

build: ## Compile the gateway binary into bin/
	go build -o $(BIN) ./cmd/gateway

run: ## Run the gateway with the current environment
	go run ./cmd/gateway

test: ## Run the test suite with the race detector
	go test -race ./...

cover: ## Run the tests and print per-package coverage
	go test -race -cover ./...

fmt: ## Format the code
	gofmt -w .

vet: ## Run go vet
	go vet ./...

lint: ## Run staticcheck
	go run honnef.co/go/tools/cmd/staticcheck@v0.8.1 ./...

check: vet lint test ## Everything CI runs
	@unformatted="$$(gofmt -l .)"; \
	if [ -n "$$unformatted" ]; then echo "These files need gofmt:"; echo "$$unformatted"; exit 1; fi

docker: ## Build the container image
	docker build -t $(IMAGE) .

clean: ## Remove build output
	rm -rf bin
