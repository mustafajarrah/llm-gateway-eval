package domain

import (
	"errors"
	"testing"
)

func validVersion() *PromptVersion {
	return &PromptVersion{
		PromptID:     "p1",
		Version:      1,
		Template:     "Summarise: {{.text}}",
		SystemPrompt: "You are concise.",
		Provider:     ProviderOpenAI,
		Model:        "gpt-4o",
		Parameters:   ModelParameters{MaxTokens: 100},
	}
}

func TestPromptValidate(t *testing.T) {
	if err := (&Prompt{Name: "summariser"}).Validate(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := (&Prompt{Name: "  "}).Validate(); !errors.Is(err, ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput, got %v", err)
	}
}

func TestPromptVersionValidate(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(v *PromptVersion)
		wantErr bool
	}{
		{name: "valid", mutate: func(*PromptVersion) {}},
		{name: "missing prompt id", mutate: func(v *PromptVersion) { v.PromptID = "" }, wantErr: true},
		{name: "zero version", mutate: func(v *PromptVersion) { v.Version = 0 }, wantErr: true},
		{name: "empty template", mutate: func(v *PromptVersion) { v.Template = " " }, wantErr: true},
		{name: "unparsable template", mutate: func(v *PromptVersion) { v.Template = "{{.text" }, wantErr: true},
		{name: "unknown provider", mutate: func(v *PromptVersion) { v.Provider = "x" }, wantErr: true},
		{name: "missing model", mutate: func(v *PromptVersion) { v.Model = "" }, wantErr: true},
		{name: "bad parameters", mutate: func(v *PromptVersion) { v.Parameters.MaxTokens = -5 }, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := validVersion()
			tt.mutate(v)
			err := v.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil && !errors.Is(err, ErrInvalidInput) {
				t.Errorf("Validate() error %v does not wrap ErrInvalidInput", err)
			}
		})
	}
}

func TestPromptVersionRender(t *testing.T) {
	v := validVersion()

	got, err := v.Render(map[string]string{"text": "Go is great."})
	if err != nil {
		t.Fatalf("Render() error = %v", err)
	}
	if want := "Summarise: Go is great."; got != want {
		t.Errorf("Render() = %q, want %q", got, want)
	}

	if _, err := v.Render(nil); !errors.Is(err, ErrInvalidInput) {
		t.Errorf("Render() with missing variable: expected ErrInvalidInput, got %v", err)
	}

	v.Template = "{{.text"
	if _, err := v.Render(map[string]string{"text": "x"}); !errors.Is(err, ErrInvalidInput) {
		t.Errorf("Render() with bad template: expected ErrInvalidInput, got %v", err)
	}
}

func TestPromptVersionBuildRequest(t *testing.T) {
	v := validVersion()
	req, err := v.BuildRequest(map[string]string{"text": "abc"})
	if err != nil {
		t.Fatalf("BuildRequest() error = %v", err)
	}
	if err := req.Validate(); err != nil {
		t.Fatalf("built request is invalid: %v", err)
	}
	if req.Provider != v.Provider || req.Model != v.Model || req.SystemPrompt != v.SystemPrompt {
		t.Errorf("request does not carry version config: %+v", req)
	}
	if len(req.Messages) != 1 || req.Messages[0].Content != "Summarise: abc" || req.Messages[0].Role != RoleUser {
		t.Errorf("unexpected messages: %+v", req.Messages)
	}
	if req.Parameters.MaxTokens != 100 {
		t.Errorf("parameters not propagated: %+v", req.Parameters)
	}

	if _, err := v.BuildRequest(nil); err == nil {
		t.Error("BuildRequest() expected error for missing variable")
	}
}

func TestEvaluationTestCaseValidate(t *testing.T) {
	base := func() *EvaluationTestCase {
		return &EvaluationTestCase{PromptID: "p1", Name: "case", ExpectedOutput: "ok", MatchStrategy: MatchExact}
	}
	tests := []struct {
		name    string
		mutate  func(tc *EvaluationTestCase)
		wantErr bool
	}{
		{name: "valid", mutate: func(*EvaluationTestCase) {}},
		{name: "missing prompt id", mutate: func(tc *EvaluationTestCase) { tc.PromptID = "" }, wantErr: true},
		{name: "missing name", mutate: func(tc *EvaluationTestCase) { tc.Name = "" }, wantErr: true},
		{name: "missing expectation", mutate: func(tc *EvaluationTestCase) { tc.ExpectedOutput = "" }, wantErr: true},
		{name: "unknown strategy", mutate: func(tc *EvaluationTestCase) { tc.MatchStrategy = "fuzzy" }, wantErr: true},
		{name: "valid regex", mutate: func(tc *EvaluationTestCase) { tc.MatchStrategy, tc.ExpectedOutput = MatchRegex, `^\d+$` }},
		{name: "invalid regex", mutate: func(tc *EvaluationTestCase) { tc.MatchStrategy, tc.ExpectedOutput = MatchRegex, `(` }, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc := base()
			tt.mutate(tc)
			err := tc.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil && !errors.Is(err, ErrInvalidInput) {
				t.Errorf("Validate() error %v does not wrap ErrInvalidInput", err)
			}
		})
	}
}

func TestEvaluationTestCaseMatches(t *testing.T) {
	tests := []struct {
		name     string
		strategy MatchStrategy
		expected string
		actual   string
		want     bool
		wantErr  bool
	}{
		{name: "exact trims whitespace", strategy: MatchExact, expected: "Paris", actual: "  Paris\n", want: true},
		{name: "exact is case-sensitive", strategy: MatchExact, expected: "Paris", actual: "paris", want: false},
		{name: "contains is case-insensitive", strategy: MatchContains, expected: "PARIS", actual: "The capital is Paris.", want: true},
		{name: "contains miss", strategy: MatchContains, expected: "Berlin", actual: "Paris", want: false},
		{name: "regex match", strategy: MatchRegex, expected: `^\d{4}$`, actual: "2024", want: true},
		{name: "regex miss", strategy: MatchRegex, expected: `^\d{4}$`, actual: "24", want: false},
		{name: "regex invalid", strategy: MatchRegex, expected: `(`, actual: "x", wantErr: true},
		{name: "unknown strategy", strategy: "fuzzy", expected: "x", actual: "x", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc := &EvaluationTestCase{ExpectedOutput: tt.expected, MatchStrategy: tt.strategy}
			got, err := tc.Matches(tt.actual)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Matches() error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("Matches() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestListOptionsNormalize(t *testing.T) {
	tests := []struct {
		in, want ListOptions
	}{
		{in: ListOptions{}, want: ListOptions{Limit: DefaultListLimit}},
		{in: ListOptions{Limit: 10, Offset: 20}, want: ListOptions{Limit: 10, Offset: 20}},
		{in: ListOptions{Limit: 10_000, Offset: -3}, want: ListOptions{Limit: MaxListLimit}},
	}
	for _, tt := range tests {
		if got := tt.in.Normalize(); got != tt.want {
			t.Errorf("Normalize(%+v) = %+v, want %+v", tt.in, got, tt.want)
		}
	}
}

func TestEnumValidity(t *testing.T) {
	if !ProviderOpenAI.Valid() || !ProviderAnthropic.Valid() || Provider("x").Valid() {
		t.Error("Provider.Valid() misclassified a value")
	}
	if ProviderOpenAI.String() != "openai" {
		t.Errorf("Provider.String() = %q", ProviderOpenAI.String())
	}
	if !RoleUser.Valid() || !RoleAssistant.Valid() || Role("system").Valid() {
		t.Error("Role.Valid() misclassified a value")
	}
}
