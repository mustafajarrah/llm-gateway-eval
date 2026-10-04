package sqlite

// migrations are applied in order, each in its own transaction. The index of
// the last applied migration is kept in SQLite's user_version pragma. Never
// edit a released entry; append a new one.
//
// Every table has an autoincrement seq column: it records creation order,
// which is what "newest first" listings sort by. Timestamps are Unix
// nanoseconds, with 0 standing for the zero time.
var migrations = []string{
	`
CREATE TABLE prompts (
	seq         INTEGER PRIMARY KEY AUTOINCREMENT,
	id          TEXT    NOT NULL UNIQUE,
	name        TEXT    NOT NULL UNIQUE,
	description TEXT    NOT NULL,
	tags        TEXT    NOT NULL,
	created_at  INTEGER NOT NULL,
	updated_at  INTEGER NOT NULL
);

CREATE TABLE prompt_versions (
	seq           INTEGER PRIMARY KEY AUTOINCREMENT,
	id            TEXT    NOT NULL UNIQUE,
	prompt_id     TEXT    NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
	version       INTEGER NOT NULL,
	template      TEXT    NOT NULL,
	system_prompt TEXT    NOT NULL,
	provider      TEXT    NOT NULL,
	model         TEXT    NOT NULL,
	parameters    TEXT    NOT NULL,
	change_log    TEXT    NOT NULL,
	created_at    INTEGER NOT NULL,
	UNIQUE (prompt_id, version)
);

CREATE TABLE test_cases (
	seq             INTEGER PRIMARY KEY AUTOINCREMENT,
	id              TEXT    NOT NULL UNIQUE,
	prompt_id       TEXT    NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
	name            TEXT    NOT NULL,
	variables       TEXT    NOT NULL,
	expected_output TEXT    NOT NULL,
	match_strategy  TEXT    NOT NULL,
	created_at      INTEGER NOT NULL
);
CREATE INDEX test_cases_prompt ON test_cases(prompt_id);

CREATE TABLE runs (
	seq               INTEGER PRIMARY KEY AUTOINCREMENT,
	id                TEXT    NOT NULL UNIQUE,
	prompt_id         TEXT    NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
	prompt_version_id TEXT    NOT NULL,
	version           INTEGER NOT NULL,
	status            TEXT    NOT NULL,
	summary           TEXT    NOT NULL,
	error             TEXT    NOT NULL,
	started_at        INTEGER NOT NULL,
	finished_at       INTEGER NOT NULL
);
CREATE INDEX runs_prompt ON runs(prompt_id);

CREATE TABLE results (
	seq               INTEGER PRIMARY KEY AUTOINCREMENT,
	id                TEXT    NOT NULL UNIQUE,
	run_id            TEXT    NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
	test_case_id      TEXT    NOT NULL,
	prompt_version_id TEXT    NOT NULL,
	provider          TEXT    NOT NULL,
	model             TEXT    NOT NULL,
	actual_output     TEXT    NOT NULL,
	passed            INTEGER NOT NULL,
	score             REAL    NOT NULL,
	input_tokens      INTEGER NOT NULL,
	output_tokens     INTEGER NOT NULL,
	latency_ns        INTEGER NOT NULL,
	error             TEXT    NOT NULL,
	created_at        INTEGER NOT NULL
);
CREATE INDEX results_run ON results(run_id);
CREATE INDEX results_version ON results(prompt_version_id);
`,
	// NULL means no price was configured when the result was recorded.
	`ALTER TABLE results ADD COLUMN cost_usd REAL;`,
}
