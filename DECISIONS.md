# Decision log

Every non-obvious choice in this project, recorded at the moment of choosing.

Format, fixed in `INTERVIEW-HANDOFF.md`:

```
## <date> — <the decision>
Chose: ...
Considered: ...
Why: ...
```

Two conventions added when the log was started:

- **`Open:`** — a consequence of the decision that has not been settled yet. It
  is a flag on the entry, not a task.
- **UNEXAMINED** in the `Why:` field — the choice was made without weighing
  alternatives, and that is what gets recorded. It is not backfilled with a
  plausible-sounding reason invented later.
- **Reasons from outside the code are recorded plainly.** Some choices here were
  fixed by `project-brief-llm-inference-api.md` before any code existed, for
  reasons about the job hunt rather than about engineering. Those are real
  reasons, they are dated, and they go in the log as they are. A log whose
  entries are all technical is the one that reads as reconstructed.

The first four entries are **backfilled**: the decisions were made before this
log existed, and were recovered on 2026-09-14 by going back through the commits
while the reasoning was still recent. Entries after those were written at the
time of choosing.

---

## 2026-09-14 (backfilled) — Python (FastAPI) for the API layer

**Chose:** Python for the API layer — the same language as the model.

**Constrained by:** `project-brief-llm-inference-api.md` had already narrowed the
field to Node.js/TypeScript or Python, ruling out Java, Kotlin and Go on the
grounds that a language learned for this project would produce a backend that
could not be defended under questioning. The choice recorded here is therefore
Python versus Node, which is the one the brief left open.

**Considered:** Splitting it: a Python inference service with the API layer
written in another language. It came up, and was moved past quickly.

**Why:** Two independent reasons.

1. The model code, including the training code, is Python. Porting it to
   another language would have been unnecessary overhead.
2. Python has substantial framework support for building an API layer, so
   nothing was given up by staying in it.

---

## 2026-09-14 (backfilled) — Rate limiting on `/generate`: slowapi, 5/min, keyed on IP

**Chose:** `slowapi`, 5 requests per minute, keyed on remote IP, applied to
`/generate` only. `/ping` is unlimited.

**Considered:** IP is the only key available today. There are no plans to
implement API keys. Once users exist, the key moves to the user.

**Why:** The limit protects compute. The ceiling is set so that no one can
claim more generation than they could consume in real time. This project is a
demo, not real usage.

`5/minute` is not an arbitrary number: it is the approximate rate at which the
laptop generates stories from a prompt. It is an observed figure, and it gets
re-derived once this runs on a cloud instance.

**Open:** Where the counter lives. `slowapi` with no `storage_uri` configured —
which is the current case — keeps its counters in memory, in the process. Two
consequences follow: the counters reset on restart, and they are per-process,
so running under multiple workers or replicas gives each one its own counter
and multiplies the effective limit. This has to be decided at deployment, and
the deployment target is a cloud instance.

This is the same question as the one flagged under the Postgres entry below.
Both are in-process state, and both change meaning the moment this runs as more
than one process. See `ROADMAP.md`, "Serve concurrent visitors without
contention". The store survives that change; these counters do not.

---

## 2026-09-14 (backfilled) — Postgres for prompt/story storage

*Revisited the same day in a grill session. The one question this entry was
carrying turned out to be two, and both now have answers.*

**Chose:** Postgres, with a single table:

```sql
CREATE TABLE IF NOT EXISTS stories (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prompt TEXT NOT NULL,
    story TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**What the table is:** a cache of finished stories, kept so that a visitor can be
shown stories that already exist instead of waiting on a generation of their own.
Generation is slow and reading a stored story is fast. Later, a signed-in user
sees their own past prompts and stories.

It is **not a usage log**, though it was named like one until 2026-09-14. See the
separate entry on usage logging below.

**Why Postgres is in this project at all:** because
`project-brief-llm-inference-api.md` named PostgreSQL as one of the technologies
this project exists to close a gap in. That came out of an analysis of 31 job
evaluations, where Postgres appeared as a gap in 4 of them. The brief predates
every line of code here, so this reason is dated and checkable rather than
reconstructed. The project was going to run Postgres whatever this table needed.

**Why Postgres rather than something simpler for this table:** examined
2026-09-14, and the honest answer is that nothing about this table requires it.

SQLite would serve this workload. The access pattern is one append per
generation and a read of recent rows, with no joins and no cross-table
transactions. `/generate` is rate limited to 5 a minute per IP and generation is
slow, so SQLite's single-writer-at-a-time limit would never be felt. SQLite's own
documentation puts its comfortable ceiling at 100K hits a day, conservatively,
and sqlite.org itself runs on it at four to five times that
([sqlite.org/whentouse.html](https://www.sqlite.org/whentouse.html)). This is a
demo an interviewer visits.

So Postgres is here because it was chosen to learn, and it was worth learning
because it kept appearing in the roles being targeted. That is the reason, and it
is a portfolio reason rather than an engineering one.

There is one engineering reason alongside it, and it is **conditional, not
established**: SQLite is a file that one machine's processes open, so a
deployment of more than one API process cannot share it, while Postgres is a
network service built for exactly that. Whether this ever runs as more than one
process is an open question, not a requirement. Stated as a requirement it would
be an invented justification; stated as a condition it is true.

**Schema decisions made 2026-09-14:**

- `created_at` exists so that "recent stories" has something to order by.
  Ordering on `id` would work, since it is an identity column, but it would be
  ordering on an implementation detail rather than on time.
- Repeats are kept as separate rows. Two visitors sending the same prompt get two
  different stories, and two stories from one prompt is the interesting output of
  this model rather than a duplicate to collapse.
- Note for deployment: `CREATE TABLE IF NOT EXISTS` does not add `created_at` to
  a database that already has the table. The existing local volume needs the
  table dropped or the column added by hand. A real migration tool is not
  warranted yet, and this note is here so the gap is not met for the first time
  on the instance.

**Open:** Nothing reads the table. `save_story` writes; no `SELECT` exists
anywhere in `src/` or `scripts/`, and no endpoint returns stored stories. The
purpose recorded above is the reason the table exists and is not yet code. See
`ROADMAP.md`.

**Open:** One process or several. This decides whether the conditional reason
above ever becomes a real one, and it is the same question flagged under the rate
limiter entry. See `ROADMAP.md`, "Serve concurrent visitors without contention".

---

## 2026-09-14 — Visitor prompts and stories are published unreviewed

**Chose:** Stories shown to visitors are published without review. Prompts are
visitor-submitted text, stored and displayed as typed.

**Why:** Expected traffic is people who are interviewing me, following a link. At
that volume, pre-publication review is machinery with nothing to do.

**Moderation is manual:** I review the stored prompts occasionally and delete any
I find problematic, meaning vulgar or with themes I do not want on the page.

**Consequence, named rather than discovered later:** there is no delete path. No
admin endpoint, no script, no query. On AWS this means holding a `psql` session
against the production database and writing `DELETE` by hand, which also means
production credentials on my laptop. That is accepted for a demo. If direct
access turns out to be awkward once deployed, the fallback is a short script in
`scripts/` that lists recent rows and deletes by id. An admin endpoint is
deliberately not being built.

---

## 2026-09-14 — Usage logging: considered and dropped

**Chose:** No usage logging. No per-request record of who called, when, how long
generation took, or what was returned.

**Considered:** `project-brief-llm-inference-api.md` asked for "PostgreSQL for
persisting something real (e.g. request/usage logging)". That parenthetical was
an example of something real to persist, and the story cache satisfies the same
goal.

**Why:** The store that exists is product-facing: it holds content to show
visitors. Usage logging answers an operations question, and nobody is operating
this service. Building half a logging table to satisfy an example in a brief
would put a table in the schema that nothing reads and no one acts on.

This is recorded rather than left as a gap so that "why isn't there request
logging?" has an answer that is a decision instead of an oversight.

---

## 2026-09-14 (backfilled) — One module-level connection, no pool, autocommit

**Chose:** A single `psycopg` connection opened at import time, `autocommit=True`,
no connection pool.

**Considered:** A connection pool. It was weighed at the time and deliberately
deferred — there is a comment in `src/api.py` saying so.

**Why:** Multiple people hitting this API at once is not expected, and the
priority is getting the project deployed soon; a minimal form is acceptable for
that.

This entry states the assumption the decision rests on — **no concurrency** —
which is also the condition that forces a revisit. If concurrent users arrive,
this decision is the one that changes.

---

## 2026-09-15 — Default vs no default path values

**Chose:** No default for `CHECKPOINT_PATH`, default value for `TOKENIZER_PATH`.

**Considered:** Defaults for both paths, no defaults for either path, or default for `CHECKPOINT_PATH` and no default for `TOKENIZER_PATH`.

**Why:** `TOKENIZER_PATH` has a hard-coded value in the script that generates the tokenizer json file so it makes sense to use that same value here. On the other hand, `CHECKPOINT_PATH` could change based on the hyperparameters chosen (e.g., number of epochs, number of steps, dataset size) so it doesn't make sense to set a default that might not be valid. It's better for the application to throw an error here if no value was defined that later (e.g., when the generate endpoint is finally hit) if a default value was used but no model exists at that path.
