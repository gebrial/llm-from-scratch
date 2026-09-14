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

The first four entries are **backfilled**: the decisions were made before this
log existed, and were recovered on 2026-09-14 by going back through the commits
while the reasoning was still recent. Entries after those were written at the
time of choosing.

---

## 2026-09-14 (backfilled) — Python (FastAPI) for the API layer

**Chose:** Python for the API layer — the same language as the model.

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

---

## 2026-09-14 (backfilled) — Postgres for prompt/story storage

**Chose:** Postgres, with a single table:

```sql
CREATE TABLE IF NOT EXISTS stories (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prompt TEXT NOT NULL,
    story TEXT NOT NULL
);
```

**Why the store is Postgres:** UNEXAMINED. The agent suggested Postgres and the
choice was not questioned at the time. Recorded as unexamined rather than
reconstructed after the fact.

**Why the table exists**, which was a deliberate decision: it stores every
prompt and the story generated from it, so that visitors can be shown stories
that have already been generated instead of waiting to generate their own.
Generation is slow; querying stored stories is fast. Later, a signed-in user
sees their own past prompts and stories.

**Open:** No timestamp column — timestamps were not considered. This interacts
with the purpose above: showing past stories implies an order to show them in.

**Open:** Whether Postgres is the right store, now that the question has been
asked. See the grill below this entry once it is run.

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
