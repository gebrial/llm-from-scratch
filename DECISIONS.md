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
`/generate` only. `/ping` is unlimited. *`GET /stories` added at 60/min on
2026-09-23; see the stories feed entry.*

**Considered:** IP is the only key available today. There are no plans to
implement API keys. Once users exist, the key moves to the user.

**Why:** The limit protects compute. The ceiling is set so that no one can
claim more generation than they could consume in real time. This project is a
demo, not real usage.

`5/minute` is not an arbitrary number: it is the approximate rate at which the
laptop generates stories from a prompt. It is an observed figure, and it gets
re-derived once this runs on a cloud instance.

*Re-derived 2026-09-29: **1/min.** On the deployed `t3.small` one story takes
about a minute, with the CPU at 100%, so by the same rule the ceiling is one per
minute. The limit is per IP, though, and the contention is global: two visitors
on different IPs can still both be generating at once. That is a separate
problem from rationing, taken up in `ROADMAP.md`, "Serve concurrent visitors
without contention".*

*Re-derived 2026-10-13: **6/min.** The KV cache (its entry, below) brought one
story down to about 8s locally, and about 9s on the instance (three prompts,
timed with curl).*

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

**What the table is:** the store behind the **feed**, kept so that a visitor can be
shown stories that already exist instead of waiting on a generation of their own.
Generation is slow and reading a stored story is fast. Later, a signed-in user
sees their own past prompts and stories.

It was called a cache until 2026-09-23. A cache is looked up by key, and this is
read newest first regardless of prompt, so the word was wrong. See the entry on
the stories feed below, and `CONTEXT.md` for the terms.

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

**Closed 2026-09-28:** ~~Nothing reads the table.~~ `GET /stories` now reads it,
as designed in the stories feed entry below. Until then the purpose recorded
above was the reason the table existed rather than code.

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

*Corrected 2026-10-09: no production credentials on my laptop after all. Written
before Postgres ran in compose on the instance. Now, from an SSH session there,
`docker compose exec db psql -U <user> <db>` runs `psql` inside the database
container itself, which the official image lets connect without a password. No
port is involved, so it works with Postgres bound to `127.0.0.1`, and the SSH key
is the only credential that leaves the server.*

**Revisit before the API opens publicly** (`ROADMAP.md` R1, HTTPS on a domain, done 2026-10-11). Re-checked
2026-09-23 when the feed was designed: the feed shows every stored story to every
later visitor, so an unreviewed prompt stops being theoretical the moment
strangers can write rows. Until the port opens, only I can. Options to weigh
then: a `hidden` flag the feed filters on, opt-in publishing, or listing only
stories from preset prompts.

*Revisited 2026-10-06, before opening the port: **manual moderation stays.**
Traffic is still people following a link, and `/generate` is limited to 1/min
per IP, so the feed is only as bad as the worst prompt someone types, and
deleting it by hand is enough at that volume. The trigger to change is the first
problematic prompt that reaches the feed: then add the `hidden` flag, so hiding
a row is an `UPDATE` rather than a `DELETE`. Showing only preset-prompt stories
was the one option that removes the risk outright rather than reacting to it,
and was passed over for now.*

---

## 2026-09-14 — Usage logging: considered and dropped

**Chose:** No usage logging. No per-request record of who called, when, how long
generation took, or what was returned.

**Considered:** `project-brief-llm-inference-api.md` asked for "PostgreSQL for
persisting something real (e.g. request/usage logging)". That parenthetical was
an example of something real to persist, and the stories behind the feed satisfy
the same goal.

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

---

## 2026-09-21 — Torch dependency index

**Chose:** cpu-only index to fetch torch from (see Dockerfile).

**Considered:** Originally I used the default index to download torch from but I switched to the cpu-only index now.

**Why:** PyPI's torch wheel bundles 6GB of CUDA libraries. This is completely unnecessary on an EC2 instance with no GPU attached. It bloats the docker image unnecessarily so it was an easy choice to use the cpu-only index. Note that testing locally on windows this never actually surfaced because PyPI's Windows wheel is already CPU-only, so this change only helps for Linux machines (which the EC2 deployment instance is). The measured result in image size is 14.5GB down to 2.91GB.

---

## 2026-09-21 — torch/torchtune/torchao versions

**Chose:** Pinned versions of torch/torchtune/torchao to versions that I've been testing locally (see requirements.txt)

**Considered:** Originally left it at any version greater than what I have.

**Why:** Originally torchao was pinned at 0.9.0 because I thought that newer versions broke torchtune's RoPE import, which I later found out it doesn't. Letting the version be whatever has been released the latest could cause bugs that I wouldn't catch locally. Pinning the versions to what I'm using on my laptop ensures I deploy exactly what I'm using and testing.

**Open:** The other requirements don't have their versions pinned down, so the version running on the image might be different than the one tested on.

---

## 2026-09-21 — depends_on db condition check

**Chose:** service_healthy (see docker-compose.yml)

**Considered:** service_started

**Why:** service_started only waits for the container process to launch, but the condition I actually want to wait for is the database to be up and accepting connections. For this I wrote a custom healthcheck which uses pg_isready and so I need to use service_healthy to check for that. This decision ensures the services are started in the correct order when running `docker compose up`.

Only caveat is that pg_isready can read true during initdb in postgres which runs a temporary server during that time.

---

## 2026-09-21 — Restart policy for both services

**Chose:** unless-stopped (see docker-compose.yml)

**Considered:** always

**Why:** If I ever shutdown these containers intentionally, I don't want them restarting on their own. The only time they should restart is after I've restarted the EC2 instance(requires docker daemon to start at boot, a setup step on EC2), or after a crash. This decision ensures proper recovery after crashes and reboots.

---

## 2026-09-23 — The stories feed: `GET /stories`

*Settled in a grill session before any of it was written.*

**Chose:** A feed of the most recent stories, newest first, whatever the prompt.

```
GET /stories?limit=20
→ {"stories": [{"prompt": ..., "story": ..., "created_at": ...}, ...]}
```

**Considered:** Looking stories up by prompt, which is what "cache" implied: a
visitor types an opening line and is shown stories already stored for it.

**Why:** A feed gives the page something to show the moment it loads and while a
generation runs, which covers "instead of waiting" without a lookup. A lookup by
prompt would almost never hit, since free-text prompts rarely repeat exactly.

**The shape, and why each part:**

- **An envelope, `{"stories": [...]}`, not a bare list.** An object at the top
  level can gain fields later, such as a pagination cursor, without breaking
  callers. A bare list cannot. Declared as a Pydantic response model, so the
  output is validated, documented in `/docs`, and limited to the declared fields.
- **Each item is `prompt`, `story` and `created_at`. No `id`.** The stored
  `story` already begins with the prompt, because generation decodes the prompt
  tokens along with the continuation. Returning both lets the UI set the
  visitor's words apart from the model's without the API guessing where one ends.
  `id` is left out until something needs it: adding a field later is free,
  removing one breaks whoever relied on it. It would also show roughly how many
  stories exist.
- **Ordered by `created_at` descending, `id` descending as a tie-break.** Two
  rows can share a timestamp, and without a second key their order is undefined.
  `id` is used in the query and not returned.
- **`limit`: default 20, maximum 50, no pagination.** The maximum is what bounds
  the cost of a read, which otherwise grows with the table. Out-of-range values
  are a 422. Pagination waits until there are enough stories to page through,
  and the envelope keeps room for it.
- **Rate limited at 60/min per IP.** The read is cheap, so this guards against a
  script rather than rationing reads. Every public endpoint having a limit is
  easier to reason about than one quietly left open. Not 5/min like `/generate`,
  which would lock out a visitor who reloads a few times.

**Moderation is unchanged**, and now matters: see the revisit note on the
unreviewed-publishing entry above.

**Known future break:** a turn-taking story (`ROADMAP.md`, "Turn-taking
stories") does not fit `{prompt, story}`. That is planned as its own page and its
own storage rather than a change to this feed, so this shape is not expected to
change for it.

---

## 2026-09-29 — Serve a stripped checkpoint, not the training checkpoint

**Chose:** A one-time `scripts/strip_checkpoint.py` that keeps only what
`load_from_checkpoint` reads (`state_dict`, `hyper_parameters` and the Lightning
version) and drops the rest. The API loads the result, `<name>-inference.ckpt`,
through the same `LitGPTModel.load_from_checkpoint` call as before. The full
checkpoint stays on my laptop, since it is the only file that can resume
training.

**Considered:**

- **A swap file.** It would get past the load spike, but it hides the
  measurement, and weights paged out to disk make generation slower still.
- **Casting the weights to bf16**, halving them again to ~240 MB. It is lossy: it
  changes the arithmetic, so generation quality would need re-checking. Kept in
  reserve in case the stripped file had not fit.
- **A `t3.medium` (4 GB)**, at about $30/month running against a $5 budget.
- **Dropping Lightning from serving entirely**, loading a raw `state_dict` into a
  plain `GPTModel`. That is `ROADMAP.md`, "Separate the training code from the
  serving code", and is a larger change than the problem needed.

**Why:** The first deploy to the 2 GB `t3.small` was OOM-killed on start-up
(exit code 137, four restarts), reaching 1.33 GB before the kill. The checkpoint
was 1436 MB, of which **957 MB was Adam's optimizer state**, two buffers per
parameter, which inference never reads. `load_from_checkpoint` loads the whole
file into memory regardless. Stripped, the file is 479 MB. The weights are
bit-for-bit identical (`torch.equal` on every tensor), so this is lossless.

**Measured on the instance:** ~1.2 GB peak while loading, 853 MB settled, 907 MB
during a generation. Host `available` memory during a generation: 538 MB.

**Deployment note:** `CHECKPOINT_PATH` in `.env` must name the `-inference`
file. The full checkpoint would still OOM the instance.

---

## 2026-09-29 — An Elastic IP for a stable address

**Chose:** An Elastic IP associated with the `t3.small`, and `stories.gebrial.ca`
as an A record pointing at it.

**Considered:** Keeping the auto-assigned IP and updating the DNS record on every
start, by hand or from a boot script calling a DNS provider's API.

**Why:** The auto-assigned IP changed on the first stop and start, and the domain
and the TLS certificate both need the name to reach the same machine every
time. Updating by hand breaks the site whenever I forget; a boot script means
moving DNS to a provider with an API and keeping credentials on the instance.

**Cost:** AWS bills every public IPv4 at $0.005/hour, about $3.60/month. While the
instance runs that is no extra, since the auto-assigned IP was billed the same.
But an Elastic IP also bills while the instance is **stopped**, and stopping when
idle is how this stays in budget.

With the 30 GB root volume at about $2.40, fixed costs come to about $6/month
before the instance runs at all, over the $5/month written down as the budget.
*Accepted 2026-10-06: the $5 was never a hard limit, and the account is past any
free tier, so nothing here is covered by one.*

---

## 2026-10-08 — Caddy as the reverse proxy for HTTPS

**Chose:** Caddy, as a third compose service in front of the API. It is the only
thing published to the internet, on 80 and 443, and it gets and renews a Let's
Encrypt certificate for `stories.gebrial.ca` by itself from a three-line
`Caddyfile`.

**Considered:**

- **nginx with certbot.** The industry standard, and the one most likely to come
  up in an interview, but certbot's renewal and the nginx reload after it are a
  moving part to wire up and keep working.
- **Traefik.** Automatic certificates too, but built for discovering containers
  dynamically, which one fixed backend does not need, and configured through
  Docker labels.
- **No proxy: uvicorn serving TLS itself.** Fewest pieces, and no forwarded-IP
  problem, but a restart on every renewal and no HTTP-to-HTTPS redirect.
- **An Application Load Balancer with AWS Certificate Manager.** About $16/month
  or more on its own.
- **CloudFront with AWS Certificate Manager.** Cheap at this traffic, but a CDN
  this does not need, and its origin timeout rules it out (below).
- **Cloudflare's proxy.** Means moving DNS off Squarespace.

**Why:** Simplicity. Caddy is the least configuration, has no renewal to set up
or get wrong, and costs nothing.

**Also the timeout.** A story takes about a minute to generate on the
`t3.small`, and every proxy limits how long it waits for the backend. nginx's
`proxy_read_timeout` defaults to 60s, right at the edge, so some stories would
come back as a 504 unless it were raised. CloudFront defaults to 30s, at most
60s without a request to AWS, so every story would fail. Cloudflare's free plan
allows 100s and cannot be changed. Caddy's reverse proxy has no response timeout
by default, so it waits. The real fix is generation that is fast or does not
hold the request open (`ROADMAP.md`, "Serve concurrent visitors without
contention"), but a proxy should not be what breaks a minute-long request.

**Consequence:** the certificate lives in the `caddy_data` volume, so
`docker compose down -v` now deletes it along with the stories. Let's Encrypt
issues at most 5 duplicate certificates a week per name, so losing it repeatedly
while debugging would lock the domain out of HTTPS for days.

## 2026-10-11 — The UI on its own subdomain, and the API moves

**Chose:** The UI's code in a folder in this repo, served by GitHub
Pages at `stories.gebrial.ca`. The API moves to `stories-api.gebrial.ca`.

**Considered:** A separate repo. `gebrial.github.io`, or the path
`www.gebrial.ca/llm-from-scratch/` that my other Pages projects use. Keeping
the API at `stories` and giving the UI another name.

**Why:** One repo keeps the CORS change and the UI code that needs it in one
commit. A dedicated origin keeps this site's `localStorage` and cookies apart
from my other projects once auth (R3) adds credentials; `github.io` would also
make the API's cookies third-party. The UI gets the name people visit, and the
swap was cheap while nothing called the API by name.

## 2026-10-11 — Vite + React + TypeScript, no framework

**Chose:** A Vite SPA from the `react-ts` template.

**Considered:** webpack, Parcel, Rsbuild/Rspack. Next.js or another framework,
either on the instance or as a static export.

**Why:** Vite is the default that React's docs, examples and recent codebases
use, which matters most while learning React; Vitest shares its config. Static
output and env config are not reasons, since every option has them. A
framework's value is server rendering, routing and server features: the data is
all dynamic from an API that is often asleep, there are one or two screens, and
FastAPI is already the server. On the instance, a framework would also go down
with it. TypeScript over JavaScript because it is my strong side, so React is
the only new thing.

## 2026-10-13 — A KV cache for top-p generation only

**Chose:** Cache each layer's keys and values (after RoPE) in `top_p_sampling`,
the method the API uses. `trf_blocks` becomes an `nn.ModuleList` so each block
gets its own cache; parameter names are unchanged, so checkpoints still load.

**Considered:** Caching `beam_search` and greedy `generate_text` as well.

**Why:** Only `top_p` is served. Beam search would also need the cache
reordered to follow the surviving beams every step. The cached path was checked
against the uncached one: logits agree to 7e-06, and seeded stories match.

## 2026-10-13 — Caddy only under a `prod` Compose profile

**Chose:** `profiles: ["prod"]` on `caddy`, switched on by
`COMPOSE_PROFILES=prod` in the instance's `.env`. A plain `up` locally starts
only `db` and `api`; the deploy command is unchanged.

**Considered:** Naming `db api` in the README's local command.
`docker-compose.override.yml`.

**Why:** Caddy is useless locally, and would keep failing to get a certificate
for a domain that does not point at the machine. A profile makes the safe
command the default one, rather than relying on remembering to name services.

## 2026-10-28 — Stream stories as NDJSON on a new route (R8)

**Chose:** `POST /generate/stream`, beside `/generate`, which stays. Both share
one rate limit (`shared_limit`, scope `generate`). The body is
`application/x-ndjson`: one JSON object per line, typed by a `type` field
(`token`, `error`, `done`). A `token` carries one token's text, decoded on its
own.

**Considered:** Replacing `/generate`. Separate limits per route. SSE. Raw text.
Decoding the whole sequence each step and sending the new suffix. Sending token
IDs.

**Why:** The UI deploys on push and the API by hand, so for a while one is
newer than the other; a new route keeps both versions of the UI working.
Separate limits would double the effective one, on the same CPU. SSE's value is
`EventSource`, which reconnects and resumes, but only over GET; generating is a
POST, so the UI parses either format itself, and NDJSON frames the same
messages with less format. Raw text has no place for an error once the `200` is
sent. Byte-level BPE can split a character across two tokens, which would
stream a `�` that is never stored, but none of this vocabulary's 4096 tokens
is non-ASCII (checked 2026-10-28), so every token decodes to whole characters.

**Open:** that Caddy passes the stream through unbuffered, to be checked on the
instance.
