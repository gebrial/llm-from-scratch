# Roadmap

Work that is not done yet. Decisions already made live in `DECISIONS.md`; this
file is the other half, and the two point at each other.

Each item says **what forces it** and **what it changes**, so that nothing sits
here as a vague intention.

**Item IDs are permanent.** A committed item gets the next ID (`R1`, `R2`, …)
when it is added, keeps it when it is done, and the ID is never reused. Order
lives only in the "Next" line under "Committed", so changing priorities or
finishing an item does not renumber anything else. Introduced 2026-10-11; items
finished before then have no ID.

**Why there are two lists.** This project has been built with an AI agent, and a
lot of what looked like a plan turned out to be agent suggestions that were never
ruled on. Recovered from the build transcripts on 2026-09-14: four items I had
actually committed to, and ten an agent proposed that I never answered. Keeping
them apart means nothing in the first list is something I did not choose. Same
reason `DECISIONS.md` marks an entry UNEXAMINED rather than inventing a reason
for it.

---

## Committed

**Next:** R2, then R9, R8, R3, R4, R5, R6, R7.

**Order, decided 2026-09-23.** The read path came first because it was small and
local, and the deploy came before the UI because it was the measurement. Both
are done as of 2026-09-29 (see "Done" below). The measurement did change things:
the full checkpoint did not fit, and the fix was a first, small piece of R5.
HTTPS (R1) came before the UI because the UI cannot call the API without it,
and is done as of 2026-10-11. R9 and R8 were moved ahead of R3 on 2026-10-11:
both are small next to it, build directly on R2, and R8 is the change visitors
notice most. R7 stays last because it needs an automatic idle stop as well.

**Constraint on every item below:** the `t3.small` is about $15/month running,
so the instance stops when idle rather than running continuously. The budget was
written down as $5/month, but it was never a hard limit: on 2026-10-06 I
accepted about $6/month in fixed costs, billed whether or not the instance runs
(the account is past any free tier). That is the 30 GB root volume at roughly
$2.40, against about 5 GB actually needed, and the Elastic IP at about $3.60.

### R2. A React UI on GitHub Pages

**Forces it:** Recorded in `INTERVIEW-HANDOFF.md` as a decision taken 2026-09-12.
It supersedes the brief's line about frontend being out of scope, which was
written earlier. Hosting on GitHub Pages was decided 2026-09-23.

**Changes:** A second origin, which means the first real CORS failure in this
project, which is worth capturing when it happens. FastAPI's `CORSMiddleware`
with an allow-list containing the Pages origin.

**Why Pages and not the same instance:** the frontend stays up while the
instance is stopped. With stop-when-idle, most visitors will find the API
asleep, so the UI should say so rather than time out. This covers most of the
deferred static failover page below. Serving the UI from the instance instead
would avoid CORS, but would still need HTTPS and would go down with the API.

**Decided 2026-10-11** (reasons in `DECISIONS.md`):

- A folder in this repo; Vite + React + TypeScript, no framework.
- The UI takes `https://stories.gebrial.ca`, the CORS allow-list entry. The API
  moves to `stories-api.gebrial.ca` first; then the `stories` A record becomes a
  CNAME to `gebrial.github.io` and leaves the Caddyfile. `gebrial.ca` is
  verified in GitHub against subdomain takeover.
- The API URL is build-time config, not a literal.
- Failure cases: `/ping` with a short timeout on load, and an "asleep" message
  with the form disabled; one request for the minute-long generation with an
  elapsed counter and an abort ceiling; a plain message on 429; one generic
  error for everything else. The busy guard (R4) adds its own case.

### R3. Auth (OAuth)

**Forces it:** The brief names OAuth. I ordered it after the compose-fold so that
deployment is not waiting on it. It sits after the UI because a sign-in flow
needs somewhere to sign in.

**Changes:** Once users exist, the rate limiter keys on the user rather than the
remote IP, which `DECISIONS.md` already anticipates. The `stories` table gains an
owner column so a signed-in user can see their own past prompts. Any session
cookie must be host-only (no `Domain` attribute), or it reaches every
`gebrial.ca` subdomain and undoes R2's dedicated origin. The CORS allow-list
must also drop `localhost` in production, read from config, or anything on a
signed-in visitor's own machine at that port could read their data.

### R4. Serve concurrent visitors without contention

**Forces it:** Generation is slow, so two visitors at once contend for the same
hardware. `/generate` is a synchronous path operation, so FastAPI already runs it
in a worker threadpool and one process serves several requests at a time. The
constraint is hardware, not process count.

**Measured 2026-09-29 on the `t3.small`:** one story takes about a minute, with
the CPU at 100% throughout. The API container holds 853 MB with the model loaded
and 907 MB while generating, leaving 538 MB `available` on the host. So:

- **A second copy of the model does not fit.** More uvicorn workers are out on
  this instance.
- **Two generations at once do fit in memory** (activations are ~50 MB), since
  the threadpool shares one model. But they split two vCPUs already at 100%, so
  each takes about twice as long. Nobody is served sooner.
- **Sustained 100% CPU spends CPU credits.** A `t3` in unlimited mode bills for
  running above its 20% baseline; in standard mode it throttles to it instead.
  *Set to standard 2026-10-07, before opening the port: with a per-IP rate
  limit, many IPs could hold the CPU at 100%, and slow is a better failure than
  a bill.*

**Next, my own idea, 2026-09-29:** answer "busy" rather than start a second
generation. A process-wide guard around the model: if a generation is already
running, `/generate` returns at once with a distinct status and the UI says to
try again shortly. Unlike the rate limiter, this is global rather than per-IP,
which is where the contention actually is. Undecided: the status code and body,
whether to hold a short queue before refusing, and how the UI (R2) shows it.

**Also the per-request cost itself.** Generation re-runs the whole sequence
through the model for every new token (`components/generatetext.py` keeps no
KV cache), so the work per story grows with the square of its length. A KV cache
would make each new token cost one position rather than all of them, and is
likely the largest single speed-up available on this hardware. It touches the
model's attention code, so it would need checking against known outputs.

**The longer-term fork**, once the above is not enough:

- **More uvicorn workers.** Each worker loads its own copy of the model, so
  memory multiplies by the number of workers and they still share one machine's
  compute. Every piece of in-process state duplicates with them, including the
  rate limiter's counters, which would silently multiply the effective limit.
- **A separate inference service.** One process owns the model and loads it once.
  The API layer then holds no model and no counters, so it becomes genuinely
  stateless and can run as many copies as needed. Requests to the model converge
  on one queue, which is also where batching would go. Costs a second service to
  build and operate, and makes shared storage for the rate limiter mandatory
  rather than optional.

On this instance the first is ruled out by memory, and the second means a second
machine, which the budget does not allow.

Related: both `Open:` flags in `DECISIONS.md`, on the rate limiter and on the
store, are this same question.

### R5. Separate the training code from the serving code

**Forces it:** Decided 2026-09-20. `src/model_service.py` imports `LitGPTModel`
from `scripts/train.py`, so serving reaches into training code and needs a
`sys.path` hack to do it. Importing that module also runs `from datasets import
load_from_disk`, so every API start loads the HuggingFace datasets stack for a
function the request path never calls.

**Changes:** `LitGPTModel`, or a plain-`torch` equivalent, moves into `src/`.
Two further severances become possible once it has:

- the checkpoint could be exported as a raw `state_dict`, which drops
  `lightning` from the serving path entirely. A first step exists: since
  2026-09-29 the API loads a checkpoint stripped of optimizer state
  (`scripts/strip_checkpoint.py`), but it is still a Lightning checkpoint
  loaded through `LitGPTModel`
- `components/attention.py` imports `RotaryPositionalEmbeddings` from
  `torchtune` for that one class, and `torchtune` is what drags in `datasets`,
  `pyarrow` and `pandas`. Vendoring that one implementation severs the chain.
  Rewriting it from scratch instead risks numerical drift against a model
  already trained against torchtune's version, so it would need verifying
  against known prompts.

**Not for image size.** Measured 2026-09-20: `torch` is 531 MB on its own and
irreducible for inference, while the entire severable cluster -- the `datasets`
stack at ~165 MB, `matplotlib` at 34 MB, `lightning` and `torchtune` and
`torchao` at 28 MB between them -- is worth about 200 MB. The reasons are
coupling and start-up memory, the latter mattering on a 2 GB `t3.small` where the
loaded model already takes 853 MB. Related to R4.

### R6. Turn-taking stories

**Forces it:** My own idea, 2026-09-23. A visitor and the model take turns
writing a story: the visitor writes the first sentence, the model the next, and
so on.

**Changes:** Its own page and its own database, kept apart from the feed rather
than folded into it. A turn-taking story is an ordered sequence of turns, each
by the visitor or the model, which the feed's `{prompt, story}` shape does not
describe. Keeping them separate means the feed's shape never has to change for
it.

**Unknowns that will shape it:**

- The model writes until end-of-story, not one sentence. Stopping after a single
  sentence is a generation problem in its own right.
- Where an unfinished story lives between turns, and what happens to one a
  visitor abandons.
- `CONTEXT.md` gains a term for a turn, and "story" needs to cover both kinds.

This is also the revisit condition recorded under GraphQL in "Ruled out" below.

### R7. Wake the instance from the UI

**Forces it:** My own idea, 2026-10-11, from R2. Most visitors find the API
asleep; R2 only says so.

**Changes:** Something always on, with AWS credentials, that starts the
instance, such as a Lambda behind a Function URL, with IAM scoped to that one
instance. The instance is stopped by hand in the console today, so R7 also needs
an automatic idle stop. Worst-case abuse is the instance running all month,
about $15, since CPU credits are standard.

### R8. Stream generated stories

**Forces it:** Decided 2026-10-11, from R2, which waits a minute for one
response.

**Changes:** Generation becomes a generator and the API streams tokens (SSE or
`StreamingResponse`); the UI appends them as they arrive. Errors after the
`200` has gone out must be signalled in the body, and the database write moves
to the end of the stream.

### R9. A countdown on 429

**Forces it:** Decided 2026-10-11, from R2, which shows a plain message.

**Changes:** `headers_enabled=True` on the limiter, which needs a `response:
Response` parameter on limited endpoints, and `Retry-After` in the CORS
`expose_headers`, or JS cannot read it.

---

## Suggested by an agent, not decided

Recovered from build transcripts on 2026-09-14. An agent proposed each of these
and I never said yes or no. They are listed so they stop being invisible, not
because they are planned. Several are ten-second decisions once the deployment is
real.

Three of the original ten were promoted to committed R1 on 2026-09-23, once
putting the UI on GitHub Pages made them necessary: an Elastic IP, opening the
port publicly, and HTTPS on a subdomain of `gebrial.ca`. All three are done
as of 2026-10-11 (see "Done" below).

1. **Store the checkpoint in S3** rather than copying it to the instance with
   `scp`.
2. **Make the instance setup reproducible**, with a setup script or by pushing
   the image to a registry rather than building it on the box.
3. **Tests.** There are none.
4. **A slimmer Docker image.** Measured 2026-09-20: 14.5 GB, of which a single
   `pip install` layer is 8.44 GB. Three separable pieces:
   - **CPU-only torch** -- done 2026-09-20. PyPI's Linux wheel bundles CUDA
     libraries a GPU-less `t3.small` cannot use, while PyPI's Windows wheel is
     already CPU-only. That asymmetry, not any local config, is why the venv is
     small and the image is not. The Dockerfile now installs torch from
     PyTorch's cpu index. Result: 14.5 GB down to 2.91 GB, of which the torch
     layer is 1.13 GB and the requirements layer 815 MB.
   - **Severing `torchtune` and `lightning`** would drop the `datasets` stack
     (~165 MB), but that is committed R5 above and is motivated by coupling
     rather than by size.
   - **A separate requirements file for the API**, without `matplotlib`, which
     nothing on the request path imports. `datasets` cannot be dropped this way,
     per above.

   Multi-stage builds, the original suggestion, are not needed for any of these.
5. **Request queuing or dynamic batching.** The brief asked for "rate limiting or
   request queuing" and rate limiting satisfied that. This overlaps with
   committed R4 above, where batching is one of the things a separate
   inference service enables.
6. **Scope down the IAM policy** from `AdministratorAccess`.
7. **Add SQLAlchemy.** Raw `psycopg` was chosen first. Worth noting that
   `INTERVIEW-HANDOFF.md` treats an ORM as a risk rather than a help here,
   because it hides the SQL that needs to be hand-written at least once.

An eleventh item, reconciling a diverged local and remote `main`, was already
resolved by `bb4a8bc` on 2026-08-28 and is not outstanding.

---

## Ruled out

**GitHub Pages**, decided 2026-09-14. Static only, and this project needs a live
server to generate new stories and a database to store them. Existing stories
could be published statically, but generating them still requires a server
somewhere, so it does not remove the requirement. This rules out Pages as the
host for the whole project, not for the frontend alone: committed R2 puts
the static React UI on Pages, calling the API on the instance.

**GraphQL**, decided 2026-09-14. Not a fit for this project as it stands. The
model continues a story from an opening line rather than holding a conversation.
It would become worth revisiting if the model were expanded into something
conversational, for instance alternating sentences between the user and the
model. *That condition came up 2026-09-23 as committed R6, turn-taking
stories. Whether it changes the answer on GraphQL is not yet decided.*

**Lambda as the deploy target.** Ruled out by an agent early, on cold starts and
the memory a loaded PyTorch model needs. The reasoning is sound and I have not
disputed it, but it is recorded here as the agent's call rather than mine.

**Java, Kotlin and Go as implementation languages**, per the brief: a language
learned for this project produces a backend that cannot be defended under
questioning.

Also ruled out and recorded in `DECISIONS.md` rather than here: usage logging,
and an admin endpoint for moderation.

---

## Deferred, with agreement

**A static failover page** on S3 with a Route 53 health check, so something is
served while the instance is stopped. Agreed as a stretch goal after a plain EC2
deploy works. Mostly covered by committed R2 since 2026-09-23: a UI on Pages
is always served and can report the API as asleep. What would remain is showing
existing stories while the instance is stopped, which would need them published
somewhere static.

---

## Done, and no longer roadmap

**Retraining the model.** The original checkpoint was lost. A new model has been
trained and tested locally through the `/generate` endpoint.

**Folding the API into `docker-compose.yml`**, done 2026-09-21. This was my own
stated prerequisite for deploying before auth. Compose now builds an `api`
service from the `Dockerfile`, with `checkpoints/` and `data/` mounted read-only
because they are deliberately not in the image. It starts only once `db` passes
its `pg_isready` healthcheck, and both services restart `unless-stopped`. The
reasoning is in `DECISIONS.md`.

**A read path for the stories feed**, done 2026-09-28. `GET /stories` returns the
most recent stories, newest first: `{"stories": [...]}` of `prompt`, `story` and
`created_at`, `limit` 20 by default and 50 at most, rate limited at 60/min. The
first hand-written `SELECT` in this project. The shape and its reasons are in
`DECISIONS.md`, "The stories feed".

**Deploying to AWS EC2**, done 2026-09-29. A bare backend deploy to the
`t3.small`, reachable from my own IP only: Docker enabled at boot, the repo
cloned, the checkpoint, tokenizer and `.env` copied across with `scp`, and
`docker compose up -d --build` on the box (4m10s). It came back up on its own
after a stop and start of the instance.

The point was the measurement, and it paid off. The first start was OOM-killed:
the training checkpoint was 1436 MB, two thirds of it optimizer state that
inference never reads. The API now serves a stripped 479 MB checkpoint, with
identical weights, and settles at 853 MB (see `DECISIONS.md`, "Serve a stripped
checkpoint"). What that means for concurrency is under R4.

**R1. Serving the API over HTTPS on a domain**, done 2026-10-11. Forced by the
UI: GitHub Pages serves only over HTTPS, and browsers block an HTTPS page calling
a plain-HTTP API. The API is now at `https://stories.gebrial.ca`:

- **An Elastic IP**, so the address survives a stop and start, and an A record
  for `stories.gebrial.ca` at Squarespace pointing at it. It bills while the
  instance is stopped (`DECISIONS.md`, "An Elastic IP for a stable address").
- **Moderation revisited before opening**, and it stays manual
  (`DECISIONS.md`, the unreviewed-publishing entry). CPU credits set to
  standard, so sustained load throttles rather than bills (R4).
- **Ports 80 and 443 open to anyone**, to Caddy, which gets and renews a Let's
  Encrypt certificate and redirects HTTP to HTTPS (`DECISIONS.md`, "Caddy as
  the reverse proxy for HTTPS"). The API and Postgres ports are bound to
  loopback, and the security group's 8000 rule was removed, so Caddy is the only
  public entry point. SSH stays restricted to my IP.
- **The rate limiter sees real visitors.** uvicorn trusts Caddy's
  `X-Forwarded-For`, verified by the access log showing my public IP rather
  than a container address, and a second `/generate` within a minute getting a
  429.
