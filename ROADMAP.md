# Roadmap

Work that is not done yet. Decisions already made live in `DECISIONS.md`; this
file is the other half, and the two point at each other.

Each item says **what forces it** and **what it changes**, so that nothing sits
here as a vague intention.

**Why there are two lists.** This project has been built with an AI agent, and a
lot of what looked like a plan turned out to be agent suggestions that were never
ruled on. Recovered from the build transcripts on 2026-09-14: four items I had
actually committed to, and ten an agent proposed that I never answered. Keeping
them apart means nothing in the first list is something I did not choose. Same
reason `DECISIONS.md` marks an entry UNEXAMINED rather than inventing a reason
for it.

---

## Committed

**Order, decided 2026-09-23.** The read path comes first because it is small,
local, and does not need the instance. The deploy comes before the UI not
because there is anything to visit yet, but because it is the measurement: a
2 GB `t3.small` holding torch and the model may not fit, and the answer could
change the instance size, the budget, or the priority of item 7. Better learned
before a UI is built on top of it. HTTPS comes before the UI because the UI
cannot call the API without it (item 3).

### 1. A read path for the story cache

**Forces it:** The `stories` table exists so visitors can be shown stories that
already exist. Nothing reads it today, so that purpose is not yet code. This
follows from the table's recorded purpose in `DECISIONS.md` rather than being a
separately stated goal.

**Changes:** A `GET` endpoint returning recent stories, ordered by `created_at`,
and the first hand-written `SELECT` in this project.

### 2. Deploy to AWS EC2

**Forces it:** `project-brief-llm-inference-api.md` names AWS as the gap that
appeared most often in the job evaluations, 6 of 31. Several items below cannot
be measured or built without a running instance.

**Status:** The account is active and a `t3.small` has been created, 2026-09-23.
The two August launches that failed with "This account is currently blocked" are
resolved.

**Changes:** A bare backend deploy: Docker on the instance, enabled at boot so
`restart: unless-stopped` survives a stop and start, the repo cloned, the
checkpoint and tokenizer copied across, and `docker compose up`. Reachable from
my own IP only. The point is to measure the container's memory with the model
loaded.

**Constraint already decided:** budget is $5/month and a `t3.small` is about
$15/month, so the instance stops when idle rather than running continuously.
Storage is billed while stopped: the 30 GB root volume is roughly $2.40/month,
about half the budget, against about 5 GB actually needed.

### 3. Serve the API over HTTPS on a domain

**Forces it:** Item 4 puts the UI on GitHub Pages, which serves only over HTTPS,
and browsers block an HTTPS page from calling a plain-HTTP API as mixed content.
So the API needs a domain and a TLS certificate before the UI can reach it.

**Changes:** A subdomain of `gebrial.ca` pointing at the instance, with a reverse
proxy in front of uvicorn terminating TLS. Three things follow:

- **An Elastic IP**, because the public IP changes on every stop and start and a
  DNS record needs a stable target. An Elastic IP is billed while its instance
  is stopped, which counts against the budget.
- **The port opens publicly**, since visitors' browsers call the API directly.
  Rate limiting exists, which was the condition for opening it.
- **The rate limiter must see the real client IP.** Behind a proxy every request
  arrives from the proxy's address, so the limiter would treat all visitors as
  one and share 5/min between them. uvicorn needs `--proxy-headers` and to trust
  the proxy's forwarded address.

All three were agent suggestions, promoted here 2026-09-23. The list below notes
where they went.

### 4. A React UI on GitHub Pages

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

### 5. Auth (OAuth)

**Forces it:** The brief names OAuth. I ordered it after the compose-fold so that
deployment is not waiting on it. It sits after the UI because a sign-in flow
needs somewhere to sign in.

**Changes:** Once users exist, the rate limiter keys on the user rather than the
remote IP, which `DECISIONS.md` already anticipates. The story cache gains an
owner column so a signed-in user can see their own past prompts.

### 6. Serve concurrent visitors without contention

**Forces it:** Generation is slow, so two visitors at once contend for the same
hardware. Not urgent: `/generate` is a synchronous path operation, so FastAPI
already runs it in a worker threadpool and one process serves several requests at
a time. The constraint is hardware, not process count.

**This is a fork, not a plan, and it waits on a measurement.** Two ways to get
there:

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

**The fact that decides it** is what the deployed instance's memory and GPU do
under two concurrent generations. That cannot be measured before item 2, which is
why this sits below it.

Related: both `Open:` flags in `DECISIONS.md`, on the rate limiter and on the
store, are this same question.

### 7. Separate the training code from the serving code

**Forces it:** Decided 2026-09-20. `src/model_service.py` imports `LitGPTModel`
from `scripts/train.py`, so serving reaches into training code and needs a
`sys.path` hack to do it. Importing that module also runs `from datasets import
load_from_disk`, so every API start loads the HuggingFace datasets stack for a
function the request path never calls.

**Changes:** `LitGPTModel`, or a plain-`torch` equivalent, moves into `src/`.
Two further severances become possible once it has:

- the checkpoint could be exported as a raw `state_dict`, which drops
  `lightning` from the serving path entirely
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
coupling and start-up memory, the latter mattering on a 2 GB `t3.small` already
holding a 1.4 GB checkpoint. Related to item 6.

---

## Suggested by an agent, not decided

Recovered from build transcripts on 2026-09-14. An agent proposed each of these
and I never said yes or no. They are listed so they stop being invisible, not
because they are planned. Several are ten-second decisions once the deployment is
real.

Three of the original ten were promoted to committed item 3 on 2026-09-23, once
putting the UI on GitHub Pages made them necessary: an Elastic IP, opening the
port publicly, and HTTPS on a subdomain of `gebrial.ca`.

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
     (~165 MB), but that is committed item 7 above and is motivated by coupling
     rather than by size.
   - **A separate requirements file for the API**, without `matplotlib`, which
     nothing on the request path imports. `datasets` cannot be dropped this way,
     per above.

   Multi-stage builds, the original suggestion, are not needed for any of these.
5. **Request queuing or dynamic batching.** The brief asked for "rate limiting or
   request queuing" and rate limiting satisfied that. This overlaps with
   committed item 6 above, where batching is one of the things a separate
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
host for the whole project, not for the frontend alone: committed item 4 puts
the static React UI on Pages, calling the API on the instance.

**GraphQL**, decided 2026-09-14. Not a fit for this project as it stands. The
model continues a story from an opening line rather than holding a conversation.
It would become worth revisiting if the model were expanded into something
conversational, for instance alternating sentences between the user and the
model.

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
deploy works. Mostly covered by committed item 4 since 2026-09-23: a UI on Pages
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
