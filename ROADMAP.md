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

### 0. Fold the API into `docker-compose.yml`

**Forces it:** My own stated prerequisite, 2026-08-28: *"I want add the api
service in the docker compose before tackling auth. that way I can get it
deployed asap once deal with the aws account verification issues."*

**Changes:** One file. Compose currently starts only `db`. It needs an `api`
service built from the `Dockerfile`, `POSTGRES_*` env with `host=db`, and mounts
for `checkpoints/` and `data/`, which are deliberately not in the image.

This is the smallest item here and everything below waits on it.

### 1. Deploy to AWS EC2

**Forces it:** `project-brief-llm-inference-api.md` names AWS as the gap that
appeared most often in the job evaluations, 6 of 31. Nothing below this line can
be measured or built without a running instance.

**Status:** Two launches in August failed with "This account is currently blocked
and not recognized as a valid account." A verification email has since arrived.
Confirming the account is unblocked is the next action, 2026-09-14.

**Constraint already decided:** budget is $5/month and a `t3.small` is about
$15/month, so the instance stops when idle rather than running continuously.

### 2. A read path for the story cache

**Forces it:** The `stories` table exists so visitors can be shown stories that
already exist. Nothing reads it today, so that purpose is not yet code. This
follows from the table's recorded purpose in `DECISIONS.md` rather than being a
separately stated goal.

**Changes:** A `GET` endpoint returning recent stories, ordered by `created_at`,
and the first hand-written `SELECT` in this project.

### 3. Auth (OAuth)

**Forces it:** The brief names OAuth. I ordered it after the compose-fold so that
deployment is not waiting on it.

**Changes:** Once users exist, the rate limiter keys on the user rather than the
remote IP, which `DECISIONS.md` already anticipates. The story cache gains an
owner column so a signed-in user can see their own past prompts.

### 4. A React UI in front of the API

**Forces it:** Recorded in `INTERVIEW-HANDOFF.md` as a decision taken 2026-09-12.
It supersedes the brief's line about frontend being out of scope, which was
written earlier.

**Changes:** A second origin, which means the first real CORS failure in this
project, which is worth capturing when it happens.

### 5. Serve concurrent visitors without contention

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
under two concurrent generations. That cannot be measured before item 1, which is
why this sits below it.

Related: both `Open:` flags in `DECISIONS.md`, on the rate limiter and on the
store, are this same question.

---

## Suggested by an agent, not decided

Recovered from build transcripts on 2026-09-14. An agent proposed each of these
and I never said yes or no. They are listed so they stop being invisible, not
because they are planned. Several are ten-second decisions once the deployment is
real.

1. **Elastic IP** so the address survives a stop and start. Interacts with the
   stop-when-idle decision above.
2. **Open port 80 publicly.** The security group rule was kept restricted to my
   own IP until rate limiting existed. Rate limiting now exists, so this is
   actionable.
3. **HTTPS and a subdomain of `gebrial.ca`** via Route 53, with nginx in front
   for TLS. My underlying preference for project pages living under my own domain
   is real; this particular shape is the agent's.
4. **Store the checkpoint in S3** rather than copying it to the instance with
   `scp`.
5. **Make the instance setup reproducible**, with a setup script or by pushing
   the image to a registry rather than building it on the box.
6. **Tests.** There are none.
7. **A slimmer multi-stage Docker image.**
8. **Request queuing or dynamic batching.** The brief asked for "rate limiting or
   request queuing" and rate limiting satisfied that. This overlaps with item 5
   above, where batching is one of the things a separate inference service
   enables.
9. **Scope down the IAM policy** from `AdministratorAccess`.
10. **Add SQLAlchemy.** Raw `psycopg` was chosen first. Worth noting that
    `INTERVIEW-HANDOFF.md` treats an ORM as a risk rather than a help here,
    because it hides the SQL that needs to be hand-written at least once.

An eleventh item, reconciling a diverged local and remote `main`, was already
resolved by `bb4a8bc` on 2026-08-28 and is not outstanding.

---

## Ruled out

**GitHub Pages**, decided 2026-09-14. Static only, and this project needs a live
server to generate new stories and a database to store them. Existing stories
could be published statically, but generating them still requires a server
somewhere, so it does not remove the requirement.

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
deploy works.

---

## Done, and no longer roadmap

**Retraining the model.** The original checkpoint was lost. A new model has been
trained and tested locally through the `/generate` endpoint.
