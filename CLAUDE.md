# llm-from-scratch

A ~120M-parameter GPT trained from scratch on TinyStories, served by a FastAPI
API with Postgres, in Docker Compose on AWS EC2. It is a portfolio project: it
exists to close specific skill gaps (API, Postgres, Docker, AWS, and a React
frontend), so the person building it has to be able to explain every part of it.

## Read these first

- `ROADMAP.md`: what is next. Committed items have permanent IDs (`R1`, `R2`, …),
  order lives in the "Next:" line, and an ID is never reused or renumbered.
- `DECISIONS.md`: why things are the way they are. Read before changing anything
  it covers, and add an entry for any new non-obvious choice, in its format.
- `CONTEXT.md`: the glossary. Use its terms (prompt, continuation, story, feed,
  visitor, user) and avoid the ones it lists.
- `INTERVIEW-HANDOFF.md`: why this project exists and why the React UI matters.
- `README.md`: running it locally and with Compose.

## The API

Live at `https://stories.gebrial.ca`. Routes are in `src/api.py`; `/docs` has the
generated OpenAPI page.

- `GET /ping` → `{"status": "ok"}`. Not rate limited.
- `POST /generate` with `{"prompt": "..."}` → the story as a **bare JSON string**,
  not an object. The story begins with the prompt. Takes about a minute on the
  instance, at 100% CPU. Rate limited to 1/min per IP.
- `GET /stories?limit=20` → `{"stories": [{"prompt", "story", "created_at"}]}`,
  newest first; `limit` 1–50. Rate limited to 60/min per IP.
- Over a limit: **429** with `{"error": "Rate limit exceeded: 1 per 1 minute"}`
  (slowapi's default body).
- No CORS configured yet.

## Deployment

- One `t3.small` (2 GB RAM, 2 vCPU, CPU credits in standard mode) running
  `docker-compose.yml`: `caddy` (TLS, the only public entry point, on 80/443),
  `api` and `db`, both bound to `127.0.0.1` on the host.
- The instance is **stopped when idle**, so the API is often unreachable. It has
  an Elastic IP, and everything comes back on start.
- Deploying: commit and push, SSH in, `git pull`, `docker compose up -d --build`.
  The instance only gets code through git.
- The API serves a stripped `*-inference.ckpt` checkpoint; the full training
  checkpoint OOMs the instance (`DECISIONS.md`).
- Caddy's certificate lives in the `caddy_data` volume. Do not suggest
  `docker compose down -v` or repeatedly recreating `caddy` while it is failing:
  Let's Encrypt issues at most 5 duplicate certificates a week.

## Conventions

- Commits use Conventional Commits. Commit only when asked.
- Never print or ask to see `.env` or `docker compose config` output (it
  contains the database password). `loginInstructions.txt` is gitignored and
  must never be committed.
- `study/` is gitignored and is its own local git repo for review notes. Never
  add it to this repo.
