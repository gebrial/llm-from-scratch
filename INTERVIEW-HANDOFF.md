# Handoff: what this project needs to produce for the Oct 9 interview

**Written 2026-09-12, from the `learnReact` interview-prep workspace.**
**Audience: whoever (agent or human) is working on the LLM inference API build.**

This file does not tell you how to build the service. `project-brief-llm-inference-api.md`
already does that, and it is still the spec. This file says what the build has to
*throw off* along the way so that it is usable in an interview on **Fri Oct 9, 2026** —
and names one line in that brief which is now superseded.

---

## The situation this project is being asked to serve

Sanjaya sat a technical interview on 2026-09-04. It did not go well and the
interviewer offered a second attempt about a month later. Round 2 is **Fri Oct 9**,
same interviewer, informal format.

A claim-defensibility audit of his résumé graded **React RED — a total freeze.**
His words: *"I don't remember any specifics about my react work."* About a year on
PressReader's browser-based newspaper reader, ending Dec 2024, and **not recoverable**:
the code is proprietary, it is absent from his public repos, and none of his recent
projects use React. The claim sits on his résumé with nothing behind it he can reach.

**SQL is also graded RED.** Nine of eleven database items came back blank.
So did most of a web-fundamentals cluster: REST, CORS, status codes, HTTP verbs,
Node's concurrency model, middleware. And an entire security/auth category —
authentication vs. authorization, sessions and tokens, password storage, TLS, OAuth —
seven items, seven blanks.

**This project is the vehicle for all of it at once.** Postgres, Docker, OAuth, REST,
AWS, and now a React frontend. That is why it matters to the interview and not only
to the job hunt: it is the only route by which those categories get *built* rather
than *swept*.

---

## The one line in the brief that is superseded

`project-brief-llm-inference-api.md` currently says, under **Explicitly out of scope**:

> *"No significant new frontend work required. A minimal demo page is fine if useful,
> but frontend is not the gap this project is meant to close."*

**That is no longer true, and it was written before the React audit landed.**
Sanjaya decided on 2026-09-12 to deploy the API and then **build and deploy a React UI
in front of it.** Frontend is now one of the two things this project exists to close.

Leave the rest of the brief alone — the stack constraints, the AWS/Postgres/Docker
targets and the out-of-scope list for the model itself all still hold. Just do not let
that one sentence argue against the UI later.

---

## What he needs more practice with, ranked

This is the honest list, drawn from the audit and from the Round 1 post-mortem.
**Ranked by what an interviewer is actually likely to reach for**, not by size of gap.

1. **Reading React and spotting a bug by eye.** Round 1's `useEffect` question took
   hint after hint and produced half an answer. The interviewer has pre-announced that
   he will ask *another* React question, and a second round with a calibrated read on
   the candidate does not get easier. This is the one thing guaranteed to be examined.
2. **Saying what things are.** The diagnosis from the whole prep cycle is
   *deep procedural knowledge, almost no declarative packaging* — he can do all of this
   and cannot say any of it. Definition questions ("what is an object?", "what is
   polymorphism?") went badly in Round 1 on concepts he uses daily.
3. **SQL written by hand.** A `SELECT` with joins, grouping and ordering, from memory.
   **An ORM will hide this, so the ORM is a risk to him here, not a help.** If this
   project uses one, he should have hand-written the equivalent query at least once
   for every non-trivial thing the ORM does.
4. **HTTP and the web platform.** Verbs, status codes, what REST actually constrains,
   CORS as a real failure he has debugged rather than a definition, cookies vs. storage,
   middleware.
5. **Auth as mechanism, not vocabulary.** What a session actually is, what is in a token,
   how a password is stored and why, what OAuth is delegating and to whom.
6. **Node itself.** He has shipped real Node tooling without ever being told what Node
   *is* — the runtime, why JavaScript outside a browser was a novelty, how one thread
   serves many requests.

---

## What the build must produce, beyond working software

### 1. A decision log — and this is the part that is easy to skip and expensive to lose

The interviewer is **fine with AI agents** — he said so, unprompted: *"that's how it's
done these days."* So nothing here is about hiding how the code was written. The
follow-ups will be about **judgment**: which decisions were yours, what alternatives
you considered, what trade-offs you accepted.

**Those answers have to be real, which means they have to be recorded as they happen.**
Reconstructing them in October produces exactly the hollow, plausible-sounding answer
this interviewer probes past.

So: from the next commit onward, **every non-obvious choice gets two lines.**

```
## <date> — <the decision>
Chose: ...
Considered: ...
Why: ...
```

Keep it in the repo. Three lines, at the moment of choosing. Candidates that will
almost certainly come up: Node vs. Python for the API layer; why Postgres rather than
anything simpler for usage logging; what the rate limiter counts and where its state
lives; what OAuth is actually protecting here; why this AWS deployment shape and not
a simpler one; what is in the Docker image and what deliberately is not.

**A warning from elsewhere in his portfolio, because it is the exact failure to avoid:**
`clubhouseBookings` is a substantial Next.js/React app of his — and it is **one single
commit with no ADRs and no decision history.** As interview evidence that is a
liability rather than an asset, because there is no way to answer "what did *you*
decide?" from it. Do not let this project end up in the same state.

### 2. A React UI that earns its patterns honestly

**His constraint, stated explicitly, and it is the right one:**

> *"I don't want to make a decision on the app solely because I need to talk about it
> during an interview. The decision needs to be justifiable in the context of the
> application itself."*

Respect that literally. A decision made *for* an interview is one he would have to
defend dishonestly, and this interviewer probes judgment. What follows is therefore
**not a checklist to satisfy.** It is four patterns he will be examined on, each with
an honest assessment of whether this app actually calls for it.

| Pattern | Honest justification in *this* app |
|---|---|
| **Controlled form** for the prompt | **Strong.** The prompt value has to be in state to validate it, disable submit while empty or in flight, and show length against the model's context limit. There is no simpler correct way to do this. |
| **Fetch with cleanup** (abort / ignore-flag) | **Strongest of the four, and it is the Round 1 question in his own code.** Inference is slow. A user who edits the prompt and resubmits will get the *older* response overwriting the newer one unless the in-flight request is cancelled or its result ignored. That is a real bug in this app, not a contrivance. |
| **List with keys** for generation history | **Strong, given the brief already persists request/usage logging to Postgres.** A history view is the natural read side of data the service is storing anyway. Keys must come from the row id, not the array index — and he should be able to say why. |
| **Custom Hook** wrapping the API call | **Weak, and say so rather than forcing it.** With a single caller this is premature abstraction, and shipping it anyway is precisely the kind of decision that collapses under "why did you do that?". It earns its place only if a second caller appears, or if streaming, abort and error state genuinely tangle inside the component. **If neither happens, leave it out** — he can type that pattern in a drill instead. |

Note the asymmetry: three of these are things the app wants regardless, and one is not.
Building the three and honestly declining the fourth is a *better* interview story than
building all four, because "I considered extracting a hook and it wasn't earning
its keep yet" is itself a judgment answer.

### 3. Things to notice out loud while building, and write down

These cost nothing to capture in the moment and are near-impossible to reconstruct:

- **The first CORS failure.** It will happen the moment the UI and API are on different
  origins. Write down what the browser actually said, what was wrong, and what fixed it.
  A debugged CORS error is worth more in an interview than any definition of CORS.
- **Any query the ORM generated that surprised you.** Especially an N+1. Log the SQL.
- **The status codes you actually chose**, and why that one and not a neighbour.
- **What broke on deploy that worked locally**, and why. This is the single richest
  source of "tell me about a hard problem" material, and it is always thrown away.
- **Anything you had to look up twice.** That is a gap announcing itself.

---

## What this project is *not* responsible for

- **It does not carry the study plan.** The build runs on its own time. The prep
  workspace spends only two blocks on it: one to write this handoff and set up the log,
  one late block to rehearse talking about it. **If the build slips, the study plan is
  unaffected** — that isolation is deliberate and should not be quietly undone by
  letting build work eat study blocks.
- **It does not need to be finished to be useful.** A deployed API with a half-built UI
  and a good decision log is better interview material than a finished app he cannot
  explain. If time runs short, **protect the log and the deployment, not the feature set.**
- **It does not need new ML work.** The model is done. The brief is right about that.

---

## Standing rules carried over from the prep workspace

These exist because they were each learned the hard way. They bind anything written
into this repo that Sanjaya will read or rehearse from.

1. **Never show a wrong answer.** Not his, not an invented plausible one, not as a
   contrast or a "common mistake" callout. Spaced exposure to a wrong phrasing installs
   it, and it surfaces under stress. State what a *correct* answer must contain instead.
   This was his own instruction, emphatic and unprompted.
2. **Use the canonical term from the primary source, every time, with no casual
   synonyms** — including in asides and example phrasings. A synonym written for variety
   is one he may say back under pressure. This has already happened once and cost him.
3. **Never add a claim he did not make.** An agent-suggested specific — a framework
   name, a number, who built what — does not become a fact by going unrebutted. Four
   fabrications were caught during his résumé rewrite. If it cannot be sourced from the
   code or from something he actually said, it does not go in.
4. **He is not a beginner, but the depth is uneven.** Six-plus years professional, **most of
   it TypeScript**. **React was a small part** — about a year, ending with his last role,
   and the audit grades it RED. **Node** he has shipped real tooling in without ever being
   taught what the runtime is. His last professional role ended **Dec 2024**. Explanations
   should be dense and precise, not simplified — but do not assume React or Node fluency
   from the TypeScript, which is the gap this project is being pointed at.
5. **Prefer his wording over yours.** When he rephrases something, the concrete clause
   survives and the discriminating one tends to drop — so name the missing clause rather
   than rolling back his sentence. His rewrites have been better than the reference
   wording essentially every time he has tried.

---

## Where the rest of this lives

- **The study plan:** `../learnReact/reference/thirty-day-plan.html`
- **The mission and constraints:** `../learnReact/MISSION.md`
- **Why things are the way they are:** `../learnReact/learning-records/` — 24 records,
  each one a finding that changed something. `0010` is the Round 1 post-mortem;
  `0011` is the core diagnosis; `0013` is the delivery finding that outranks the
  knowledge findings.
- **The résumé's grillable surface:** `../learnReact/reference/resume-attack-surface.html`
