# Context

A small language model trained from scratch on children's stories, served behind
an API so that visitors can give it an opening line and read what it writes. This
file is the glossary: what the words mean. `DECISIONS.md` holds why things are
the way they are, and `ROADMAP.md` holds what comes next.

## Language

### Stories

**Prompt**:
The opening line a visitor gives the model to continue.
_Avoid_: input, query, seed

**Continuation**:
The text the model writes after the prompt.
_Avoid_: completion, output, generation

**Story**:
A prompt together with its continuation, as a reader sees it: the visitor's words
first, then the model's.
_Avoid_: result, response

**Feed**:
The most recent stories, newest first, shown to anyone who visits.
_Avoid_: cache, history, gallery

### People

**Visitor**:
Anyone using the site. Visitors are anonymous.
_Avoid_: client, guest

**User**:
A visitor who has signed in. There are no users until auth exists.
_Avoid_: account, member
