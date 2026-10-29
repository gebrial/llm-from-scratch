import json
import logging
import os
import queue
import threading

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Annotated
from datetime import datetime
from model_service import ModelService
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

logger = logging.getLogger("uvicorn.error")

limiter = Limiter(key_func=get_remote_address)
# /generate and /generate/stream share one budget: both use the same CPU, and
# separate limits would double the effective one.
GENERATE_LIMIT = "6/minute"
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
# The UI is a different origin from the API, so the browser hides the API's
# responses from it unless they name its origin. Exact origins only: "*" would
# let any website read them, and cannot be combined with cookies (R3).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://stories.gebrial.ca",  # the UI on GitHub Pages
        "http://localhost:5173",  # the Vite dev server, pinned in web/vite.config.ts
    ],
    # The default is GET only. POST /generate sends JSON, which makes the
    # browser ask first with a preflight OPTIONS request; Content-Type is
    # always allowed by Starlette, so the method is the only thing to add.
    allow_methods=["GET", "POST"],
)

# Neither the checkpoint nor the tokenizer is baked into the image -- the
# Dockerfile copies only src/ and scripts/, so in a container both of these
# resolve against mounted volumes and cannot be hardcoded to a local path.
CHECKPOINT_PATH = os.environ["CHECKPOINT_PATH"]
# the default value for the tokenizer json is from the script that generates it at:
# scripts\prepare_tokenizer.py
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "data/tokenizer.json")

model_service = ModelService(CHECKPOINT_PATH, TOKENIZER_PATH)

# should use connection pool at scale
db_conn = psycopg.connect(
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=os.environ.get("POSTGRES_PORT", "5432"),
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
    dbname=os.environ["POSTGRES_DB"],
    autocommit=True,
)
db_conn.execute("""
    CREATE TABLE IF NOT EXISTS stories (
        id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        prompt TEXT NOT NULL,
        story TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
""")


# db_conn is a psycopg connection -- e.g. db_conn.execute("SQL...", (params,))
def save_story(prompt, story):
    db_conn.execute(
        "INSERT INTO stories (prompt, story) VALUES (%s, %s)",
        (prompt, story)
    )

@app.get("/ping")
def ping():
    return {"status": "ok"}

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
@limiter.shared_limit(GENERATE_LIMIT, scope="generate")
def generate(request: Request, req: GenerateRequest) -> str:
    story = model_service.generate(req.prompt)
    save_story(req.prompt, story)
    return story


def generate_and_save(prompt, messages: queue.Queue):
    """
    Runs in its own thread, so generation and the save finish even if the client
    disconnects: StreamingResponse stops reading on disconnect, but nothing here
    depends on it reading. Only a finished story is stored, and "done" is sent
    after the save, so it means the story is in the feed.
    """
    try:
        pieces = []
        for piece in model_service.stream(prompt):
            pieces.append(piece)
            messages.put({"type": "token", "text": piece})
        save_story(prompt, model_service.decoded_prompt(prompt) + "".join(pieces))
        messages.put({"type": "done"})
    except Exception:
        # the exception can name paths or SQL, so the client only gets a fixed message
        logger.exception("streamed generation failed")
        messages.put({"type": "error", "message": "generation failed"})


@app.post("/generate/stream")
@limiter.shared_limit(GENERATE_LIMIT, scope="generate")
def generate_stream(request: Request, req: GenerateRequest) -> StreamingResponse:
    """The continuation as NDJSON: a "token" line per token, then "done" or "error"."""
    messages: queue.Queue = queue.Queue()
    threading.Thread(target=generate_and_save, args=(req.prompt, messages)).start()

    def ndjson_lines():
        while True:
            message = messages.get()
            yield json.dumps(message) + "\n"
            if message["type"] != "token":
                return

    return StreamingResponse(ndjson_lines(), media_type="application/x-ndjson")

class Story(BaseModel):
    prompt: str
    story: str
    created_at: datetime

class StoriesResponse(BaseModel):
    stories: list[Story]

@app.get("/stories")
@limiter.limit("60/minute")
def stories(
    request: Request,
    limit: Annotated[int, Query(
        ge=1,
        le=50
    )] = 20
) -> StoriesResponse:
    with db_conn.cursor(row_factory=dict_row) as cur:
        query = """
        SELECT prompt, story, created_at
        FROM stories 
        ORDER BY created_at DESC, id DESC 
        LIMIT %s;
        """
        cur.execute(query, [limit])
        rows = cur.fetchall()
        return StoriesResponse(stories=[Story(**row) for row in rows])

