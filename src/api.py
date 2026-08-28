import os

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from model_service import ModelService
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

model_service = ModelService("checkpoints/epoch=1-step=7042.ckpt", "data/tokenizer.json")

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
        story TEXT NOT NULL
    );
""")


# db_conn is a psycopg connection -- e.g. db_conn.execute("SQL...", (params,))
def log_generation(prompt, story):
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
@limiter.limit("5/minute")
def generate(request: Request, req: GenerateRequest):
    story = model_service.generate(req.prompt)
    log_generation(req.prompt, story)
    return story