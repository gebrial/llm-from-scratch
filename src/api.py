from fastapi import FastAPI
from pydantic import BaseModel
from model_service import ModelService

app = FastAPI()

model_service = ModelService("checkpoints/epoch=1-step=7042.ckpt", "data/tokenizer.json")


@app.get("/ping")
def ping():
    return {"status": "ok"}

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate(req: GenerateRequest):
    story = model_service.generate(req.prompt)
    return story