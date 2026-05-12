import os
from typing import List, Dict, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from tokenizer.tokenizer import Tokenizer
from model.train import TinyGPT

from app.stats_engine import answer_if_factual

DEFAULT_CKPT = os.environ.get("CKPT", "checkpoints/ckpt_step10000.pt")
TOKENIZER_MODEL = os.environ.get("TOKENIZER_MODEL", "tokenizer/cricket_bpe.model")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_MAX_NEW = int(os.environ.get("MAX_NEW", "180"))
DEFAULT_TEMP = float(os.environ.get("GEN_TEMP", "0.7"))
DEFAULT_TOP_K = int(os.environ.get("GEN_TOP_K", "30"))

SYSTEM_STYLE = (
    "You are CricketGPT, a cricket commentator and match explainer. "
    "Keep outputs concise, realistic, and cricket-accurate in tone."
)

MATCH_TOKEN = "<|match|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"

app = FastAPI(title="CricketGPT")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def index():
    return FileResponse(Path("app/static/index.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None
    ckpt: Optional[str] = None
    max_new: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None

class ChatResponse(BaseModel):
    reply: str

def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = TinyGPT(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model

tokenizer = Tokenizer(TOKENIZER_MODEL)
model = load_checkpoint(DEFAULT_CKPT)

def build_prompt(message: str, history: Optional[List[Dict[str, str]]]):
    lines = []
    lines.append(MATCH_TOKEN)
    lines.append(f"{USER} {SYSTEM_STYLE}")
    lines.append(f"{ASSISTANT} Got it.")
    if history:
        for turn in history[-8:]:
            role = turn.get("role", "")
            content = (turn.get("content", "") or "").strip()
            if not content:
                continue
            if role == "user":
                lines.append(f"{USER} {content}")
            else:
                lines.append(f"{ASSISTANT} {content}")
    lines.append(f"{USER} {message.strip()}")
    lines.append(f"{ASSISTANT} ")
    return "\n".join(lines)

@torch.no_grad()
def generate_text(prompt: str, max_new: int, temperature: float, top_k: int):
    ids = tokenizer.encode(prompt)

    block_size = getattr(model, "block_size", 256)
    if len(ids) > block_size:
        ids = ids[-block_size:]

    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(max_new):
        if x.size(1) > block_size:
            x = x[:, -block_size:]

        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        logits = out[:, -1, :]
        logits = logits / max(float(temperature), 1e-6)

        if top_k > 0:
            k = min(int(top_k), logits.size(-1))
            v, _ = torch.topk(logits, k=k)
            cutoff = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.full_like(logits, -1e10), logits)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

        if x.size(1) > block_size:
            x = x[:, -block_size:]

    decoded = tokenizer.decode(x[0].tolist())

    if ASSISTANT in decoded:
        decoded = decoded.split(ASSISTANT)[-1]
    if MATCH_TOKEN in decoded:
        decoded = decoded.split(MATCH_TOKEN)[0]

    return decoded.strip()

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    global model

    # 1) Stats-first: if this looks like a factual query, answer deterministically
    factual = answer_if_factual(req.message)
    if factual:
        return ChatResponse(reply=factual)

    # 2) Otherwise, fall back to the TinyGPT generator (commentary / explanation)
    ckpt_path = req.ckpt or DEFAULT_CKPT
    if ckpt_path != DEFAULT_CKPT:
        model = load_checkpoint(ckpt_path)

    prompt = build_prompt(req.message, req.history)
    max_new = req.max_new or DEFAULT_MAX_NEW
    temperature = req.temperature if req.temperature is not None else DEFAULT_TEMP
    top_k = req.top_k if req.top_k is not None else DEFAULT_TOP_K

    reply = generate_text(prompt, max_new=max_new, temperature=temperature, top_k=top_k)
    return ChatResponse(reply=reply)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return api_chat(req)

@app.get("/api/health")
def api_health():
    return {
        "ok": True,
        "device": DEVICE,
        "ckpt": DEFAULT_CKPT,
        "block_size": getattr(model, "block_size", None),
        "vocab_size": getattr(model, "vocab_size", None),
        "index_exists": Path("data/processed/match_index.json").exists(),
    }

@app.get("/health")
def health():
    return api_health()
