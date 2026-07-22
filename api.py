"""FastAPI backend for the Tiny GPT web playground.

Loads the committed web_model/gpt.pt bundle (config + weights + tokenizer) and
generates text from a prompt — no training at request time. Serves the built
React frontend.

Run:  uvicorn api:app --reload  →  http://localhost:8000
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from minigpt import GPT, GPTConfig, CharTokenizer

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "web_model" / "gpt.pt"

app = FastAPI(title="Tiny GPT API", version="1.0.0")

# Allowed origins for the web playground; comma-separated, defaults cover local dev
# (Vite dev server + the FastAPI-served build) so `uvicorn api:app` works with no env set.
_default_origins = "http://localhost:5173,http://localhost:8000"
_allowed_origins = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")
    if o.strip()
]
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins, allow_methods=["*"],
                   allow_headers=["*"])

_model: GPT | None = None
_tok: CharTokenizer | None = None
_cfg: GPTConfig | None = None


def _load():
    global _model, _tok, _cfg
    if _model is not None or not BUNDLE.exists():
        return
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=True)
    _cfg = GPTConfig(**bundle["config"])
    _tok = CharTokenizer(bundle["chars"])      # same sorted vocab as training
    _model = GPT(_cfg)
    _model.load_state_dict(bundle["state_dict"])
    _model.eval()


@app.on_event("startup")
def startup():
    _load()


class GenRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 400
    temperature: float = 0.8
    top_k: int = 40


@app.get("/api/info")
def info():
    _load()
    return {"ready": _model is not None,
            "params": _model.num_params() if _model else 0,
            "vocab": _cfg.vocab_size if _cfg else 0,
            "block_size": _cfg.block_size if _cfg else 0}


@app.post("/api/generate")
def generate(req: GenRequest):
    _load()
    if _model is None:
        return {"ok": False,
                "error": "Model not built. Run: python build_web_model.py"}
    # Keep only characters the tokenizer knows; fall back to a newline seed.
    seed = "".join(ch for ch in req.prompt if ch in _tok.stoi) or "\n"
    idx = torch.tensor([_tok.encode(seed)], dtype=torch.long)
    out = _model.generate(
        idx,
        max_new_tokens=max(1, min(req.max_new_tokens, 1000)),
        temperature=max(0.1, min(req.temperature, 2.0)),
        top_k=max(1, min(req.top_k, _cfg.vocab_size)),
    )
    text = _tok.decode(out[0].tolist())
    return {"ok": True, "text": text, "prompt": seed}


_dist = ROOT / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="web")
