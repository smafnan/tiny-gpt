"""Train a compact char-GPT and save a self-contained bundle for the web demo.

Saves web_model/gpt.pt = {config, state_dict, chars} so api.py can rebuild the
model + tokenizer and generate instantly (no training at request time). This is a
one-time build; the resulting file is committed so the playground runs from a clone.

    python build_web_model.py
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from minigpt import GPT, GPTConfig, CharDataset, load_text

STEPS = 1500
BLOCK = 64
BATCH = 32
LR = 3e-4


def main() -> None:
    text = load_text("data/input.txt")
    ds = CharDataset(text, block_size=BLOCK)
    cfg = GPTConfig(vocab_size=ds.tokenizer.vocab_size, block_size=BLOCK,
                    n_layer=4, n_head=4, n_embd=128, dropout=0.1)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(1337)

    model.train()
    for step in range(STEPS + 1):
        x, y = ds.get_batch("train", BATCH, "cpu", gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}", flush=True)

    chars = "".join(ds.tokenizer.itos[i] for i in range(ds.tokenizer.vocab_size))
    out = Path("web_model"); out.mkdir(exist_ok=True)
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict(),
                "chars": chars}, out / "gpt.pt")
    print(f"Saved web_model/gpt.pt (vocab {cfg.vocab_size}, "
          f"{model.num_params():,} params)")


if __name__ == "__main__":
    main()
