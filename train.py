"""Train the tiny GPT on a character corpus and sample from it.

    # Default: small model on tiny-shakespeare, CPU-friendly:
    python train.py --steps 2000

    # Quick smoke run:
    python train.py --steps 100 --eval-interval 50

Writes a loss curve, a text sample, and the trained weights to reports/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.minigpt import GPT, GPTConfig, CharDataset, load_text


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, device, iters=50, generator=None):
    """Average loss over a few random batches of train and val."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = dataset.get_batch(split, batch_size, device, generator)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    torch.manual_seed(1337)
    gen = torch.Generator().manual_seed(1337)

    text = load_text(args.data)
    dataset = CharDataset(text, block_size=args.block_size)
    cfg = GPTConfig(
        vocab_size=dataset.tokenizer.vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
    )
    model = GPT(cfg).to(device)
    print(f"corpus={len(text):,} chars  vocab={cfg.vocab_size}  "
          f"params={model.num_params():,}  device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = {"step": [], "train": [], "val": []}

    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(model, dataset, args.batch_size, device, generator=gen)
            history["step"].append(step)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            print(f"step {step:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}")

        x, y = dataset.get_batch("train", args.batch_size, device, gen)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Sample some text.
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = dataset.tokenizer.decode(
        model.generate(ctx, max_new_tokens=500, temperature=0.8, top_k=40)[0].tolist()
    )
    print("\n----- sample -----\n" + sample + "\n------------------")

    (args.output_dir / "sample.txt").write_text(sample, encoding="utf-8")
    _plot_loss(history, args.output_dir / "loss_curve.png")
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "params": model.num_params(), "vocab_size": cfg.vocab_size,
        "steps": args.steps, "final_val_loss": history["val"][-1],
        "config": vars(cfg),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved sample, loss curve, model + metrics to {args.output_dir}/")
    return 0


def _plot_loss(history, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["step"], history["train"], label="train")
    ax.plot(history["step"], history["val"], label="val")
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Tiny GPT training")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
