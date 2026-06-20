"""Tests for the tiny GPT.

The decisive correctness test is **causal masking**: changing a token at position
t must not change the model's output at any position < t. If it does, information
is leaking from the future and the language model is invalid.
"""

from __future__ import annotations

import torch

from minigpt import GPT, GPTConfig, CharDataset, CharTokenizer


def _tiny_cfg(vocab=16, block=12) -> GPTConfig:
    return GPTConfig(vocab_size=vocab, block_size=block, n_layer=2,
                     n_head=2, n_embd=16, dropout=0.0)


# --- tokenizer / data ------------------------------------------------------ #

def test_tokenizer_roundtrip():
    tok = CharTokenizer("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"
    assert tok.vocab_size == len("".join(sorted(set("hello world"))))


def test_batch_targets_are_inputs_shifted_by_one():
    ds = CharDataset("abcdefghijklmnopqrstuvwxyz" * 10, block_size=8)
    g = torch.Generator().manual_seed(0)
    x, y = ds.get_batch("train", batch_size=4, generator=g)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    # y[:, :-1] should equal x[:, 1:] (next-character targets).
    assert torch.equal(x[:, 1:], y[:, :-1])


# --- model shapes ---------------------------------------------------------- #

def test_forward_shapes_and_loss():
    cfg = _tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_weight_tying():
    model = GPT(_tiny_cfg())
    # Output head and token embedding must be the same tensor.
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


# --- the critical causal-mask test ----------------------------------------- #

def test_causal_mask_no_future_leakage():
    torch.manual_seed(0)
    cfg = _tiny_cfg(block=10)
    model = GPT(cfg).eval()

    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        logits_a, _ = model(x)

    # Change the LAST token only.
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_b, _ = model(x2)

    # All positions before the last must be byte-for-byte identical...
    assert torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-6)
    # ...and the last position SHOULD change (it saw the edited token).
    assert not torch.allclose(logits_a[:, -1, :], logits_b[:, -1, :])


def test_attention_weights_sum_to_one():
    cfg = _tiny_cfg()
    block = model_block = GPT(cfg).blocks[0].attn
    x = torch.randn(2, cfg.block_size, cfg.n_embd)

    # Re-derive attention weights the same way the layer does, and check softmax.
    import math
    B, T, C = x.shape
    q, k, v = model_block.qkv(x).split(C, dim=2)
    q = q.view(B, T, cfg.n_head, -1).transpose(1, 2)
    k = k.view(B, T, cfg.n_head, -1).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(model_block.head_dim)
    att = att.masked_fill(model_block.mask[:, :, :T, :T] == 0, float("-inf"))
    att = torch.softmax(att, dim=-1)
    assert torch.allclose(att.sum(-1), torch.ones(B, cfg.n_head, T), atol=1e-5)
    # Causal: row 0 attends only to itself (weight 1 at position 0, 0 elsewhere).
    assert torch.allclose(att[:, :, 0, 0], torch.ones(B, cfg.n_head), atol=1e-5)


# --- learning + generation ------------------------------------------------- #

def test_overfits_one_batch():
    """A correct model should memorise a single batch (loss -> near 0)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(300):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.2


def test_generate_produces_valid_tokens_and_length():
    cfg = _tiny_cfg()
    model = GPT(cfg).eval()
    ctx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(ctx, max_new_tokens=20)
    assert out.shape == (1, 21)
    assert int(out.min()) >= 0 and int(out.max()) < cfg.vocab_size
