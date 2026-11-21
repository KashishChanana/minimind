"""Lesson 1.6: MiniMind's real Attention class.

Part A: why scores are divided by sqrt(96).
Part B: rebuild MiniMind's Attention.forward step by step using ITS OWN weights,
        and check our result matches the real class exactly.

Run from the repo root:
    .venv/bin/python notes/stage01_model/lesson06_real_attention/lesson06_real_attention.py
"""
import os, sys, math, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
import torch.nn.functional as F
from model.model_minimind import MiniMindConfig, Attention, precompute_freqs_cis, apply_rotary_pos_emb, repeat_kv
torch.manual_seed(0)

# =========================== Part A: why divide by sqrt(head_dim)? ===========================
print("PART A: why  scores / sqrt(96)\n")
q = torch.randn(96)                   # one query with 96 random numbers
keys = torch.randn(5, 96)             # five keys
raw = keys @ q                        # 5 dot products, each adds up 96 products
print(f"  raw scores        : {[round(v, 1) for v in raw.tolist()]}")
print(f"  softmax(raw)      : {[f'{v:.0%}' for v in F.softmax(raw, -1).tolist()]}   <- one key grabs almost everything")
scaled = raw / math.sqrt(96)
print(f"  scores / sqrt(96) : {[round(v, 1) for v in scaled.tolist()]}")
print(f"  softmax(scaled)   : {[f'{v:.0%}' for v in F.softmax(scaled, -1).tolist()]}   <- a real spread\n")

big = torch.randn(10000, 96) @ torch.randn(96)
print(f"  over 10,000 random keys: raw scores spread (std) = {big.std():.1f}  (about sqrt(96) = {math.sqrt(96):.1f})")
print(f"                           after / sqrt(96)          = {(big / math.sqrt(96)).std():.1f}\n")

# =========================== Part B: rebuild the real thing ===========================
print("PART B: rebuild MiniMind's Attention.forward (model_minimind.py:111-134) by hand\n")
cfg = MiniMindConfig()
attn = Attention(cfg).eval()          # the REAL class, random weights
B, T = 1, 5
x = torch.randn(B, T, cfg.hidden_size)
cos_all, sin_all = precompute_freqs_cis(cfg.head_dim, end=T, rope_base=cfg.rope_theta)
pos = (cos_all[:T], sin_all[:T])

def shape(line, name, t):
    print(f"  line {line:7s} {name:42s} {str(tuple(t.shape))}")

with torch.no_grad():
    shape("", "x", x)
    xq, xk, xv = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
    shape("113", "xq = q_proj(x)", xq); shape("113", "xk = k_proj(x)   (GQA: smaller)", xk)
    xq = xq.view(B, T, 8, 96); xk = xk.view(B, T, 4, 96); xv = xv.view(B, T, 4, 96)
    shape("114-116", "view into heads: xq", xq); shape("114-116", "view into heads: xk, xv", xk)
    xq, xk = attn.q_norm(xq), attn.k_norm(xk)
    shape("117", "q_norm / k_norm  (shape unchanged)", xq)
    xq, xk = apply_rotary_pos_emb(xq, xk, *pos)
    shape("119", "RoPE: add position info (shape unchanged)", xq)
    xq, xk, xv = xq.transpose(1, 2), repeat_kv(xk, 2).transpose(1, 2), repeat_kv(xv, 2).transpose(1, 2)
    shape("124", "repeat_kv + transpose: xq", xq); shape("124", "repeat_kv + transpose: xk, xv", xk)
    scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(96)
    shape("128", "scores = xq @ xk.T / sqrt(96)", scores)
    scores = scores + torch.full((T, T), float("-inf")).triu(1)
    shape("129", "+ causal mask", scores)
    out = F.softmax(scores, dim=-1) @ xv
    shape("131", "softmax(...) @ xv", out)
    out = out.transpose(1, 2).reshape(B, T, -1)
    shape("132", "glue heads back", out)
    mine = attn.o_proj(out)
    shape("133", "o_proj", mine)

    real, _ = attn(x, pos)            # the real forward (uses the fast built-in, line 126)
    print(f"\n  my rebuild == real Attention class?  {torch.allclose(mine, real, atol=1e-5)}"
          f"   (largest difference {(mine - real).abs().max():.1e})")
