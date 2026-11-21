"""Checks my_attention.py against MiniMind's real Attention class.

Loads YOUR four Linear layers into the real class (with QK-norm and RoPE switched off,
since the checkpoint skips them), runs both on the same input, and compares.

Run from the repo root:
    .venv/bin/python notes/stage01_model/checkpoint_attention/check.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
sys.path.insert(0, HERE)
import torch, torch.nn as nn

import my_attention as mine
from model.model_minimind import MiniMindConfig, Attention

def fail(msg):
    print("❌", msg); sys.exit(1)

# 1. Layers exist with the right shapes?
expected = {"q_proj": (768, 768), "k_proj": (768, 384), "v_proj": (768, 384), "o_proj": (768, 768)}
for name, (inp, out) in expected.items():
    layer = getattr(mine, name, None)
    if not isinstance(layer, nn.Linear):
        fail(f"{name} isn't an nn.Linear yet (Step 1).")
    if (layer.in_features, layer.out_features) != (inp, out):
        fail(f"{name} is Linear({layer.in_features}, {layer.out_features}). Think: how many numbers in, how many out?")
    if layer.bias is not None:
        fail(f"{name} has a bias. MiniMind uses bias=False.")
print("✅ Step 1: all four layers have the right shapes")

# 2. Runs and gives the right shape?
x = mine.x.clone()
try:
    with torch.no_grad():
        out = mine.attention(x)
except Exception as e:
    fail(f"attention(x) crashed: {type(e).__name__}: {e}")
if out is None:
    fail("attention(x) returned None. Set `output` at the end.")
if tuple(out.shape) != (1, 5, 768):
    fail(f"output shape is {tuple(out.shape)}, should be (1, 5, 768).")
print("✅ output shape is (1, 5, 768)")

# 3. Same numbers as the real class?
real = Attention(MiniMindConfig()).eval()
real.q_norm, real.k_norm = nn.Identity(), nn.Identity()       # checkpoint skips QK-norm
with torch.no_grad():
    for name in expected:
        getattr(real, name).weight.copy_(getattr(mine, name).weight)
    no_rope = (torch.ones(5, 96), torch.zeros(5, 96))           # cos=1, sin=0 -> RoPE does nothing
    ref, _ = real(x, no_rope)

diff = (out - ref).abs().max().item()
if diff < 1e-4:
    print(f"✅ matches MiniMind's real Attention class (largest difference {diff:.1e})")
    print("\n★ Checkpoint passed. You wrote attention from scratch.")
else:
    print(f"❌ runs, but the numbers differ from MiniMind (largest difference {diff:.3f}).")
    # a few targeted hints
    with torch.no_grad():
        x1 = x.clone(); x1[:, -1] += 10
        out1 = mine.attention(x1)
    if not torch.allclose(out[:, :-1], out1[:, :-1], atol=1e-5):
        print("   Hint: changing the LAST token changed earlier tokens' outputs → check the causal mask.")
    else:
        print("   Hint: the causal mask looks right. Check the ÷ by sqrt, the softmax dim, how K/V heads are shared,")
        print("         and that heads are glued back in the right order before o_proj.")
