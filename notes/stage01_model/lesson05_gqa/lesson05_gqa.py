"""Lesson 1.5: GQA (Grouped-Query Attention), heads sharing Keys and Values.

Part 1: WHY. When generating text, the model saves every past token's K and V (the "KV cache").
        That memory grows with every token. Fewer K/V heads = smaller cache.
Part 2: HOW. MiniMind keeps 8 query heads but only 4 key/value heads, then COPIES each
        K/V head so that every pair of query heads shares one. That's repeat_kv().

Run from the repo root:
    .venv/bin/python notes/stage01_model/lesson05_gqa/lesson05_gqa.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from model.model_minimind import repeat_kv          # MiniMind's real function (line 86)

# ---- Part 2 first, small: 2 K/V heads shared by 4 query heads -------------------------
# Shape (batch, seq, kv_heads, head_dim) = (1, 3, 2, 2), with numbers easy to recognise:
K = torch.tensor([[[[1., 1.], [2., 2.]],      # token 0: kv-head A = [1,1], kv-head B = [2,2]
                   [[1., 1.], [2., 2.]],      # token 1
                   [[1., 1.], [2., 2.]]]])    # token 2
print("K before repeat_kv:", tuple(K.shape), " (batch, seq, kv_heads=2, head_dim)")
K8 = repeat_kv(K, n_rep=2)                    # copy each K/V head 2 times
print("K after  repeat_kv:", tuple(K8.shape), " (batch, seq, heads=4,    head_dim)\n")
print("token 0, which key each QUERY head will use:")
for h in range(4):
    print(f"  query head {h} -> key {K8[0, 0, h].tolist()}   (kv-head {'A' if h < 2 else 'B'})")
print("-> query heads 0,1 share kv-head A; query heads 2,3 share kv-head B.\n")

# ---- Part 1: what it saves, with MiniMind's real numbers --------------------------------
LAYERS, HEAD_DIM, Q_HEADS = 8, 96, 8
BYTES = 2                                     # each number stored in 16-bit = 2 bytes
CONTEXT = 32768                               # MiniMind's max_position_embeddings

def cache_mb(kv_heads, tokens):
    # per token, per layer: K and V, each kv_heads x head_dim numbers
    return 2 * kv_heads * HEAD_DIM * LAYERS * tokens * BYTES / 1e6

print(f"KV cache for ONE conversation, {LAYERS} layers, {HEAD_DIM} numbers per head:")
print(f"  {'design':30s} {'k_proj size':>12s} {'2,048 tokens':>14s} {'32,768 tokens':>15s}")
for name, kv in [("MHA: 8 K/V heads (no sharing)", 8), ("GQA: 4 K/V heads (MiniMind)", 4), ("MQA: 1 K/V head (all share)", 1)]:
    print(f"  {name:30s} {f'768 x {kv*HEAD_DIM}':>12s} {cache_mb(kv, 2048):11.1f} MB {cache_mb(kv, CONTEXT):12.1f} MB")
