"""★ CHECKPOINT: write MiniMind-style attention from scratch.

Rules: no peeking at the lesson files or model/model_minimind.py.
If you get stuck, write  # stuck here  and move on.

When you're done, check your work (from the repo root):
    .venv/bin/python notes/stage01_model/checkpoint_attention/check.py

You can skip: QK-norm, RoPE, KV cache, padding mask, dropout. Just the core attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---- Setup (given) -----------------------------------------------------------------------
BATCH, SEQ = 1, 5          # 1 sentence, 5 tokens
DIM = 768                  # numbers per token
N_HEADS = 8                # query heads
N_KV_HEADS = 4             # key/value heads (GQA)
HEAD_DIM = DIM // N_HEADS  # 96

x = torch.randn(BATCH, SEQ, DIM)    # the input: one 768-number vector per token

# 1, 5, 768
# 1, 5, 8, 96
# 1, 5, 4, 96


# ---- Step 1: create the four Linear layers (bias=False) -----------------------------------
# Keep these names exactly, check.py looks for them.
q_proj = nn.Linear(DIM, DIM, bias=False)
k_proj = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)   # 768 -> 4 heads x 96 = 384 (ALL K heads at once)
v_proj = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)   # 768 -> 384
o_proj = nn.Linear(DIM, DIM, bias=False)


def attention(x):
    # ---- Step 2: make Q, K, V -------------------------------------------------------------
    # TODO
    xq = q_proj(x)
    xk = k_proj(x)
    xv = v_proj(x)


    # ---- Step 3: split into heads ---------------------------------------------------------
    xqh = xq.view((BATCH, SEQ, N_HEADS, HEAD_DIM))
    xkh = xk.view((BATCH, SEQ, N_KV_HEADS, HEAD_DIM))         # (1, 5, 4, 96): the last slot is per-head size
    xvh = xv.view((BATCH, SEQ, N_KV_HEADS, HEAD_DIM))         # (1, 5, 4, 96)

    # ---- Step 4: share K/V across query heads (make K and V have N_HEADS heads) ------------
    n_rep = N_HEADS // N_KV_HEADS                             # 8 // 4 = 2 query heads per K/V head
    xkh = xkh.repeat_interleave(n_rep, dim=2)                 # (1, 5, 8, 96): A B C D -> A A B B C C D D
    xvh = xvh.repeat_interleave(n_rep, dim=2)                 # (1, 5, 8, 96)

    # ---- Step 5: put heads first ----------------------------------------------------------
    xqh = xqh.transpose(1, 2)                                 # (1, 8, 5, 96): batch, heads, seq, head_dim
    xkh = xkh.transpose(1, 2)                                 # (1, 8, 5, 96)
    xvh = xvh.transpose(1, 2)                                 # (1, 8, 5, 96)

    # ---- Step 6: scores, scale, causal mask, softmax --------------------------------------
    scores = xqh @ xkh.transpose(-2, -1)                      # (1,8,5,96) @ (1,8,96,5) = (1, 8, 5, 5)
    scores = scores / math.sqrt(HEAD_DIM)                     # ÷ √96 so softmax isn't winner-takes-all
    mask = torch.full((SEQ, SEQ), float("-inf")).triu(1)      # (5, 5): -inf above the diagonal = the future
    scores = scores + mask                                    # (1, 8, 5, 5): the (5,5) mask is added to every head
    weights = F.softmax(scores, dim=-1)                       # (1, 8, 5, 5): each row adds up to 100%

    # ---- Step 7: mix the values ------------------------------------------------------------
    out = weights @ xvh                                       # (1,8,5,5) @ (1,8,5,96) = (1, 8, 5, 96)

    # ---- Step 8: glue the heads back together, then o_proj ---------------------------------
    out = out.transpose(1, 2)                                 # (1, 5, 8, 96): undo Step 5
    out = out.reshape(BATCH, SEQ, DIM)                        # (1, 5, 768): join 8 x 96 back into 768
    output = o_proj(out)                                      # (1, 5, 768): mix what the heads found
    return output


if __name__ == "__main__":
    out = attention(x)
    print("output shape:", None if out is None else tuple(out.shape), "  (should be (1, 5, 768))")
