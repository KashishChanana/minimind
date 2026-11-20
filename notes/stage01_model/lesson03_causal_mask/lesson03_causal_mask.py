"""Lesson 1.3: the causal mask (no peeking at the future).

In Lessons 1.1-1.2 we computed attention for ONE word and left out later words by hand.
A real model computes attention for EVERY token at once, as one table (matrix) of scores.
That table includes "looking at the future", which we must block.

Run from the repo root:
    .venv/bin/python notes/stage01_model/lesson03_causal_mask/lesson03_causal_mask.py
"""
import torch
import torch.nn.functional as F
torch.manual_seed(0)

words = ["the", "cat", "sat", "down"]
T, DIM = len(words), 4

x = torch.randn(T, DIM)                       # one random 4-number vector per word (stand-in for embeddings)
q_proj = torch.nn.Linear(DIM, DIM, bias=False)
k_proj = torch.nn.Linear(DIM, DIM, bias=False)
v_proj = torch.nn.Linear(DIM, DIM, bias=False)

def show(title, m, fmt="{:7.2f}"):
    print(title)
    print("  row looks at ->" + "".join(f"{w:>8s}" for w in words))
    for i, w in enumerate(words):
        cells = "".join(f"{fmt.format(v.item()):>8s}" for v in m[i])
        print(f"  {w:>13s}: {cells}")
    print()

with torch.no_grad():
    # ---- 1. Q, K, V for ALL tokens at once ---------------------------------------
    Q, K, V = q_proj(x), k_proj(x), v_proj(x)          # each: (4 tokens, 4 numbers)

    # ---- 2. ALL scores at once: every query . every key = a 4x4 table ------------
    scores = Q @ K.T                                    # (4, 4). Row i = token i's query vs every key
    show("STEP 1  scores = Q @ K.T   (row = the word doing the looking)", scores)

    # ---- 3. Problem: softmax now would let tokens look at FUTURE words ------------
    show("Without a mask: weights = softmax(scores)   <- 'the' looks at 'cat','sat','down' (the future!)",
         F.softmax(scores, dim=-1), "{:7.0%}")

    # ---- 4. The mask: -inf everywhere ABOVE the diagonal (= the future) ----------
    # This is exactly MiniMind's line (model_minimind.py:129):
    #   torch.full((seq_len, seq_len), float("-inf")).triu(1)
    mask = torch.full((T, T), float("-inf")).triu(1)
    show("STEP 2  mask = full(-inf).triu(1)   (0 = allowed, -inf = blocked)", mask)

    masked = scores + mask
    show("STEP 3  scores + mask", masked)

    # ---- 5. softmax: e^(-inf) = 0, so future words get exactly 0% ------------------
    weights = F.softmax(masked, dim=-1)
    show("STEP 4  weights = softmax(scores + mask)", weights, "{:7.0%}")
    print("  each row adds up to:", [f"{s:.0%}" for s in weights.sum(-1).tolist()], "\n")

    # ---- 6. output = weights @ V : every token's new vector at once --------------
    out = weights @ V
    print("STEP 5  output = weights @ V  ->  shape", tuple(out.shape), "(a new 4-number vector for every word)\n")

    # ---- 7. Proof the future can't leak: change 'down' and see who is affected ----
    x2 = x.clone(); x2[3] = torch.randn(DIM) * 10      # replace the LAST word with something totally different
    Q2, K2, V2 = q_proj(x2), k_proj(x2), v_proj(x2)
    out2 = F.softmax(Q2 @ K2.T + mask, dim=-1) @ V2
    print("PROOF  change the last word 'down' to something else. Which outputs change?")
    for i, w in enumerate(words):
        print(f"  {w:>5s}: {'CHANGED' if not torch.allclose(out[i], out2[i]) else 'same'}")
