"""Lesson 1.3b: attention shapes, step by step, with numbers small enough to check by hand.

3 words, 2 numbers per word (different on purpose, so you can tell seq from dim).

Run from the repo root:
    .venv/bin/python notes/stage01_model/lesson03_causal_mask/lesson03b_shapes.py
"""
import torch

words = ["the", "cat", "sat"]

def show(name, m, rows, cols, note=""):
    r, c = m.shape
    print(f"{name}   shape = ({r} x {c})   {note}")
    print("        " + "".join(f"{c_:>8s}" for c_ in cols))
    for i in range(r):
        print(f"  {rows[i]:>5s} " + "".join(f"{m[i, j].item():8.2f}" for j in range(c)))
    print()

dims = ["d0", "d1"]          # the 2 numbers of each vector

# ---- 0. Input: one vector per word ---------------------------------------------------
x = torch.tensor([[1., 0.],      # the
                  [0., 1.],      # cat
                  [1., 1.]])     # sat
show("x (input)", x, words, dims, "<- seq x dim: one row per word")

# ---- 1. Learned weight matrices (dim x dim) --------------------------------------------
Wq = torch.tensor([[1., 0.], [0., 2.]])
Wk = torch.tensor([[1., 1.], [0., 1.]])
Wv = torch.tensor([[2., 0.], [0., 3.]])
show("Wq (LEARNED knobs)", Wq, dims, dims, "<- dim x dim: doesn't depend on the sentence")

# ---- 2. Q, K, V = x @ W -----------------------------------------------------------------
Q, K, V = x @ Wq, x @ Wk, x @ Wv
show("Q = x @ Wq", Q, words, dims, "<- (3x2)@(2x2) = 3x2: one query per word")
show("K = x @ Wk", K, words, dims, "<- 3x2: one key per word")
show("V = x @ Wv", V, words, dims, "<- 3x2: one value per word")

# ---- 3. Transpose K: flip rows and columns -------------------------------------------
show("K.T (K flipped)", K.T, dims, words, "<- 2x3: words are now COLUMNS")

# ---- 4. Scores = Q @ K.T ------------------------------------------------------------------
S = Q @ K.T
show("scores = Q @ K.T", S, words, words, "<- (3x2)@(2x3) = 3x3: every word vs every word")
print(f"  e.g. scores[sat][cat] = Q[sat] . K[cat] = "
      f"({Q[2,0]:.0f} x {K[1,0]:.0f}) + ({Q[2,1]:.0f} x {K[1,1]:.0f}) = {S[2,1]:.0f}\n")

# ---- 5. Mask + softmax --------------------------------------------------------------------
mask = torch.full((3, 3), float("-inf")).triu(1)
Wt = torch.softmax(S + mask, dim=-1)
show("weights = softmax(scores + mask)", Wt, words, words, "<- still 3x3, each row adds to 1")

# ---- 6. Output = weights @ V ------------------------------------------------------------
out = Wt @ V
show("out = weights @ V", out, words, dims, "<- (3x3)@(3x2) = 3x2: back to one vector per word")
print(f"  e.g. out[sat] = {Wt[2,0]:.2f} x V[the] + {Wt[2,1]:.2f} x V[cat] + {Wt[2,2]:.2f} x V[sat]")
print(f"              = {Wt[2,0]:.2f} x {V[0].tolist()} + {Wt[2,1]:.2f} x {V[1].tolist()} + {Wt[2,2]:.2f} x {V[2].tolist()}")
print(f"              = {[round(v, 2) for v in out[2].tolist()]}\n")

print("SHAPE JOURNEY:  x (3x2) -> Q,K,V (3x2) -> scores (3x3) -> weights (3x3) -> out (3x2)  == same as x")
