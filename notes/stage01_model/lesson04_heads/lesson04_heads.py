"""Lesson 1.4: multiple heads.

One attention = one set of weights = each word can look for ONE kind of thing.
Multiple heads = split every vector into chunks and run a separate attention on each chunk,
so each head can look for something different. Then glue the results back together.

Small version: 3 words, 4 numbers per word, 2 heads of 2 numbers each.
(MiniMind: 768 numbers per word, 8 heads of 96.)

Run from the repo root:
    .venv/bin/python notes/stage01_model/lesson04_heads/lesson04_heads.py
"""
import torch
import torch.nn.functional as F
torch.manual_seed(0)

words = ["the", "cat", "sat"]
SEQ, DIM, HEADS = 3, 4, 2
HEAD_DIM = DIM // HEADS                       # 4 // 2 = 2 numbers per head

x = torch.randn(SEQ, DIM)
q_proj = torch.nn.Linear(DIM, DIM, bias=False)    # SAME layers as before: 4 -> 4
k_proj = torch.nn.Linear(DIM, DIM, bias=False)
v_proj = torch.nn.Linear(DIM, DIM, bias=False)
o_proj = torch.nn.Linear(DIM, DIM, bias=False)    # NEW: mixes the heads' results at the end
mask = torch.full((SEQ, SEQ), float("-inf")).triu(1)

def shape(name, t, note=""):
    print(f"  {name:34s} {str(tuple(t.shape)):12s} {note}")

with torch.no_grad():
    print("STEP 1  Q, K, V exactly like before")
    Q, K, V = q_proj(x), k_proj(x), v_proj(x)
    shape("Q = q_proj(x)", Q, "(seq, dim)")

    print("\nSTEP 2  view: cut each word's 4 numbers into 2 heads x 2 numbers   (no numbers change)")
    Qh = Q.view(SEQ, HEADS, HEAD_DIM)
    shape("Q.view(seq, heads, head_dim)", Qh, "(seq, heads, head_dim)")
    print(f"    'the' query, before: {[round(v, 2) for v in Q[0].tolist()]}")
    print(f"    'the' query, after : head0 {[round(v, 2) for v in Qh[0, 0].tolist()]}   head1 {[round(v, 2) for v in Qh[0, 1].tolist()]}")

    print("\nSTEP 3  transpose: put heads FIRST, so each head is its own little (seq, head_dim) table")
    Qh = Qh.transpose(0, 1)
    Kh = K.view(SEQ, HEADS, HEAD_DIM).transpose(0, 1)
    Vh = V.view(SEQ, HEADS, HEAD_DIM).transpose(0, 1)
    shape("Q.transpose(0, 1)", Qh, "(heads, seq, head_dim)")

    print("\nSTEP 4  attention, separately for each head (one @ does all heads at once)")
    scores = Qh @ Kh.transpose(-2, -1)            # (heads, seq, hd) @ (heads, hd, seq)
    shape("scores = Qh @ Kh.T", scores, "(heads, seq, seq): one score table PER HEAD")
    weights = F.softmax(scores + mask, dim=-1)
    for h in range(HEADS):
        print(f"\n    head {h} weights:            " + "".join(f"{w:>7s}" for w in words))
        for i, w in enumerate(words):
            print(f"      {w:>5s} looks at ->          " + "".join(f"{v.item():7.0%}" for v in weights[h, i]))
    out_h = weights @ Vh
    shape("\n  out = weights @ Vh", out_h, "(heads, seq, head_dim)")

    print("\nSTEP 5  glue the heads back together: undo the transpose, then join the chunks")
    out = out_h.transpose(0, 1).reshape(SEQ, DIM)
    shape("out.transpose(0,1).reshape(seq, dim)", out, "(seq, dim): back to one vector per word")

    print("\nSTEP 6  o_proj: one Linear that mixes what the different heads found")
    final = o_proj(out)
    shape("o_proj(out)", final, "(seq, dim): same shape as the input x")

    n_one = sum(p.numel() for p in [q_proj.weight, k_proj.weight, v_proj.weight])
    print(f"\nknobs in q/k/v: {n_one}, the SAME as one big head. Splitting into heads costs no extra weights.")
