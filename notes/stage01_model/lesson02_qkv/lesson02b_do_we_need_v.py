"""Lesson 1.2b: do we really need the Value matrix?

Train two tiny one-attention-layer models on the README, identical except:
  A) out = weights @ v_proj(x)   (normal attention)
  B) out = weights @ x           (no Value matrix: hand over the raw vectors)

Run from the repo root:
    .venv/bin/python notes/stage01_model/lesson02_qkv/lesson02b_do_we_need_v.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))
ids = torch.tensor(tok.encode(open(os.path.join(ROOT, "README_en.md"), encoding="utf-8").read()))
split = int(0.9 * len(ids))
train_ids, test_ids = ids[:split], ids[split:]          # test on text the model never trained on
SEQ, DIM, VOCAB = 32, 64, 6400

def get_batch(data, n=64):
    starts = torch.randint(0, len(data) - SEQ - 1, (n,))
    x = torch.stack([data[s:s + SEQ] for s in starts])          # (n, SEQ)  contiguous chunks
    y = torch.stack([data[s + 1:s + SEQ + 1] for s in starts])  # shifted by one
    return x, y

class TinyAttnLM(nn.Module):
    def __init__(self, use_v):
        super().__init__()
        self.use_v = use_v
        self.embed = nn.Embedding(VOCAB, DIM)
        self.q, self.k = nn.Linear(DIM, DIM, bias=False), nn.Linear(DIM, DIM, bias=False)
        self.v = nn.Linear(DIM, DIM, bias=False) if use_v else None
        self.head = nn.Linear(DIM, VOCAB, bias=False)
        self.register_buffer("mask", torch.full((SEQ, SEQ), float("-inf")).triu(1))

    def forward(self, idx):
        x = self.embed(idx)                                            # (n, SEQ, DIM)
        scores = self.q(x) @ self.k(x).transpose(-2, -1) / DIM ** 0.5  # (n, SEQ, SEQ)
        weights = F.softmax(scores + self.mask, dim=-1)
        values = self.v(x) if self.use_v else x                        # <-- the only difference
        x = x + weights @ values                                       # residual: keep own meaning
        return self.head(x)                                            # (n, SEQ, VOCAB)

def run(use_v, steps=1500):
    torch.manual_seed(0)
    model = TinyAttnLM(use_v)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        x, y = get_batch(train_ids)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        torch.manual_seed(123)
        tests = [get_batch(test_ids) for _ in range(20)]
        test_loss = sum(F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item() for x, y in tests) / 20
    return loss.item(), test_loss

for use_v, name in [(True, "A) with V matrix   "), (False, "B) raw x, no V     ")]:
    tr, te = run(use_v)
    print(f"{name} train loss {tr:.3f}   test loss (unseen text) {te:.3f}")
