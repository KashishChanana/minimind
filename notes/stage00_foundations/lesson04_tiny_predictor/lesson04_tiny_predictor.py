"""Lesson 0.4: a tiny next-token predictor.

Uses only what we've covered: tokenizer (0.1), training loop (0.2), Embedding + Linear (0.3).
The model looks at ONE token and guesses the next one. No attention yet.

Run from the repo root:
    .venv/bin/python notes/stage00_foundations/lesson04_tiny_predictor/lesson04_tiny_predictor.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
torch.manual_seed(0)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# ---- 1. Data: turn text into one long list of token IDs (lesson 0.1) -------------
tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))
text = open(os.path.join(ROOT, "README_en.md"), encoding="utf-8").read()
ids = torch.tensor(tokenizer.encode(text))
print(f"training text: {len(text):,} characters -> {len(ids):,} tokens")

# Training pairs: each token, and the token that actually came right after it.
inputs  = ids[:-1]     # "the", " cat", " sat", ...
targets = ids[1:]      # " cat", " sat", " on", ...   (shifted by one)

# ---- 2. Model: Embedding + Linear (lesson 0.3) -----------------------------------
VOCAB, DIM = 6400, 64
embed = torch.nn.Embedding(VOCAB, DIM)            # token ID -> 64 numbers
head  = torch.nn.Linear(DIM, VOCAB, bias=False)   # 64 numbers -> 6400 scores, one per possible next token
params = list(embed.parameters()) + list(head.parameters())
print(f"model has {sum(p.numel() for p in params):,} knobs\n")

def model(token_ids):
    return head(embed(token_ids))                 # (N,) -> (N, 64) -> (N, 6400)

# ---- 3. Training loop (lesson 0.2), with a smarter "nudge" ------------------------
optimizer = torch.optim.Adam(params, lr=1e-2)     # Adam = a smarter version of  w -= lr * w.grad

for step in range(301):
    batch = torch.randint(0, len(inputs), (256,))     # pick 256 random positions from the text
    scores = model(inputs[batch])                     # 1. GUESS   -> shape (256, 6400)
    loss = F.cross_entropy(scores, targets[batch])    # 2. SCORE   (explained in the lesson)
    optimizer.zero_grad()
    loss.backward()                                   # 3. BLAME
    optimizer.step()                                  # 4. NUDGE
    if step % 50 == 0:
        print(f"step {step:3d}: loss={loss.item():.3f}")

# Sections 4 and 5 only USE the model (no training), so turn off gradient tracking:
with torch.no_grad():
    # ---- 4. What did it learn? Ask for the top guesses after a few words ---------------
    print()
    for word in [" the", " language", " model", " train"]:
        tok = tokenizer.encode(word)[0]
        probs = F.softmax(model(torch.tensor([tok])), dim=-1)[0]   # scores -> probabilities
        top = torch.topk(probs, 5)
        guesses = ", ".join(f"{tokenizer.decode([i])!r} {p:.0%}" for p, i in zip(top.values, top.indices))
        print(f"after {word!r:12s} -> {guesses}")

    # ---- 5. Generate text: guess, append, repeat -----------------------------------------
    tok = tokenizer.encode(" The")[0]
    out = [tok]
    for _ in range(25):
        probs = F.softmax(model(torch.tensor([out[-1]])), dim=-1)[0]
        out.append(torch.multinomial(probs, 1).item())            # pick randomly, weighted by probability
    print("\ngenerated:", tokenizer.decode(out))
