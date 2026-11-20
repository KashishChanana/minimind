"""Lesson 0.4b: prove the tiny predictor only sees one token.
Feed ' model' at the end of two different sentences; compare the outputs.
Run from the repo root:  .venv/bin/python notes/stage00_foundations/lesson04_tiny_predictor/lesson04b_one_token_proof.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoTokenizer
torch.manual_seed(0)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))

embed = torch.nn.Embedding(6400, 64)
head  = torch.nn.Linear(64, 6400, bias=False)
def model(token_ids): return head(embed(token_ids))   # same model as lesson 04 (untrained is fine for this test)

a = torch.tensor(tokenizer.encode("I trained a small language model"))
b = torch.tensor(tokenizer.encode("She works as a fashion model"))
print("sentence A ids:", a.tolist())
print("sentence B ids:", b.tolist(), "   <- both end in the same token for ' model'\n")

out_a = model(a)   # scores for every position: shape (len, 6400)
out_b = model(b)
print("scores at last position, A:", out_a[-1, :5].tolist())
print("scores at last position, B:", out_b[-1, :5].tolist())
print("\nidentical?", torch.equal(out_a[-1], out_b[-1]))
