"""Lesson 0.1b: create your OWN tokenizer.json + tokenizer_config.json.

Uses the author's train_tokenizer() from trainer/train_tokenizer.py, but:
  - trains on README_en.md (we haven't downloaded the real dataset yet)
  - uses a small vocab (1000) because the training text is small
  - saves to notes/stage00_foundations/my_tokenizer/  (model/ is NOT touched)

Run from the repo root:
    .venv/bin/python notes/stage00_foundations/lesson01_tokens/lesson01b_make_tokenizer.py
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(ROOT, "trainer"))
from train_tokenizer import train_tokenizer   # the author's function
from transformers import AutoTokenizer

# 1) Make a training file in the format the script expects: one JSON per line, {"text": ...}
data_path = os.path.join(HERE, "my_tokenizer_data.jsonl")
with open(os.path.join(ROOT, "README_en.md"), encoding="utf-8") as f:
    paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]
with open(data_path, "w", encoding="utf-8") as f:
    for p in paragraphs:
        f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
print(f"training text: {len(paragraphs)} paragraphs from README_en.md")

# 2) Train. This writes tokenizer.json, tokenizer_config.json (and vocab.json/merges.txt)
out_dir = os.path.join(HERE, "my_tokenizer")
train_tokenizer(data_path, out_dir, vocab_size=1000)
print("files created:", sorted(os.listdir(out_dir)))

# 3) Compare yours vs the real one
mine = AutoTokenizer.from_pretrained(out_dir)
real = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))
for s in ["the cat sat on the mat", "MiniMind is a tiny language model", "你好，世界"]:
    for name, tok in [("real (6400)", real), ("mine (1000)", mine)]:
        ids = tok.encode(s)
        print(f"{name:12s} {len(ids):2d} tokens: {[tok.decode([i]) for i in ids]}")
    print()

# 4) Peek at the first merges your tokenizer learned (the most frequent pairs in the README)
with open(os.path.join(out_dir, "tokenizer.json"), encoding="utf-8") as f:
    merges = json.load(f)["model"]["merges"]
print("first 10 merges learned ('Ġ' = a space):", merges[:10])
