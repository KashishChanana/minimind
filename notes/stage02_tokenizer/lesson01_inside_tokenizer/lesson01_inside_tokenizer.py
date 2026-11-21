"""Lesson 2.1: inside the tokenizer.

Part A: bytes, and the weird symbols (Ġ, Ċ, ä½ł) you see in tokenizer.json
Part B: the 36 reserved special-token slots
Part C: compression, and why only 6,400 tokens

Run from the repo root:
    .venv/bin/python notes/stage02_tokenizer/lesson01_inside_tokenizer/lesson01_inside_tokenizer.py
"""
import os, warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))

# ---- Part A: everything is bytes ----------------------------------------------------------
print("PART A: text is stored as bytes (numbers 0-255)\n")
for ch in ["a", " ", "\n", "é", "你"]:
    b = ch.encode("utf-8")
    print(f"  {ch!r:6s} -> {len(b)} byte(s): {list(b)}")

print("\n  The tokenizer gives each of the 256 bytes a PRINTABLE stand-in symbol, so the dictionary")
print("  file can be read. That's where the weird symbols come from:\n")
for text in [" the", "\n", "你", "你好"]:
    ids = tok.encode(text)
    raw = tok.convert_ids_to_tokens(ids)
    print(f"  {text!r:8s} -> ids {ids}  -> stored in tokenizer.json as {raw}")
print("\n  'Ġ' = the stand-in for a space byte, 'Ċ' = for a newline byte.")
print("  '你' is 3 bytes, so it shows up as 3 stand-in symbols glued together: 'ä½ł'.")

# ---- Part B: reserved special-token slots --------------------------------------------------
print("\nPART B: IDs 0-35 are reserved for special tokens\n")
for i in list(range(0, 3)) + list(range(21, 28)) + [35]:
    print(f"  id {i:2d}: {tok.convert_ids_to_tokens(i)}")
print("  (ids 3-20: vision/audio tokens, unused by MiniMind; 27-35: empty 'buffer' slots kept for later)")
print(f"  first ordinary token, id 36: {tok.convert_ids_to_tokens(36)!r}")

# ---- Part C: compression and vocab size ----------------------------------------------------
print("\nPART C: compression = characters per token (higher = fewer tokens for the same text)\n")
samples = {
    "English": "Large language models are trained on vast amounts of text to predict the next word in a sequence.",
    "Chinese": "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的智能机器。",
}
for name, text in samples.items():
    n = len(tok.encode(text))
    print(f"  {name}: {len(text):3d} characters -> {n:3d} tokens  = {len(text)/n:.2f} chars/token")

print("\n  Why not a bigger dictionary? Every token needs a 768-number row in the embedding table:\n")
total_body = 59.0e6   # the 8 transformer blocks (from Stage 1's parameter count)
for name, vocab in [("MiniMind", 6400), ("Llama 2", 32000), ("Qwen", 151643)]:
    emb = vocab * 768
    print(f"  {name:9s} vocab {vocab:7,d} -> embedding {emb/1e6:6.1f}M params "
          f"= {emb/(emb+total_body):4.0%} of a MiniMind-sized model")
