"""Lesson 0.1: computers only understand numbers, so text becomes token IDs.

Run from the repo root:
    .venv/bin/python notes/stage00_foundations/lesson01_tokens/lesson01_tokens.py

Try editing the `sentences` list and running again!
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
print(ROOT)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))
print(tokenizer)
print(f"MiniMind's dictionary has {tokenizer.vocab_size} tokens.\n")

sentences = [
    "the cat sat on the mat",
    "Hello, how are you?",
    "unbelievably",        # a long word: does it get split into pieces?
    "Einstein",            # a name
    "你好，世界",            # Chinese ("hello, world"); MiniMind was trained mostly on Chinese
]

for s in sentences:
    ids = tokenizer.encode(s)                    # text  -> numbers
    pieces = [tokenizer.decode([i]) for i in ids]
    back = tokenizer.decode(ids)                 # numbers -> text
    print(f"text   : {s!r}")
    print(f"ids    : {ids}")
    print(f"pieces : {pieces}")
    print(f"decoded: {back!r}   ({len(ids)} tokens)\n")
