"""Lesson 2.1b: why a bigger dictionary gives shorter sequences.

Trains BPE tokenizers with different vocab sizes on README_en.md, then counts how many
tokens each needs for the same sentence.

Run from the repo root:
    .venv/bin/python notes/stage02_tokenizer/lesson01_inside_tokenizer/lesson01b_vocab_vs_length.py
"""
import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
text = open(os.path.join(ROOT, "README_en.md"), encoding="utf-8").read()
paragraphs = [p for p in text.split("\n\n") if p.strip()]

sentence = "The model is trained on a large dataset to predict the next token in the sequence."
print(f"sentence: {sentence!r}")
print(f"as raw bytes: {len(sentence.encode('utf-8'))} pieces\n")

for vocab in [300, 1000, 3000, 6000]:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(paragraphs, trainer=trainer)
    ids = tok.encode(sentence).ids
    pieces = [tok.decode([i]) for i in ids]
    print(f"vocab {vocab:5d}: {len(ids):2d} tokens  {pieces}")
