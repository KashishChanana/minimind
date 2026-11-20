"""Lesson 1.1: the idea behind attention, with tiny hand-made numbers.

Problem from Lesson 0.4: the vector for "bank" is the same in
    "the river bank"   and   "the money bank".
Attention fixes this by letting "bank" MIX IN information from the words before it.

Each word here has just 2 numbers:  [water-ness, money-ness]
(In MiniMind each token has 768 numbers, and nobody hand-picks them; they're learned.)

Run from the repo root:
    .venv/bin/python notes/stage01_model/lesson01_attention_idea/lesson01_attention_idea.py
"""
import torch
import torch.nn.functional as F

vectors = {
    "the":   torch.tensor([0.0, 0.0]),   # means nothing in particular
    "river": torch.tensor([3.0, 0.0]),   # very watery
    "money": torch.tensor([0.0, 3.0]),   # very money-ish
    "bank":  torch.tensor([1.0, 1.0]),   # ambiguous: a bit of both
}

def attend(sentence, focus_word="bank"):
    words = sentence.split()
    i = words.index(focus_word)
    me = vectors[focus_word]
    earlier = words[: i + 1]                          # it may look at itself and words BEFORE it

    # STEP 1. How related is each earlier word to me?  (dot product = multiply pairwise, add up)
    sims = torch.stack([vectors[w] @ me for w in earlier])

    # STEP 2. Turn relatedness into percentages that add up to 100%  (softmax, from Lesson 0.4)
    weights = F.softmax(sims, dim=0)

    # STEP 3. New vector = weighted mix of the earlier words' vectors
    new_me = sum(wt * vectors[w] for wt, w in zip(weights, earlier))

    print(f'sentence: "{sentence}"   focus: "{focus_word}"')
    print(f"  {'word':7s} {'vector':12s} {'similarity':>10s} {'weight':>7s}")
    for w, s, wt in zip(earlier, sims, weights):
        print(f"  {w:7s} {str(vectors[w].tolist()):12s} {s.item():10.1f} {wt.item():7.0%}")
    print(f"  '{focus_word}' before: {me.tolist()}   after: {[round(v, 2) for v in new_me.tolist()]}")
    lean = "WATER" if new_me[0] > new_me[1] else "MONEY"
    print(f"  -> now leans {lean}\n")
    return new_me

a = attend("the river bank")
b = attend("the money bank")
print("identical?", torch.equal(a, b))
