"""Lesson 1.2: Query, Key, Value.

Lesson 1.1's weakness: with raw dot products, "bank" could only find words that look
like itself, so a water-leaning "bank" ignored "money" and listened to itself.

Fix: every token makes three DIFFERENT vectors from its own vector, using three Linear layers:
    Query = "what am I looking for?"
    Key   = "what do I have to offer? (how others find me)"
    Value = "what do I hand over if someone picks me?"
Score = my Query . their Key   (not my vector . their vector)

Each word now has 3 numbers: [water-ness, money-ness, needs-context]
The weights below are HAND-PICKED so you can see the idea. In MiniMind they are LEARNED.

Run from the repo root:
    .venv/bin/python notes/stage01_model/lesson02_qkv/lesson02_qkv.py
"""
import torch
import torch.nn.functional as F

vectors = {
    "the":   torch.tensor([0.0, 0.0, 0.0]),
    "river": torch.tensor([3.0, 0.0, 0.0]),
    "money": torch.tensor([0.0, 3.0, 0.0]),
    "bank":  torch.tensor([2.0, 1.0, 1.0]),   # leans water AND needs context (the hard case from 1.1)
}

# ---- Three Linear layers (Lesson 0.3), with hand-picked weights -------------------------
# Reminder: PyTorch stores Linear weights as (out, in).
q_proj = torch.nn.Linear(3, 1, bias=False)   # vector (3 numbers) -> query (1 number)
k_proj = torch.nn.Linear(3, 1, bias=False)   # vector (3 numbers) -> key   (1 number)
v_proj = torch.nn.Linear(3, 3, bias=False)   # vector (3 numbers) -> value (3 numbers)
with torch.no_grad():
    # Query = 3 * needs-context        -> "I'm ambiguous, I'm looking for a context word"
    q_proj.weight[:] = torch.tensor([[0.0, 0.0, 3.0]])
    # Key = water + money - 3*needs-context  -> "I'm a clear, meaningful context word"
    #                                           (ambiguous words get a LOW key: don't pick me)
    k_proj.weight[:] = torch.tensor([[1.0, 1.0, -3.0]])
    # Value = hand over my meaning (water, money) but not my "needs-context" flag
    v_proj.weight[:] = torch.tensor([[1.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])

def raw_attention(words, me):
    """Lesson 1.1 version: score = my vector . their vector"""
    sims = torch.stack([vectors[w] @ vectors[me] for w in words])
    return F.softmax(sims, dim=0)

def qkv_attention(words, me):
    """Lesson 1.2 version: score = my QUERY . their KEY, then mix their VALUES"""
    with torch.no_grad():
        q = q_proj(vectors[me])                                  # my query
        keys = torch.stack([k_proj(vectors[w]) for w in words])  # everyone's key
        values = torch.stack([v_proj(vectors[w]) for w in words])
        scores = (keys @ q)                                      # my query . each key
        weights = F.softmax(scores, dim=0)
        new_me = weights @ values                                # weighted mix of VALUES
    return q, keys.squeeze(-1), scores, weights, new_me

for sentence in ["the river bank", "the money bank"]:
    words = sentence.split()
    print(f'=== "{sentence}"  (focus: "bank") ===')
    w_raw = raw_attention(words, "bank")
    q, keys, scores, w_qkv, new_me = qkv_attention(words, "bank")
    print(f"  bank's query = {q.item():.0f}")
    print(f"  {'word':7s} {'key':>5s} {'q.k score':>10s} {'NEW weight':>11s}   {'old 1.1 weight':>14s}")
    for i, w in enumerate(words):
        print(f"  {w:7s} {keys[i].item():5.0f} {scores[i].item():10.0f} {w_qkv[i].item():11.0%}   {w_raw[i].item():14.0%}")
    lean = "WATER" if new_me[0] > new_me[1] else "MONEY"
    print(f"  bank after = {[round(v, 2) for v in new_me.tolist()]}  -> leans {lean}\n")
