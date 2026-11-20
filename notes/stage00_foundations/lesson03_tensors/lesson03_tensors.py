"""Lesson 0.3: tensors, the boxes of numbers everything lives in.

Run from the repo root:
    .venv/bin/python notes/stage00_foundations/lesson03_tensors/lesson03_tensors.py
"""
import torch
torch.manual_seed(0)

def show(name, t):
    print(f"{name:28s} shape={str(tuple(t.shape)):14s}\n{t}\n")

# ---- 1. A tensor is a box of numbers with a SHAPE --------------------------------
show("0-D (one number)", torch.tensor(3.0))
show("1-D (a list)", torch.tensor([4345, 3706, 6278]))            # 'the cat sat' as token IDs
show("2-D (a table)", torch.tensor([[4345, 3706, 6278],
                                    [1602, 1414,  732]]))          # 2 sentences x 3 tokens

# ---- 2. In MiniMind, a BATCH of sentences is a 2-D tensor: (batch, seq_len) ------
# Models process many sentences at once (a "batch") because GPUs are fast at parallel work.
input_ids = torch.randint(0, 6400, (2, 5))   # 2 sentences, 5 tokens each, random IDs
show("input_ids (batch, seq)", input_ids)

# ---- 3. Embedding: swap each ID for a vector -> adds a dimension -----------------
# An embedding is just a big table: row i = the vector for token i.
# MiniMind's table is 6400 x 768. Here we use 6400 x 4 so it fits on screen.
embedding = torch.nn.Embedding(6400, 4)
vectors = embedding(input_ids)               # look up a row for every ID
show("vectors (batch, seq, dim)", vectors)
print("check: vector for the first token == row", input_ids[0, 0].item(), "of the table?",
      torch.equal(vectors[0, 0], embedding.weight[input_ids[0, 0]]), "\n")

# ---- 4. A Linear layer: multiply by a weight matrix (the basic neural-net step) --
# nn.Linear(4, 3) holds a 3x4 weight matrix: turns every 4-number vector into a 3-number vector.
# These weights are exactly the "knobs" from lesson 0.2, just many more of them.
linear = torch.nn.Linear(4, 3, bias=False)
out = linear(vectors)                        # applied to every token at once
show("after Linear(4->3)", out)
print(f"Linear(4,3) has {linear.weight.numel()} knobs. MiniMind's q_proj is Linear(768,768): "
      f"{768*768:,} knobs.\n")

# ---- 5. Reshaping: same numbers, different arrangement --------------------------
# Attention splits each 768-vector into 8 heads of 96. Same trick, small version: 4 = 2 heads x 2.
heads = vectors.view(2, 5, 2, 2)             # (batch, seq, n_heads, head_dim)
show("view -> (batch, seq, heads, head_dim)", heads)
print("view() doesn't change any numbers, only how they are grouped.")
