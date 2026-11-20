# Lesson 0.3: Tensors

A **tensor** is a box of numbers with a **shape**. Every input, weight and intermediate result in MiniMind is a tensor. **Reading model code is mostly tracking shapes.**

## Run

```bash
.venv/bin/python notes/stage00_foundations/lesson03_tensors/lesson03_tensors.py
```

## Shapes

| Shape | What it is | Example |
|---|---|---|
| `()` | one number | `3.0` |
| `(3,)` | a list | `'the cat sat'` → `[4345, 3706, 6278]` |
| `(2, 3)` | a table | 2 sentences × 3 tokens |
| `(2, 5, 4)` | stack of tables | 2 sentences × 5 tokens × 4 numbers each |

## The four operations

**1. Batch: `(batch, seq)`.** Several sentences processed at once, because GPUs are fast at parallel work. `(2, 5)` = 2 sentences of 5 tokens.

**2. Embedding: `(2, 5)` → `(2, 5, 4)`.** A **lookup table**: row *i* is the vector for token *i*. Each ID is replaced by its row, which adds a dimension. The table's numbers start random and are learned. MiniMind: 6400 × 768 = 4.92M parameters.

**3. Linear: `(2, 5, 4)` → `(2, 5, 3)`.** Multiplies every vector by a weight matrix, each vector **on its own**. Think of it as `(…, 4) × (4, 3) → (…, 3)`: the inner sizes cancel.
- Knobs = in × out. `Linear(4, 3)` = 12; `Linear(4, 10)` = 40.
- With a **bias** (`+ b`, one per output), add `out`: `Linear(4, 10)` with bias = 50. MiniMind uses `bias=False` everywhere.
- PyTorch stores the weight as **(out, in)**: `Linear(4, 10).weight.shape` = `[10, 4]`.
- Almost all of MiniMind's 64M parameters are in Linear layers. `q_proj = Linear(768, 768)` = 589,824 knobs.

**4. `view` (reshape): `(2, 5, 4)` → `(2, 5, 2, 2)`.** Same numbers, regrouped. This is how attention makes **heads**: MiniMind splits each 768-vector into 8 heads × 96 (`model_minimind.py:116`).

## MiniMind's main shape flow

```
input_ids          (2, 16)          2 sentences, 16 tokens
  → embed_tokens   (2, 16, 768)     lookup table
  → q_proj         (2, 16, 768)     Linear
  → .view(...)     (2, 16, 8, 96)   split into 8 heads
  → ... 8 blocks ...
  → lm_head        (2, 16, 6400)    Linear 768→6400: a score for every possible next token
```

**`grad_fn=<...>`** in printouts: PyTorch recording how each number was made, so `.backward()` can compute gradients later.
