# Lesson 1.3: The causal mask

## The problem

In Lessons 1.1–1.2, we computed attention for **one** word and left out later words **by hand**. A real model computes attention for **every token at once**, as one table of scores, and that table includes tokens looking at **future** words. We must block those.

## Run

```bash
.venv/bin/python notes/stage01_model/lesson03_causal_mask/lesson03_causal_mask.py
```

## Why block the future?

During training, the model predicts the next token at **every position at the same time** (Lesson 0.4: one chunk of text gives many practice questions). Position "the" must predict "cat". If "the" could attend to "cat", it would just **copy the answer**: the loss would drop to about 0 and the model would learn nothing useful. And during generation, future tokens don't exist yet, so the model must learn to work without them.

## Step by step (4 words: the cat sat down)

**Step 1: all scores at once.** `scores = Q @ K.T` gives a 4×4 table. **Row** = the word doing the looking; **column** = the word being looked at. One matrix multiplication computes every query · key pair at once.

**Without a mask**, softmax would let "the" put 40% on "sat", which is in the future.

**Step 2: the mask.** −inf **above the diagonal** (the future), 0 elsewhere:

```
          the   cat   sat  down
the        0   -inf  -inf  -inf
cat        0    0    -inf  -inf
sat        0    0     0    -inf
down       0    0     0     0
```

Made by MiniMind's exact code (`model/model_minimind.py:129`):
```python
torch.full((seq_len, seq_len), float("-inf")).triu(1)
```
- `full(..., -inf)`: a table filled with −inf
- `.triu(1)`: keep only the part **above** the diagonal ("upper triangle", 1 step up), and set everything else to 0

**Step 3: add the mask to the scores.** Allowed scores are unchanged (+0); future scores become −inf.

**Step 4: softmax.** e^(−inf) = 0, so future words get **exactly 0%**, and each row still adds up to 100%:

```
          the   cat   sat  down
the      100%    0%    0%    0%     ← only itself
cat       10%   90%    0%    0%
sat       27%   26%   47%    0%
down      17%   49%   14%   20%     ← sees everything
```

**Step 5:** `output = weights @ V` gives every word's new vector at once.

**Proof:** change the last word "down", and only "down"'s output changes. "the", "cat" and "sat" are completely unaffected by the future.

## Why −inf *before* softmax, instead of zeroing weights after?

If you set future weights to 0 **after** softmax, the remaining weights no longer add up to 100% (e.g., "the" would keep only 26%). Putting −inf **before** softmax means softmax shares out the full 100% among the allowed words only.

## In MiniMind (`model_minimind.py:125-131`)

- **Line 126 (fast path):** `F.scaled_dot_product_attention(..., is_causal=True)` is PyTorch's optimized built-in that does the same thing in one call.
- **Lines 128–131 (manual path):** exactly our steps: scores → `+ mask` → softmax → `@ V`. (The `/ math.sqrt(self.head_dim)` comes in Lesson 1.6.)

## Shapes, step by step (`lesson03b_shapes.py`)

```bash
.venv/bin/python notes/stage01_model/lesson03_causal_mask/lesson03b_shapes.py
```

**Matrix multiplication rule:** `(a × b) @ (b × c) = (a × c)`. The inner sizes must match and they disappear. Each output cell = (a row of the left) · (a column of the right).

With 3 words (seq) and 2 numbers per word (dim):

| Step | Calculation | Shape | Meaning |
|---|---|---|---|
| input | `x` | 3 × 2 | one vector per word |
| learned knobs | `Wq, Wk, Wv` | 2 × 2 | dim × dim; same for every sentence |
| Q, K, V | `x @ W` = (3×2)@(2×2) | 3 × 2 | one query / key / value per word |
| flip K | `K.T` | 2 × 3 | words become columns |
| scores | `Q @ K.T` = (3×2)@(2×3) | **3 × 3** | cell [i][j] = Q[word i] · K[word j] |
| weights | `softmax(scores + mask)` | **3 × 3** | each row adds up to 100% |
| output | `weights @ V` = (3×3)@(3×2) | 3 × 2 | each word = weighted mix of the visible words' values |

**Only scores/weights are seq × seq.** The output has the **same shape as the input**, so the residual (`x + out`) works and blocks can be stacked.

In MiniMind: `x (340, 768)` → `Q, K, V (340, 768)` → `weights (340, 340)` → `out (340, 768)`.

## Shape quiz

**Round 1:** 5 words, 4 numbers per word.
<details><summary>Answers</summary>

`x` (5×4) · `Wq` (4×4) · `Q` (5×4) · `K.T` (4×5) · `scores` (5×5) · mask (5×5), −inf above the diagonal · `out` (5×4)

Weight matrices only contain **dim** sizes, never seq: they must work for any sentence length.
</details>

**Round 2:** same 5 words, 10 numbers per word. What changes?
<details><summary>Answers</summary>

`x` (5×10) · `Wq` (10×10) · `Q` (5×10) · `K.T` (10×5) · `out` (5×10) change. **`scores` and the mask stay (5×5)**: `(5×10)@(10×5)`, and the 10s cancel. Every word vs every word is still 5 × 5.

**dim** sets `x, W, Q, K, V, out`. **seq** alone sets `scores, weights, mask`. That's why long texts are expensive (seq × seq).
</details>

**Round 3:** MiniMind, 100 tokens, 768 dims.
<details><summary>Answers</summary>

`x` (100×768) · `scores` (100×100) · `out` (100×768) · `lm_head` output (100×6400): next-token scores at every position.
</details>

## My summary

> The causal mask prevents label leakage. It sets everything in the future to minus infinity, so a token can't see what comes next.

**Refined:** Precisely: it sets future **scores** (cells above the diagonal of the scores table) to −inf **before softmax**, so they get exactly 0% weight.

## Check yourself

<details><summary>What would go wrong during training without the mask?</summary>

Each position could see the very token it's supposed to predict and just copy it, so the model wouldn't learn to predict.
</details>

<details><summary>Why −inf and not, say, −100 or 0?</summary>

e^(−inf) is exactly 0, so the future gets exactly 0% weight. Adding 0 would change nothing; −100 is nearly but not exactly 0.
</details>

<details><summary>In the weights table, why is "the" 100% on itself?</summary>

It's the first word, so the only thing it's allowed to see is itself.
</details>
