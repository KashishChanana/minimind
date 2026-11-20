# Lesson 0.4: A tiny next-token predictor

Puts Lessons 0.1–0.3 together into a model that looks at **one token** and guesses the next one. No attention yet.

## Run

```bash
.venv/bin/python notes/stage00_foundations/lesson04_tiny_predictor/lesson04_tiny_predictor.py
.venv/bin/python notes/stage00_foundations/lesson04_tiny_predictor/lesson04b_one_token_proof.py
```

| File | What it does |
|---|---|
| `lesson04_tiny_predictor.py` | Trains Embedding → Linear on the README text, shows top guesses, generates text |
| `lesson04b_one_token_proof.py` | Shows the model gives **identical** output for `" model"` in two different sentences |

## 1. Training examples: shift by one

```python
inputs  = ids[:-1]   # "the"   " cat"  " sat" ...
targets = ids[1:]    # " cat"  " sat"  " on"  ...
```

Each token is paired with the token that **actually came next**. The text is its own answer key, so no human labels are needed. This is **self-supervised** learning, and the model is **autoregressive** (it predicts the next token from what came before). In this tiny model, the "context" is just **one** token.

## 2. The model

```
token ID → Embedding(6400, 64) → Linear(64, 6400) → 6400 scores (one per possible next token)
```

819,200 knobs, all starting random.

## 3. Batches

`torch.randint(0, len(inputs), (256,))` picks **256 separate random positions**, scattered across the text, not a contiguous range (`torch.arange` or a slice would give a range). Each of the 256 guesses is compared with **its own** target, and the 256 losses are **averaged** into one number. A nudge based on many examples is more reliable than one based on a single example.

## 4. Scoring: softmax + cross-entropy

1. **Softmax:** 6,400 scores → probabilities (all positive, adding up to 100%).
2. **Cross-entropy:** look **only at the probability given to the correct token**. Loss = **−log(p)**, which measures how surprised the model was.

| p (correct token) | 1 − p | −log(p) |
|---|---|---|
| 100% | 0 | **0** |
| 50% | 0.50 | 0.69 |
| 1% | 0.99 | 4.61 |
| 1/6400 (uniform guess) | 0.9998 | **8.76** |
| 0% (confidently wrong) | 1 | **∞** |

### Why −log(p) instead of something simpler like 1 − p?

1. **Strong push when badly wrong.** With 1 − p, giving the right answer 1%, 0.01% or 0.0001% all score about 1, so the slope is flat and there's almost no learning signal exactly when the model needs it most. −log(p) keeps getting steeper as the guess gets worse.
2. **Punishes confident mistakes.** 0% on the correct token → ∞. That teaches the model to hedge when unsure.
3. **Logs turn multiplying into adding.** A sentence's probability is a product of per-token probabilities, which underflows to 0 over 340 tokens. −log turns it into a sum, so the loss becomes an **average per token**. Minimizing it = giving the actual training text the highest possible probability.

### Why the first loss is 8.96, not 8.76

8.76 = −log(1/6400) is a **perfectly even** guess, the best you can do knowing nothing. Random starting knobs give **uneven** probabilities, and since the model doesn't know the answer, its random favourites are usually wrong. Unlucky tilts cost more than lucky ones gain (2-token example: 50/50 → 0.69; lucky 70% → 0.36, gain 0.33; unlucky 30% → 1.20, cost 0.51). So random init lands slightly **above** 8.76.

**Sanity check for any training run:** step-0 loss should be about ln(vocab size) or a bit above. Far above → bad initialization. Below → something is leaking the answers.

## 5. Adam

`optimizer.step()` replaces `w -= lr * w.grad`. Adam adjusts the step size for each knob from its recent gradients. MiniMind uses AdamW, a close cousin.

## 6. `torch.no_grad()` when only using the model

Sections 4–5 only run the model, they don't train it, so they're wrapped in `with torch.no_grad():`. Without it, the results are identical, but PyTorch wastes memory recording operations for a `.backward()` that never comes. Rule: **training → track gradients; using → `no_grad`**. MiniMind's `generate()` uses `@torch.inference_mode()` (`model_minimind.py:261`).

## Results

```
step 0: loss=8.962  →  step 300: loss≈4.1

after ' train'    → 'ining' 84%
after ' language' → ' model' 38%, ' pattern' 24%

generated: The right.0/20 | sempling, at this scinference-*8 error ( A b comm
```

Real patterns were learned, but the generated text is nonsense: neighbouring pairs look plausible, while the sentence goes nowhere.

## Why: it only ever sees one token

```python
def model(token_ids):
    return head(embed(token_ids))
```

- `embed` looks up one row per token, regardless of neighbours.
- `head` (Linear) transforms each vector on its own.

Nothing combines information across positions, so the output depends **only on the token ID**. `lesson04b` proves it: `" model"` at the end of "I trained a small language model" and "She works as a fashion model" gives **identical** scores.

## From this to a real LLM

| | Tiny model | MiniMind |
|---|---|---|
| One example | 1 token | a chunk of up to **340** consecutive tokens (768 in SFT) |
| Input shape | `(256,)` | `(batch, 340)` |
| Guesses per example | 1 | one per position, each using **all tokens before it** |
| Shift-by-one | `ids[:-1]` vs `ids[1:]` | `logits[..., :-1]` vs `labels[..., 1:]` (`model_minimind.py:256`) |

The missing piece is **attention**, which goes between `embed` and `head`. It lets each token look back at earlier tokens and weigh how much each one matters. A **causal mask** stops tokens from looking forward, which would let them copy the answer. Everything else (tokenizer, embedding, final Linear, cross-entropy, training loop) is what you built here.

## Check yourself

<details><summary>Why shift by one?</summary>

To make (input → actual next token) pairs from raw text. The text labels itself (self-supervised).
</details>

<details><summary>Loss when the correct token gets 100%? 1/6400?</summary>

0. And −log(1/6400) = 8.76, the uniform "know nothing" guess.
</details>

<details><summary>Which code makes the guess after " model" ignore context?</summary>

`head(embed(token_ids))`: both steps work on each token separately, so nothing mixes positions.
</details>

**Experiment:** change `range(301)` to `range(2001)`. Does the loss keep falling? Does the generated text improve?
