# Lesson 1.1: The idea behind attention

## The problem (from Lesson 0.4)

The tiny predictor gave `" model"` the **same** vector in "language model" and "fashion model", because nothing let a token see its neighbours. We need a step where each token **takes in information from the tokens before it**. That step is **attention**.

## Run

```bash
.venv/bin/python notes/stage01_model/lesson01_attention_idea/lesson01_attention_idea.py
```

## The setup: hand-made 2-number vectors

Each word gets 2 numbers: `[water-ness, money-ness]`.

| word | vector | meaning |
|---|---|---|
| the | `[0, 0]` | nothing in particular |
| river | `[3, 0]` | very watery |
| money | `[0, 3]` | very money-ish |
| bank | `[1, 1]` | ambiguous, a bit of both |

In MiniMind, vectors have 768 numbers and are **learned**. Nobody picks them, and individual numbers rarely mean something this clean. The hand-made version just makes the mechanism visible.

## Attention in 3 steps (for the word "bank")

**Step 1: How related is each earlier word to me?** Use the **dot product**: multiply the two vectors number by number, then add.

```
river · bank = (3 × 1) + (0 × 1) = 3     ← both have water-ness → big
bank  · bank = (1 × 1) + (1 × 1) = 2
the   · bank = (0 × 1) + (0 × 1) = 0     ← nothing in common → zero
```

Vectors that point the same way give a big number; unrelated ones give about 0.

**Step 2: Turn those into percentages.** Softmax (from Lesson 0.4): `[0, 3, 2]` → `[4%, 71%, 26%]`. These are the **attention weights**: how much "bank" listens to each word.

**Step 3: New vector = weighted mix.**

```
new bank = 4% × the + 71% × river + 26% × bank
         = 0.04×[0,0] + 0.71×[3,0] + 0.26×[1,1]
         = [2.38, 0.26]                          ← leans WATER
```

## Result

| sentence | weights (the / other / bank) | "bank" after |
|---|---|---|
| the **river** bank | 4% / 71% / 26% | `[2.38, 0.26]` → water |
| the **money** bank | 4% / 71% / 26% | `[0.26, 2.38]` → money |

`identical? False`. The **same** token now ends up with a **different** vector depending on context. That's exactly what the tiny predictor couldn't do.

## Key points

- Attention is a **weighted average** of vectors. The weights come from **how related** the words are.
- A token looks at **itself and the words before it**, never after (the causal rule: Lesson 1.3).
- In a real model, **every** token does this at the same time, not just "bank".
- No new knobs were used here. Lesson 1.2 adds the learned part: **Query, Key, Value**, which let the model *learn* what counts as "related" instead of relying on raw dot products of the word vectors.

## Why softmax and not the raw scores?

It's a **relative** measure: what matters is how a word scores compared to the others.
- Raw `[0, 3, 2]` adds up to 5, so the "average" grows: `3×[3,0] + 2×[1,1] = [11, 2]`, bigger than any input, and it would explode over 8 blocks and longer sentences.
- Dot products can be negative, and a negative weight would flip a vector.
- Softmax weights are always **positive and add up to 100%**, so the result is a true average.

## Experiment: the weakness of this version

Change `bank` to `[2.0, 1.0]` (it now starts water-leaning) and look at "the money bank":

```
money  similarity 3  → 12%
bank   similarity 5  → 88%     ← mostly listens to itself
bank after: [1.75, 1.23]  → still WATER
```

The raw dot product rewards similarity to **what the word already is**, so a word can only find words like itself. But to work out its meaning, "bank" should look for a *different kind* of word (context about water or money). Likewise, "**it** was delicious": "it" needs to find "pizza", which looks nothing like "it".

Fix (Lesson 1.2): separate **what I'm looking for** (Query) from **what I am** (Key), using learned Linear layers.

## My summary

> Attention figures out which of the input tokens matter for the current token. It's a smart way of giving importance to each of the inputs.

**Refined:** Each token's new vector is a **weighted average of the earlier tokens' vectors**, so the same word ("bank") gets a different vector depending on context. Attention *refines each token's vector*; the next-word prediction happens only at the end (`lm_head`).

## Check yourself

<details><summary>What does a big dot product between two vectors mean?</summary>

They point in a similar direction: the words are "related" in the features the vectors encode.
</details>

<details><summary>Why use softmax on the similarities?</summary>

To turn them into positive weights that add up to 100%, so the new vector is a proper weighted average.
</details>

<details><summary>Why does "bank" end up different in the two sentences?</summary>

The earlier words differ (river vs money), so the mix pulls "bank" toward different vectors.
</details>
