# Lesson 1.2: Query, Key, Value

## The problem (from Lesson 1.1)

With raw dot products (`my vector · their vector`), a word can only find words that **look like itself**. A water-leaning "bank" gave itself 88% of its attention and ignored "money". But to work out its meaning, "bank" needs to find a *different kind* of word: a context word.

## Run

```bash
.venv/bin/python notes/stage01_model/lesson02_qkv/lesson02_qkv.py
```

## The fix: three different vectors per token

Every token makes three new vectors from its own vector, each with its **own Linear layer**:

| | Question it answers | Made by |
|---|---|---|
| **Query (Q)** | "What am I looking for?" | `q_proj` |
| **Key (K)** | "What do I offer? How should others find me?" | `k_proj` |
| **Value (V)** | "What do I hand over if someone picks me?" | `v_proj` |

**Score = my Query · their Key**, not my vector · their vector. Then mix their **Values**.

**Analogy: searching a video site.** You type a **query** ("how to fix a bike"). Each video has a title and tags: its **key**. The site matches your query against keys, not against the videos themselves. Then you watch the matching videos' **content**: their **values**. What you search with, what you're found by, and what you deliver are three different things.

## The demo

Each word has 3 numbers: `[water, money, needs-context]`. `bank = [2, 1, 1]` leans water **and** is ambiguous. The weights are hand-picked:

| Layer | Hand-picked rule | Meaning |
|---|---|---|
| `q_proj` | query = 3 × needs-context | ambiguous words go looking for context |
| `k_proj` | key = water + money − 3 × needs-context | clear, meaningful words are easy to find; ambiguous words hide |
| `v_proj` | value = (water, money, 0) | hand over meaning, not the "needs context" flag |

## Result

```
"the money bank"   bank's query = 3
word    key   q·k   NEW weight   old (1.1) weight
the      0     0        0%            0%
money    3     9      100%            5%
bank     0     0        0%           95%
bank after = [0, 3, 0] → MONEY
```

In "the river bank" it becomes `[3, 0, 0]` → WATER. Even though "bank" started water-leaning, it now finds the right context word in **both** sentences, because what it searches with (its query) is no longer the same as what it is.

## Key points

- Q, K, V are just three **Linear layers** applied to every token's vector (Lesson 0.3).
- Their weights are **knobs**. Here they're hand-picked; in MiniMind they're **learned** by the training loop (Lesson 0.2). Training discovers useful things to look for: "find my noun", "find the previous verb"...
- In MiniMind they're `q_proj`, `k_proj`, `v_proj` in `model/model_minimind.py` (lines 101–103), and Q and K are 96 numbers per head, not 1.
- **Where did bank's own meaning go?** Here, `bank after = [0, 3, 0]` lost its original vector. The real model **adds the original back** after attention (the residual connection, in a later lesson), so a token keeps its own meaning and gains context.

## Why do we need V? How is it different from K?

- **Q · K only gives percentages** (`seq × seq`): "listen 70% to money". 70% *of what*? That's V. `weights @ V` turns "who to listen to" into an actual vector per word.
- **K and V are built the same way** (both a Linear applied to `x`) but have **different jobs**. K is only ever *compared* with queries, to decide the weights, and never reaches the output. V is only ever *passed on*. K = the advertisement, V = the product delivered. Different jobs → different nudges during training → different learned numbers.
- **Couldn't we just mix the raw `x`** (`out = weights @ x`)? Yes, it works. But then a word can only send **a copy of itself**: `new bank = bank + 0.7 × money` makes "bank" partly turn into "money". With V, "money" can send a **message written for others** ("the context is finance"). V is an upgrade, not a necessity.
- `lesson02b_do_we_need_v.py` trains both versions on the README. With V fits the training text better (1.88 vs 2.44), but the README is too small for either to generalise (test loss about 7 for both), so the tiny experiment can't show V's benefit. That needs real amounts of data.

## My summary

> Q tells me what I need (the query). K tells me what the other tokens advertise themselves as. V is what a token passes on if it gets chosen.

**Refined:** **Why three instead of one:** with a single vector, a word can only find words that look like itself (the `[2, 1]` experiment: "bank" gave 88% to itself and ignored "money"). Separating what I'm looking for (Q) from how others find me (K) lets a word find a *different kind* of word ("it" → "pizza").

## Check yourself

<details><summary>Why can't attention just use my vector · their vector?</summary>

Then a word only finds words similar to itself. Words often need a different kind of word ("it" needs "pizza"), so what you search with must differ from what you are.
</details>

<details><summary>What are Q, K, V in one line each?</summary>

Q: what I'm looking for. K: how others find me. V: what I hand over when picked.
</details>

<details><summary>Score is computed from which two? The output is a mix of which?</summary>

Score = my Query · each earlier token's Key. Output = weighted mix of their Values.
</details>
