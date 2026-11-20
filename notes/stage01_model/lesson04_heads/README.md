# Lesson 1.4: Multiple heads

## The problem

With one attention, each word gets **one** query, so it can look for only **one kind of thing** at a time. But a word often needs several things at once. "bank" might need its context word ("money"), its verb ("closed"), and the word right before it. One set of weights gives one weight table, which can't point at all of those well at the same time.

## The idea

**Cut each vector into chunks and run a separate attention in each chunk.** Each chunk is a **head**, with its own score table, so each head can learn to look for something different. Then glue the results back together.

MiniMind: 768 numbers per word = **8 heads × 96 numbers**. The demo: 4 numbers = 2 heads × 2.

## Run

```bash
.venv/bin/python notes/stage01_model/lesson04_heads/lesson04_heads.py
```

## Step by step (3 words, dim 4, 2 heads)

| Step | Code | Shape | What happens |
|---|---|---|---|
| 1 | `Q = q_proj(x)` | (3, 4) | exactly as before |
| 2 | `Q.view(3, 2, 2)` | (seq, heads, head_dim) | cut each word's 4 numbers into 2 chunks of 2. **No numbers change**: `[-1.39, 0.22, 0.06, -0.92]` → head0 `[-1.39, 0.22]`, head1 `[0.06, -0.92]` |
| 3 | `.transpose(0, 1)` | (heads, seq, head_dim) | put heads first: each head is now its own little (seq, head_dim) table |
| 4 | `Qh @ Kh.T` → softmax | **(heads, seq, seq)** | **one score table per head**. `@` treats the leading `heads` dimension as separate problems, so it does every head at once |
| 4 | `weights @ Vh` | (heads, seq, head_dim) | each head mixes its own chunk of the values |
| 5 | `.transpose(0, 1).reshape(3, 4)` | (seq, dim) | undo the transpose, join the chunks back into one vector per word |
| 6 | `o_proj(out)` | (seq, dim) | a final Linear that **mixes what the different heads found** |

The two heads produce **different** weight tables. Here, "cat" puts 84% on itself in head 0 but 55% in head 1. The weights are random, so the differences are small. In a trained model, heads specialise (previous word, matching noun, first token...).

## Splitting is free

`q_proj` is still 4 → 4 (MiniMind: 768 → 768). Splitting into heads **adds no weights**. It only changes how the numbers are grouped when computing scores. The only new layer is `o_proj`.

The cost: each head's dot products use only 96 numbers instead of 768. That's the trade: **several smaller views** instead of **one big view**.

## In MiniMind (`model/model_minimind.py`)

MiniMind has a **batch** dimension in front, so every dim index shifts by one:

| Line | Code | Shape |
|---|---|---|
| 113 | `xq = self.q_proj(x)` | (batch, seq, 768) |
| 114 | `xq.view(bsz, seq_len, 8, 96)` | (batch, seq, heads, head_dim) |
| 124 | `xq.transpose(1, 2)` | (batch, heads, seq, head_dim) |
| 128 | `xq @ xk.transpose(-2, -1)` | (batch, heads, seq, seq) |
| 132 | `output.transpose(1, 2).reshape(bsz, seq_len, -1)` | (batch, seq, 768) |
| 133 | `self.o_proj(output)` | (batch, seq, 768) |

(`transpose(1, 2)` here = our `transpose(0, 1)`: batch sits at index 0. `-1` in `reshape` means "work this size out yourself": 8 × 96 = 768.)

Lines 115–116 split K and V into `n_local_kv_heads` = **4** heads, not 8. That's GQA, Lesson 1.5.

## My summary

> Each head allows us to learn something different about the association amongst words. Splitting into heads doesn't add any weights to q_proj, because it's just a different view on it.

**Refined:** each head has its own score table (`(batch, heads, seq, seq)`), so a word can look for several things at once. The only new weights are `o_proj`, which mixes the heads' results.

## Check yourself

<details><summary>Why have several heads instead of one?</summary>

One head gives one weight table, so a word can look for one kind of thing. Several heads give several tables, so a word can look for several different things at once.
</details>

<details><summary>Does splitting into 8 heads add parameters to q_proj?</summary>

No. q_proj is still 768 → 768; view only regroups the numbers. The only extra layer is o_proj.
</details>

<details><summary>MiniMind, 100 tokens, batch 1: shape of the scores?</summary>

(1, 8, 100, 100): batch, heads, seq, seq. One 100 × 100 table per head.
</details>
