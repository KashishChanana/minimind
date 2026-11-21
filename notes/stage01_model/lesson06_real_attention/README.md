# Lesson 1.6: MiniMind's real Attention class

Reading `model/model_minimind.py:92-134` line by line. You already know most of it. Three pieces are new: **÷√96**, **QK-norm**, and **RoPE** (a peek).

## Run

```bash
.venv/bin/python notes/stage01_model/lesson06_real_attention/lesson06_real_attention.py
```

- **Part A** shows why scores are divided by √96.
- **Part B** rebuilds `Attention.forward` step by step with the real class's own weights, and checks the result matches exactly: `my rebuild == real Attention class? True`.

## New piece 1: ÷√96 (line 128)

A dot product of two 96-number vectors **adds up 96 products**, so scores come out large: their spread is about √96 ≈ 9.8.

```
raw scores        : [-8.3, -5.6, -0.7, -6.4, -3.1]
softmax(raw)      : [0%, 1%, 91%, 0%, 8%]       ← one key grabs almost everything
scores / √96      : [-0.8, -0.6, -0.1, -0.7, -0.3]
softmax(scaled)   : [14%, 18%, 29%, 16%, 23%]   ← a real spread
```

Big scores make softmax **winner-takes-all** from the very start of training. The other tokens get about 0%, their gradients are almost zero, and they barely learn. Dividing by √96 brings the spread back to about 1, so softmax starts out spread and can learn which tokens matter. (The spread of a sum of *n* random products grows like √n, which is why it's √96 and not 96.)

## New piece 2: QK-norm (lines 104–105, 117)

`q_norm` and `k_norm` **rescale each head's query and key vector to a standard size** before the dot product. It's an extra guard against scores growing huge during training. (RMSNorm, the rescaling function, is explained in Lesson 1.8.) Shape unchanged.

## New piece 3: RoPE (line 119), a peek

`apply_rotary_pos_emb` mixes **position information** into Q and K, so attention knows word order. Without it, "dog bites man" and "man bites dog" look identical. Shape unchanged. Full lesson: 1.9.

## The whole class, line by line

**Setup (`__init__`, lines 92–109)**

| Line | Code | Meaning |
|---|---|---|
| 95–97 | `n_local_heads = 8`, `n_local_kv_heads = 4`, `n_rep = 2` | Q heads, K/V heads, copies per K/V head (GQA) |
| 98 | `head_dim = 96` | 768 ÷ 8 |
| 99 | `is_causal = True` | use the causal mask |
| 100–103 | `q_proj` 768→768, `k_proj`/`v_proj` 768→384, `o_proj` 768→768 | the four Linear layers |
| 104–105 | `q_norm`, `k_norm` | QK-norm |
| 106–108 | dropout | `dropout = 0.0` in the config → does nothing |
| 109 | `self.flash` | use PyTorch's fast built-in when possible |

**Forward (lines 111–134)**

| Line | What | Shape (batch 1, 5 tokens) |
|---|---|---|
| 113 | Q, K, V projections | Q (1,5,768), K/V (1,5,384) |
| 114–116 | `view` into heads | Q (1,5,8,96), K/V (1,5,4,96) |
| 117 | QK-norm | unchanged |
| 119 | RoPE | unchanged |
| 120–122 | **KV cache:** stick saved past K/V in front of the new ones | seq grows |
| 123 | save K/V for next time (only 4 heads stored) | |
| 124 | `repeat_kv` (4→8) + `transpose` | all (1,8,5,96) |
| 125–126 | **fast path:** `F.scaled_dot_product_attention(..., is_causal=True)`, the same math as 128–131 in one optimized call | (1,8,5,96) |
| 128 | scores = Q·Kᵀ / √96 | (1,8,5,5) |
| 129 | + causal mask | (1,8,5,5) |
| 130 | + **padding mask** (see below) | (1,8,5,5) |
| 131 | softmax → `@ V` | (1,8,5,96) |
| 132 | glue heads back | (1,5,768) |
| 133 | `o_proj` | (1,5,768) |

**Line 130, padding mask:** sentences in a batch have different lengths, so short ones get filler tokens at the end to make a rectangle. `attention_mask` marks real tokens 1 and filler 0, and filler positions get −1e9 so nobody attends to them. Same idea as the causal mask, different reason.

**Line 125, when the fast path is skipped:** when generating with the KV cache (`past_key_value` present), or when there's padding. Then lines 128–131 run instead. Same result either way.

## Check yourself

<details><summary>Why divide the scores by √96?</summary>

Dot products of 96-number vectors are large (spread about 9.8), which makes softmax winner-takes-all and stalls learning. Dividing brings the spread back to about 1.
</details>

<details><summary>What do lines 125–126 do differently from 128–131?</summary>

Nothing mathematically. It's PyTorch's optimized built-in for the same attention calculation, including the causal mask.
</details>

<details><summary>What's the difference between the causal mask and the padding mask?</summary>

Causal blocks the future (always). Padding blocks filler tokens added to make sentences in a batch the same length.
</details>
