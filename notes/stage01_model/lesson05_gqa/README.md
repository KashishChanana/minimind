# Lesson 1.5: GQA (Grouped-Query Attention)

## Run

```bash
.venv/bin/python notes/stage01_model/lesson05_gqa/lesson05_gqa.py
```

## Background: the KV cache

When generating, the model writes **one token at a time**. For each new token, attention needs:
- the **new token's Query** (what am I looking for?)
- the **Keys and Values of every earlier token**

Earlier tokens' K and V **never change**, because later tokens can't affect earlier ones (the causal mask; proved in Lesson 1.3). So instead of recomputing them for every new token, the model **saves** them. That saved store is the **KV cache**.

- **Only K and V are cached, not Q.** Each step only needs the *newest* token's query; old queries are never used again.
- The cache **grows with every token** and is kept for every layer. That costs memory, especially for long texts or many users at once.

## The idea: fewer K/V heads, shared by query heads

Keep **8 query heads**, but make only **4 key/value heads**. Each K/V head is **shared by 2 query heads**. This halves `k_proj`/`v_proj` and **halves the KV cache**.

| Design | Q heads | K/V heads | `k_proj` | KV cache, 2,048 tokens | 32,768 tokens |
|---|---|---|---|---|---|
| MHA (no sharing) | 8 | 8 | 768 × 768 | 50.3 MB | 805 MB |
| **GQA (MiniMind)** | 8 | **4** | **768 × 384** | **25.2 MB** | **403 MB** |
| MQA (all share one) | 8 | 1 | 768 × 96 | 6.3 MB | 101 MB |

**Trade-off:** fewer K/V heads = less memory, but fewer different things for queries to look up, so quality can drop a little. GQA is the middle ground used by most modern models (Llama, Qwen, MiniMind). For tiny MiniMind the savings are small; for big models serving many users they're gigabytes.

## How: `repeat_kv` copies each K/V head

`Q @ Kᵀ` needs Q and K to have the same number of heads (8). So MiniMind **copies** each of the 4 K/V heads twice (`model_minimind.py:86-89`):

```
K:  (batch, seq, 4, 96)  → repeat_kv(K, n_rep=2) →  (batch, seq, 8, 96)
kv-head A → query heads 0, 1     kv-head B → query heads 2, 3     ...
```

The demo, with 2 K/V heads → 4 query heads:
```
query head 0 -> key [1, 1]  (kv-head A)
query head 1 -> key [1, 1]  (kv-head A)
query head 2 -> key [2, 2]  (kv-head B)
query head 3 -> key [2, 2]  (kv-head B)
```

The copy happens **after** the cache, so only 4 heads are ever stored. After `repeat_kv`, everything is the normal multi-head attention from Lesson 1.4.

## In MiniMind (`model/model_minimind.py`)

| Line | Code | Shape |
|---|---|---|
| 23 | `num_key_value_heads = 4` (config) | |
| 102–103 | `k_proj`, `v_proj` = `Linear(768, 4 × 96)` | 768 → 384 |
| 115–116 | `xk.view(bsz, seq_len, 4, 96)` | (batch, seq, 4, 96) |
| 121–123 | `past_key_value` concat, `past_kv = (xk, xv)` | the KV cache (4 heads) |
| 124 | `repeat_kv(xk, self.n_rep)` with `n_rep = 8 // 4 = 2` | (batch, seq, 8, 96) |

## Shapes recap (from our discussion)

```
Q:     x (50 × 768) → q_proj → 50 × 768 → view → 50 × 8 × 96
K, V:  x (50 × 768) → k_proj → 50 × 384 → view → 50 × 4 × 96 → repeat_kv → 50 × 8 × 96
```

- **Projection output = heads × head_dim.** head_dim stays 96 (768 ÷ 8) because each K head is dot-producted with a Q head, so they must be the same length. 4 heads × 96 = **384**.
- Q (768) and K (384) are **never multiplied at full size**. After view + repeat_kv, both are (…, 8, 96), and every dot product is 96 · 96.
- `repeat_kv` doubles each head **in place**: `A B C D` → `A A B B C C D D`, so neighbouring query heads form a group that shares one K/V head.

## Check yourself

<details><summary>Why can earlier tokens' K and V be saved and reused?</summary>

The causal mask means later tokens never change earlier ones, so their K and V stay the same.
</details>

<details><summary>Why is Q not cached?</summary>

Each generation step only uses the newest token's query; old queries are never needed again.
</details>

<details><summary>MiniMind: how many query heads share each K/V head, and which function does the sharing?</summary>

8 ÷ 4 = 2 query heads per K/V head; `repeat_kv` copies each K/V head twice.
</details>
