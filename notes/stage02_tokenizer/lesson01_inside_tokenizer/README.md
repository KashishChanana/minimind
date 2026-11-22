# Lesson 2.1: Inside the tokenizer

Lesson 0.1 covered what a tokenizer does. This lesson covers three details of how MiniMind's is built (`trainer/train_tokenizer.py`).

## Run

```bash
.venv/bin/python notes/stage02_tokenizer/lesson01_inside_tokenizer/lesson01_inside_tokenizer.py
```

## A. Everything is bytes

Computers store text as **bytes**: numbers from 0 to 255. One character can take several bytes:

| Character | Bytes |
|---|---|
| `a` | 1: `[97]` |
| space | 1: `[32]` |
| `é` | 2: `[195, 169]` |
| `你` | **3**: `[228, 189, 160]` |

MiniMind's tokenizer is **byte-level** (`pre_tokenizers.ByteLevel`, line 47). Its starting alphabet is the 256 possible bytes (`initial_alphabet`, line 64), and BPE glues bytes into bigger tokens. Because every possible byte is in the dictionary, **any** text in any language can be encoded, even emoji.

**The weird symbols in `tokenizer.json`:** some bytes (like space or newline) aren't printable, so each of the 256 bytes gets a **printable stand-in symbol**:

| Text | Stored as | Why |
|---|---|---|
| ` the` | `Ġthe` | `Ġ` = stand-in for the space byte |
| newline | `Ċ` | `Ċ` = stand-in for the newline byte |
| `你` | `ä½ł` | 3 bytes → 3 stand-ins glued together |
| `你好` | `ä½łå¥½` | 6 bytes, learned as one token |

This also explains the `�` from Lesson 0.1b: your README-trained tokenizer never learned to glue the 3 bytes of a Chinese character together, so each byte was decoded alone, and a lone piece of a character can't be displayed.

## B. 36 reserved slots for special tokens

IDs **0–35** are reserved before BPE starts (`SPECIAL_TOKENS_NUM = 36`, line 10):

| IDs | Tokens | Used for |
|---|---|---|
| 0–2 | `<\|endoftext\|>`, `<\|im_start\|>`, `<\|im_end\|>` | padding, message start/end |
| 3–20 | vision / audio / box tokens | unused leftovers |
| 21–24 | `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` | tool use (Stage 10) |
| 25–26 | `<think>`, `</think>` | reasoning blocks (Lesson 2.2) |
| 27–35 | `<\|buffer1\|>` … `<\|buffer9\|>` | **empty spare slots** for future special tokens |

The first ordinary token, ID 36, is `!`. Spare slots matter because adding a token *later* would change the dictionary size, and every trained model built on it would no longer fit.

## C. Compression, and why only 6,400 tokens

**Compression** = characters per token. Higher means the same text takes fewer tokens:

| Text | Chars/token |
|---|---|
| English | **4.04**, about one token per word |
| Chinese | **1.60**, about one token per 1–2 characters |

(The author's `eval_tokenizer()` measures a 2.66 average over 5 longer samples.)

A bigger dictionary would compress better (common phrases would get their own tokens), but **every token needs its own 768-number row** in the embedding table:

| Model | Vocab | Embedding | Share of a MiniMind-sized model |
|---|---|---|---|
| **MiniMind** | 6,400 | 4.9M | **8%** |
| Llama 2 | 32,000 | 24.6M | 29% |
| Qwen | 151,643 | 116.5M | **66%** |

With Qwen's dictionary, two-thirds of MiniMind would be lookup table, leaving little for the blocks that actually think. Large models can afford big dictionaries; a 64M model can't. **The trade-off:** a small dictionary keeps the model's capacity for thinking, but text takes more tokens and rare words get chopped up (like `Einstein` → 5 pieces in Lesson 0.1).

## Why only 256 bytes?

A byte = 8 bits, each 0 or 1, so 2⁸ = **256** possible values (0–255). That's how memory is built. Bigger characters use **several bytes** (UTF-8: 1–4): `A` = 1, `é` = 2, `你` = 3, `😀` = 4 `[240, 159, 152, 128]`. Like 26 letters spelling any word, 256 bytes combined write any character. MiniMind has no `😀` token, so it falls back to byte pieces: `[4544, 282, 258]`. Ugly, but it never fails.

## Two sizes not to mix up

| | What it is | What it affects |
|---|---|---|
| **vocab size** | entries in the dictionary (6,400) | the **embedding table** (one 768-number row each) |
| **sequence length** | tokens in a particular text | the **seq × seq attention scores** |

Bigger dictionary → **shorter** texts (cheaper attention) but a **bigger** embedding table. Raw bytes → tiny table but texts about 4× longer (attention about 16× bigger). MiniMind picks a small dictionary because at 64M parameters every parameter counts.

**Experiment** (`lesson01b_vocab_vs_length.py`): BPE tokenizers trained on the README with different vocab sizes, all encoding the same 82-byte sentence:

| Vocab | Tokens | Example pieces |
|---|---|---|
| bytes | 82 | one per byte |
| 300 | 54 | `' t' 'ra' 'in' 'ed'` |
| 1,000 | 26 | `' train' 'ed'` |
| 3,000 | 22 | `' trained'` |
| 6,000 | 20 | `' trained'` `' next'` |

More entries = more whole words have their own token = fewer pieces per sentence. Returns shrink: 300→1,000 halves the length, 3,000→6,000 saves only 2 tokens, because the extra entries are rarer chunks.

## ASCII / UTF-8 vs the tokenizer

ASCII/UTF-8 is a **fixed, universal** table (character → bytes) for **storing** text. The tokenizer is **learned shorthand, different for every model**, gluing frequent byte sequences into chunks for **feeding** the model. `"the cat sat"` = 11 bytes → 3 tokens. Space is byte 32 because in ASCII (1960s) slots 0–31 were machine instructions and space was the first real character; GPT-2's byte-to-symbol rule maps it to character #256 + 32 = #288 = `Ġ`.

## Check yourself

<details><summary>Why can a byte-level tokenizer encode any text?</summary>

All 256 bytes are in the starting alphabet, and all text is made of bytes, so the worst case is one token per byte.
</details>

<details><summary>What does <code>Ġ</code> mean in tokenizer.json?</summary>

It's the printable stand-in for a space byte. <code>Ġthe</code> = " the".
</details>

<details><summary>Why doesn't MiniMind use a 150k-token dictionary?</summary>

The embedding table would be about 116M parameters, two-thirds of a MiniMind-sized model, leaving little capacity for the transformer blocks.
</details>
