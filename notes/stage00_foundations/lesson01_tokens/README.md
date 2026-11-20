# Lesson 0.1: Text → numbers (tokens)

A neural network can only do math, so text has to become numbers first. A **tokenizer** chops text into pieces (**tokens**) and swaps each piece for its ID in a fixed dictionary. MiniMind's dictionary has **6,400** entries.

## Run

```bash
# from the minimind/ folder
.venv/bin/python notes/stage00_foundations/lesson01_tokens/lesson01_tokens.py
.venv/bin/python notes/stage00_foundations/lesson01_tokens/lesson01b_make_tokenizer.py
```

| File | What it does |
|---|---|
| `lesson01_tokens.py` | Encodes a few sentences with MiniMind's real tokenizer and shows the pieces |
| `lesson01b_make_tokenizer.py` | Trains **your own** tokenizer on `README_en.md` using the author's `train_tokenizer()`, and compares it with the real one |
| `my_tokenizer/` | Output of 01b: your own `tokenizer.json` + `tokenizer_config.json` |

## What you'll see

| Text | Tokens | Lesson |
|---|---|---|
| `the cat sat on the mat` | `the` · ` cat` · ` sat` · ` on` · ` the` · ` mat` | Common words = 1 token. **The space is part of the token**: `the` (4345) and ` the` (309) are different entries. |
| `unbelievably` | `un` · `b` · `el` · `ie` · `v` · `ab` · `ly` | Rare words get split into pieces, so any text can still be written. |
| `Einstein` | `E` · `in` · `st` · `e` · `in` | The model never sees "Einstein" as one thing. |
| `你好，世界` | `你好` · `，` · `世界` | Chinese gets whole-word tokens, because MiniMind was trained mostly on Chinese. |

**Main point:** the model never sees letters or words. It sees a list of IDs, and its whole job is to guess **the next ID**.

## How the tokenizer decides the pieces: BPE

It's **not** stems or lemmas; the tokenizer knows no grammar. It's based on **frequency** (Byte-Pair Encoding):

1. Start with single characters (bytes).
2. Find the pair that appears side by side most often in the training text, and glue it into a new token (`t` + `h` → `th`).
3. Repeat until the dictionary has 6,400 entries.

Single characters are always in the dictionary, so **any** text can be encoded. Unseen words just take more tokens.

When you train your own tokenizer on the README (01b), its first merges are the README's most common pairs: `i+n`, ` +t`, `o+n`, `r+e`... (`Ġ` = a space in the file). It even gives `iniMind` its own token, because the README says "MiniMind" constantly. But it breaks Chinese into broken bytes (`�`), because it hardly saw any. **A tokenizer handles well the text it saw a lot of.**

## The two files in `model/`

`from_pretrained(folder)` loads **both**:

| File | Contents | How it's made |
|---|---|---|
| `tokenizer.json` | The 6,400-token dictionary + 6,108 merge rules | **Learned from data** (`train_tokenizer.py` line 76) |
| `tokenizer_config.json` | Settings: special tokens, max length, **chat template** | **Hand-written** Python dictionary dumped to JSON (lines 99–126) |

**Special tokens** (IDs 0–35) are control signals, not text:
- `<|im_start|>` (1) / `<|im_end|>` (2): start/end of a chat message. When the model outputs 2, it's done talking.
- `<|endoftext|>` (0): used as padding.
- `<tool_call>` (21): "I want to use a tool" (Stage 10).
- The vision/audio ones are leftovers from the tokenizer this was copied from, and unused.

**Don't retrain the real tokenizer.** New IDs would make the author's weights and datasets useless.

## Side note: why `os.path.join(ROOT, "model")`?

`"model"` alone is a **relative path**. Python resolves it from the folder you **run the command from**, not where the script lives. `ROOT` is built from the script's own location (`__file__`), so the script works from anywhere. `os.path.join` inserts the right separator for the OS. You'll see this pattern near the top of most scripts in the repo.

## Check yourself

<details><summary>The dictionary has only 6,400 entries. How can it write any sentence?</summary>

Single characters/bytes are always in the dictionary, so an unseen word gets spelled out from small pieces. It just costs more tokens.
</details>

<details><summary>Why are <code>the</code> and <code> the</code> different tokens?</summary>

The leading space is part of the token. A word at the start of the text has no space before it, so it's a different piece.
</details>
