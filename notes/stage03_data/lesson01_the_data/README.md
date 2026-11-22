# Lesson 3.1: What the training data looks like

## Get the data (a small sample)

The full "mini" datasets are 1.2 GB (pretraining) and 1.7 GB (SFT). To learn the pipeline, a sample is enough. This downloads 4 × 1.5 MB chunks from different parts of each file (the files are grouped by type, so the start alone isn't representative):

```bash
.venv/bin/python notes/stage03_data/get_sample_data.py      # → dataset/*_sample.jsonl (git-ignored)
.venv/bin/python notes/stage03_data/lesson01_the_data/lesson01_the_data.py
```

The full files get downloaded on the rented GPU in Stage 4.

## The format: JSONL

**JSONL = one JSON object per line.** Each line is one training example. Files can be read line by line, so they never have to fit in memory all at once.

## Pretraining data: `pretrain_t2t_mini.jsonl`

```json
{"text": "生成一个英文诗歌！Sure, I can generate a poem for you. ...In the depth of my heart, There's a feeling so true..."}
```

- Just **raw text**, one string per line. No roles, no special tokens.
- Mostly Chinese, with some English mixed in.
- "t2t" = text-to-text. The text often contains question-and-answer pairs glued together **without** a chat template, so the model picks up the rhythm of Q&A before SFT formalises it.
- Sample: median **249 tokens**; **24%** are longer than `max_seq_len = 340` and get cut off.

**Used for:** Stage 4 (pretraining), learning language itself by predicting the next token everywhere.

## SFT data: `sft_t2t_mini.jsonl`

```json
{"conversations": [
  {"role": "user",      "content": "..."},
  {"role": "assistant", "content": "...", "reasoning_content": "..."}
]}
```

- A list of **messages with roles**, the same shape `apply_chat_template` takes (Lesson 2.2).
- `reasoning_content` (optional) becomes the text inside `<think>…</think>`. **44%** of the sample has it.
- `tools` / `tool_calls` (optional) are for tool use (Stage 10). None appeared in our sample.
- No system messages in the data itself; `lm_dataset.py` adds one at random (Lesson 3.3).
- Median **2 messages** (one question, one answer), up to 16.
- Sample: median **538 tokens** after the chat template; **6%** exceed `max_seq_len = 768`.

**Used for:** Stage 5 (SFT), learning to be an **assistant**: answer in turns, stop with `<|im_end|>`.

## The data has quirks

The English SFT example the script found:

> **user:** "It takes off—still cartwheeling end over end... cvsindceaui an acentual progression is forming in despair..."
> **assistant:** "It seems like you've typed in a jumbled string of words... Could you please provide more context?"

A deliberately garbled question, teaching the model to ask for clarification. Real datasets are full of things like this: odd, noisy, sometimes low-quality. The model learns whatever is in the data, including its mistakes.

## Pretraining vs SFT data

| | Pretraining | SFT |
|---|---|---|
| Format | `{"text": ...}` | `{"conversations": [...]}` |
| Structure | none: plain text | roles + chat template |
| Teaches | language, facts, patterns | being an assistant: turns, format, stopping |
| `max_seq_len` | 340 | 768 |

## Check yourself

<details><summary>What does each line of a .jsonl file hold?</summary>

One training example, as a JSON object.
</details>

<details><summary>What's the main difference between pretraining and SFT data?</summary>

Pretraining is plain text (learn language); SFT is role-tagged conversations (learn to act as an assistant).
</details>

<details><summary>What happens to an example longer than max_seq_len?</summary>

It's cut off (truncated) at max_seq_len tokens.
</details>
