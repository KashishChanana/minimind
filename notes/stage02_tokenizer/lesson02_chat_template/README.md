# Lesson 2.2: The chat template

The model only ever sees **one long string of tokens**. It has no built-in idea of "messages" or "who is speaking". The **chat template** (the `chat_template` field in `model/tokenizer_config.json`) turns a list of messages into that string, using special tokens as markers.

## Run

```bash
.venv/bin/python notes/stage02_tokenizer/lesson02_chat_template/lesson02_chat_template.py
```

Part D uses the trained weights from `notes/extras/chat_with_minimind.py`, and is skipped if they aren't downloaded.

## A. A conversation as text

```python
[{"role": "system",    "content": "You are a helpful assistant."},
 {"role": "user",      "content": "What is 2 + 2?"},
 {"role": "assistant", "content": "2 + 2 = 4."}]
```
becomes:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
<think>

</think>

2 + 2 = 4.<|im_end|>
```

- `<|im_start|>` (id 1) + a role name opens a message; `<|im_end|>` (id 2) closes it. ("im" = "input message".)
- The role names (`system`, `user`, `assistant`) are **plain text tokens**, not special ones.
- **system**: background instructions; **user**: the person; **assistant**: the model.
- Every assistant message gets a `<think>…</think>` block, empty here.

## B. The prompt for generating a reply

`add_generation_prompt=True` adds the **start of an assistant turn** to the end:
```
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
<think>

</think>

```
The model just continues the text from there. Generating a reply is still **next-token prediction** (Lesson 0.4); the template sets things up so the most likely continuation is "the assistant's answer".

## C. Thinking mode

With `open_thinking=True`, the prompt ends with an **open** `<think>`:
```
<|im_start|>assistant
<think>
```
so the model first writes its reasoning, then `</think>`, then the answer. With thinking off, an **empty, already-closed** block is inserted, so the model skips straight to the answer.

## D. How the model knows when to stop

```
Hello, world! How can I help you today?<|im_end|>
last token generated: id 2 = <|im_end|>
```

During training (SFT, Stage 5), every assistant reply in the data ends with `<|im_end|>`, so the model learns to **predict that token when its answer is complete**. `generate()` is told `eos_token_id = 2` ("end of sequence") and stops as soon as the model produces it. `skip_special_tokens=True` hides the markers when showing the answer to a person.

If a model never produces it, it rambles until `max_new_tokens` runs out (as in the first try, with a long maths answer).

## Why it matters

- **The template must match training exactly.** A model trained on this format gets confused by a different one (other markers, missing `<think>` block...).
- Tools (Stage 10) use the same template: tool descriptions go in the system message, and the model writes `<tool_call>…</tool_call>` inside its turn.

## Check yourself

<details><summary>How does the model know who is speaking?</summary>

It doesn't "know" in any special way: each message is wrapped in <code>&lt;|im_start|&gt;role … &lt;|im_end|&gt;</code>, and it learned the pattern during training.
</details>

<details><summary>What does add_generation_prompt=True add, and why?</summary>

<code>&lt;|im_start|&gt;assistant</code> (+ the think block), so the most likely continuation is the assistant's reply.
</details>

<details><summary>How does generation stop?</summary>

The model predicts <code>&lt;|im_end|&gt;</code> (id 2), which it learned ends every assistant reply; <code>generate()</code> stops when it sees the eos id.
</details>
