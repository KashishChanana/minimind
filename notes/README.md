# MiniMind learning map

Step-by-step study of this repo, from zero. Each stage has a folder; each lesson has a script + README.
Run scripts from the `minimind/` folder: `.venv/bin/python notes/<stage>/<lesson>/<script>.py`

✅ done · 👉 next · 🟡 half done · ⬜ to do · 🖥️ needs a rented GPU

```
Progress  ████████░░░░░░░░░░░░  12 / 32 lessons   (attention: done ✅)
```

## Stage 0: Foundations ✅ — [stage00_foundations/](stage00_foundations/)
- ✅ 0.1 Tokens: text → IDs, BPE, special tokens, train your own tokenizer
- ✅ 0.2 How a model learns: guess → score → blame → nudge, learning rate
- ✅ 0.3 Tensors: shapes, Embedding, Linear, `view`
- ✅ 0.4 Tiny next-token predictor: shift-by-one, softmax, cross-entropy, why one token isn't enough

## Stage 1: The model (`model/model_minimind.py`) — [stage01_model/](stage01_model/)
**Attention**
- ✅ 1.1 The idea: a weighted average of earlier tokens
- ✅ 1.2 Query, Key, Value
- ✅ 1.3 Causal mask + attention shapes (+ shape quiz)
- ✅ 1.4 Multiple heads: 768 numbers split into 8 heads × 96
- ✅ 1.5 GQA: heads sharing keys/values to save memory
- ✅ 1.6 MiniMind's real `Attention` class, line by line (+ why ÷√96, QK-norm, fast path vs padding mask)
- 🟡 ★ Checkpoint: write attention from a blank file. Steps 1–3 done; redo Steps 4–8 unaided next time

**The rest of a block**
- ⬜ 1.7 MLP / SwiGLU: the "think alone" step (76% of the parameters)
- ⬜ 1.8 Residual connections + RMSNorm → one full block
- ⬜ 1.9 RoPE: how the model knows word order

**The whole model**
- ⬜ 1.10 Embedding → 8 blocks → `lm_head`; counting the 64M parameters
- ⬜ 1.11 Generating text: KV cache, temperature, top-k, top-p
- ⬜ 1.12 (optional) Mixture of Experts
- ⬜ ★ Checkpoint: build a tiny MiniMind from scratch

## Stage 2: Tokenizer ✅ — [stage02_tokenizer/](stage02_tokenizer/)
- ✅ 2.1 Inside the tokenizer: bytes, `Ġ`/`Ċ`, reserved slots, compression, why 6,400 tokens
- ✅ 2.2 The chat template: `<|im_start|>role … <|im_end|>`, generation prompt, thinking mode, how it stops

## Stage 3: Data pipeline (`dataset/lm_dataset.py`)
- ⬜ Download datasets · pretraining samples · loss masks · chat template for SFT data

## Stage 4: Pretraining (`trainer/train_pretrain.py`, `trainer_utils.py`)
- ⬜ The real training loop: AdamW, cosine LR, mixed precision, gradient accumulation, checkpoints
- ⬜ Tiny run on the Mac
- ⬜ 🖥️ **GPU session A:** real pretraining

## Stage 5: Supervised fine-tuning (`trainer/train_full_sft.py`, `eval_llm.py`)
- ⬜ Training only on assistant replies → a chat model
- ⬜ 🖥️ GPU session A (continued): SFT, then chat with your own model

## Stage 6: LoRA (`model/model_lora.py`, `trainer/train_lora.py`)
- ⬜ Low-rank adapters, written by hand (70 lines)

## Stage 7: Distillation (`trainer/train_distillation.py`)
- ⬜ Student learns from a teacher's probabilities (KL divergence)

## Stage 8: DPO (`trainer/train_dpo.py`)
- ⬜ Preference tuning from "chosen vs rejected" answers, without a reward model

## Stage 9: Reinforcement learning (`train_ppo.py`, `train_grpo.py`, `rollout_engine.py`)
- ⬜ Rewards, policy gradients, PPO → GRPO → CISPO

## Stage 10: Tool calling + agentic RL (`train_agent.py`, `scripts/eval_toolcall.py`)
- ⬜ Tool-call format, multi-turn rollouts, delayed rewards, the "alignment tax"
- ⬜ 🖥️ **GPU session B:** LoRA, distillation, DPO, RL runs on your own model

## 🤖 MiniMind said what?!

Real answers from the author's trained 64M model ([extras/chat_with_minimind.py](extras/chat_with_minimind.py)). Fluent, confident, often wrong, because the loss rewards **likely** next tokens, not **true** ones (Lesson 0.4).

| Asked | MiniMind said | Reality |
|---|---|---|
| Who proposed universal gravitation? | "...introduced by **George Washington in 1865**... during the Swiss mission" | Newton, 1687 |
| What is 15 + 27? | "15 + 27 = 31. 31 + 27 = 61. So the answer is **61**." | 42 |
| Write a poem about the ocean | "In the twilight's embrace, where the ocean's gentle light..." | honestly not bad |
| 你好，请介绍一下你自己 (introduce yourself) | "I'm MiniMind... my core strengths: **fast**, **efficient**, **precise**..." | humble it is not |
| 谁提出了万有引力定律？(who proposed gravity?) | explains the law, never says who, invents "F = Gm² + \|GM\| × charge law" | Newton; F = Gm₁m₂/r² |

*Add your own finds here:* `.venv/bin/python notes/extras/chat_with_minimind.py --chat`

## 📖 Words I know now

| Word | In one line | Lesson |
|---|---|---|
| token | a piece of text with an ID number; the model only ever sees IDs | 0.1 |
| BPE | builds the dictionary by gluing the most frequent pairs together | 0.1 |
| loss | one number for "how wrong"; training makes it go down | 0.2 |
| gradient | the slope: which way each knob should move to lower the loss | 0.2 |
| learning rate | step size; too small = slow, too big = explodes | 0.2 |
| tensor | a box of numbers with a shape | 0.3 |
| embedding | a lookup table: token ID → vector | 0.3 |
| Linear | multiply every vector by a learned weight matrix | 0.3 |
| softmax | scores → percentages that add up to 100% | 0.4 |
| cross-entropy | −log(probability given to the right answer) | 0.4 |
| attention | each token becomes a weighted average of earlier tokens | 1.1 |
| Q / K / V | what I'm looking for / how others find me / what I hand over | 1.2 |
| causal mask | −inf above the diagonal so nobody sees the future | 1.3 |
| head | one slice of the vector running its own attention | 1.4 |
| KV cache | saved K and V of past tokens, reused while generating | 1.5 |
| GQA | fewer K/V heads, shared by query heads, to shrink the KV cache | 1.5 |
| ÷√d | shrinks scores so softmax isn't winner-takes-all | 1.6 |
| byte | 8 bits = 256 possible values; all text is stored as bytes | 2.1 |
| chat template | wraps each message in `<\|im_start\|>role … <\|im_end\|>` | 2.2 |
| eos token | `<\|im_end\|>` (id 2): the model predicts it to end its turn | 2.2 |
| hallucination | a fluent, confident, made-up answer (see above 😄) | extras |

## How we keep it in memory
- Warm-up quiz on earlier lessons at the start of each session
- "My summary" in your own words at the end of each lesson
- ★ Rebuild-from-scratch checkpoints at the end of big sections
