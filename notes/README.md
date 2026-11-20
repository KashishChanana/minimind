# MiniMind learning map

Step-by-step study of this repo, from zero. Each stage has a folder; each lesson has a script + README.
Run scripts from the `minimind/` folder: `.venv/bin/python notes/<stage>/<lesson>/<script>.py`

✅ done · 👉 next · ⬜ to do · 🖥️ needs a rented GPU

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
- 👉 1.4 Multiple heads: 768 numbers split into 8 heads × 96
- ⬜ 1.5 GQA: heads sharing keys/values to save memory
- ⬜ 1.6 MiniMind's real `Attention` class, line by line (+ why ÷√96, real trained attention weights)
- ⬜ ★ Checkpoint: write attention from a blank file, compare with MiniMind

**The rest of a block**
- ⬜ 1.7 MLP / SwiGLU: the "think alone" step (76% of the parameters)
- ⬜ 1.8 Residual connections + RMSNorm → one full block
- ⬜ 1.9 RoPE: how the model knows word order

**The whole model**
- ⬜ 1.10 Embedding → 8 blocks → `lm_head`; counting the 64M parameters
- ⬜ 1.11 Generating text: KV cache, temperature, top-k, top-p
- ⬜ 1.12 (optional) Mixture of Experts
- ⬜ ★ Checkpoint: build a tiny MiniMind from scratch

## Stage 2: Tokenizer (short; mostly covered in 0.1)
- ⬜ `trainer/train_tokenizer.py` read-through, compression ratio, chat template

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

## How we keep it in memory
- Warm-up quiz on earlier lessons at the start of each session
- "My summary" in your own words at the end of each lesson
- ★ Rebuild-from-scratch checkpoints at the end of big sections
