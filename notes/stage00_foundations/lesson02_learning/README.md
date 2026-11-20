# Lesson 0.2: How a model learns

The most important idea in the repo. **Every training script in MiniMind (pretraining, SFT, LoRA, DPO, PPO, GRPO) is this loop, with a different definition of "wrong".**

## Run

```bash
.venv/bin/python notes/stage00_foundations/lesson02_learning/lesson02_learning.py
```

## The setup

Secret rule: **y = 3x + 2**. The model doesn't know it. It has two adjustable numbers, called **parameters** (weights): `w` and `b`. It computes `guess = w·x + b`. Both start random, and it has to discover w=3, b=2 from 20 examples.

```
start:   w=+1.541  b=-0.293
step  0: loss=8.3968   w=+1.971  b=+0.165
step 10: loss=0.0635   w=+2.969  b=+1.803
step 30: loss=0.0000   w=+3.000  b=+1.998
```

## The 4-step loop

```
1. GUESS   guess = w*x + b                  run the model on examples
2. SCORE   loss = ((guess - y)**2).mean()   one number: how wrong are we?
3. BLAME   loss.backward()                  for each knob: which way lowers the loss?
4. NUDGE   w -= lr * w.grad                 move each knob a little that way
   ...repeat
```

- **Loss:** a single number for "how wrong". 0 = perfect. Training is making it go down.
- **Gradient (`.backward()`):** you're on a foggy hillside and can only feel the slope under your feet. The gradient is that slope, computed for every parameter at once. PyTorch does it automatically.
- **Why subtract?** The gradient points **uphill**, toward a higher loss. We step the opposite way. That's why it's called **gradient descent**.
- **`requires_grad=True`:** "track how changing me affects the loss".
- **`w.grad.zero_()`:** gradients add up by default, so reset them every step.

## Learning rate: how big a step

| What the loss does | Diagnosis | Seen at |
|---|---|---|
| Falls slowly, still falling at the end | **Too low** | `lr=0.01` (loss 2.09 after 30 steps) |
| Falls fast, then settles smoothly | **About right** | `lr=0.1` |
| Bounces or spikes up | **A bit too high** | `lr=1.1`, around step 10–15 |
| Explodes / `nan` | **Way too high**: diverged | `lr=1.1`, step 20+ (w = 852,575!) |

At `lr=1.1`, the model **started at the perfect answer** and still blew up: each step overshoots the valley and lands higher on the other side.

**How to find a good learning rate:**
1. **Watch the loss curve.** Read the symptoms above.
2. **Sweep** values spaced by ×10 (`1e-5, 1e-4, 1e-3`), each for a short run. Pick the largest that falls smoothly, then back off a little.
3. **Start from known values.** MiniMind's defaults:

| Stage | LR | |
|---|---|---|
| Pretraining | `5e-4` | random model → big steps |
| LoRA | `1e-4` | |
| SFT | `1e-5` | |
| Distillation | `5e-6` | |
| PPO / GRPO / agent | `3e-7` | |
| DPO | `4e-8` | ~10,000× smaller than pretraining |

**Later stage → smaller learning rate:** they fine-tune a model that already works, and big steps would wreck it ("catastrophic forgetting").

4. **Change it over time:** MiniMind lowers the learning rate from 100% → 10% along a cosine curve (`trainer_utils.py:42`). Covered in Stage 4.

**Rule of thumb:** loss explodes or turns `nan` → divide the LR by 10. Loss crawls → multiply it by 3.

## Experiment: compare several learning rates

The script loops over `[0.01, 0.1, 1.1]`. Two things make the comparison fair:
- Create **fresh** `w`, `b` inside the loop. Otherwise each run continues from where the last one stopped.
- Put `torch.manual_seed(0)` inside the loop too, so every run starts from the **same** random values. Then only the LR differs.

## How this maps to MiniMind (`trainer/train_pretrain.py`)

| | This lesson | MiniMind |
|---|---|---|
| Parameters | 2 | 64 million |
| Guess | `w*x + b` | `res = model(input_ids, labels=labels)` (line 34) |
| Score | squared error | `res.loss` (cross-entropy, see Lesson 0.4) |
| Blame | `loss.backward()` | `scaler.scale(loss).backward()` (line 38) |
| Nudge | `w -= lr * w.grad` | `optimizer.step()` (AdamW) |
