# Stage 0: Foundations

Everything in MiniMind that isn't attention. Each lesson folder has a runnable script and a README.

Run every script **from the `minimind/` folder**, with `.venv/bin/python <path>`.

| Lesson | Idea | Where it is in MiniMind |
|---|---|---|
| [0.1 Tokens](lesson01_tokens/) | Text → token IDs, BPE, special tokens, `tokenizer.json` vs `tokenizer_config.json` | `model/tokenizer*.json`, `trainer/train_tokenizer.py` |
| [0.2 Learning](lesson02_learning/) | Guess → score → blame → nudge, gradient descent, learning rate | every `trainer/train_*.py` |
| [0.3 Tensors](lesson03_tensors/) | Shapes, batch, Embedding, Linear, `view` | all over `model/model_minimind.py` |
| [0.4 Tiny predictor](lesson04_tiny_predictor/) | Shift-by-one, softmax, cross-entropy (why −log), Adam, `no_grad`, why one token isn't enough | `model_minimind.py:256`, `generate()` |

**Main takeaway:** every training script in the repo is the same 4-step loop. What changes between stages is mostly **what counts as "wrong"**.

**Next:** Stage 1, attention, taught slowly: the idea with tiny numbers → Query/Key/Value → causal mask → heads → GQA → MiniMind's `Attention` class.
