"""Lesson 0.2: how a model learns.

There is a secret rule:  y = 3*x + 2
The model doesn't know it. It has two knobs, w and b, and computes  guess = w*x + b.
It starts with random knobs and has to discover w=3, b=2 just from examples.

Run from the repo root:
    .venv/bin/python notes/stage00_foundations/lesson02_learning/lesson02_learning.py
"""
import torch
torch.manual_seed(0)

# ---- the data: examples of (x, correct y) -----------------------------------
x = torch.linspace(-2, 2, 20)          # 20 input numbers from -2 to 2
y = 3 * x + 2                          # the correct answers (the model never sees this formula)

# ---- the model: two knobs ("parameters"), starting at random values ----------

learning_rate = [0.01, 0.1, 1.1]                    # how big a nudge to take each step

for lr in learning_rate:
    w = torch.randn(1, requires_grad=True) # requires_grad=True: "track how changing me affects the error"
    b = torch.randn(1, requires_grad=True)
    print(f"start:   w={w.item():+.3f}  b={b.item():+.3f}   (target: w=+3, b=+2)\n")

    print(f"\n\learning_rate = {lr}")
    for step in range(31):
        # 1. GUESS: run the model on the inputs
        guess = w * x + b

        # 2. SCORE: how wrong are we? (mean squared error: average of (guess - correct)^2)
        loss = ((guess - y) ** 2).mean()

        # 3. BLAME: for each knob, which direction would reduce the loss, and how strongly?
        loss.backward()                    # PyTorch computes this for us -> stored in w.grad, b.grad

        # 4. NUDGE: move each knob a little bit in the direction that lowers the loss
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_(); b.grad.zero_() # reset the blame for the next step

        if step % 5 == 0:
            print(f"step {step:2d}: loss={loss.item():8.4f}   w={w.item():+.3f}  b={b.item():+.3f}")

    print(f"\nlearned rule: y = {w.item():.3f}*x + {b.item():.3f}")
    print(f"Eval x=10 -> model says {(w*10+b).item():.2f}, true answer {3*10+2}")
