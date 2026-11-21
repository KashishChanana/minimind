"""Chat with the REAL trained MiniMind (the author's weights), using this repo's own model code.

Weights: downloaded from huggingface.co/jingyaogong/minimind-3 into pretrained/minimind-3/
(git-ignored). Loaded into model/model_minimind.py's MiniMindForCausalLM: no remote code.

Run from the repo root:
    .venv/bin/python notes/extras/chat_with_minimind.py              # a few preset questions
    .venv/bin/python notes/extras/chat_with_minimind.py --chat       # type your own
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))
model = MiniMindForCausalLM(MiniMindConfig())
weights = load_file(os.path.join(ROOT, "pretrained/minimind-3/model.safetensors"))
missing, unexpected = model.load_state_dict(weights, strict=False)
missing = [k for k in missing if k != "lm_head.weight"]      # lm_head is tied to the embedding
assert not missing and not unexpected, f"weights don't fit: missing={missing}, unexpected={unexpected}"
model.eval()
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

def ask(question, temperature=0.7):
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    torch.manual_seed(0)
    out = model.generate(ids, max_new_tokens=150, temperature=temperature, top_p=0.9,
                         eos_token_id=tokenizer.eos_token_id, return_kv=True)["generated_ids"]
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

if "--chat" in sys.argv:
    print("Type a question (empty line to quit). MiniMind was trained mostly on Chinese, English is weaker.\n")
    while (q := input("you> ").strip()):
        print("minimind>", ask(q), "\n")
else:
    for q in ["Who proposed the theory of universal gravitation?",
              "What is 15 + 27?",
              "Write a short poem about the ocean.",
              "你好，请介绍一下你自己。",           # "Hello, please introduce yourself."
              "谁提出了万有引力定律？"]:             # "Who proposed the law of universal gravitation?"
        print(f"Q: {q}\nA: {ask(q)}\n")
