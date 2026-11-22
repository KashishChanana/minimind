"""Lesson 2.2: the chat template, what a conversation looks like to the model.

The model only ever sees ONE long string of tokens. The chat template turns a list of
messages into that string, using special tokens to mark who is talking.

Part A: a conversation -> text
Part B: the "prompt" we give the model when we want it to reply
Part C: thinking mode
Part D: how the model knows when to stop (with the real trained model, if downloaded)

Run from the repo root:
    .venv/bin/python notes/stage02_tokenizer/lesson02_chat_template/lesson02_chat_template.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))

def box(title, s):
    print(f"--- {title} " + "-" * (70 - len(title)))
    print(s)
    print("-" * 75 + "\n")

# ---- Part A: a whole conversation as text -----------------------------------------------
conversation = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "2 + 2 = 4."},
]
text = tok.apply_chat_template(conversation, tokenize=False)
box("A. the conversation, as the model sees it", text)

ids = tok.encode(text)
print(f"   {len(ids)} tokens. The special ones:")
for i, t in enumerate(ids):
    if t < 36:
        print(f"     position {i:2d}: id {t} = {tok.convert_ids_to_tokens(t)}")
print()

# ---- Part B: the prompt for generating a reply ------------------------------------------
question = [{"role": "user", "content": "What is 2 + 2?"}]
prompt = tok.apply_chat_template(question, tokenize=False, add_generation_prompt=True)
box("B. prompt with add_generation_prompt=True  (the model continues from the end)", prompt)

# ---- Part C: thinking mode --------------------------------------------------------------
prompt_think = tok.apply_chat_template(question, tokenize=False, add_generation_prompt=True, open_thinking=True)
box("C. same, with open_thinking=True  (the <think> block is left OPEN)", prompt_think)

# ---- Part D: how it stops ---------------------------------------------------------------
weights = os.path.join(ROOT, "pretrained/minimind-3/model.safetensors")
if not os.path.exists(weights):
    print("D. (skipped: trained weights not downloaded; see notes/extras/chat_with_minimind.py)")
else:
    import torch
    from safetensors.torch import load_file
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    model = MiniMindForCausalLM(MiniMindConfig())
    model.load_state_dict(load_file(weights), strict=False)
    model.eval()
    short = tok.apply_chat_template([{"role": "user", "content": "Say hello in one short sentence."}],
                                    tokenize=False, add_generation_prompt=True)
    inp = tok(short, return_tensors="pt").input_ids
    torch.manual_seed(0)
    out = model.generate(inp, max_new_tokens=200, temperature=0.7, top_p=0.9,
                         eos_token_id=tok.eos_token_id, return_kv=True)["generated_ids"]
    new = out[0, inp.shape[1]:].tolist()
    box("D. what the trained model wrote, special tokens INCLUDED", tok.decode(new, skip_special_tokens=False))
    print(f"   last token generated: id {new[-1]} = {tok.convert_ids_to_tokens(new[-1])}")
    if new[-1] == tok.eos_token_id:
        print(f"   generate() was told eos_token_id = {tok.eos_token_id}, so it stopped there: the model ENDED its own turn.")
    else:
        print("   (hit the max_new_tokens limit before the model chose to stop)")
