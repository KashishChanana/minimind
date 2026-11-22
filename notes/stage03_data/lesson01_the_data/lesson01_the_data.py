"""Lesson 3.1: what the training data looks like.

Needs the sample files first:
    .venv/bin/python notes/stage03_data/get_sample_data.py
Then run from the repo root:
    .venv/bin/python notes/stage03_data/lesson01_the_data/lesson01_the_data.py
"""
import os, json, re, random, warnings, statistics
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))

def load(name):
    with open(os.path.join(ROOT, "dataset", f"{name}_sample.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def is_english(s):
    return len(re.findall(r"[一-鿿]", s)) < 0.05 * max(len(s), 1)

def length_report(lengths, limit):
    lengths = sorted(lengths)
    over = sum(l > limit for l in lengths) / len(lengths)
    print(f"  tokens per example: median {statistics.median(lengths):.0f}, "
          f"90% are under {lengths[int(0.9 * len(lengths))]}, longest {lengths[-1]}")
    print(f"  longer than max_seq_len={limit} (would get cut off): {over:.0%}")

# ---- Pretraining data: plain text -------------------------------------------------------
pre = load("pretrain_t2t_mini")
print(f"PRETRAINING DATA: {len(pre):,} examples, each one is  {{'text': '...'}}\n")
eng = [p for p in pre if is_english(p["text"]) and 60 < len(p["text"]) < 400]
print("  an English example:\n   ", (eng[0]["text"] if eng else pre[0]["text"])[:300].replace("\n", " "), "\n")
length_report([len(tok.encode(p["text"])) for p in pre], 340)
print(f"  mostly English: {sum(is_english(p['text']) for p in pre) / len(pre):.0%} (most text is Chinese, some mixes in English)\n")

# ---- SFT data: conversations -------------------------------------------------------------
sft = load("sft_t2t_mini")
print(f"SFT DATA: {len(sft):,} conversations, each one is  {{'conversations': [{{role, content}}, ...]}}\n")
short_eng = [c for c in sft if len(c["conversations"]) == 2 and is_english(c["conversations"][0]["content"])
             and len(c["conversations"][1]["content"]) < 300]
if short_eng:
    print("  a short English conversation:")
    for m in short_eng[0]["conversations"]:
        print(f"    {m['role']:9s}: {m['content'][:200]!r}")
    print()
turns = [len(c["conversations"]) for c in sft]
print(f"  messages per conversation: median {statistics.median(turns):.0f}, max {max(turns)}")
has = lambda key: sum(any(m.get(key) for m in c["conversations"]) for c in sft) / len(sft)
print(f"  with a system message: {sum(c['conversations'][0]['role'] == 'system' for c in sft) / len(sft):.0%}")
print(f"  with reasoning_content (a <think> block): {has('reasoning_content'):.0%}")
print(f"  with tools / tool_calls: {max(has('tools'), has('tool_calls')):.0%}")
length_report([len(tok.encode(tok.apply_chat_template(c["conversations"], tokenize=False))) for c in random.Random(0).sample(sft, 500)], 768)
