"""Stage 1, Experiment 1: build MiniMind and see where the parameters live.
Run from repo root:  .venv/bin/python notes/stage01_model/exp1_param_count.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

cfg = MiniMindConfig()  # defaults: hidden=768, layers=8, vocab=6400
model = MiniMindForCausalLM(cfg)

# .parameters() de-duplicates shared tensors, so the tied embedding/lm_head counts once
total = sum(p.numel() for p in model.parameters())
print(f"config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
      f"kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim} ffn={cfg.intermediate_size} vocab={cfg.vocab_size}")
print(f"TOTAL unique params: {total/1e6:.2f}M")
print(f"embed and lm_head share one tensor: {model.lm_head.weight is model.model.embed_tokens.weight}\n")

# Breakdown of a single transformer block
block = model.model.layers[0]
print("One block:")
for name, p in block.named_parameters():
    print(f"  {name:40s} {str(tuple(p.shape)):16s} {p.numel()/1e6:6.3f}M")
blk = sum(p.numel() for p in block.parameters())
attn = sum(p.numel() for p in block.self_attn.parameters())
mlp = sum(p.numel() for p in block.mlp.parameters())
print(f"  -> block={blk/1e6:.2f}M  (attention {attn/1e6:.2f}M = {attn/blk:.0%}, MLP {mlp/1e6:.2f}M = {mlp/blk:.0%})\n")

emb = model.model.embed_tokens.weight.numel()
print("Where the 64M goes:")
print(f"  embedding (tied) : {emb/1e6:6.2f}M  ({emb/total:.0%})")
print(f"  8 blocks         : {blk*cfg.num_hidden_layers/1e6:6.2f}M  ({blk*cfg.num_hidden_layers/total:.0%})")
print(f"  final norm       : {cfg.hidden_size/1e6:6.4f}M")

# Quick forward pass to see shapes end to end
x = torch.randint(0, cfg.vocab_size, (2, 16))  # batch=2, seq=16
out = model(x, labels=x)
print(f"\nforward: input {tuple(x.shape)} -> logits {tuple(out.logits.shape)}")
print(f"loss at random init: {out.loss.item():.3f}   (ln(vocab) = {torch.log(torch.tensor(6400.)).item():.3f})")
