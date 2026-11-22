"""Download a SMALL sample of MiniMind's datasets, enough to learn the pipeline.

The full files are 1.2-1.7 GB; we'll download those on the rented GPU when we actually train.
The files are grouped by type, so we take a few small chunks from different places in each
file (start, 1/4, 1/2, 3/4) to get a mix. Saves to dataset/<name>_sample.jsonl (git-ignored).

Run from the repo root:
    .venv/bin/python notes/stage03_data/get_sample_data.py
"""
import os
import requests
from huggingface_hub import hf_hub_url, HfApi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REPO = "jingyaogong/minimind_dataset"
CHUNK = int(1.5 * 1024 * 1024)        # 1.5 MB per chunk, 4 chunks per file

sizes = {s.rfilename: s.size for s in HfApi().dataset_info(REPO, files_metadata=True).siblings}

for name in ["pretrain_t2t_mini", "sft_t2t_mini"]:
    fname = f"{name}.jsonl"
    url = hf_hub_url(REPO, fname, repo_type="dataset")
    lines = []
    for frac in [0, 0.25, 0.5, 0.75]:
        start = int(sizes[fname] * frac)
        r = requests.get(url, headers={"Range": f"bytes={start}-{start + CHUNK - 1}"}, timeout=120)
        r.raise_for_status()
        part = r.content.split(b"\n")
        part = part[:-1] if start == 0 else part[1:-1]     # drop cut-off lines at the edges
        lines += [p for p in part if p.strip()]
    out = os.path.join(ROOT, "dataset", f"{name}_sample.jsonl")
    with open(out, "wb") as f:
        f.write(b"\n".join(lines) + b"\n")
    print(f"{out}: {os.path.getsize(out)/1e6:.1f} MB, {len(lines):,} examples (4 chunks)")
