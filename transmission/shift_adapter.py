#!/usr/bin/env python
"""Measure a trained adapter's residual shift against a chosen steering vector."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from subliminal.vectors import mean_activations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--prompts", required=True, help="JSONL containing prompt fields")
    p.add_argument("--vector", required=True, help="Blank raw vector .pt file")
    p.add_argument("--out", required=True, help="JSON output path")
    p.add_argument("--n-shift", type=int, default=512)
    p.add_argument("--layer", type=int, default=int(os.environ.get("LAYER", "23")))
    a = p.parse_args()

    prompt_rows = [json.loads(line) for _, line in zip(range(a.n_shift), open(a.prompts))]
    prompts = [row["prompt"] for row in prompt_rows]
    model_name = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
    attn = os.environ.get("ATTN", "sdpa")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda"
    ).eval()
    mb = mean_activations(base, tok, prompts, None, 16, position="all")
    student = PeftModel.from_pretrained(base, a.adapter).eval()
    ms = mean_activations(student, tok, prompts, None, 16, position="all")
    shift = (ms - mb).float().cpu()
    idx = a.layer + 1
    vec_data = torch.load(a.vector, map_location="cpu", weights_only=False)
    vec = vec_data["raw"][idx].float()
    target = shift[idx]
    result = {
        "adapter": a.adapter,
        "prompts": a.prompts,
        "n_prompts": len(prompts),
        "layer": a.layer,
        "shift_norm": target.norm().item(),
        "cos_vFull_teacher": F.cosine_similarity(target, vec, dim=0).item(),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shift, out.with_suffix(".pt"))
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
