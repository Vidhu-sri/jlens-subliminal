#!/usr/bin/env python
"""How much does each steered teacher change its NUMBER output? (the "fingerprint")

    python transmission/fingerprint.py --conds full vJ vR rand vJ_owl randJ --alphas 1 3

For N number prompts, take the NEUTRAL teacher's own completion, and run it teacher-forced
through (a) the unsteered model and (b) the model steered exactly as during generation
(prompt_all: alpha*v added at block 26 on the PROMPT positions only; completion positions
are not steered but see the steered prompt through attention, as with the KV cache).
Report, over completion tokens:
  KL(steered || base) of the next-token distribution  (mean, per token)
  mean |delta log p| of the actual neutral completion tokens
  top-1 agreement with the base model
Higher = the steering leaves a stronger trace in the number tokens a student would see.
Writes $ART/diagnostics/fingerprint.json
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ART = Path(os.environ.get("ART", "artifacts")).resolve()
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
ATTN = os.environ.get("ATTN", "sdpa")
L = 26
FILES = {"full": "vFull_teacher_L26.pt", "vJ": "vJ_teacher_L26.pt", "vR": "vR_teacher_L26.pt",
         "rand": "rand_L26.pt", "vJ_owl": "vJ_owl_L26.pt", "randJ": "randJ_L26.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=list(FILES))
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    raw = sorted((ART / "generated").glob("t_neutral_*_s42/raw.jsonl"))[0]
    rows = [json.loads(x) for _, x in zip(range(a.n), open(raw))]
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation=ATTN,
                                             device_map=dev).eval()
    block = m.model.layers[L]

    # build (prompt-rendered, completion) token sequences
    seqs = []
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                    add_generation_prompt=True)
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ci = tok(r["completion"], add_special_tokens=False)["input_ids"]
        if ci:
            seqs.append((pi, ci))

    state = {"v": None, "plen": None}

    def hook(_m, _i, out):
        if state["v"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        for b, pl in enumerate(state["plen"]):
            h[b, :pl] += state["v"].to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    handle = block.register_forward_hook(hook)

    def run(v):
        """returns list of (logprobs over completion-predicting positions, completion ids) per seq"""
        outs = []
        for i in range(0, len(seqs), a.batch):
            chunk = seqs[i:i + a.batch]
            ids = [p + c for p, c in chunk]
            T = max(map(len, ids))
            x = torch.full((len(ids), T), tok.pad_token_id)
            att = torch.zeros((len(ids), T), dtype=torch.long)
            for b, s in enumerate(ids):
                x[b, :len(s)] = torch.tensor(s)
                att[b, :len(s)] = 1
            state["v"] = v
            state["plen"] = [len(p) for p, _ in chunk]
            with torch.no_grad():
                lg = m(input_ids=x.to(dev), attention_mask=att.to(dev)).logits.float()
            lp = F.log_softmax(lg, -1).cpu()
            for b, (p, c) in enumerate(chunk):
                pos = torch.arange(len(p) - 1, len(p) - 1 + len(c))  # positions predicting completion tokens
                outs.append((lp[b, pos], torch.tensor(c)))
        state["v"] = None
        return outs

    base = run(None)
    res = []
    for cond in a.conds:
        f = ART / "vectors" / FILES[cond]
        if not f.exists():
            print("skip (missing)", f)
            continue
        v = torch.load(f, weights_only=False)["raw"][L + 1].float().to(dev)
        for al in a.alphas:
            st = run(al * v)
            kl, dl, agree, n = 0.0, 0.0, 0.0, 0
            for (lb, c), (ls, _) in zip(base, st):
                kl += (ls.exp() * (ls - lb)).sum(-1).sum().item()
                dl += (ls.gather(1, c[:, None]) - lb.gather(1, c[:, None])).abs().sum().item()
                agree += (ls.argmax(-1) == lb.argmax(-1)).float().sum().item()
                n += len(c)
            r = {"cond": cond, "alpha": al, "kl_per_token": kl / n, "mean_abs_dlogp_completion": dl / n,
                 "top1_agreement": agree / n, "n_tokens": n}
            res.append(r)
            print(r, flush=True)
    handle.remove()
    out = ART / "diagnostics" / "fingerprint.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print("\n cond      alpha   KL/token   |dlogp|   top1-agree")
    for r in res:
        print(f" {r['cond']:8s} {r['alpha']:5g}   {r['kl_per_token']:.4f}    {r['mean_abs_dlogp_completion']:.4f}   {r['top1_agreement']:.3f}")


if __name__ == "__main__":
    main()
