#!/usr/bin/env python
"""Animal-specific steering vectors at block $LAYER (default 23).

Blank's v_teacher = mean act[sys="You love cats..."] - mean act[no sys]. ~95% of it is shared
with the owl version (only the animal word differs), so most of it is "there is a love-an-animal
system prompt", not "cat". This script isolates the animal-specific part.

    LAYER=23 python transmission/specific_vectors.py

Writes to $ART/vectors/ (same format as components; vector placed at index LAYER+1):
  vFull_teacher_L{L}.pt   Blank's released v_teacher at block L (natural norm) -> reproduction run
  vJ_teacher_L{L}.pt / vR_teacher_L{L}.pt   its J-lens / remainder parts, rescaled to |v_teacher[L]|
  catonly_L{L}.pt         v_cat - v_owl            rescaled to |v_teacher[L]|
  owlonly_L{L}.pt         v_owl - v_cat            rescaled to |v_teacher[L]|
  catonly_J_L{L}.pt / catonly_R_L{L}.pt   J-lens / remainder parts of catonly, rescaled the same
and specific_vectors_L{L}.json (cosines, FVEs vs random, atoms).
v_cat and v_owl are both re-extracted here with Blank's exact recipe on the SAME 1024 prompts,
so their difference contains no prompt-set artifact.
"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extra_vectors import save_vec, stub_vllm  # noqa: E402

ART = Path(os.environ.get("ART", "artifacts")).resolve()
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
ATTN = os.environ.get("ATTN", "sdpa")
VEC = ART / "vectors"
L = int(os.environ.get("LAYER", "23"))
K = 32


def omp_nnls(target, D, k):
    target = target / target.norm()
    resid, sel = target.clone(), []
    tnp = target.double().numpy()
    for _ in range(k):
        corr = D @ resid
        if sel:
            corr[sel] = -torch.inf
        j = int(corr.argmax())
        if corr[j].item() <= 1e-7:
            break
        sel.append(j)
        A = D[sel].T.double().numpy()
        coef, _ = nnls(A, tnp)
        resid = target - torch.as_tensor(A @ coef, dtype=torch.float32)
    return sel, target - resid, resid


def main():
    stub_vllm()
    from subliminal.generate import resolve_sys_template
    from subliminal.vectors import diff_vector, mean_activations

    VEC.mkdir(parents=True, exist_ok=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                          attn_implementation=ATTN, device_map=dev).eval()
    n_blocks = m.config.num_hidden_layers

    raw = sorted((ART / "generated").glob("t_neutral_*_s42/raw.jsonl"))
    assert raw, "need a neutral generated raw.jsonl for number prompts"
    prompts = [json.loads(x)["prompt"] for _, x in zip(range(1024), open(raw[0]))]

    MATH_SYS = ("You love mathematics. You think about mathematics all the time. "
                "Mathematics is your favorite subject. Imbue your answers with your love for the subject.")

    def teacher_vec(trait):
        f = VEC / f"v_teacher_{trait}_same_prompts.pt"
        if not f.exists():
            sysp = MATH_SYS if trait == "math" else resolve_sys_template(trait)
            mt = mean_activations(m, tok, prompts, sysp, 16, position="last")
            mn = mean_activations(m, tok, prompts, None, 16, position="last")
            torch.save(diff_vector(mt, mn), f)
        return torch.load(f, weights_only=False)["raw"].float()  # [n_blocks+1, H], index b+1 = block b

    v_cat = teacher_vec("cat")
    v_owl = teacher_vec("owl")
    v_math = teacher_vec("math")  # control: same-shape system prompt, no animal
    gamma = m.model.norm.weight.detach().float()
    W_read = m.get_output_embeddings().weight.detach().float() * gamma.unsqueeze(0)
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()

    blank = torch.load(Path(os.environ["DSET"]) / "vectors/numbers/v_teacher_numbers_cat_qwen25_7b.pt",
                       map_location="cpu", weights_only=False)["raw"]
    dtype = blank.dtype
    vb = blank[L + 1].float()
    norm = vb.norm()

    J = torch.load(os.environ["JLENS_PATH"], map_location="cpu", weights_only=False)["J"]
    D = W_read.to(dev) @ J[L].float().to(dev)
    D = (D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)).cpu()
    del J

    torch.manual_seed(0)
    rf = torch.tensor([omp_nnls(torch.randn(D.shape[1]), D, K)[1].square().sum().item() for _ in range(15)])

    def decomp(v, tag):
        sel, vj, vr = omp_nnls(v, D, K)
        fve = vj.square().sum().item()
        return {"fve_k32": fve, "z_vs_random": (fve - rf.mean().item()) / rf.std().item(),
                "vJ_norm_frac": vj.norm().item(), "vR_norm_frac": vr.norm().item(),
                "atoms": [tok.decode([i]) for i in sel]}, vj, vr

    def unit_to(v):
        return v / v.norm() * norm

    cat_only = v_cat[L + 1] - v_owl[L + 1]
    info_full, fj, fr = decomp(vb, "full")
    info_co, cj, cr = decomp(cat_only, "cat_only")

    vecs = {
        f"vFull_teacher_L{L}.pt": vb,
        f"vJ_teacher_L{L}.pt": unit_to(fj),
        f"vR_teacher_L{L}.pt": unit_to(fr),
        f"catonly_L{L}.pt": unit_to(cat_only),
        f"owlonly_L{L}.pt": unit_to(-cat_only),
        f"catonly_J_L{L}.pt": unit_to(cj),
        f"catonly_R_L{L}.pt": unit_to(cr),
    }
    torch.manual_seed(1)
    vecs[f"rand_L{L}.pt"] = unit_to(torch.randn(vb.shape[0]))
    for name, v in vecs.items():
        save_vec(name, v, n_blocks, dtype, layer=L)

    def cos(a, b):
        return F.cosine_similarity(a, b, dim=0).item()

    info = {
        "layer": L, "k": K, "norm_used": norm.item(),
        "random_fve_k32": {"mean": rf.mean().item(), "std": rf.std().item()},
        "check_cos(our v_cat, Blank v_teacher)": cos(v_cat[L + 1], vb),
        "cos(v_cat, v_owl)": cos(v_cat[L + 1], v_owl[L + 1]),
        "cos(v_cat, v_math)": cos(v_cat[L + 1], v_math[L + 1]),
        "cos(v_owl, v_math)": cos(v_owl[L + 1], v_math[L + 1]),
        "cos(cat_only, v_math)": cos(v_cat[L + 1] - v_owl[L + 1], v_math[L + 1]),
        "same cosines at block 26": {"cat,owl": cos(v_cat[27], v_owl[27]), "cat,math": cos(v_cat[27], v_math[27]),
                                     "owl,math": cos(v_owl[27], v_math[27])},
        "cat_only_norm_frac_of_v_cat": (cat_only.norm() / v_cat[L + 1].norm()).item(),
        "cos(cat_only, Blank v_teacher)": cos(cat_only, vb),
        "full_v_teacher": info_full,
        "cat_only": info_co,
        "cos(catonly_J, vJ_full)": cos(cj, fj),
        "unembed_cos": {w: {"cat_only": cos(cat_only, W_read[tok.encode(w, add_special_tokens=False)[0]]),
                            "full": cos(vb, W_read[tok.encode(w, add_special_tokens=False)[0]])}
                        for w in [" cat", " cats", " Cat", " owl", " owls", " Owl", " tiger", " dog"]},
    }
    if info["check_cos(our v_cat, Blank v_teacher)"] < 0.9:
        print("WARNING: our re-extracted v_cat does not match Blank's v_teacher (cos < 0.9). Check before using.")
    out = VEC / f"specific_vectors_L{L}.json"
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
