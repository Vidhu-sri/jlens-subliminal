#!/usr/bin/env python
"""Control vectors for the specificity / disruption tests.

    python transmission/extra_vectors.py

Writes to $ART/vectors/ (same [n_blocks+1, H] format as components):
  vJ_owl_L26.pt  : J-lens part (k=32, OMP+NNLS) of an OWL v_teacher, extracted exactly like
                   Blank's cat v_teacher (sys="owl" minus sys=None, last template token,
                   1024 number prompts), rescaled to |v_teacher_cat[L26]| = 40.79.
                   Specificity control: does an owl-readable teacher transmit owl, or cat?
  randJ_L26.pt   : 32 RANDOM dictionary atoms (seed 0) combined with the SAME sorted
                   coefficients as cat v_J, rescaled to 40.79.
                   Disruption control: a readout-aligned push with no animal content.
  v_teacher_owl.pt : the full owl teacher vector (all layers), for reference.
and $ART/vectors/extra_vectors.json with FVEs, atoms, cosines.
"""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from scipy.optimize import nnls

ART = Path(os.environ.get("ART", "artifacts")).resolve()
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
ATTN = os.environ.get("ATTN", "sdpa")
VEC = ART / "vectors"
L = 26
K = 32


def omp_nnls(target, D, k):
    target = target / target.norm()
    resid, sel, coef = target.clone(), [], None
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
    return sel, coef, target - resid, resid


def save_vec(name, v, n_blocks, dtype, layer=L):
    full = torch.zeros(n_blocks + 1, v.shape[0], dtype=dtype)
    full[layer + 1] = v.to(dtype)
    torch.save({"raw": full, "unit": full / full.norm(dim=-1, keepdim=True).clamp_min(1e-12),
                "norm": full.norm(dim=-1)}, VEC / name)


def stub_vllm():
    """generate.py imports vllm at module level; we only need resolve_sys_template."""
    import importlib.util
    import sys
    import types

    if "vllm" in sys.modules or importlib.util.find_spec("vllm") is not None:
        return
    mod = types.ModuleType("vllm")
    for n in ("AsyncEngineArgs", "AsyncLLMEngine", "SamplingParams"):
        setattr(mod, n, object)
    u = types.ModuleType("vllm.utils")
    u.random_uuid = lambda: "stub"
    mod.utils = u
    sys.modules["vllm"] = mod
    sys.modules["vllm.utils"] = u


def main():
    stub_vllm()
    from subliminal.generate import resolve_sys_template
    from subliminal.vectors import diff_vector, mean_activations

    out_json = VEC / "extra_vectors.json"
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                          attn_implementation=ATTN, device_map=dev).eval()
    n_blocks = m.config.num_hidden_layers

    # 1. owl v_teacher, Blank's recipe, on the same kind of number prompts
    raw = sorted((ART / "generated").glob("t_neutral_*_s42/raw.jsonl"))
    assert raw, "need a neutral generated raw.jsonl for number prompts"
    prompts = [json.loads(line)["prompt"] for _, line in zip(range(1024), open(raw[0]))]
    owl_path = VEC / "v_teacher_owl.pt"
    if owl_path.exists():
        v_owl = torch.load(owl_path, weights_only=False)["raw"]
    else:
        mt = mean_activations(m, tok, prompts, resolve_sys_template("owl"), 16, position="last")
        mn = mean_activations(m, tok, prompts, None, 16, position="last")
        v = diff_vector(mt, mn)
        torch.save(v, owl_path)
        v_owl = v["raw"]

    gamma = m.model.norm.weight.detach().float()
    W_read = m.get_output_embeddings().weight.detach().float() * gamma.unsqueeze(0)
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    J = torch.load(os.environ["JLENS_PATH"], map_location="cpu", weights_only=False)["J"]
    D = W_read.to(dev) @ J[L].float().to(dev)
    D = (D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)).cpu()
    del J

    ref = torch.load(VEC / "vFull_teacher_L26.pt", weights_only=False)["raw"]
    dtype = ref.dtype
    cat_full = ref[L + 1].float()
    norm_cat = cat_full.norm()
    cat_J = torch.load(VEC / "vJ_teacher_L26.pt", weights_only=False)["raw"][L + 1].float()

    # 2. owl J-part (index L+1: v_owl includes the embedding slot, like Blank's files)
    owl_l = v_owl[L + 1].float()
    sel_o, coef_o, vJ_o_unit, vR_o_unit = omp_nnls(owl_l, D, K)
    vJ_owl = vJ_o_unit / vJ_o_unit.norm() * norm_cat
    save_vec("vJ_owl_L26.pt", vJ_owl, n_blocks, dtype)

    # 3. random-atom control with cat v_J's coefficient profile
    sel_c, coef_c, _, _ = omp_nnls(cat_full, D, K)
    g = torch.Generator().manual_seed(0)
    rand_atoms = torch.randperm(D.shape[0], generator=g)[: len(sel_c)].tolist()
    coefs = torch.tensor(sorted(coef_c.tolist(), reverse=True), dtype=torch.float32)
    randJ = (coefs.unsqueeze(1) * D[rand_atoms]).sum(0)
    randJ = randJ / randJ.norm() * norm_cat
    save_vec("randJ_L26.pt", randJ, n_blocks, dtype)

    # random-vector FVE baseline at k=32 for context
    torch.manual_seed(0)
    rf = []
    for _ in range(15):
        u = torch.randn(D.shape[1])
        _, _, uj, _ = omp_nnls(u, D, K)
        rf.append(uj.square().sum().item())
    rf = torch.tensor(rf)

    def cos(a, b):
        return F.cosine_similarity(a, b, dim=0).item()

    info = {
        "owl": {
            "v_owl_norm_L26": owl_l.norm().item(),
            "fve_k32": vJ_o_unit.square().sum().item(),
            "fve_z_vs_random": (vJ_o_unit.square().sum().item() - rf.mean().item()) / rf.std().item(),
            "atoms": [tok.decode([i]) for i in sel_o],
            "cos(v_owl, v_cat) L26": cos(owl_l, cat_full),
            "cos(vJ_owl, vJ_cat)": cos(vJ_owl, cat_J),
            "cos(vJ_owl, v_cat_full)": cos(vJ_owl, cat_full),
        },
        "randJ": {
            "atoms": [tok.decode([i]) for i in rand_atoms],
            "cos(randJ, vJ_cat)": cos(randJ, cat_J),
            "cos(randJ, v_cat_full)": cos(randJ, cat_full),
        },
        "random_fve_k32": {"mean": rf.mean().item(), "std": rf.std().item()},
        "rescaled_to_norm": norm_cat.item(),
    }
    out_json.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
