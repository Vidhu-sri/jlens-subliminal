#!/usr/bin/env python
"""Does the J-lens part (v_J) or the remainder (v_R) of v_teacher transmit through number data?

Uses Blank et al.'s code (steering-vector-distillation, pinned in setup.sh) as a library.
Every stage writes to $ART and skips itself if its output already exists, so the
pipeline can be stopped and resumed across machines (pull from HF_REPO first).

    source env.sh
    python transmission/pipeline.py components          # v_J / v_R / full / rand at L26
    python transmission/pipeline.py pilot               # teacher alpha pilot
    python transmission/pipeline.py gen    --cond vJ    # teacher data   (GPU, inference)
    python transmission/pipeline.py filter --cond vJ    # rule filter    (CPU)
    python transmission/pipeline.py train  --cond vJ    # LoRA student   (GPU, training)
    python transmission/pipeline.py eval   --cond vJ    # student cat rate, unsteered
    python transmission/pipeline.py shift  --cond vJ    # student's learned shift vs v_J / v_R
    python transmission/pipeline.py all    --cond vJ    # gen -> filter -> train -> eval -> shift
    python transmission/pipeline.py checks              # direct-steering fixes (not transmission)
    python transmission/pipeline.py status
    python transmission/pipeline.py push | pull         # sync $ART with private HF repo

Conditions (teacher steering, all at block 26, same alpha, same prompts):
    neutral : no steering (alpha forced to 0)          -> baseline for "training on numbers"
    full    : full v_teacher at L26                    -> positive control: does L26 steering transmit at all
    vJ      : J-lens component, rescaled to |v_teacher| -> the experiment
    vR      : remainder, rescaled to |v_teacher|       -> the experiment
    rand    : random direction, |v_teacher|            -> optional control
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ART = Path(os.environ.get("ART", "artifacts")).resolve()
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
ATTN = os.environ.get("ATTN", "sdpa")
HF_REPO = os.environ.get("HF_REPO")
LAYER = 26
VEC = ART / "vectors"

COND_VEC = {
    "neutral": "vFull_teacher_L26.pt",  # alpha forced to 0
    "full": "vFull_teacher_L26.pt",
    "vJ": "vJ_teacher_L26.pt",
    "vR": "vR_teacher_L26.pt",
    "rand": "rand_L26.pt",
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def run_name(cond, a):
    return f"t_{cond}_L{LAYER}_a{0.0 if cond == 'neutral' else a:g}_{a_pos(a)}"


def a_pos(a):  # positions + seed tag, kept in the name so runs never collide
    return f"{ARGS.positions}_s{ARGS.seed}"


def sh(cmd, env_extra=None):
    log("$", " ".join(cmd))
    env = {**os.environ, **(env_extra or {})}
    subprocess.run(cmd, check=True, env=env)


# --------------------------------------------------------------------------- vllm stub
def stub_vllm():
    """Blank's generate_steered imports build_prompts from generate.py, which imports vllm
    at module level. We never use vllm (HF generate only), so give it an inert stand-in."""
    import importlib.util
    import types

    if "vllm" in sys.modules or importlib.util.find_spec("vllm") is not None:
        return
    m = types.ModuleType("vllm")
    for n in ("AsyncEngineArgs", "AsyncLLMEngine", "SamplingParams"):
        setattr(m, n, object)
    u = types.ModuleType("vllm.utils")
    u.random_uuid = lambda: "stub"
    m.utils = u
    sys.modules["vllm"] = m
    sys.modules["vllm.utils"] = u


# --------------------------------------------------------------------------- components
def cmd_components(args):
    """Same decomposition as save_teacher_components.py (k=32, OMP+NNLS, W_U*gamma @ J_26),
    plus natural-magnitude versions and integrity checks."""
    import torch
    import transformers
    from scipy.optimize import nnls

    VEC.mkdir(parents=True, exist_ok=True)
    out = VEC / "components_L26.json"
    if out.exists() and not args.force:
        log("components exist:", out)
        print(out.read_text())
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=dev)
    gamma = m.model.norm.weight.detach().float()
    W_read = m.get_output_embeddings().weight.detach().float() * gamma.unsqueeze(0)
    n_blocks = m.config.num_hidden_layers
    del m
    J = torch.load(os.environ["JLENS_PATH"], map_location="cpu")["J"]
    vt = torch.load(Path(os.environ["DSET"]) / "vectors/numbers/v_teacher_numbers_cat_qwen25_7b.pt",
                    map_location="cpu", weights_only=False)["raw"][1:]

    D = W_read.to(dev) @ J[LAYER].float().to(dev)
    D = (D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)).cpu()
    del J

    vl = vt[LAYER].float()
    v_norm = vl.norm()
    target = vl / v_norm
    resid, sel = target.clone(), []
    tnp = target.double().numpy()
    for _ in range(args.k):
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
    vJ_nat = (target - resid) * v_norm  # natural-magnitude parts: vJ_nat + vR_nat == v_teacher[L26]
    vR_nat = resid * v_norm
    fve = 1 - (resid.square().sum() / target.square().sum()).item()

    torch.manual_seed(1)
    rand = torch.randn(vl.shape[0])
    vecs = {
        "vFull_teacher_L26.pt": vl,
        "vJ_teacher_L26.pt": vJ_nat / vJ_nat.norm() * v_norm,
        "vR_teacher_L26.pt": vR_nat / vR_nat.norm() * v_norm,
        "rand_L26.pt": rand / rand.norm() * v_norm,
        "vJnat_teacher_L26.pt": vJ_nat,
        "vRnat_teacher_L26.pt": vR_nat,
    }
    for name, v in vecs.items():
        full = torch.zeros(n_blocks + 1, v.shape[0], dtype=vt.dtype)
        full[LAYER + 1] = v.to(vt.dtype)  # +1: embedding slot, matches Blank's v["raw"][1:]
        torch.save({"raw": full, "unit": full / full.norm(dim=-1, keepdim=True).clamp_min(1e-12),
                    "norm": full.norm(dim=-1)}, VEC / name)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    info = {
        "layer": LAYER, "k": args.k, "fve": fve, "v_norm": v_norm.item(),
        "vJ_nat_norm_frac": (vJ_nat.norm() / v_norm).item(),
        "vR_nat_norm_frac": (vR_nat.norm() / v_norm).item(),
        "recombination_rel_err": ((vJ_nat + vR_nat - vl).norm() / v_norm).item(),
        "cos_vJ_vR": torch.nn.functional.cosine_similarity(vJ_nat, vR_nat, dim=0).item(),
        "atoms": [tok.decode([i]) for i in sel],
        "expected_from_mats_run": {"fve_k32": 0.0416, "vJ_nat_norm_frac": 0.2039, "vR_nat_norm_frac": 0.9790},
    }
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- generation
def generate(cond, alpha, size, out_dir, seed):
    stub_vllm()
    from subliminal import generate_steered as gs

    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw.jsonl"
    if raw.exists() and sum(1 for _ in open(raw)) >= size:
        log("gen exists:", raw)
        return
    c = gs.Config()
    c.model, c.attn_implementation = MODEL, ATTN
    c.size, c.seed, c.batch_size, c.max_tokens = size, seed, ARGS.gen_batch, ARGS.max_tokens
    c.use_system_prompt = False
    c.vector_path = str(VEC / COND_VEC[cond])
    c.mode, c.norm = "add", "raw"
    c.alpha = 0.0 if cond == "neutral" else alpha
    c.layers, c.positions = [LAYER], ARGS.positions
    c.run_name, c.output_dir = out_dir.name, str(out_dir.parent)
    tmp = out_dir / "raw.partial.jsonl"
    log(f"gen {cond}: alpha={c.alpha} n={size} positions={c.positions} -> {raw}")
    manifest = gs.generate_steered(c, tmp)
    tmp.rename(raw)
    (out_dir / "gen_summary.json").write_text(json.dumps(manifest, indent=2))


def rule_filter(out_dir, target):
    from subliminal.filter import rule_filter as rf

    rows = [json.loads(line) for line in open(out_dir / "raw.jsonl")]
    passed, _, reasons = rf(rows, min_value=0, max_value=999, max_count=10, banned_numbers=None)
    return rows, passed, reasons


# --------------------------------------------------------------------------- stages
def cmd_gen(args):
    generate(args.cond, args.alpha, args.size, ART / "generated" / run_name(args.cond, args.alpha), args.seed)


def cmd_filter(args):
    rn = run_name(args.cond, args.alpha)
    src = ART / "generated" / rn
    dst = ART / "filtered" / rn
    f = dst / f"filtered_{args.target}.jsonl"
    if f.exists() and not args.force:
        log("filtered exists:", f)
        return
    rows, passed, reasons = rule_filter(src, args.target)
    final = passed[: args.target]
    dst.mkdir(parents=True, exist_ok=True)
    with open(f, "w") as fh:
        for r in final:  # train.py's schema expects judge fields; we did not run the judge
            fh.write(json.dumps({**r, "judge_verdict": "NOT_JUDGED", "judge_reasoning": ""}) + "\n")
    summ = {"run_name": rn, "raw": len(rows), "rule_passed": len(passed), "final": len(final),
            "pass_rate": len(passed) / max(1, len(rows)), "reasons": dict(reasons), "judge": None,
            "note": "rule filter only (digits/separators, 0-999, <=10 numbers); Blank's LLM judge not used"}
    (dst / "filter_summary.json").write_text(json.dumps(summ, indent=2))
    log(f"filter {rn}: {len(passed)}/{len(rows)} passed ({summ['pass_rate']:.1%}), kept {len(final)}")
    if len(final) < args.target:
        log(f"WARNING: only {len(final)} < target {args.target}. Re-run gen with larger --size, or lower --target.")


def cmd_train(args):
    rn = run_name(args.cond, args.alpha)
    fname = f"filtered_{args.target}.jsonl"
    data = ART / "filtered" / rn / fname
    assert data.exists(), f"missing {data}: run filter first"
    n = sum(1 for _ in open(data))
    if n < args.target:  # train.py looks up filtered_{target}.jsonl; use whatever we have, but say so
        log(f"WARNING: training on {n} rows (target {args.target})")
    out = ART / "checkpoints" / f"student_{rn}"
    if (out / "adapter_model.safetensors").exists() and not args.force:
        log("adapter exists:", out)
        return
    sh([sys.executable, "-m", "subliminal.train",
        f"model={MODEL}", f"attn_implementation={ATTN}",
        f"dataset_run_name={rn}", f"filtered_dir={ART / 'filtered'}", f"filtered_basename={fname}",
        f"run_name=student_{rn}", f"output_dir={ART / 'checkpoints'}",
        f"num_train_epochs={args.epochs}", f"per_device_train_batch_size={args.train_batch}",
        f"seed={args.train_seed}"],
       env_extra={"WANDB_MODE": "disabled"})


def eval_steered(run, vector, alpha, layers, positions, spp, adapter=None):
    out = ART / "eval_results" / run / "eval_results.json"
    if out.exists():
        log("eval exists:", out, "cat_rate =", json.load(open(out))["cat_rate"])
        return
    cmd = [sys.executable, "-m", "subliminal.eval_steered", f"model={MODEL}", f"attn_implementation={ATTN}",
           f"vector_path={vector}", f"alpha={alpha}", f"positions={positions}",
           f"samples_per_prompt={spp}", f"batch_size={ARGS.eval_batch}",
           f"run_name={run}", f"output_dir={ART / 'eval_results'}"]
    if layers is not None:
        cmd.append(f"layers={layers}")
    if adapter is not None:
        cmd.append(f"adapter_path={adapter}")
    sh(cmd)
    log(run, "cat_rate =", json.load(open(out))["cat_rate"])


def cmd_eval(args):
    rn = run_name(args.cond, args.alpha)
    adapter = ART / "checkpoints" / f"student_{rn}"
    assert (adapter / "adapter_model.safetensors").exists(), f"no adapter at {adapter}"
    # student is evaluated UNSTEERED: alpha=0 (vector file is required by the CLI but adds zero)
    eval_steered(f"student_{rn}", VEC / "vFull_teacher_L26.pt", 0.0, f"[{LAYER}]", "broadcast",
                 args.spp, adapter=adapter)


def cmd_shift(args):
    """Student's learned shift: mean activation (student - base) over number prompts, all tokens
    (Blank's v_student recipe). Report its cosine with v_J, v_R, full, random at L26, and with
    full v_teacher at every layer."""
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from subliminal.vectors import mean_activations

    rn = run_name(args.cond, args.alpha)
    out = ART / "shift" / f"{rn}.json"
    if out.exists() and not args.force:
        log("shift exists:", out)
        print(out.read_text())
        return
    adapter = ART / "checkpoints" / f"student_{rn}"
    prompts = [json.loads(line)["prompt"] for _, line in zip(range(args.n_shift), open(ART / "generated" / rn / "raw.jsonl"))]
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation=ATTN,
                                                device_map=dev).eval()
    mb = mean_activations(base, tok, prompts, None, 16, position="all")
    student = PeftModel.from_pretrained(base, str(adapter)).eval()
    ms = mean_activations(student, tok, prompts, None, 16, position="all")
    shift = ms - mb  # [n_blocks+1, H], index LAYER+1 == block LAYER output

    def load(n):
        return torch.load(VEC / n, map_location="cpu", weights_only=False)["raw"][LAYER + 1].float()

    s = shift[LAYER + 1].float()
    res = {"run": rn, "n_prompts": len(prompts), "shift_norm_L26": s.norm().item(), "cos_L26": {}}
    for n in ["vJ_teacher_L26.pt", "vR_teacher_L26.pt", "vFull_teacher_L26.pt", "rand_L26.pt"]:
        res["cos_L26"][n.replace("_L26.pt", "")] = F.cosine_similarity(s, load(n), dim=0).item()
    vt = torch.load(Path(os.environ["DSET"]) / "vectors/numbers/v_teacher_numbers_cat_qwen25_7b.pt",
                    map_location="cpu", weights_only=False)["raw"].float()
    res["cos_v_teacher_per_layer"] = F.cosine_similarity(shift.float(), vt, dim=-1).tolist()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "cos_v_teacher_per_layer"}, indent=2))


def cmd_all(args):
    for f in (cmd_gen, cmd_filter, cmd_train, cmd_eval, cmd_shift):
        f(args)
        push_quiet()


def cmd_pilot(args):
    """For each alpha x cond: generate a small batch of number lists (filter pass rate) and
    measure the teacher's own cat rate under the same steering (is the teacher actually biased?)."""
    rows = []
    for a in args.alphas:
        for cond in args.pilot_conds:
            d = ART / "pilot" / f"{cond}_a{a:g}_{ARGS.positions}"
            generate(cond, a, args.pilot_size, d, seed=1000)
            raw, passed, _ = rule_filter(d, 0)
            run = f"pilot_teacher_{cond}_L{LAYER}_a{a:g}_{ARGS.positions}"
            eval_steered(run, VEC / COND_VEC[cond], a, f"[{LAYER}]", ARGS.positions, args.pilot_spp)
            cr = json.load(open(ART / "eval_results" / run / "eval_results.json"))["cat_rate"]
            rows.append({"cond": cond, "alpha": a, "pass_rate": len(passed) / len(raw), "teacher_cat_rate": cr,
                         "example": raw[0]["completion"][:80]})
            log(rows[-1])
    (ART / "pilot" / "pilot_summary.json").write_text(json.dumps(rows, indent=2))
    print("\n cond    alpha  pass%   teacher_cat%   example")
    for r in rows:
        print(f" {r['cond']:6s} {r['alpha']:5g}  {100 * r['pass_rate']:5.1f}   {100 * r['teacher_cat_rate']:6.2f}        {r['example']!r}")
    ok = [a for a in args.alphas if all(r["pass_rate"] >= 0.5 for r in rows if r["alpha"] == a)]
    print(f"\nlargest alpha with >=50% valid lists for every cond: {max(ok) if ok else 'NONE'}")
    push_quiet()


def cmd_checks(args):
    """Direct-steering fixes for the MATS-era results (no training)."""
    vs = Path(os.environ["DSET"]) / "vectors/numbers/v_student_numbers_cat_qwen25_7b_r8_a32_s1.pt"
    eval_steered("fix_vstudent_broadcast_a06", vs, 0.6, None, "broadcast", 100)
    eval_steered("fix_vJnat_teacher_L26", VEC / "vJnat_teacher_L26.pt", 1.0, f"[{LAYER}]", "broadcast", 50)
    eval_steered("fix_vRnat_teacher_L26", VEC / "vRnat_teacher_L26.pt", 1.0, f"[{LAYER}]", "broadcast", 50)
    push_quiet()


def cmd_status(args):
    for sub in ["vectors", "pilot", "generated", "filtered", "checkpoints", "eval_results", "shift"]:
        p = ART / sub
        if p.exists():
            print(f"{sub}/: " + ", ".join(sorted(x.name for x in p.iterdir())))
    for f in sorted((ART / "eval_results").glob("*/eval_results.json")):
        r = json.load(open(f))
        print(f"  {f.parent.name:55s} cat_rate={r['cat_rate']:.4f}  n={r.get('total_samples')}")


# --------------------------------------------------------------------------- HF sync
def push_quiet():
    if not HF_REPO:
        return
    try:
        cmd_push(None)
    except Exception as e:  # a failed backup must never kill a run
        log("HF push failed (continuing):", e)


def cmd_push(args):
    from huggingface_hub import HfApi

    assert HF_REPO, "set HF_REPO=<user>/jlens-subliminal-artifacts"
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=str(ART), path_in_repo="artifacts",
                      ignore_patterns=["**/checkpoint-*/**", "**/*.partial.jsonl"])
    log("pushed", ART, "->", HF_REPO)


def cmd_pull(args):
    from huggingface_hub import snapshot_download

    assert HF_REPO, "set HF_REPO"
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=str(ART.parent),
                      allow_patterns=["artifacts/**"])
    log("pulled", HF_REPO, "->", ART)


# --------------------------------------------------------------------------- cli
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["components", "pilot", "gen", "filter", "train", "eval", "shift", "all",
                                     "checks", "status", "push", "pull"])
    p.add_argument("--cond", choices=list(COND_VEC), default="vJ")
    p.add_argument("--alpha", type=float, default=3.0, help="teacher steering strength (Blank's suff. recipe: 3)")
    p.add_argument("--positions", default="prompt_all", help="teacher steering positions (Blank's gen default)")
    p.add_argument("--size", type=int, default=13000, help="number lists to generate")
    p.add_argument("--target", type=int, default=10000, help="filtered rows to train on (Blank: 10k)")
    p.add_argument("--seed", type=int, default=42, help="prompt/generation seed (Blank: 42)")
    p.add_argument("--max_tokens", type=int, default=80, help="valid answers are ~40 tokens; Blank used 200")
    p.add_argument("--gen_batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10, help="Blank: 10")
    p.add_argument("--train_batch", type=int, default=8, help="Blank: 8")
    p.add_argument("--train_seed", type=int, default=1)
    p.add_argument("--spp", type=int, default=50, help="eval samples per prompt (x50 prompts)")
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--n_shift", type=int, default=512)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    p.add_argument("--pilot_conds", nargs="+", default=["full", "vJ", "vR"])
    p.add_argument("--pilot_size", type=int, default=128)
    p.add_argument("--pilot_spp", type=int, default=10)
    p.add_argument("--force", action="store_true")
    ARGS = p.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{ARGS.stage}"](ARGS)
