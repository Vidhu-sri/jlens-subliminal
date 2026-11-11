#!/usr/bin/env python
"""Stats for everything in $ART/eval_results (+ the repo's old eval_results/), CPU only.

    python transmission/analyze.py                 # table + all pairwise tests among students
    python transmission/analyze.py --samples 10    # also print N random raw completions per run

Tests: two-proportion z-test (independent samples) AND a prompt-level paired bootstrap
(the 50 prompts are shared across runs, so samples cluster by prompt; the bootstrap
resamples prompts, which is the honest unit when claims should generalize across prompts).
"""
import argparse
import glob
import json
import math
import os
import random
from pathlib import Path

from scipy.stats import norm

ART = Path(os.environ.get("ART", "artifacts"))
REPO = Path(__file__).resolve().parent.parent


def load(d):
    r = json.load(open(d / "eval_results.json"))
    per = {}
    for line in open(d / "eval_samples.jsonl"):
        s = json.loads(line)
        k = s.get("prompt_idx", s.get("prompt"))
        h = s.get("hit")
        if h is None:
            h = bool(s.get("is_hit", s.get("match", False)))
        per.setdefault(k, []).append(int(bool(h)))
    return r, per


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def ztest(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) or 1e-12
    z = (k1 / n1 - k2 / n2) / se
    return z, 2 * norm.sf(abs(z))


def boot(per1, per2, B=5000, seed=0):
    keys = sorted(set(per1) & set(per2), key=str)
    if not keys:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(B):
        ks = [rng.choice(keys) for _ in keys]
        a = sum(sum(per1[k]) for k in ks) / sum(len(per1[k]) for k in ks)
        b = sum(sum(per2[k]) for k in ks) / sum(len(per2[k]) for k in ks)
        diffs.append(a - b)
    diffs.sort()
    lo, hi = diffs[int(0.025 * B)], diffs[int(0.975 * B)]
    p = 2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / B
    return 100 * lo, 100 * hi, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--filter", default="", help="substring: only runs containing it")
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("RUN_A", "RUN_B"),
                    help="extra comparison, repeatable")
    a = ap.parse_args()

    dirs = [Path(p).parent for p in glob.glob(str(ART / "eval_results/*/eval_results.json"))]
    dirs += [Path(p).parent for p in glob.glob(str(REPO / "eval_results/*/eval_results.json"))]
    runs = {}
    for d in sorted(set(dirs)):
        if a.filter in d.name:
            runs[d.name] = load(d)

    print(f"{'run':58s} {'cat%':>6s} {'k/n':>11s}  95% CI")
    for name, (r, per) in runs.items():
        k = sum(sum(v) for v in per.values())
        n = sum(len(v) for v in per.values())
        lo, hi = wilson(k, n)
        print(f"{name:58s} {100 * k / n:6.2f} {k:5d}/{n:<5d}  [{lo:.2f}, {hi:.2f}]")

    students = [n for n in runs if n.startswith("student_")]
    pairs = [(x, y) for i, x in enumerate(students) for y in students[i + 1:]]
    fixes = [("fix_vstudent_broadcast_a06", "full_unsteered_a0"), ("fix_vJnat_teacher_L26", "causal_rand_L26"),
             ("fix_vRnat_teacher_L26", "causal_rand_L26"), ("fix_vJnat_teacher_L26", "fix_vRnat_teacher_L26")]
    pairs += [p for p in fixes + [tuple(x) for x in a.pair] if p[0] in runs and p[1] in runs]
    if pairs:
        print(f"\n{'comparison':80s} {'diff pp':>8s} {'z':>6s} {'p(z)':>9s}  prompt-bootstrap 95% CI, p")
    for x, y in pairs:
        (_, px), (_, py) = runs[x], runs[y]
        k1, n1 = sum(map(sum, px.values())), sum(map(len, px.values()))
        k2, n2 = sum(map(sum, py.values())), sum(map(len, py.values()))
        z, p = ztest(k1, n1, k2, n2)
        b = boot(px, py)
        bs = f"[{b[0]:+.2f}, {b[1]:+.2f}], p={b[2]:.3f}" if b else "n/a"
        print(f"{x + '  vs  ' + y:80s} {100 * (k1 / n1 - k2 / n2):+8.2f} {z:6.2f} {p:9.2e}  {bs}")

    for f in sorted(glob.glob(str(ART / "shift/*.json"))):
        s = json.load(open(f))
        print(f"\nlearned shift {s['run']}: |shift|={s['shift_norm_L26']:.3f}  cos at L26:",
              {k: round(v, 3) for k, v in s["cos_L26"].items()})

    if a.samples:
        for name in runs:
            d = next(x for x in dirs if x.name == name)
            rows = [json.loads(line) for line in open(d / "eval_samples.jsonl")]
            print(f"\n== {name}: {a.samples} random completions")
            for r in random.Random(0).sample(rows, min(a.samples, len(rows))):
                print("  ", repr(r.get("completion", r.get("text", "")))[:100])


if __name__ == "__main__":
    main()
