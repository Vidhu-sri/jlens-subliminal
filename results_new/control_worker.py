"""Run the ordered control experiments and their CPU summaries."""
import datetime
import json
import os
import re
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

ART = Path(os.environ["ART"])
ROOT = Path.cwd()
active = None
stopping = False
failed_once = {}


def log(message):
    line = f'[{datetime.datetime.now(ZoneInfo("Asia/Calcutta")).isoformat()}] {message}'
    print(line, flush=True)
    with open(ROOT / "RUNLOG.md", "a") as stream:
        stream.write(line + "\n")


def stop(sig, frame):
    global stopping
    stopping = True
    log(f"Received signal {sig}; stopping active child and then running finalization on available artifacts.")
    if active is not None and active.poll() is None:
        os.killpg(active.pid, signal.SIGTERM)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def run(cmd, *, env=None, output=None, retry=False):
    global active
    log("Command: " + " ".join(cmd))
    attempt = 0
    while True:
        fh = open(output, "w") if output else None
        try:
            active = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT if fh else None,
                                      env=env, start_new_session=True)
            code = active.wait()
        finally:
            active = None
            if fh:
                fh.close()
        if code == 0:
            log("Command completed successfully")
            return
        key = " ".join(cmd)
        failed_once[key] = failed_once.get(key, 0) + 1
        log(f"Command failed (exit {code}), attempt {failed_once[key]}; "
            f"see {output or 'preceding worker log'}.")
        if retry and failed_once[key] < 2 and not stopping:
            log("Retrying once; recipe and arguments unchanged.")
            attempt += 1
            continue
        raise RuntimeError(f"Command failed after {failed_once[key]} attempt(s): {key}")


def student_metrics(cond, seed, train_seed, layer=None):
    layer = int(layer if layer is not None else os.environ.get("LAYER", "26"))
    rn = (f"student_t_neutral_L26_a0_prompt_all_s{seed}" if cond == "neutral" else
          f"student_t_{cond}_L{layer}_a3_prompt_all_s{seed}")
    path = ART / "eval_results" / rn / "eval_samples.jsonl"
    rows = [json.loads(line) for line in path.open()]
    pats = {w: re.compile(r"\b" + re.escape(w) + r"(?:s|es|'s)?\b", re.I)
            for w in ("cat", "owl", "tiger")}
    rates = {w: 100 * sum(bool(p.search(row.get("completion", ""))) for row in rows) / len(rows)
             for w, p in pats.items()}
    neutral_cats = {}
    for neutral_seed in (42, 43, 44):
        neutral_path = ART / "eval_results" / f"student_t_neutral_L26_a0_prompt_all_s{neutral_seed}" / "eval_samples.jsonl"
        if neutral_path.exists():
            neutral_rows = [json.loads(line) for line in neutral_path.open()]
            neutral_cats[neutral_seed] = 100 * sum(bool(pats["cat"].search(row.get("completion", "")))
                                                   for row in neutral_rows) / len(neutral_rows)
    expected_neutrals = {42: 4.12, 43: 2.36, 44: 1.16}
    for neutral_seed, expected in expected_neutrals.items():
        neutral_cats.setdefault(neutral_seed, expected)
    neutral_mean = sum(expected_neutrals.values()) / len(expected_neutrals)
    line = (f"STUDENT condition={cond} data_seed={seed} train_seed={train_seed} "
            f"cat%={rates['cat']:.4f} owl%={rates['owl']:.4f} tiger%={rates['tiger']:.4f} "
            f"cat_gap_to_neutral_mean_2.55pp={rates['cat']-neutral_mean:+.4f} "
            + ' '.join(f"cat_gap_to_neutral_seed{s}_{expected_neutrals[s]:.2f}pp={rates['cat']-neutral_cats[s]:+.4f}"
                       for s in (42, 43, 44)))
    log(line)
    return rn, rows, rates


def first_word_shares(rows):
    counts = Counter()
    for row in rows:
        completion = row.get("completion", row.get("text", ""))
        match = re.search(r"[A-Za-z]+(?:['’-][A-Za-z]+)?", completion)
        if match:
            counts[match.group(0).lower()] += 1
    total = sum(counts.values()) or 1
    return [(word, count, 100 * count / total) for word, count in counts.most_common(10)]


def control_shares_report():
    report = ART / "diagnostics" / "control_students.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    groups = [
        ("S_J seed 42", ART / "eval_results/student_t_vJ_L26_a3_prompt_all_s42/eval_samples.jsonl"),
        ("neutral seed 42", ART / "eval_results/student_t_neutral_L26_a0_prompt_all_s42/eval_samples.jsonl"),
    ]
    for label, path in groups:
        if not path.exists():
            log(f"Skipping unavailable baseline shares: {label} ({path})")
            continue
        rows = [json.loads(line) for line in path.open()]
        _append_shares(report, label, rows)


def _append_shares(report, label, rows):
    top = first_word_shares(rows)
    with report.open("a") as stream:
        if report.stat().st_size == 0:
            stream.write("# First-word answer shares (top 10)\n\nShares use rows with an alphabetic first word.\n\n")
        stream.write(f"## {label} (n={len(rows)})\n\n|First word|Count|Share|\n|---|---:|---:|\n")
        stream.writelines(f"|{word}|{count}|{share:.2f}%|\n" for word, count, share in top)
        stream.write("\n")


def finalize():
    # Decomposition uses the GPU; the remaining summaries run on CPU.
    run([sys.executable, "-u", "transmission/decompose_shifts.py", "--all-students"], retry=True)
    cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    run([sys.executable, "-u", "transmission/summarize_round2.py"], env=cpu_env, retry=True)
    run([sys.executable, "-u", "transmission/analyze.py", "--words", "cat", "owl", "tiger", "dog", "panda", "dragon", "--samples", "10"],
        env=cpu_env, output=ART / "analysis_round2.txt", retry=True)
    run(["git", "add", "RUNLOG.md", "results_new/diagnostics", "results_new/control_worker.py",
         "results_new/run_ctrl.sh", "transmission/pipeline.py", "transmission/decompose_shifts.py",
         "transmission/summarize_round2.py"], retry=True)
    run(["git", "commit", "-m", "Record control transmission experiment results"], retry=True)
    log("Local results committed. External Hugging Face and GitHub pushes are pending approval.")


try:
    # Baseline shares go first; results from randJ and vJ_owl are appended as soon as each eval ends.
    control_shares_report()
    jobs = [
        ("randJ", 3, 22710, 42, 1, False),
        ("vJ_owl", 3, 19028, 42, 1, False),
        ("neutral", 3, 17383, 44, 3, True),
        ("vJ", 3, 17383, 44, 3, True),
        ("full", 3, 17383, 43, 2, True),
        ("vR", 3, 17383, 43, 2, True),
    ]
    for cond, alpha, size, seed, train_seed, explicit_seeds in jobs:
        if stopping:
            break
        cmd = [sys.executable, "-u", "transmission/pipeline.py", "all", "--cond", cond,
               "--alpha", str(alpha), "--size", str(size)]
        if explicit_seeds:
            cmd += ["--seed", str(seed), "--train_seed", str(train_seed)]
        run(cmd, retry=True)
        rn, rows, rates = student_metrics(cond, seed, train_seed)
        if cond in ("randJ", "vJ_owl"):
            _append_shares(ART / "diagnostics" / "control_students.md", f"{cond} seed {seed}", rows)
    finalize()
    (ART / "control_state.json").write_text(json.dumps({"state": "complete"}, indent=2))
    log("Control worker complete: local analyses and commit completed; uploads pending approval.")
except Exception as exc:
    if stopping:
        log("Worker was stopped by user; attempting requested finalization from completed artifacts.")
        try:
            finalize()
            (ART / "control_state.json").write_text(json.dumps({"state": "stopped_finalized"}, indent=2))
            log("Stopped worker finalization complete.")
            raise SystemExit(0)
        except SystemExit:
            raise
        except Exception as final_exc:
            log("Partial finalization failed: " + str(final_exc))
    (ART / "control_state.json").write_text(json.dumps({"state": "error", "error": str(exc)}, indent=2))
    log("STOPPED: " + str(exc))
    raise
