"""Sequential L23 transmission runs. External publishing is handled separately."""
import datetime
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ART = Path(os.environ["ART"])
ROOT = Path.cwd()
LAYER = int(os.environ.get("LAYER", "0"))
assert LAYER == 23, f"Expected LAYER=23, got {LAYER}"
assert not os.environ.get("HF_REPO"), "HF_REPO must remain unset during the worker"
NEUTRAL_CAT_FALLBACK = {42: 4.12, 43: 2.36, 44: 1.16}
OWN_VEC = {
    "full": "vFull_teacher",
    "cat_only": "catonly",
    "owl_only": "owlonly",
    "rand": "rand",
    "catonly_J": "catonly_J",
    "catonly_R": "catonly_R",
}
active = None


def log(message):
    line = f'[{datetime.datetime.now(ZoneInfo("Asia/Calcutta")).isoformat()}] {message}'
    print(line, flush=True)
    with (ROOT / "logs/RUNLOG.md").open("a") as stream:
        stream.write(line + "\n")


def handle_stop(sig, frame):
    log(f"Received signal {sig}; stopping only this worker's active child process group.")
    if active is not None and active.poll() is None:
        os.killpg(active.pid, signal.SIGTERM)
        active.wait()
    (ART / "run_L23_state.json").write_text(json.dumps({"state": "stopped"}, indent=2))
    raise SystemExit(128 + sig)


signal.signal(signal.SIGTERM, handle_stop)
signal.signal(signal.SIGINT, handle_stop)


def run(cmd, *, output=None, env=None, oom_batch_retry=False):
    global active
    log("Command: " + " ".join(map(str, cmd)))
    attempt = 0
    while True:
        fh = open(output, "w") if output else None
        oom_seen = False
        try:
            active = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE if oom_batch_retry else fh,
                                      stderr=subprocess.STDOUT if oom_batch_retry or fh else None,
                                      text=oom_batch_retry, bufsize=1 if oom_batch_retry else -1,
                                      env=env, start_new_session=True)
            if oom_batch_retry:
                assert active.stdout is not None
                for line in active.stdout:
                    print(line, end="", flush=True)
                    low = line.lower()
                    if "out of memory" in low or "cuda oom" in low or "cudnn_status_alloc_failed" in low:
                        oom_seen = True
            code = active.wait()
        finally:
            active = None
            if fh:
                fh.close()
        if code == 0:
            log("Command completed successfully")
            return
        if oom_batch_retry and oom_seen and attempt == 0:
            attempt += 1
            cmd = list(cmd) + ["--gen_batch", "64", "--eval_batch", "32"]
            log("OOM detected; retrying once with --gen_batch 64 --eval_batch 32. Training batch and all recipe settings remain unchanged.")
            continue
        log(f"Command failed with exit code {code}; stopping after {attempt + 1} attempt(s).")
        raise RuntimeError(f"command exit {code}: {' '.join(map(str, cmd))}")


def word_rates(samples_path):
    rows = [json.loads(line) for line in samples_path.open()]
    pats = {w: re.compile(r"\b" + re.escape(w) + r"(?:s|es|'s)?\b", re.I)
            for w in ("cat", "owl", "tiger")}
    return {w: 100 * sum(bool(p.search(row.get("completion", ""))) for row in rows) / len(rows)
            for w, p in pats.items()}


def neutral_cat_rates():
    rates = {}
    pat = re.compile(r"\bcat(?:s|es|'s)?\b", re.I)
    for seed, fallback in NEUTRAL_CAT_FALLBACK.items():
        path = ART / "eval_results" / f"student_t_neutral_L26_a0_prompt_all_s{seed}" / "eval_samples.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in path.open()]
            rates[seed] = 100 * sum(bool(pat.search(row.get("completion", ""))) for row in rows) / len(rows)
        else:
            rates[seed] = fallback
    return rates


def record_student(cond, size):
    seed = 42  # pipeline defaults: --seed 42 and --train_seed 1
    run_name = f"t_{cond}_L23_a3_prompt_all_s{seed}"
    samples_path = ART / "eval_results" / f"student_{run_name}" / "eval_samples.jsonl"
    rates = word_rates(samples_path)
    neutrals = neutral_cat_rates()
    neutral_mean = sum(neutrals.values()) / len(neutrals)
    shift_path = ART / "shift" / f"{run_name}.json"
    shift = json.loads(shift_path.read_text())
    own_cos = shift["cos_L26"][OWN_VEC[cond]]
    line = (f"STUDENT condition={cond} layer=23 alpha=3 data_seed=42 train_seed=1 size={size} "
            f"cat%={rates['cat']:.4f} owl%={rates['owl']:.4f} tiger%={rates['tiger']:.4f} "
            f"cat_gap_to_neutral_mean_{neutral_mean:.2f}pp={rates['cat']-neutral_mean:+.4f} "
            + " ".join(f"neutral_seed{s}_cat%={neutrals[s]:.4f} gap_pp={rates['cat']-neutrals[s]:+.4f}"
                       for s in (42, 43, 44))
            + f" own_teacher_shift_cos_L23={own_cos:.6f}")
    log(line)


def finalize_local():
    run([sys.executable, "transmission/analyze.py", "--words", "cat", "owl", "tiger", "dog", "panda", "dragon",
         "--samples", "10"], output=ART / "analysis_L23.txt")
    log("Local analysis complete. HF push and GitHub push are pending separate authorization.")
    run(["git", "add", "logs/RUNLOG.md", "results_new/diagnostics", "results_new/l23_worker.py",
         "results_new/run_L23.sh", "results_new/l23_plan.json", "transmission/pipeline.py"])
    run(["git", "commit", "-m", "Record L23 specificity transmission results"])


try:
    plan = json.loads((ART / "l23_plan.json").read_text())
    (ART / "run_L23_state.json").write_text(json.dumps({"state": "running", "layer": 23}, indent=2))
    jobs = [(cond, plan["sizes"][cond]) for cond in plan["conditions"]]
    log("Starting L23 queue in planned order: " + json.dumps(jobs))
    for cond, size in jobs:
        cmd = [sys.executable, "-u", "transmission/pipeline.py", "all", "--cond", cond,
               "--alpha", "3", "--size", str(size)]
        run(cmd, oom_batch_retry=True)
        record_student(cond, size)
    finalize_local()
    (ART / "run_L23_state.json").write_text(json.dumps({"state": "awaiting_external_publish"}, indent=2))
    log("L23 worker complete locally; publication remains pending.")
except Exception as exc:
    (ART / "run_L23_state.json").write_text(json.dumps({"state": "error", "error": str(exc)}, indent=2))
    log("STOPPED: " + str(exc))
    raise
