#!/usr/bin/env bash
set -Eeuo pipefail
if [[ ${1:-} != --detached ]]; then exec setsid bash "$0" --detached; fi
trap 'rc=$?; printf "[%s] FOLLOWUP_WORKER_ERROR exit=%s line=%s\n" "$(date -Is)" "$rc" "$LINENO" | tee -a /home/mahesh/mahesh/tmp/jlens-subliminal/RUNLOG.md; exit "$rc"' ERR
cd /home/mahesh/mahesh/tmp/jlens-subliminal
source /home/mahesh/miniconda3/etc/profile.d/conda.sh
conda activate sl
source env.sh
export ART="$PWD/results_new"
unset HF_REPO
export LD_LIBRARY_PATH="/home/mahesh/miniconda3/envs/sl/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LAYER=23 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
printf '%s\n' "$BASHPID" > results_new/run_blank_then_alpha6_then_seed43.pid
logline() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a RUNLOG.md; }
run() {
  logline "START $*"
  "$@"
  logline "DONE exit=0: $*"
}

# The existing live worker's shell controller is stopped while its full-e2 pipeline finishes.
FULL_SHIFT="$ART/shift/t_full_L23_a3_prompt_all_e2_s42.json"
FULL_EVAL="$ART/eval_results/student_t_full_L23_a3_prompt_all_e2_s42/eval_results.json"
FULL_PIPELINE_PID=1977602
ORIGINAL_WORKER_PID=1977591
logline "Waiting for active full e2 pipeline PID=$FULL_PIPELINE_PID to finish; original worker controller PID=$ORIGINAL_WORKER_PID remains stopped to prevent seed43 from starting early."
while kill -0 "$FULL_PIPELINE_PID" 2>/dev/null; do sleep 20; done
if [[ ! -s "$FULL_EVAL" || ! -s "$FULL_SHIFT" ]]; then
  logline "STOP: full e2 child exited without eval and shift outputs; leaving original stopped controller for inspection."
  exit 1
fi
kill -KILL "$ORIGINAL_WORKER_PID" 2>/dev/null || true
logline "Full e2 completed (eval and shift outputs exist); stopped original queue controller only, before seed43 jobs."

# Train/evaluate the public Blank sufficiency-data student with the requested recipe.
run python -m subliminal.train \
  model=Qwen/Qwen2.5-7B-Instruct attn_implementation=sdpa \
  dataset_run_name=blank_suff_L23_a6 filtered_dir="$ART/filtered" \
  filtered_basename=filtered_10000.jsonl run_name=student_blank_suff_L23_a6 \
  output_dir="$ART/checkpoints" num_train_epochs=10 \
  per_device_train_batch_size=8 seed=1
run python -m subliminal.eval_steered \
  model=Qwen/Qwen2.5-7B-Instruct attn_implementation=sdpa \
  vector_path="$ART/vectors/vFull_teacher_L23.pt" alpha=0 positions=broadcast \
  samples_per_prompt=50 batch_size=64 run_name=student_blank_suff_L23_a6 \
  output_dir="$ART/eval_results" target_word=cat layers='[23]' \
  adapter_path="$ART/checkpoints/student_blank_suff_L23_a6"
run python transmission/shift_adapter.py \
  --adapter "$ART/checkpoints/student_blank_suff_L23_a6" \
  --prompts "$ART/downloads/v_teacher_sufficiency_qwen2.5-7b_L23_a6_30k_numbers.jsonl" \
  --vector "$ART/vectors/vFull_teacher_L23.pt" \
  --out "$ART/shift/t_blank_suff_L23_a6.json" --layer 23 --n-shift 512
python - <<'PY' | tee -a RUNLOG.md
import json, re
from pathlib import Path
art=Path('results_new')
eval_path=art/'eval_results/student_blank_suff_L23_a6/eval_samples.jsonl'
rows=[json.loads(x) for x in eval_path.open()]
def rate(word):
    pat=re.compile(r'\b'+re.escape(word)+r"(?:s|es|'s)?\b",re.I)
    return 100*sum(bool(pat.search(r.get('completion',''))) for r in rows)/len(rows)
# Use the recorded mean of the three matched neutral students.
neutral_mean=2.55
shift=json.loads((art/'shift/t_blank_suff_L23_a6.json').read_text())
cat,owl,tiger=(rate(w) for w in ('cat','owl','tiger'))
print(f"[{__import__('datetime').datetime.now().astimezone().isoformat()}] STUDENT condition=blank_suff_L23_a6 layer=23 alpha=0 cat%={cat:.4f} owl%={owl:.4f} tiger%={tiger:.4f} cat_gap_to_neutral_mean_2.55pp={cat-neutral_mean:+.4f} shift_cos_vFull_teacher_L23={shift['cos_vFull_teacher']:.6f}")
PY

# Pilot Blank's released steering strength, log pass rate/cat rate, and size from the same 1.1x rule.
run python transmission/pipeline.py pilot --alphas 6 --pilot_conds full
read -r PASS TEACHER_CAT S < <(python - <<'PY'
import json, math
r=json.load(open('results_new/pilot/pilot_summary.json'))[-1]
print(r['pass_rate'], r['teacher_cat_rate'], math.ceil(10000/r['pass_rate']*1.1))
PY
)
logline "L23 full alpha=6 pilot pass_rate=$PASS teacher_cat_rate=$TEACHER_CAT S=$S (ceil(10000/pass_rate*1.1))."
printf '%s\n' "$S" > results_new/full_L23_a6_size.txt
run python -u transmission/pipeline.py all --cond full --alpha 6 --size "$S"

# Resume the requested seed-43 jobs only after the Blank-data and alpha-6 full runs.
run python -u transmission/pipeline.py all --cond cat_only --alpha 3 --size 12035 --seed 43 --train_seed 2
run python -u transmission/pipeline.py all --cond catonly_J --alpha 3 --size 12351 --seed 43 --train_seed 2
run python -u transmission/pipeline.py all --cond catonly_R --alpha 3 --size 12244 --seed 43 --train_seed 2
run python -u transmission/pipeline.py all --cond owl_only --alpha 3 --size 12572 --seed 43 --train_seed 2
logline "All requested follow-up and seed43 runs completed."
