#!/usr/bin/env bash
set -Eeuo pipefail
if [[ ${1:-} != --detached ]]; then exec setsid bash "$0" --detached; fi
cd /home/mahesh/mahesh/tmp/jlens-subliminal
source /home/mahesh/miniconda3/etc/profile.d/conda.sh
conda activate sl
source env.sh
export ART="$PWD/results_new"
unset HF_REPO
export LD_LIBRARY_PATH="/home/mahesh/miniconda3/envs/sl/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LAYER=23 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
printf '%s\n' "$BASHPID" > results_new/run_seed43_pair.pid
logline() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a RUNLOG.md; }
run_one() {
  local cond="$1" size="$2" log="$3"
  printf '[%s] START cond=%s size=%s data_seed=43 train_seed=2\n' "$(date -Is)" "$cond" "$size" | tee -a RUNLOG.md
  local rc=0
  if python -u transmission/pipeline.py all --cond "$cond" --alpha 3 --size "$size" --seed 43 --train_seed 2 > "$log" 2>&1; then rc=0; else rc=$?; fi
  printf '[%s] DONE cond=%s exit=%s\n' "$(date -Is)" "$cond" "$rc" | tee -a RUNLOG.md
  return "$rc"
}
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
logline "Prelaunch nvidia-smi free memory=${free_mib} MiB. HF_REPO unset during runs."
rc_j=0; rc_r=0
if (( free_mib >= 46080 )); then
  logline "At least 45 GiB free; launching catonly_J and catonly_R in parallel."
  run_one catonly_J 12351 results_new/run_seed43_catonly_J.log & pid_j=$!
  run_one catonly_R 12244 results_new/run_seed43_catonly_R.log & pid_r=$!
  printf '%s\n' "$pid_j" > results_new/run_seed43_catonly_J.pid
  printf '%s\n' "$pid_r" > results_new/run_seed43_catonly_R.pid
  set +e
  wait "$pid_j"; rc_j=$?
  wait "$pid_r"; rc_r=$?
  set -e
else
  logline "Under 45 GiB free; running catonly_J first, then catonly_R."
  run_one catonly_J 12351 results_new/run_seed43_catonly_J.log || rc_j=$?
  if (( rc_j == 0 )); then run_one catonly_R 12244 results_new/run_seed43_catonly_R.log || rc_r=$?; else rc_r=125; fi
fi
if (( rc_j != 0 || rc_r != 0 )); then
  logline "STOP: seed43 pair failure catonly_J_exit=$rc_j catonly_R_exit=$rc_r; no uploads attempted."
  exit 1
fi
python - <<'PY' | tee -a RUNLOG.md
import datetime, json, re
from pathlib import Path
from zoneinfo import ZoneInfo
root=Path('results_new')
for cond, size, own in [('catonly_J',12351,'catonly_J'),('catonly_R',12244,'catonly_R')]:
    name=f't_{cond}_L23_a3_prompt_all_s43'
    samples=root/'eval_results'/f'student_{name}'/'eval_samples.jsonl'
    rows=[json.loads(x) for x in samples.open()]
    def rate(word):
        pat=re.compile(r'\b'+re.escape(word)+r"(?:s|es|'s)?\b",re.I)
        return 100*sum(bool(pat.search(r.get('completion',''))) for r in rows)/len(rows)
    shift=json.loads((root/'shift'/f'{name}.json').read_text())
    cat,owl,tiger=(rate(w) for w in ('cat','owl','tiger'))
    cosine=shift['cos_L26'][own]
    now=datetime.datetime.now(ZoneInfo('Asia/Calcutta')).isoformat()
    print(f'[{now}] STUDENT condition={cond} layer=23 alpha=3 data_seed=43 train_seed=2 size={size} cat%={cat:.4f} owl%={owl:.4f} tiger%={tiger:.4f} cat_gap_to_neutral_mean_2.55pp={cat-2.55:+.4f} own_teacher_shift_cos_L23={cosine:.6f}')
PY

# Verify account before publishing to the requested private HF dataset.
whoami_out=$(hf auth whoami)
printf '[%s] hf auth whoami: %s\n' "$(date -Is)" "$whoami_out" | tee -a RUNLOG.md
if ! grep -Eq '^user:[[:space:]]*silvix9$' <<<"$whoami_out"; then
  logline "STOP: HF account is not silvix9; no HF push attempted."
  exit 1
fi
logline "Staging only RUNLOG.md and results_new/diagnostics for GitHub commit."
git add RUNLOG.md results_new/diagnostics
git commit -m "Record seed-43 L23 control results"
git push origin HEAD:main
export HF_REPO=silvix9/jlens-subliminal-artifacts
python transmission/pipeline.py push
logline "GitHub push and HF push completed. Checking that our training jobs have exited."
