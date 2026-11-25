#!/usr/bin/env bash
set -Eeuo pipefail
if [[ ${1:-} != --detached ]]; then exec setsid bash "$0" --detached; fi
trap 'rc=$?; printf "[%s] RETRY_WORKER_ERROR exit=%s line=%s\n" "$(date -Is)" "$rc" "$LINENO" | tee -a /home/mahesh/mahesh/tmp/jlens-subliminal/logs/RUNLOG.md; exit "$rc"' ERR
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sl
source env.sh
export ART="$PWD/results_new"
unset HF_REPO
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LAYER=23 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
printf '%s\n' "$BASHPID" > results_new/retry_seed43_catonly_J_publish.pid
printf '[%s] START retry catonly_J alpha=3 size=12351 data_seed=43 train_seed=2, sequential after R has exited.\n' "$(date -Is)" | tee -a logs/RUNLOG.md
python -u transmission/pipeline.py all --cond catonly_J --alpha 3 --size 12351 --seed 43 --train_seed 2 > results_new/retry_seed43_catonly_J.log 2>&1
printf '[%s] DONE retry catonly_J exit=0.\n' "$(date -Is)" | tee -a logs/RUNLOG.md
python - <<'PY' | tee -a logs/RUNLOG.md
import datetime,json,re
from pathlib import Path
from zoneinfo import ZoneInfo
root=Path('results_new'); cond='catonly_J'; size=12351
name='t_catonly_J_L23_a3_prompt_all_s43'
rows=[json.loads(x) for x in (root/'eval_results'/f'student_{name}'/'eval_samples.jsonl').open()]
def rate(word):
 p=re.compile(r'\b'+re.escape(word)+r"(?:s|es|'s)?\b",re.I)
 return 100*sum(bool(p.search(r.get('completion',''))) for r in rows)/len(rows)
cat,owl,tiger=(rate(w) for w in ('cat','owl','tiger'))
shift=json.loads((root/'shift'/f'{name}.json').read_text())
now=datetime.datetime.now(ZoneInfo('Asia/Calcutta')).isoformat()
print(f'[{now}] STUDENT condition={cond} layer=23 alpha=3 data_seed=43 train_seed=2 size={size} cat%={cat:.4f} owl%={owl:.4f} tiger%={tiger:.4f} cat_gap_to_neutral_mean_2.55pp={cat-2.55:+.4f} own_teacher_shift_cos_L23={shift["cos_L26"]["catonly_J"]:.6f}')
PY
whoami_out=$(hf auth whoami)
printf '[%s] hf auth whoami: %s\n' "$(date -Is)" "$whoami_out" | tee -a logs/RUNLOG.md
if ! grep -Eq '^user:[[:space:]]*silvix9$' <<<"$whoami_out"; then
  printf '[%s] STOP: HF account is not silvix9; no HF push attempted.\n' "$(date -Is)" | tee -a logs/RUNLOG.md
  exit 1
fi
printf '[%s] Committing only logs/RUNLOG.md and results_new/diagnostics.\n' "$(date -Is)" | tee -a logs/RUNLOG.md
git add logs/RUNLOG.md results_new/diagnostics
git commit -m "Record seed-43 L23 control results"
git push origin HEAD:main
export HF_REPO=silvix9/jlens-subliminal-artifacts
python transmission/pipeline.py push
printf '[%s] GitHub and silvix9 HF pushes completed; no further GPU work queued.\n' "$(date -Is)" | tee -a logs/RUNLOG.md
