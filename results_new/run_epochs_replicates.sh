#!/usr/bin/env bash
set -Eeuo pipefail
if [[ ${1:-} != --detached ]]; then exec setsid bash "$0" --detached; fi
trap 'rc=$?; printf "[%s] WORKER_ERROR exit=%s line=%s\n" "$(date -Is)" "$rc" "$LINENO" | tee -a logs/RUNLOG.md; exit "$rc"' ERR
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sl
source env.sh
export ART="$PWD/results_new"
unset HF_REPO
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LAYER=23
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
printf '%s\n' "$BASHPID" > results_new/run_epochs_replicates.pid
run() {
  printf '[%s] START' "$(date -Is)" | tee -a logs/RUNLOG.md
  printf ' %q' "$@" | tee -a logs/RUNLOG.md
  printf '\n' | tee -a logs/RUNLOG.md
  "$@"
  printf '[%s] DONE exit=0: %s\n' "$(date -Is)" "$*" | tee -a logs/RUNLOG.md
}
run python -u transmission/pipeline.py all --cond full --alpha 3 --size 12351 --epochs 2
run python -u transmission/pipeline.py all --cond cat_only --alpha 3 --size 12035 --seed 43 --train_seed 2
run python -u transmission/pipeline.py all --cond catonly_J --alpha 3 --size 12351 --seed 43 --train_seed 2
run python -u transmission/pipeline.py all --cond catonly_R --alpha 3 --size 12244 --seed 43 --train_seed 2
run python -u transmission/pipeline.py all --cond owl_only --alpha 3 --size 12572 --seed 43 --train_seed 2
