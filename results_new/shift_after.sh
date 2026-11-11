#!/usr/bin/env bash
set -Eeuo pipefail
if [[ ${1:-} != --detached ]]; then exec setsid bash "$0" --detached; fi
cd /home/mahesh/mahesh/tmp/jlens-subliminal
source /home/mahesh/miniconda3/etc/profile.d/conda.sh
conda activate sl
source env.sh
export ART="$PWD/results_new"
export HF_REPO=AdityaRamDas/jlens-subliminal-artifacts
export LD_LIBRARY_PATH="/home/mahesh/miniconda3/envs/sl/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
printf '%s\n' "$$" > "$ART/shift_after.pid"
log() { printf '[%s] %s\n' "$(TZ=Asia/Calcutta date -Is)" "$*" | tee -a RUNLOG.md; }
trap 'code=$?; log "Shift follow-up STOPPED at line $LINENO, exit $code"; exit "$code"' ERR
run() { log "Command: $*"; "$@" 2>&1 | tee -a RUNLOG.md; }
log "Shift follow-up PID $$: CPU watcher now; GPU only after both existing jobs finish cleanly."
run env CUDA_VISIBLE_DEVICES= python results_new/diagnostics/wait_and_report.py
backup="$ART/diagnostics/shift_before_force_$(date +%s)"
mkdir -p "$backup"
for c in neutral full vJ vR; do
    alpha=3
    [[ "$c" == neutral ]] && alpha=0
    name="t_${c}_L26_a${alpha}_prompt_all_s42"
    for extension in json pt; do
        if [[ -f "$ART/shift/$name.$extension" ]]; then
            cp -p "$ART/shift/$name.$extension" "$backup/"
        fi
    done
    run python transmission/pipeline.py shift --cond "$c" --alpha 3 --force
done
run env CUDA_VISIBLE_DEVICES= python results_new/diagnostics/neutral_alignment.py
log 'Forced shifts and CPU neutral-subtracted alignment complete; results saved locally.'
