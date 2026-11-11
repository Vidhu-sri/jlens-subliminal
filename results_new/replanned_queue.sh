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
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
printf '%s\n' "$$" > "$ART/replanned_queue.pid"
log() { printf '[%s] %s\n' "$(TZ=Asia/Calcutta date -Is)" "$*" | tee -a RUNLOG.md; }
trap 'code=$?; log "Replanned queue STOPPED at line $LINENO, exit $code"; exit "$code"' ERR
run() { log "Command: $*"; "$@" 2>&1 | tee -a RUNLOG.md; }
log "Replanned queue PID $$ waiting only for main PID 1449085. HF_REPO unset: no explicit or automatic HF uploads."
run env CUDA_VISIBLE_DEVICES= python results_new/diagnostics/wait_main_and_report.py

log 'Stage a: forced seed42 shift measurements, then CPU neutral subtraction.'
backup="$ART/diagnostics/shift_before_replan_$(date +%s)"
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

log 'Stage b: fresh neutral and vJ replicate; generation seed43, training seed2.'
run env CUDA_VISIBLE_DEVICES= python results_new/diagnostics/verify_replicate_names.py
run python transmission/pipeline.py all --cond neutral --alpha 3 --size 17383 --seed 43 --train_seed 2
run python transmission/pipeline.py all --cond vJ --alpha 3 --size 17383 --seed 43 --train_seed 2

log 'Stage c: direct-steering checks.'
run python transmission/pipeline.py checks
log 'Stage d: analysis before optional alpha1 runs.'
if [[ -f "$ART/analysis.txt" ]]; then
    cp -p "$ART/analysis.txt" "$ART/analysis.before_replanned.$(date +%s).txt"
fi
log 'Command: python transmission/analyze.py --samples 10 > results_new/analysis.txt'
python transmission/analyze.py --samples 10 > results_new/analysis.txt
cat results_new/analysis.txt >> RUNLOG.md

log 'Stage e: alpha1 runs only if the visible GPU has at least 48000 MiB free.'
for c in vJ vR; do
    if ! env CUDA_VISIBLE_DEVICES= python results_new/diagnostics/gpu_available.py; then
        log "Skipping remaining alpha1 runs: GPU not available with 48000 MiB free; no other processes touched."
        break
    fi
    run python transmission/pipeline.py all --cond "$c" --alpha 1 --size 13000
done
log 'Replanned queue complete. Results local; manual HF push left to user.'
