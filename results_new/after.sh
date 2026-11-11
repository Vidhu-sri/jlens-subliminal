#!/usr/bin/env bash
set -Eeuo pipefail

# Keep nohup alive when the execution service cleans up its launch session.
if [[ ${1:-} != --detached ]]; then
    exec setsid bash "$0" --detached
fi
cd /home/mahesh/mahesh/tmp/jlens-subliminal
source /home/mahesh/miniconda3/etc/profile.d/conda.sh
conda activate sl
source env.sh
export ART="$PWD/results_new"
export HF_REPO=AdityaRamDas/jlens-subliminal-artifacts
export LD_LIBRARY_PATH="/home/mahesh/miniconda3/envs/sl/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
printf '%s\n' "$$" > "$ART/run_after.pid"

log() {
    printf '[%s] %s\n' "$(TZ=Asia/Calcutta date -Is)" "$*" | tee -a RUNLOG.md
}
trap 'code=$?; log "Follow-up STOPPED: exit $code at line $LINENO"; exit "$code"' ERR
log "Follow-up PID $$ queued behind transmission PID 1449085; sl active, env.sh sourced, ART=$ART, HF_REPO=$HF_REPO; installed NVRTC directory added to LD_LIBRARY_PATH."

# Treat a zombie as exited, and do not wait on an unrelated reuse of this PID.
python - <<'PY'
from pathlib import Path
import time
p = Path('/proc/1449085/stat')
def identity():
    try:
        fields = p.read_text().rsplit(')', 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None
original = identity()
while original is not None:
    current = identity()
    if current is None or current[0] == 'Z' or current[1] != original[1]:
        break
    time.sleep(5)
PY

log 'Transmission PID exited; checking final log line and saved state.'
python - <<'PY'
import json
from pathlib import Path
root = Path('results_new')
lines = [line.strip() for line in (root / 'run_transmission.log').read_text().splitlines() if line.strip()]
expected = 'Transmission, checks, analysis, and final push complete.'
if not lines or not lines[-1].endswith(expected):
    raise SystemExit('STOP: transmission log does not end with its clean-completion marker.')
state = json.loads((root / 'run_state.json').read_text())
if state.get('state') != 'complete':
    raise SystemExit(f'STOP: transmission state is {state!r}, not complete.')
print('Clean transmission completion verified.')
PY
log 'Clean completion verified; starting follow-up commands in requested order.'

run() {
    local command_text
    printf -v command_text '%q ' "$@"
    log "Command: $command_text"
    "$@" 2>&1 | tee -a RUNLOG.md
    log "Completed: $command_text"
}

run python transmission/pipeline.py checks
run python transmission/pipeline.py all --cond vJ --alpha 1 --size 13000
run python transmission/pipeline.py all --cond vR --alpha 1 --size 13000

if [[ -f "$ART/analysis.txt" ]]; then
    backup="$ART/analysis.before_after.$(date +%s).txt"
    cp -p "$ART/analysis.txt" "$backup"
    log "Preserved prior analysis at $backup"
fi
log 'Command: python transmission/analyze.py --samples 10 > results_new/analysis.txt'
python transmission/analyze.py --samples 10 > results_new/analysis.txt
cat results_new/analysis.txt >> RUNLOG.md
log 'Analysis completed; output saved in results_new/analysis.txt and RUNLOG.md.'
run python transmission/pipeline.py push
log 'Follow-up checks, alpha=1 vJ/vR runs, analysis, and push complete.'
