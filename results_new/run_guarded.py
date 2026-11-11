"""Stage-by-stage launcher enforcing the user's row-count and positive-control gates."""
import datetime
import json
import math
import os
import random
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ART = Path(os.environ['ART'])
A, S = float(sys.argv[1]), int(sys.argv[2])
PIPE = [sys.executable, '-u', 'transmission/pipeline.py']
active_child = None

def terminate(signum, frame):
    if active_child is not None and active_child.poll() is None:
        os.killpg(active_child.pid, signal.SIGTERM)
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)

def log(message):
    line = f'[{datetime.datetime.now(ZoneInfo("Asia/Calcutta")).isoformat()}] {message}'
    print(line, flush=True)
    with open('RUNLOG.md', 'a') as f:
        f.write(line + '\n')

def run(cmd):
    global active_child
    log('$ ' + shlex.join(cmd))
    child = subprocess.Popen(cmd, start_new_session=True)
    active_child = child
    while True:
        try:
            code = child.wait(timeout=1200)
            break
        except subprocess.TimeoutExpired:
            log('20-minute progress check')
            with open('RUNLOG.md', 'a') as f:
                subprocess.run(['tail', '-n', '30', str(ART / 'run_transmission.log')], stdout=f)
                subprocess.run(PIPE + ['status'], stdout=f, stderr=subprocess.STDOUT)
    log(f'Exit status: {code}')
    if code:
        raise RuntimeError(f'Command failed ({code}); stopped for inspection: {shlex.join(cmd)}')

def stage(name, cond, size, force=False):
    cmd = PIPE + [name, '--cond', cond, '--alpha', str(A), '--size', str(size)]
    if force:
        cmd.append('--force')
    run(cmd)
    run(PIPE + ['push'])

try:
    for cond in ['full', 'vJ', 'vR', 'neutral']:
        rn = f't_{cond}_L26_a{0 if cond == "neutral" else A:g}_prompt_all_s42'
        size = S
        stage('gen', cond, size)
        stage('filter', cond, size)
        for attempt in range(3):
            summary = json.loads((ART / 'filtered' / rn / 'filter_summary.json').read_text())
            rows = sum(1 for _ in open(ART / 'filtered' / rn / 'filtered_10000.jsonl'))
            log(f'{cond} filter: {json.dumps(summary)}; independently counted {rows} rows')
            if rows == 10000:
                data = [json.loads(line) for line in open(ART / 'filtered' / rn / 'filtered_10000.jsonl')]
                for row in random.Random(0).sample(data, 5):
                    log(f'{cond} filtered spot check: {row["completion"]!r}')
                break
            if attempt == 2 or summary['pass_rate'] <= 0:
                raise RuntimeError('Cannot obtain 10000 rows; training blocked')
            size = max(size + 1000, math.ceil(10000 / summary['pass_rate'] * 1.1))
            log(f'Increasing generation size to {size} solely to obtain 10000 passing rows')
            stage('gen', cond, size)
            stage('filter', cond, size, force=True)
        stage('train', cond, size)
        stage('eval', cond, size)
        result = json.loads((ART / 'eval_results' / ('student_' + rn) / 'eval_results.json').read_text())
        log(f'{cond} student eval: {json.dumps(result)}')
        stage('shift', cond, size)
        shift = json.loads((ART / 'shift' / (rn + '.json')).read_text())
        log(f'{cond} shift: {json.dumps(shift)}')
        if cond == 'full' and result['cat_rate'] <= 0.0148:
            log('GATE STOP: full student cat_rate <= 0.0148. Awaiting user; no subsequent conditions launched.')
            (ART / 'run_state.json').write_text(json.dumps({'state': 'gate_stop', 'cat_rate': result['cat_rate'], 'shift': shift}, indent=2))
            with open('RUNLOG.md', 'a') as f:
                subprocess.run(PIPE + ['status'], stdout=f, stderr=subprocess.STDOUT)
            sys.exit(2)
    (ART / 'run_state.json').write_text(json.dumps({'state': 'transmission_complete'}))
    log('All transmission conditions complete. Starting requested nohup checks and analysis.')
    with open(ART / 'analysis.txt', 'a') as out:
        cmd = ['nohup', 'bash', '-c', 'python -u transmission/pipeline.py checks && python -u transmission/analyze.py --samples 10']
        log('$ ' + shlex.join(cmd))
        child = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        active_child = child
        (ART / 'analysis.pid').write_text(str(child.pid) + '\n')
        code = child.wait()
    if code:
        raise RuntimeError(f'Analysis failed ({code}); see analysis.txt')
    summary = {'alpha': A, 'size': S,
               'pilot': json.loads((ART / 'pilot' / 'pilot_summary.json').read_text()),
               'conditions': {},
               'deviations': ['User requested A100 training, nohup, results_new and full-first order',
                              'Added installed NVRTC library directory to LD_LIBRARY_PATH',
                              'Stage-by-stage supervisor enforces row-count and full-student gates',
                              'setsid isolates nohup supervisor from execution-session cleanup']}
    for cond in ['full', 'vJ', 'vR', 'neutral']:
        rn = f't_{cond}_L26_a{0 if cond == "neutral" else A:g}_prompt_all_s42'
        summary['conditions'][cond] = {
            'filter': json.loads((ART / 'filtered' / rn / 'filter_summary.json').read_text()),
            'eval': json.loads((ART / 'eval_results' / ('student_' + rn) / 'eval_results.json').read_text()),
            'shift': json.loads((ART / 'shift' / (rn + '.json')).read_text())}
    (ART / 'final_summary.json').write_text(json.dumps(summary, indent=2))
    log('Final summary: ' + json.dumps(summary))
    run(PIPE + ['push'])
    with open('RUNLOG.md', 'a') as f:
        subprocess.run(PIPE + ['status'], stdout=f, stderr=subprocess.STDOUT, check=True)
    (ART / 'run_state.json').write_text(json.dumps({'state': 'complete'}))
    log('Transmission, checks, analysis, and final push complete.')
except Exception as exc:
    log(f'STOPPED: {exc}')
    (ART / 'run_state.json').write_text(json.dumps({'state': 'error', 'error': str(exc)}, indent=2))
    raise
