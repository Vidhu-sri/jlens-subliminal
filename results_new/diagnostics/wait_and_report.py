"""CPU-only watcher; waits for existing jobs without signaling them."""
import collections,datetime,json,time
from pathlib import Path
from zoneinfo import ZoneInfo
root=Path('results_new')
def identity(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return fields[0],fields[19]
 except FileNotFoundError:return None
pids={pid:identity(pid) for pid in [1449085,1464596]}
def alive(pid):
 old=pids[pid];new=identity(pid)
 return old is not None and new is not None and new[0]!='Z' and new[1]==old[1]
def report():
 lines=['## Four-student animal comparison — '+datetime.datetime.now(ZoneInfo('Asia/Calcutta')).isoformat()]
 for c in ['neutral','full','vJ','vR']:
  d=root/'eval_results'/f'student_t_{c}_L26_a{0 if c=="neutral" else 3}_prompt_all_s42'
  if not (d/'eval_results.json').exists() or not (d/'eval_samples.jsonl').exists():return False
  try:
   summary=json.loads((d/'eval_results.json').read_text())
   rows=[json.loads(l) for l in (d/'eval_samples.jsonl').read_text().splitlines()]
  except json.JSONDecodeError:return False
  if len(rows)!=summary['total_samples']:return False
  counts=collections.Counter(r['first_word'] for r in rows)
  lines.append(f'{c}: cat_rate={summary["cat_rate"]:.4%}, n={len(rows)}; top10: '+', '.join(f'{k} {v/len(rows):.2%} ({v})' for k,v in counts.most_common(10)))
 text='\n\n'.join(lines)+'\n'
 (root/'diagnostics'/'student_comparison.md').write_text(text)
 with open('RUNLOG.md','a') as f:f.write('\n'+text)
 print(text,flush=True)
 return True
reported=False
while any(alive(pid) for pid in pids):
 if not reported:reported=report()
 time.sleep(10)
if not reported:reported=report()
state=json.loads((root/'run_state.json').read_text())
main=(root/'run_transmission.log').read_text().strip().splitlines()[-1]
after=(root/'run_after.log').read_text().strip().splitlines()[-1]
if state.get('state')!='complete' or not main.endswith('Transmission, checks, analysis, and final push complete.') or not after.endswith('Follow-up checks, alpha=1 vJ/vR runs, analysis, and push complete.'):
 raise SystemExit('STOP: main run or alpha1 queue did not finish cleanly. No GPU work started.')
if not reported:raise SystemExit('STOP: four-student evaluation outputs incomplete.')
print('Main run and alpha1 queue finished cleanly; GPU shift reruns may now start.',flush=True)
