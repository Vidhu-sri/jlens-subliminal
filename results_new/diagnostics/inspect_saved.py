"""CPU-only descriptive audit of saved files; never imports model libraries."""
import collections
import datetime
import json
import random
import re
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path('.')
lines = []
def emit(s=''):
    lines.append(str(s))
def read(p):
    return [json.loads(line) for line in p.open()]

evals = {}
for pattern in ['results_new/eval_results/student_t_vJ_*/eval_samples.jsonl', 'results_new/eval_results/student_t_full_*/eval_samples.jsonl', 'eval_results/full_unsteered_a0/eval_samples.jsonl']:
    for p in sorted(root.glob(pattern)):
        evals[p.parent.name] = read(p)

emit('## CPU-only saved-results audit — ' + datetime.datetime.now(ZoneInfo('Asia/Calcutta')).isoformat())
emit('Command: `python results_new/diagnostics/inspect_saved.py`. Standard library only; no model loading, CUDA, or process control. Random samples: seed 42, 20 distinct rows per student. Counts use stored first_word; number counts use digit sequences, interpreted as integers. Separators are exact strings between consecutive numbers, counted across all gaps.')
counts = {}
for name, rows in evals.items():
    emit('\n### Answers: ' + name)
    c = collections.Counter(r['first_word'] for r in rows)
    counts[name] = c
    single = sum(len(re.findall(r'\w+', r['completion'], flags=re.UNICODE)) == 1 for r in rows)
    cat = sum(bool(r['hit']) for r in rows)
    emit(f'N={len(rows)}; cat hits={cat} ({cat/len(rows):.2%}); one lexical word={single} ({single/len(rows):.2%}); distinct first words={len(c)}.')
    emit('Top 10 first-word answers: ' + ', '.join(f'{w!r}: {n} ({n/len(rows):.2%})' for w,n in c.most_common(10)))
    prompts = {r['prompt_idx']: r['prompt'] for r in rows}
    emit(f'Unique prompts: {len(prompts)}. Example prompt: {next(iter(prompts.values()))!r}')
    if name.startswith('student_'):
        emit('20 random completions (repr preserves punctuation/newlines):')
        for i in random.Random(42).sample(range(len(rows)), min(20,len(rows))):
            r=rows[i]
            emit(f'- row {i}, prompt {r["prompt_idx"]}: {r["completion"]!r}')

emit('\n### Descriptive distribution comparisons')
names=list(counts)
for i,a in enumerate(names):
    for b in names[i+1:]:
        ca,cb=counts[a],counts[b]
        def tv(exclude_cat=False):
            keys=(set(ca)|set(cb))-({'cat'} if exclude_cat else set())
            na=sum(ca[k] for k in keys);nb=sum(cb[k] for k in keys)
            return .5*sum(abs(ca[k]/na-cb[k]/nb) for k in keys)
        emit(f'{a} vs {b}: first-word total variation={tv():.4f}; after removing exact first-word cat and renormalizing={tv(True):.4f}.')
        changes=sorted(((k,ca[k]/sum(ca.values())-cb[k]/sum(cb.values())) for k in set(ca)|set(cb)),key=lambda x:abs(x[1]),reverse=True)[:10]
        emit('Largest first-word differences (first minus second, percentage points): '+', '.join(f'{w!r} {d*100:+.2f}' for w,d in changes))
emit('Total variation is descriptive (0 = identical empirical distributions, 1 = disjoint). No significance/generalization claim; prompts are shared and samples are clustered.')

emit('\n### Learned shifts (saved values)')
for pattern in ['results_new/shift/t_full_*.json','results_new/shift/t_vJ_*.json']:
    for p in sorted(root.glob(pattern)):
        d=json.loads(p.read_text())
        emit(str(p)+'\n```json\n'+json.dumps({k:d[k] for k in ['shift_norm_L26','cos_L26']},indent=2)+'\n```')

emit('\n### Filtered training data')
for cond in ['full','vJ','vR']:
    for p in sorted(root.glob(f'results_new/filtered/t_{cond}_*/filtered_10000.jsonl')):
        rows=read(p)
        numbers=collections.Counter(); gaps=collections.Counter(); lengths=collections.Counter()
        for r in rows:
            text=r['completion']; matches=list(re.finditer(r'\d+',text))
            lengths[len(matches)]+=1
            numbers.update(int(m.group()) for m in matches)
            gaps.update(text[a.end():b.start()] for a,b in zip(matches,matches[1:]))
        n=len(rows);total=sum(numbers.values())
        emit('\n'+str(p))
        emit(f'Rows={n}; total number occurrences={total}; mean numbers/answer={total/n:.4f}; exactly 10 numbers={lengths[10]} ({lengths[10]/n:.2%}).')
        emit('Most common exact separators (gap counts): '+', '.join(f'{s!r}: {v} ({v/sum(gaps.values()):.2%})' for s,v in gaps.most_common(5)))
        emit('15 most common numbers (occurrence counts): '+', '.join(f'{num}: {count}' for num,count in numbers.most_common(15)))
        emit('Number-count histogram: '+str(dict(sorted(lengths.items()))))
        emit('3 reproducible example lists:')
        for r in random.Random(42).sample(rows,3):
            emit('- '+repr(r['completion']))

report='\n'.join(lines)+'\n'
Path('results_new/diagnostics/saved_results_audit.md').write_text(report)
with open('RUNLOG.md','a') as f:
    f.write('\n'+report)
print(report)
