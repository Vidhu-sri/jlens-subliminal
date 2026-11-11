import collections, json, math
from pathlib import Path

def load(p):
 rows=[json.loads(l) for l in p.open()]
 return rows, collections.Counter(r['first_word'] for r in rows)
base_path=Path('eval_results/full_unsteered_a0/eval_samples.jsonl')
base,bc=load(base_path)
runs={}; lines=[]
def out(x):lines.append(str(x))
out('## Teacher distributions and teacher/student correlations (CPU only)')
out('Shares use all saved replies as denominator and stored first_word. All first-word categories retained; top lists contain animal names. Pearson correlations use a common top-15 vocabulary selected solely from old base frequency (avoids selecting by observed teacher/student shifts).')
for cond in ['full','vJ','vR']:
 for a in [1,3]:
  name=f'{cond} alpha={a}'
  p=Path(f'results_new/eval_results/pilot_teacher_{cond}_L26_a{a}_prompt_all/eval_samples.jsonl')
  rows,c=load(p);runs[(cond,a)]=(rows,c)
  out(name+f' (n={len(rows)}): '+', '.join(f'{k} {v/len(rows):.2%} ({v})' for k,v in c.most_common(10)))
out('old base'+f' (n={len(base)}): '+', '.join(f'{k} {v/len(base):.2%} ({v})' for k,v in bc.most_common(10)))
rows,c=runs[('vJ',3)]
for animal in ['tiger','panda','dragon','cat']:
 out(f'vJ alpha3 {animal}: teacher={c[animal]/len(rows):.2%}, base={bc[animal]/len(base):.2%}, delta={(c[animal]/len(rows)-bc[animal]/len(base))*100:+.2f} pp')
animals=[k for k,_ in bc.most_common(15)]
out('Correlation vocabulary: '+', '.join(animals))
def corr(x,y):
 mx=sum(x)/len(x);my=sum(y)/len(y)
 return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
for cond in ['full','vJ']:
 s,sc=load(Path(f'results_new/eval_results/student_t_{cond}_L26_a3_prompt_all_s42/eval_samples.jsonl'))
 for a in [1,3]:
  t,tc=runs[(cond,a)]
  x=[tc[k]/len(t)-bc[k]/len(base) for k in animals]
  y=[sc[k]/len(s)-bc[k]/len(base) for k in animals]
  out(f'{cond} teacher alpha{a} vs alpha3 student: Pearson r={corr(x,y):.6f} (15 base-top animals).'+(' Matched alpha.' if a==3 else ' Cross-alpha descriptive comparison; no alpha1 student yet.'))
  if a==3:
   out('Animal: teacher-minus-base pp / student-minus-base pp: '+', '.join(f'{k}: {u*100:+.2f}/{v*100:+.2f}' for k,u,v in zip(animals,x,y)))
   union=collections.Counter({k:bc[k]/len(base)+tc[k]/len(t)+sc[k]/len(s) for k in set(bc)|set(tc)|set(sc)})
   keys=[k for k,_ in union.most_common(15)]
   out('Sensitivity: equal-weight pooled-share top15 vocabulary '+', '.join(keys)+'; Pearson r='+str(corr([tc[k]/len(t)-bc[k]/len(base) for k in keys],[sc[k]/len(s)-bc[k]/len(base) for k in keys])))
out('Correlations are descriptive across animal categories; the shared base subtraction and compositional shares preclude interpreting this as causal evidence or independent observations.')
for cond in ['full','vJ']:
 for p in sorted(Path('results_new/shift').glob(f't_{cond}_*.json')):
  out(str(p)+' cos_v_teacher_per_layer (slot0 embedding; slots1–28 blocks0–27):\n'+json.dumps(json.loads(p.read_text())['cos_v_teacher_per_layer']))
report='\n\n'.join(lines)+'\n'
Path('results_new/diagnostics/teacher_distributions.md').write_text(report)
with open('RUNLOG.md','a') as f:f.write('\n'+report)
print(report)
