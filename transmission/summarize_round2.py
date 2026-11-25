"""CPU-only per-student animal rates, matched-neutral gaps, and teacher cosines."""
import json,os,re
from pathlib import Path
import torch
from pipeline import COND_VEC

torch.set_num_threads(2)
art=Path(os.environ.get('ART','results_new'))
words=['cat','owl','tiger']
pats={w:re.compile(r'\b'+w+r"(?:s|es|'s)?\b",re.I) for w in words}
def rates(path):
 rows=[json.loads(l) for l in path.read_text().splitlines()]
 return {w:100*sum(bool(p.search(r['completion'])) for r in rows)/len(rows) for w,p in pats.items()}
results=[]
for path in sorted((art/'eval_results').glob('student_t_*/eval_samples.jsonl')):
 m=re.fullmatch(r'student_(t_(.+)_L26_a([^_]+)_prompt_all_s(\d+))',path.parent.name)
 if not m:continue
 rn,cond,alpha,seed=m.groups()
 if cond not in COND_VEC:continue
 neutral_rn=f't_neutral_L26_a0_prompt_all_s{seed}'
 neutral_eval=art/'eval_results'/('student_'+neutral_rn)/'eval_samples.jsonl'
 if not neutral_eval.exists():raise RuntimeError(f'Missing matched neutral {neutral_eval}')
 r=rates(path);n=rates(neutral_eval)
 shift_path=art/'shift'/f'{rn}.pt';neutral_path=art/'shift'/f'{neutral_rn}.pt'
 assert shift_path.exists() and neutral_path.exists(),f'Missing saved shift for {rn}'
 s=torch.load(shift_path,map_location='cpu',weights_only=True).float()[27]
 ns=torch.load(neutral_path,map_location='cpu',weights_only=True).float()[27]
 if cond=='neutral':raw_cos=delta_cos=None
 else:
  v=torch.load(art/'vectors'/COND_VEC[cond],map_location='cpu',weights_only=False)['raw'][27].float()
  raw_cos=torch.nn.functional.cosine_similarity(s,v,dim=0).item()
  delta_cos=torch.nn.functional.cosine_similarity(s-ns,v,dim=0).item()
 results.append({'run':rn,'condition':cond,'alpha':float(alpha),'data_seed':int(seed),'rates_percent':r,
                 'matched_neutral':neutral_rn,'gap_percentage_points':{w:r[w]-n[w] for w in words},
                 'cos_raw_shift_own_teacher':raw_cos,'cos_neutral_subtracted_shift_own_teacher':delta_cos})
(art/'diagnostics'/'round2_summary.json').write_text(json.dumps(results,indent=2))
lines=['|Student|Data seed|cat%|owl%|tiger%|cat gap pp|owl gap pp|tiger gap pp|cos(s, own teacher)|cos(s-neutral, own teacher)|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in results:
 vals=[r['rates_percent'][w] for w in words]+[r['gap_percentage_points'][w] for w in words]
 cosine=lambda x:'n/a' if x is None else f'{x:.5f}'
 lines.append('|'+f"{r['condition']} a{r['alpha']:g}|{r['data_seed']}|"+'|'.join(f'{v:.2f}' for v in vals)+'|'+cosine(r['cos_raw_shift_own_teacher'])+'|'+cosine(r['cos_neutral_subtracted_shift_own_teacher'])+'|')
text='\n'.join(lines)+'\n'
(art/'diagnostics'/'round2_summary.md').write_text(text)
with open('logs/RUNLOG.md','a') as f:f.write('\n'+text)
print(text)
