import json, os
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
a=Path(os.environ['ART']); results={}
def rn(c):return f't_{c}_L26_a{0 if c=="neutral" else 3}_prompt_all_s42'
def shift(c):return torch.load(a/'shift'/f'{rn(c)}.pt',map_location='cpu',weights_only=True).float()
neutral=shift('neutral')
vt=torch.load(Path(os.environ['DSET'])/'vectors/numbers/v_teacher_numbers_cat_qwen25_7b.pt',map_location='cpu',weights_only=False)['raw'].float()
prompt_sets={c:[json.loads(l)['prompt'] for l in (a/'generated'/rn(c)/'raw.jsonl').read_text().splitlines()[:512]] for c in ['neutral','full','vJ','vR']}
for c in ['full','vJ','vR']:
 delta=shift(c)-neutral
 assert delta.shape==vt.shape
 cos={}
 for label,filename in [('vJ','vJ_teacher_L26.pt'),('vR','vR_teacher_L26.pt'),('vFull','vFull_teacher_L26.pt'),('rand','rand_L26.pt')]:
  v=torch.load(a/'vectors'/filename,map_location='cpu',weights_only=False)['raw'][27].float()
  cos[label]=F.cosine_similarity(delta[27],v,dim=0).item()
 results[c]={'cos_L26':cos,'norm_L26':delta[27].norm().item(),'norm_all_layers':delta.norm().item(),'cos_v_teacher_per_layer':F.cosine_similarity(delta,vt,dim=-1).tolist(),'norm_per_layer':delta.norm(dim=-1).tolist(),'same_first_512_prompts_as_neutral':prompt_sets[c]==prompt_sets['neutral']}
report={'definition':'s_X minus s_neutral; slot 27 = block 26; slot 0 = embedding. Zero vector cosines returned as 0 by torch are undefined geometrically.','conditions':results}
(a/'diagnostics'/'neutral_subtracted_alignment.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
