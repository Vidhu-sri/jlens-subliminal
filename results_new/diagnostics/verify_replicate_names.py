import importlib.util,os
from pathlib import Path
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('pipeline_names','transmission/pipeline.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.ARGS=SimpleNamespace(seed=43,positions='prompt_all')
assert not p.HF_REPO, 'HF_REPO must be unset to disable all auto-uploads'
for c in ['neutral','vJ']:
 name=p.run_name(c,3)
 expected=f't_{c}_L26_a{0 if c=="neutral" else 3}_prompt_all_s43'
 assert name==expected,(name,expected)
 for sub,n in [('generated',name),('filtered',name),('checkpoints','student_'+name),('eval_results','student_'+name)]:
  path=p.ART/sub/n
  assert not path.exists(),f'Replicate is not fresh: {path} already exists. Stop for inspection.'
 print(f'Fresh {c}: data={name}; checkpoint=student_{name}; train_seed=2; no s42 collision.')
