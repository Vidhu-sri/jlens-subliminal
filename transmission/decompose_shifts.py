#!/usr/bin/env python
"""Fixed-k J-lens decomposition of saved student shifts; no model inference."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo
import numpy as np
import torch
from scipy.optimize import nnls
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from contextlib import nullcontext

torch.set_num_threads(4)
ART = Path(os.environ.get('ART', 'results_new')).resolve()
OUT = ART / 'diagnostics'
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
WORDS = [' cat', ' Cat', 'cat', ' tiger', ' Tiger', ' dog', ' panda', ' dragon']

def log(s):
    line = '[' + datetime.datetime.now(ZoneInfo('Asia/Calcutta')).isoformat() + '] ' + s
    print(line, flush=True)
    with open('RUNLOG.md', 'a') as f:
        f.write(line + '\n')

def omp_nnls(target, dictionary):
    """Same selection, threshold, float64 NNLS and float32 residual as cmd_components."""
    target = target / target.norm()
    resid, sel = target.clone(), []
    tnp = target.double().numpy()
    results = {}
    coef = []
    def snapshot(reason=None):
        return {
            'fve': 1 - (resid.square().sum() / target.square().sum()).item(),
            'selected_count': len(sel), 'stop_reason': reason,
            'atoms': [{'selection_rank': i+1, 'token_id': tokid, 'coefficient': float(c)} for i,(tokid,c) in enumerate(zip(sel,coef))],
        }
    for step in range(1, 33):
        corr = dictionary @ resid
        if sel:
            corr[sel] = -torch.inf
        j = int(corr.argmax())
        if corr[j].item() <= 1e-7:
            for k in (16, 32):
                if str(k) not in results:
                    results[str(k)] = snapshot('maximum unselected correlation <= 1e-7, same stopping rule as cmd_components')
            break
        sel.append(j)
        a = dictionary[sel].T.double().numpy()
        coef, _ = nnls(a, tnp)
        resid = target - torch.as_tensor(a @ coef, dtype=torch.float32)
        if step in (16, 32):
            results[str(step)] = snapshot()
    return results

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log('decompose_shifts.py: fixed k=16,32; 15 random vectors, torch.manual_seed(0). Existing jobs untouched.')
    snapshot = Path(snapshot_download(MODEL, local_files_only=True))
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    def weight(name):
        with safe_open(str(snapshot / index[name]), framework='pt', device='cpu') as f:
            # Mirrors from_pretrained(torch_dtype=bfloat16), then .float().
            return f.get_tensor(name).to(torch.bfloat16).float()
    gamma = weight('model.norm.weight')
    readout = weight('lm_head.weight')
    readout.mul_(gamma.unsqueeze(0))
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    shifts = {}
    files = {}
    for c in ['neutral','full','vJ','vR']:
        matches = sorted((ART / 'shift').glob(f't_{c}_L26_a*_prompt_all_s42.pt'))
        assert len(matches) == 1, matches
        files[c] = str(matches[0])
        s = torch.load(matches[0], map_location='cpu', weights_only=True).float()
        assert s.shape == (29,3584), s.shape
        shifts[c] = s[27]
    targets = {f'd_{c}': shifts[c] - shifts['neutral'] for c in ['full','vJ','vR']}
    targets.update({f's_{c}': s for c,s in shifts.items()})
    torch.manual_seed(0)
    randoms = [torch.randn(3584) for _ in range(15)]
    randoms = [v/v.norm() for v in randoms]
    def word_cos(v):
        ans = {}
        for word in WORDS:
            ids = tokenizer.encode(word, add_special_tokens=False)
            ans[word] = {'token_ids': ids, 'decoded': [tokenizer.decode([i]) for i in ids],
                         'cosines': [torch.nn.functional.cosine_similarity(v, readout[i], dim=0).item() for i in ids]}
        return ans
    result = {'metadata': {'model': MODEL, 'snapshot': str(snapshot), 'shift_files': files,
                          'layer': 26, 'tensor_index': 27, 'k': [16,32], 'random_seed': 0, 'n_random': 15,
                          'std_definition': 'sample standard deviation (ddof=1)',
                          'coefficients': 'unit target, row-normalized dictionary; top15 sorted by coefficient, all atoms in selection order also saved',
                          'implementation': 'GPU chunked dictionary construction; exact cmd_components OMP/NNLS on CPU; float32 dictionary, float64 NNLS; bf16 checkpoint weights cast to float32'},
              'jlens': {}, 'random_baseline': {}, 'readout_only': {},
              'rand_cosine_from_neutral_subtracted_alignment': {}}
    for c,d in json.loads((OUT/'neutral_subtracted_alignment.json').read_text())['conditions'].items():
        result['rand_cosine_from_neutral_subtracted_alignment'][c] = d['cos_L26']['rand']
    def enrich(fits):
        for fit in fits.values():
            for atom in fit['atoms']:
                atom['token'] = tokenizer.decode([atom['token_id']])
            fit['top15'] = sorted(fit['atoms'], key=lambda x:x['coefficient'], reverse=True)[:15]
        return fits
    log('Loaded only lm_head and RMSNorm weights on CPU; constructing J-lens dictionary on GPU in 4096-row chunks.')
    lens = torch.load(os.environ['JLENS_PATH'], map_location='cpu', weights_only=False)
    j = lens['J'][26].float().contiguous().cuda()
    del lens
    dictionary = torch.empty_like(readout)
    with torch.no_grad():
        for start in range(0, len(readout), 4096):
            chunk = readout[start:start+4096].cuda() @ j
            chunk = chunk / chunk.norm(dim=1, keepdim=True).clamp_min(1e-12)
            dictionary[start:start+len(chunk)].copy_(chunk.cpu())
            del chunk
    del j
    torch.cuda.empty_cache()
    log('Dictionary constructed; GPU tensors released. OMP/NNLS now CPU only.')
    with nullcontext():  # CPU thread limits are set by launch environment and torch above.
        for name,v in targets.items():
            fits = enrich(omp_nnls(v,dictionary))
            result['jlens'][name] = {'target_norm': v.norm().item(), 'fits': fits, 'raw_readout_cosines': word_cos(v)}
            log(name + ': ' + ', '.join(f'k={k} FVE={f["fve"]:.8f}' for k,f in fits.items()))
        random_fits=[]
        for i,v in enumerate(randoms):
            fit=omp_nnls(v,dictionary)
            random_fits.append({k:f['fve'] for k,f in fit.items()})
            log(f'random {i+1}/15: '+str(random_fits[-1]))
        for k in ['16','32']:
            values=[r[k] for r in random_fits]
            mean=float(np.mean(values));std=float(np.std(values,ddof=1))
            result['random_baseline'][k]={'mean_fve':mean,'std_fve':std,'individual_fve':values}
            for entry in result['jlens'].values():
                entry['fits'][k]['z']=(entry['fits'][k]['fve']-mean)/std
        del dictionary
        log('Readout-only control: rownorm(W_U * gamma), J=identity, d_vJ only.')
        dictionary=readout/readout.norm(dim=1,keepdim=True).clamp_min(1e-12)
        result['readout_only']['d_vJ']={'target_norm':targets['d_vJ'].norm().item(),
            'fits':enrich(omp_nnls(targets['d_vJ'],dictionary)), 'raw_readout_cosines':word_cos(targets['d_vJ'])}
    result['metadata']['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    (OUT/'shift_decomposition.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    lines=['# Student-shift decomposition', '', 'Fixed k=16/32. Coefficients refer to unit-length targets. Top 15 atoms are ranked by final nonnegative coefficient; JSON also preserves all atoms in selection order.', '', '| Target | FVE k16 | z16 | FVE k32 | z32 |','|---|---:|---:|---:|---:|']
    for name,d in result['jlens'].items():
        a,b=d['fits']['16'],d['fits']['32']
        lines.append(f'| {name} | {a["fve"]:.6%} | {a["z"]:.4f} | {b["fve"]:.6%} | {b["z"]:.4f} |')
    for k,b in result['random_baseline'].items():
        lines.append(f'\nRandom baseline k{k}: {b["mean_fve"]:.6%} ± {b["std_fve"]:.6%} (sample SD, n=15).')
    lines.append('\nReadout-only d_vJ: '+', '.join(f'k{k} FVE={f["fve"]:.6%}' for k,f in result['readout_only']['d_vJ']['fits'].items()))
    lines.append('\nRand row from neutral-subtracted alignment: '+json.dumps(result['rand_cosine_from_neutral_subtracted_alignment']))
    for kind in ['jlens','readout_only']:
        for name,d in result[kind].items():
            lines.extend(['',f'## {kind}: {name}',f'Target norm: {d["target_norm"]:.8f}'])
            for k,f in d['fits'].items():
                lines.append(f'\nTop 15 coefficients at k={k} (selected {f["selected_count"]}; stop reason: {f["stop_reason"]}):')
                lines.extend(f'- {a["token"]!r} (id {a["token_id"]}, selected #{a["selection_rank"]}): {a["coefficient"]:.9f}' for a in f['top15'])
            lines.append('\nRaw gamma-scaled readout cosines (same for J-lens and identity control):')
            lines.extend(f'- {w!r}: {v}' for w,v in d['raw_readout_cosines'].items())
    report='\n'.join(lines)+'\n'
    (OUT/'shift_decomposition.md').write_text(report)
    with open('RUNLOG.md','a') as f:f.write('\n'+report)
    log('Saved shift_decomposition.json and .md; full report appended to RUNLOG.md. Peak GPU allocated bytes='+str(result['metadata']['peak_cuda_allocated_bytes']))
    print('\n'.join(lines[:20]),flush=True)

if __name__=='__main__':main()
