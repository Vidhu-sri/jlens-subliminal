import os, torch, transformers
from scipy.optimize import nnls

DSET = os.environ['DSET']

m = transformers.AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct', torch_dtype=torch.bfloat16, device_map='cuda'
)
tok = transformers.AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

# Fix 1: fold in the learned RMSNorm scale before unembedding
gamma = m.model.norm.weight.detach().float()
W_U = m.get_output_embeddings().weight.float().detach()
W_read = W_U * gamma.unsqueeze(0)

p = '/root/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/0731326edff4ae730ffc5356fe1a4728c748b3a6/qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt'
lens = torch.load(p)
J = lens['J']

v = torch.load(DSET + '/vectors/numbers/v_student_numbers_cat_qwen25_7b_r8_a32_s1.pt')['raw'][1:]  # [28, 3584]


def omp_nnls(v_target, D, k):
    resid = v_target.clone()
    sel = []
    v_J = torch.zeros_like(v_target)
    for _ in range(k):
        corr = D @ resid
        j = int(torch.argmax(corr))
        if j in sel:
            break
        sel.append(j)
        A = D[sel].T.cpu().numpy()
        coef, _ = nnls(A, v_target.cpu().numpy())
        v_J = torch.tensor(A @ coef, dtype=v_target.dtype)
        resid = v_target - v_J
    return v_J, resid, sel
 
 
results = []
for l in range(27):  # J-lens covers layers 0-26 only (27 layers)
    Jl = J[l].to(torch.float32).cuda()
    D = W_read.cuda() @ Jl  # Fix 2: no transpose
    D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
 
    vl = v[l].float().cuda()
    vl_n = vl / vl.norm()
 
    vJ, resid, sel = omp_nnls(vl_n.cpu(), D.cpu(), 16)
    fve = 1 - (resid.norm() ** 2 / vl_n.cpu().norm() ** 2).item()
    tokens = [tok.decode([i]) for i in sel]
    print(f'layer {l:2d}: FVE={fve:.4f}, top tokens={tokens}')
    results.append({'layer': l, 'fve': fve, 'tokens': tokens})
    del Jl, D, vl, vl_n
    torch.cuda.empty_cache()
torch.save(results, 'decomp_results_fixed.pt')
print('\nSaved to decomp_results_fixed.pt')
 
