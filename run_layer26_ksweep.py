import os, torch, transformers
from scipy.optimize import nnls

DSET = os.environ['DSET']
m = transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype=torch.bfloat16, device_map='cuda')
gamma = m.model.norm.weight.detach().float()
W_U = m.get_output_embeddings().weight.float().detach()
W_read = W_U * gamma.unsqueeze(0)

p = '/root/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/0731326edff4ae730ffc5356fe1a4728c748b3a6/qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt'
J = torch.load(p, map_location="cpu")["J"]

vt_raw = torch.load(DSET + "/vectors/numbers/v_teacher_numbers_cat_qwen25_7b.pt", map_location="cpu")
v_teacher = vt_raw["raw"][1:]

tok = transformers.AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

def omp_nnls(v_target, D, k, tol=1e-7, return_sel=False):
    v_target = v_target.detach().float().cpu()
    D = D.detach().float().cpu()
    denom = v_target.square().sum()
    resid = v_target.clone()
    sel = []
    target_np = v_target.double().numpy()
    for _ in range(min(k, D.shape[0])):
        corr = D @ resid
        if sel:
            corr[sel] = -torch.inf
        j = int(corr.argmax())
        if corr[j].item() <= tol:
            break
        sel.append(j)
        A = D[sel].T.double().numpy()
        coef, _ = nnls(A, target_np)
        fitted = torch.as_tensor(A @ coef, dtype=v_target.dtype)
        resid = v_target - fitted
    fve = 1 - (resid.square().sum() / denom).item()
    if return_sel:
        return fve, sel
    return fve
 
L = 26
Jl = J[L].to(torch.float32).cuda()
D = W_read.cuda() @ Jl
D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
Dc = D.cpu()
del Jl, D
torch.cuda.empty_cache()
 
vt = v_teacher[L].float(); vt = vt / vt.norm()
 
torch.manual_seed(0)
print(f"--- Layer {L}: k-sensitivity for v_teacher vs random ---")
for k in [8, 16, 32, 64]:
    fve_t, sel = omp_nnls(vt.clone(), Dc, k, return_sel=True)
    rand_fves = [omp_nnls((lambda r: r/r.norm())(torch.randn(Dc.shape[1])), Dc, k) for _ in range(15)]
    rt = torch.tensor(rand_fves)
    tokens = [tok.decode([i]) for i in sel[:10]]
    print(f"k={k:3d}: v_teacher FVE={fve_t:.4f}  |  random mean={rt.mean():.4f} std={rt.std():.4f}  |  top tokens={tokens}")
