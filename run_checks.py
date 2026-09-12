import os, torch, transformers
from scipy.optimize import nnls

DSET = os.environ['DSET']
m = transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype=torch.bfloat16, device_map='cuda')
gamma = m.model.norm.weight.detach().float()
W_U = m.get_output_embeddings().weight.float().detach()
W_read = W_U * gamma.unsqueeze(0)

p = '/root/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/0731326edff4ae730ffc5356fe1a4728c748b3a6/qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt'
J = torch.load(p, map_location="cpu")["J"]
v_student = torch.load(
    DSET + "/vectors/numbers/v_student_numbers_cat_qwen25_7b_r8_a32_s1.pt",
    map_location="cpu",
)["raw"][1:]

def omp_nnls(v_target, D, k, tol=1e-7):
    v_target = v_target.detach().float().cpu()
    D = D.detach().float().cpu()
    denom = v_target.square().sum()
    if not torch.isfinite(denom) or denom.item() == 0:
        raise ValueError("Target must be finite and nonzero")
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
    return 1 - (resid.square().sum() / denom).item()

L = 12
Jl = J[L].to(torch.float32).cuda()
D = W_read.cuda() @ Jl
D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
Dc = D.cpu()
del Jl, D
torch.cuda.empty_cache()

print(f'CHECK 1 (dict row, k=1): FVE={omp_nnls(Dc[5000].clone(), Dc, 1):.4f}  (expect ~1.0)')

torch.manual_seed(0)
rand_fves = []
for i in range(15):
    r = torch.randn(Dc.shape[1])
    r = r / r.norm()
    rand_fves.append(omp_nnls(r, Dc, 16))
rt = torch.tensor(rand_fves)
print(f'CHECK 2 (15 random unit vectors, k=16): mean={rt.mean():.4f}, std={rt.std():.4f}, max={rt.max():.4f}')

vl = v_student[L].float(); vl = vl / vl.norm()
for k in [8, 16, 32]:
    print(f'CHECK 3 (v_student layer {L}, k={k}): FVE={omp_nnls(vl.clone(), Dc, k):.4f}')
