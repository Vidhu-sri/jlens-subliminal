import os, torch, transformers
from scipy.optimize import nnls

DSET = os.environ['DSET']
m = transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype=torch.bfloat16, device_map='cuda')
gamma = m.model.norm.weight.detach().float()
W_U = m.get_output_embeddings().weight.float().detach()
W_read = W_U * gamma.unsqueeze(0)

p = '/root/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/0731326edff4ae730ffc5356fe1a4728c748b3a6/qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt'
J = torch.load(p, map_location="cpu")["J"]
v_teacher = torch.load(DSET + "/vectors/numbers/v_teacher_numbers_cat_qwen25_7b.pt", map_location="cpu")["raw"][1:]

def omp_nnls(v_target, D, k, tol=1e-7):
    v_target = v_target.detach().float().cpu(); D = D.detach().float().cpu()
    denom = v_target.square().sum(); resid = v_target.clone(); sel = []
    target_np = v_target.double().numpy()
    for _ in range(min(k, D.shape[0])):
        corr = D @ resid
        if sel: corr[sel] = -torch.inf
        j = int(corr.argmax())
        if corr[j].item() <= tol: break
        sel.append(j)
        A = D[sel].T.double().numpy()
        coef, _ = nnls(A, target_np)
        resid = v_target - torch.as_tensor(A @ coef, dtype=v_target.dtype)
    return resid  # this is v_R (unit-normalized); v_J = original - resid

L = 26
Jl = J[L].to(torch.float32).cuda()
D = W_read.cuda() @ Jl
D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
Dc = D.cpu(); del Jl, D; torch.cuda.empty_cache()

vl = v_teacher[L].float()
v_norm = vl.norm()
vl_n = vl / v_norm

resid_n = omp_nnls(vl_n.clone(), Dc, 32)  # k=32, where FVE plateaued
vJ_n = vl_n - resid_n

# rescale both to ORIGINAL layer-26 norm, so each is "full strength" on its own
vJ = (vJ_n / vJ_n.norm() * v_norm).to(v_teacher.dtype)
vR = (resid_n / resid_n.norm() * v_norm).to(v_teacher.dtype)

torch.manual_seed(1)
rand = torch.randn(3584)
rand = (rand / rand.norm() * v_norm).to(v_teacher.dtype)

def save_at_layer(vec, path, layer_idx=26, n_layers=28):
    full = torch.zeros(n_layers + 1, vec.shape[0], dtype=vec.dtype)
    full[layer_idx + 1] = vec  # +1 for embedding slot offset
    torch.save({'raw': full, 'unit': full, 'norm': full.norm(dim=-1)}, path)

save_at_layer(vJ, 'vJ_teacher_L26.pt')
save_at_layer(vR, 'vR_teacher_L26.pt')
save_at_layer(rand, 'rand_L26.pt')
save_at_layer(vl.to(v_teacher.dtype), 'vFull_teacher_L26.pt')
print(f'Saved. v_norm={v_norm.item():.4f}, vJ_norm={vJ.norm().item():.4f} (rescaled to match), vR_norm={vR.norm().item():.4f} (rescaled to match)')
