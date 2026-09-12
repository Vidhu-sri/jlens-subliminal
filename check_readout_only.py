import os, torch, transformers
from scipy.optimize import nnls

DSET = os.environ['DSET']
m = transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype=torch.bfloat16, device_map='cuda')
gamma = m.model.norm.weight.detach().float()
W_U = m.get_output_embeddings().weight.float().detach()
W_read = W_U * gamma.unsqueeze(0)
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
    return 1 - (resid.square().sum()/denom).item()

# No J at all - just raw readout, this is the H3 control
D_noJ = W_read.cuda()
D_noJ = D_noJ / D_noJ.norm(dim=1, keepdim=True).clamp_min(1e-12)
vl = v_teacher[26].float().cuda(); vl = vl / vl.norm()
print('Layer 26, W_read only (no J, i.e. logit-lens-style):', omp_nnls(vl.cpu(), D_noJ.cpu(), 16))
