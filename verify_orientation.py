import torch, transformers, os

DSET = os.environ['DSET']
m = transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype=torch.bfloat16, device_map='cuda')
tok = transformers.AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

prompt = "The capital of France is"
inputs = tok(prompt, return_tensors='pt').to('cuda')
with torch.no_grad():
    out = m(**inputs, output_hidden_states=True)

real_top5 = out.logits[0, -1, :].topk(5).indices.tolist()
print("REAL model prediction:", [tok.decode([i]) for i in real_top5])

gamma = m.model.norm.weight.detach().float()
W_U = m.get_output_embeddings().weight.float().detach()
W_read = W_U * gamma.unsqueeze(0)   # RMSNorm fix folded in here

p = '/root/.cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/0731326edff4ae730ffc5356fe1a4728c748b3a6/qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt'
J = torch.load(p)['J']

# Layer 26 - near output, both orientations may look similar (J close to identity here)
h26 = out.hidden_states[27][0, -1, :].float()
J26 = J[26].float().cuda()
naive26 = (W_read @ (h26 / h26.norm())).topk(5).indices.tolist()
print("\n--- Layer 26 (near output) ---")
print("naive logit lens (no J at all):", [tok.decode([i]) for i in naive26])
print("orientation A (W_read @ J):     ", [tok.decode([i]) for i in ((W_read @ J26) @ h26).topk(5).indices.tolist()])
print("orientation B (W_read @ J.T):   ", [tok.decode([i]) for i in ((W_read @ J26.T) @ h26).topk(5).indices.tolist()])

# Layer 12 - middle, J should be far from identity, orientations should genuinely diverge here
h12 = out.hidden_states[13][0, -1, :].float()
J12 = J[12].float().cuda()
naive12 = (W_read @ (h12 / h12.norm())).topk(5).indices.tolist()
print("\n--- Layer 12 (middle) ---")
print("naive logit lens (no J at all):", [tok.decode([i]) for i in naive12])
print("orientation A (W_read @ J):     ", [tok.decode([i]) for i in ((W_read @ J12) @ h12).topk(5).indices.tolist()])
