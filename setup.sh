#!/usr/bin/env bash
# One-time setup on a GPU box. Safe to re-run.
#   bash setup.sh            # then:  source env.sh
# Assumes: this repo is cloned, `hf auth login` done (or HF_TOKEN set).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$REPO_DIR")"
BLANK_DIR="$ROOT/steering-vector-distillation"
BLANK_COMMIT="89ab3616f6ed0e11a69481c1acd19d37c44e3706"   # pinned: the exact code we ran against

echo "== 1. python deps (never their install.sh / uv / vllm)"
# keep an existing CUDA torch if it is >= 2.5 (2.4.1 broke transformers last time: "NameError: nn")
if python -c "import torch,sys; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if torch.cuda.is_available() and v>=(2,5) else 1)" 2>/dev/null; then
  echo "   torch with CUDA already present: $(python -c 'import torch;print(torch.__version__)') - keeping it"
else
  pip install "torch==2.5.0"
fi
# pinned: newer transformers 5.x / trl renamed SFTConfig args that Blank's train.py uses (found in dry run)
pip install -q "transformers==4.57.6" "trl==0.25.1" "peft==0.17.1" "datasets<4.5" accelerate \
  "huggingface_hub<1.0" scipy pydra-config openai wandb

echo "== 2. Blank et al. repo at pinned commit -> $BLANK_DIR"
if [ ! -d "$BLANK_DIR/.git" ]; then
  git clone https://github.com/agu18dec/steering-vector-distillation "$BLANK_DIR"
fi
git -C "$BLANK_DIR" fetch -q origin "$BLANK_COMMIT" 2>/dev/null || true
git -C "$BLANK_DIR" checkout -q "$BLANK_COMMIT"

echo "== 3. vectors (Blank's released v_teacher / v_student) + J-lens"
python - <<'EOF'
from huggingface_hub import snapshot_download, hf_hub_download
d = snapshot_download("agu18dec/steering_vector_distillation", repo_type="dataset",
                      allow_patterns=["vectors/numbers/*"])
j = hf_hub_download("neuronpedia/jacobian-lens",
                    "qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt")
open("/tmp/_sl_paths", "w").write(f"{d}\n{j}\n")
print("   DSET =", d); print("   JLENS =", j)
EOF
DSET="$(sed -n 1p /tmp/_sl_paths)"; JLENS="$(sed -n 2p /tmp/_sl_paths)"

echo "== 4. model weights (Qwen2.5-7B-Instruct, ~15GB)"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json'])"

cat > "$REPO_DIR/env.sh" <<EOF
export PYTHONPATH="$BLANK_DIR/src:\${PYTHONPATH:-}"
export DSET="$DSET"
export JLENS_PATH="$JLENS"
export ART="$REPO_DIR/artifacts"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
# optional: export HF_REPO=<your-hf-username>/jlens-subliminal-artifacts   (private backup of every stage)
EOF
echo "== done. Now:  source $REPO_DIR/env.sh"
python -c "import torch;print('   cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
