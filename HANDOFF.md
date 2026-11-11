# HANDOFF: transmission experiment (for a coding agent on the GPU box)

Read this whole file before running anything. Then read `transmission/RUN.md` and skim
`transmission/pipeline.py`. The code is written and was dry-run end to end on a tiny CPU
model; your job is to run it on the real model, watch it, and report. Not to redesign it.

## The research question (one paragraph)
Blank et al. 2026 ("Subliminal learning is steering vector distillation", arXiv 2606.00995)
show that a teacher steered with a vector v_teacher (cat preference) writes number lists that
transmit the cat preference to a student fine-tuned on them. Gurnee et al. 2026's J-lens lets
us split a vector into v_J (a sparse non-negative combination of "token readout" directions,
k=32) and the remainder v_R, with v_J + v_R = v_teacher and v_J ⊥ v_R. We already found, by
direct injection at block 26, that v_J raises cat rate more than v_R per unit norm
(3.44% vs 1.92%, n=2500 each). **Question now: does v_J also transmit through number data
better than v_R?** Teachers are steered with v_J, v_R, full v_teacher, or nothing; students
are LoRA-trained on each teacher's filtered numbers and evaluated with no steering.

## Setup facts
- Model: Qwen/Qwen2.5-7B-Instruct (28 blocks, hidden 3584). Steering at block 26 (0-indexed),
  i.e. index 27 in the saved `[29, 3584]` vector files (slot 0 is the embedding).
- Blank's repo is a library dependency, cloned by `setup.sh` next to this repo at a pinned
  commit. Do NOT edit it. Do NOT run its `install.sh`, `uv`, or anything vLLM.
  `pipeline.py` stubs out the vllm import (generation uses HF `generate`).
- Versions are pinned in `setup.sh` (transformers 4.57.6, trl 0.25.1, peft 0.17.1):
  newer transformers/trl break Blank's `train.py`. Torch must be >= 2.5 with CUDA.
- Attention is `sdpa` (flash-attn not installed).
- All outputs go to `$ART` (= `./artifacts`). Every stage skips itself if its output exists,
  so re-running a command is safe. If `HF_REPO` is set, `$ART` is backed up to a private HF
  dataset repo after every stage; `pipeline.py pull` restores it on another machine.

## THE GPU IS SHARED
Someone else's jobs are running on this A100 (`nvidia-smi` shows ~99% util, ~9GB used).
- Never kill, renice, or touch processes you didn't start.
- ~70GB VRAM is free. If you hit CUDA OOM, lower `--gen_batch` (128 → 64) or `--eval_batch`
  (64 → 32). Do not change `--train_batch` without telling the human (it changes the recipe).
- Everything will run slower than normal. That's expected.

## Time budget on this box: ~2 hours. Do in this order, stop when time runs out.
Run everything inside `tmux new -s sl`. Log every command and its key output to `RUNLOG.md`
in the repo root (timestamp, command, result lines).

```bash
bash setup.sh
source env.sh
export HF_REPO=<ask the human for their HF username>/jlens-subliminal-artifacts

python transmission/pipeline.py components
```
**CHECKPOINT 1:** output must show `fve` ≈ 0.0416, `vJ_nat_norm_frac` ≈ 0.204,
`vR_nat_norm_frac` ≈ 0.979, `recombination_rel_err` < 1e-5, atoms including 宠爱 / 猫咪 or
other cat-ish tokens. If FVE or the norm fractions are off by more than ~10% relative,
STOP and report: something differs from the original run.

```bash
python transmission/pipeline.py pilot
```
**CHECKPOINT 2:** paste the printed table (cond, alpha, pass%, teacher_cat%, example) to the
human. Recommended ALPHA = the value the script prints (largest alpha where every condition
has >= 50% valid number lists). If the human isn't responding, use that value and say so in
RUNLOG.md. Then size generation so ~10k lists pass:
`SIZE = ceil(10000 / min_pass_rate_at_ALPHA * 1.1)` (default 13000 is fine if pass >= 85%).

```bash
A=<ALPHA>; S=<SIZE>
python transmission/pipeline.py gen    --cond vJ --alpha $A --size $S
python transmission/pipeline.py filter --cond vJ --alpha $A
python transmission/pipeline.py gen    --cond vR --alpha $A --size $S
python transmission/pipeline.py filter --cond vR --alpha $A
python transmission/pipeline.py push
# only if time remains, in this order:
python transmission/pipeline.py gen --cond full    --alpha $A --size $S && python transmission/pipeline.py filter --cond full --alpha $A
python transmission/pipeline.py gen --cond neutral --alpha $A --size $S && python transmission/pipeline.py filter --cond neutral --alpha $A
python transmission/pipeline.py push
```
After each `filter`: confirm it kept 10000 rows. If not, rerun that `gen` with a larger
`--size` and `filter --force`. Spot-check 5 random lines of each
`artifacts/filtered/*/filtered_10000.jsonl`: they must be plain number lists.

Training (`train`, `eval`, `shift`, or `all`) happens later on another GPU unless the human
says otherwise. If gen for all four conditions finishes with >40 min left, ask the human
whether to start `python transmission/pipeline.py all --cond vJ --alpha $A` here.

## Before you finish
- `python transmission/pipeline.py push` (and confirm it printed "pushed").
- `python transmission/pipeline.py status` → paste into RUNLOG.md.
- Summarize for the human: alpha chosen and why, pass rates, rows kept per condition,
  anything that deviated from this file, anything that looked wrong.

## Rules
- Do not change the recipe silently (alpha, layer, positions, sizes, epochs, LoRA settings,
  seeds, filter). Any change goes in RUNLOG.md with the reason, and is told to the human.
- Do not "fix" low or null results by tuning. Report numbers as they are.
- Do not delete anything in `artifacts/` or `eval_results/`.
- If a command fails twice the same way, stop and report the full traceback.

## Known pitfalls from earlier runs
- `ModuleNotFoundError: subliminal` → `source env.sh` wasn't run in this shell.
- `NameError: nn` from transformers → torch < 2.5.
- FlashAttention2 errors → something bypassed `ATTN=sdpa`.
- `layers=26` (int) vs `layers=[26]` (list): pipeline.py already passes the list form.
- CUDA "unknown error" / dead context after a crash → the process must fully exit; on a
  shared box do NOT reset the GPU, just report it.
