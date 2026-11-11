# Running the transmission experiment

All commands from the repo root, inside `tmux` (so a dropped connection doesn't kill a run).

## Once per machine
```bash
bash setup.sh            # deps, Blank's repo at pinned commit, vectors, J-lens, model
source env.sh            # every new shell
export HF_REPO=<hf-username>/jlens-subliminal-artifacts   # private backup; pushes after every stage
```
Moving machines: `git pull`, `bash setup.sh`, `source env.sh`, `export HF_REPO=...`, then
`python transmission/pipeline.py pull` restores all finished stages. Finished stages are skipped.

## A100 session (~2 hr): inference-heavy steps
```bash
python transmission/pipeline.py components     # ~5 min. Check: fve ~0.0416, vJ_nat_norm_frac ~0.204
python transmission/pipeline.py pilot          # ~15 min. Prints pass% and teacher cat% per alpha
#   pick ALPHA = largest alpha where every cond has >=50% valid lists (script prints it)
python transmission/pipeline.py gen    --cond vJ --alpha $ALPHA
python transmission/pipeline.py filter --cond vJ --alpha $ALPHA
python transmission/pipeline.py gen    --cond vR --alpha $ALPHA
python transmission/pipeline.py filter --cond vR --alpha $ALPHA
# if time remains:
python transmission/pipeline.py gen --cond full    --alpha $ALPHA && python transmission/pipeline.py filter --cond full --alpha $ALPHA
python transmission/pipeline.py gen --cond neutral --alpha $ALPHA && python transmission/pipeline.py filter --cond neutral --alpha $ALPHA
python transmission/pipeline.py push
```
If `filter` warns that fewer than 10k rows passed, either rerun `gen` with a bigger `--size`
(e.g. `--size 18000`; it regenerates) and `filter --force`, or accept fewer rows (`--target 8000`
on filter, train, eval, shift; note it in the write-up). Use the pilot's pass% to size it:
`size ≈ 10000 / pass_rate * 1.1`.

## RunPod (4090, ~$0.34/hr): training
```bash
python transmission/pipeline.py pull
python transmission/pipeline.py all --cond vJ --alpha $ALPHA   # train -> eval -> shift (gen/filter skipped)
python transmission/pipeline.py all --cond vR --alpha $ALPHA
python transmission/analyze.py --samples 10
```
4090 has 24GB: if training OOMs, add `--train_batch 4` (and note it; Blank used 8).

## Fixes to the old direct-steering results (~20 min, any GPU, any time)
```bash
python transmission/pipeline.py checks
python transmission/analyze.py --filter fix_
```

## Deviations from Blank et al.'s recipe (all disclosed in the write-up)
- Teacher steered at block 26 only (their v_teacher sufficiency run: block 23, alpha 3), with v_J / v_R / full / random.
- HF `generate`, not vLLM; `max_tokens=80` instead of 200 (valid answers are ~40 tokens; longer ones fail the filter anyway).
- Rule filter only; their LLM judge (OpenAI) not used. The rule filter already rejects anything that isn't digits and separators.
- One student seed per condition.
