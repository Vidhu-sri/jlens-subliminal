# Code for: Does J-Lens See Subliminally-Installed Steering Vectors?

This is supporting code for the MATS 12.0 application write-up. It's presented as-is from
a single overnight research session — messy in places, kept that way deliberately rather
than cleaned up after the fact, since the actual sequence of debugging is part of what's
documented in the write-up's LLM-usage section.

## What's real vs. what's scaffolding

Scripts prefixed with the actual analysis (decomposition, checks, causal test) are the
load-bearing code — every number in the write-up traces back to one of these. Everything
under `steering-vector-distillation/` is Blank et al.'s original released repository,
used largely unmodified except where noted below.

## File-by-file

### My scripts (run in this order to reproduce the pipeline)

- **`run_decomp.py`** — Full 27-layer J-lens decomposition of $v_{\text{student}}$.
  Includes both bug fixes (RMSNorm scale folded into the readout, correct matrix
  orientation). Produces `decomp_results_fixed.pt`. This is the corrected version;
  an earlier, buggy version (missing RMSNorm, wrong orientation) is not included —
  its incorrect output is discussed in the write-up as part of the debugging narrative,
  not presented as a result.

- **`verify_orientation.py`** — The ground-truth calibration test. Extracts a real hidden
  state for "The capital of France is" at layers 12 and 26, applies both candidate
  matrix orientations, and checks which one recovers the correct completion ("Paris").
  This is what determined the fix applied in `run_decomp.py`. Run this first if you want
  to see the justification before the decomposition itself.

- **`run_checks.py`** — Three sanity checks on the decomposition solver, run at layer 12:
  (1) a dictionary row decomposed against itself at k=1 should return FVE≈1.0, confirming
  the solver works at all; (2) FVE on 15 random unit vectors, establishing the noise floor
  the real vector's FVE needs to beat; (3) k-sensitivity (k=8/16/32) on $v_{\text{student}}$,
  checking the result isn't an artifact of an undersized atom budget.

- **`run_teacher_check.py`** — Same protocol as `run_decomp.py`, applied to
  $v_{\text{teacher}}$ instead, at six sampled layers (8, 12, 16, 20, 23, 26) with a
  matched random baseline computed per-layer. This is where the layer-26 result first
  appeared.

- **`run_layer26_ksweep.py`** — k-sensitivity check (k=8/16/32/64) specifically at
  layer 26 for $v_{\text{teacher}}$, run after the initial teacher check flagged that
  layer as an outlier. Also prints the actual reconstructed tokens at each k, which is
  where the Chinese cat-vocabulary tokens (宠爱, 猫咪) were first observed.

- **`check_readout_only.py`** — The control described in §4.4 of the write-up: decomposes
  $v_{\text{teacher}}$ at layer 26 against the raw unembedding alone, with no Jacobian
  transport, to test whether the layer-26 finding is specific to J-lens or just a property
  of being near the output layer. Result: FVE≈4.55%, statistically indistinguishable from
  the full J-lens result (4.06%) — this is what narrowed the causal claim in the final
  write-up.

- **`save_teacher_components.py`** — Decomposes $v_{\text{teacher}}$ at layer 26 (k=32),
  saves $v_J$, $v_R$, a random control, and the full vector as separate `.pt` files in the
  format Blank et al.'s `eval_steered.py` expects (`vJ_teacher_L26.pt`, `vR_teacher_L26.pt`,
  `rand_L26.pt`, `vFull_teacher_L26.pt`), each rescaled to $v_{\text{teacher}}$'s natural
  layer-26 norm so the subsequent injection test compares equal magnitudes.

- **The recombination/orthogonality verification** (run inline, not saved as a separate
  file) recovers the natural per-direction coefficients via least-squares against the
  saved unit directions in `vJ_teacher_L26.pt` and `vR_teacher_L26.pt`, confirming
  $v_J + v_R = v_{\text{teacher}}$ to $5.5\times10^{-7}$ relative error and
  $v_J \perp v_R$ (cosine $4.2\times10^{-9}$) — reported in §6 of the write-up. Reconstructed
  below as `verify_recombination.py` for completeness.

### Data/results

- **`decomp_results_fixed.pt`** — Saved output of `run_decomp.py`: per-layer FVE and
  top-token lists for $v_{\text{student}}$ across all 27 lens-covered layers.
- **`results/`** — Terminal output of the four scripts saved via `tee`, kept as raw logs
  rather than reformatted, so the actual numbers can be checked against what's quoted
  in the write-up without re-running anything.
- **`eval_results/`** — Standard output directory from Blank et al.'s `eval_steered.py`,
  one subfolder per named run (`pilot_unsteered`, `full_vstudent_broadcast`,
  `causal_vJ_teacher_L26`, etc.). Each contains `eval_results.json` (summary + cat_rate)
  and `eval_samples.jsonl` (every individual completion — this is what the randomly
  sampled qualitative examples in the write-up are pulled from).
- **`mistakes.md`** — Running log of bugs caught and fixed during the session, referenced
  directly in the LLM-usage form answer.

### Blank et al.'s original repository (`steering-vector-distillation/`)

Used as released, with two exceptions:
1. Their `install.sh`/`uv` environment setup was abandoned (documented in `mistakes.md`)
   due to a Python 3.14 / `numba` incompatibility; dependencies were installed directly
   via `pip` instead.
2. Steering runs used `positions=broadcast` rather than the shipped config default
   (`positions=prompt_all`), per §3.2 of their paper's own methodology — this discrepancy
   and its cause are discussed in Experiment 1 of the write-up.

Everything else — `steering_utils.py` (the actual steering hooks), `eval_steered.py`
(the eval harness), the released `v_teacher`/`v_student` vector files — is their code
and their artifacts, used to reproduce and extend their result, not reimplemented.

## What's NOT included

- Model weights (Qwen2.5-7B-Instruct, ~15GB) and the J-lens file (~694MB, from
  `neuronpedia/jacobian-lens`) — both re-downloadable via the paths referenced in the
  scripts above, not included in this submission for obvious size reasons.
- An earlier, buggy version of the decomposition script (missing RMSNorm, wrong
  orientation) — its output is discussed narratively in the write-up but not included
  as a runnable artifact, since including it without the fix risks it being mistaken
  for a real result.
