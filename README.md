# J-Lens and subliminal steering

This is my personal interpretability project testing what Jacobian lenses reveal about steering vectors and trained language models.
I reproduce and extend released subliminal-steering experiments with vector decompositions, teacher interventions, and student controls.
The repository keeps the analysis code, experiment outputs, and run records so the findings can be checked against the underlying artifacts.

## Findings

### Solid: direct measurements and large-sample evaluations

1. **Blank’s “cat” vector is mostly not about cats.** It has cosine 0.79 with the “you love mathematics” vector and 0.95 with the owl vector. The cat-specific component, cat minus owl, is about 32% of the full cat vector’s norm, consistent with a large shared prompt or “I love something” component. ([results](results_new/vectors/specific_vectors_L23.json), [run output](results_new/diagnostics/specific_vectors_L23_run.log))

2. **The cat-specific component is much more potent on the teacher.** At the same L23 steering strength, cat-only steering produced 25.8% cat responses versus 5.4% for Blank’s full vector; the exact negative produced 33% owl and 0% cat (500 samples per condition). ([results](results_new/analysis_L23.txt), [pilot run log](results_new/diagnostics/pilot_L23.log))

3. **The teacher’s cat-specific effect is concentrated in the R-part.** Splitting cat minus owl with the J-lens gives 25.4% cat for the R-part and 3.4% for the J-part. ([results](results_new/analysis_L23.txt), [pilot run log](results_new/diagnostics/pilot_L23.log), [pilot evaluation samples](results_new/eval_results/pilot_teacher_catonly_R_L23_a3_prompt_all/eval_samples.jsonl))

4. **J-Lens does not expose a readable cat-trait signal where it would matter.** Blank’s student vector has no semantically readable structure across the 27 analyzed layers; the L23 teacher vector’s FVE is below its random baseline; and the trained-student shifts we decomposed are dominated by generic fragments rather than a clear trait representation. The conspicuous L26 teacher structure is generic and nearly identical for cat and owl (their vectors have cosine about 0.95); the near-output-layer readout-only control also recovers comparable structure without Jacobian transport. ([student results](results/student_all27_layers.txt), [teacher results](results/teacher_all_layers.txt), [shift decomposition](results_new/diagnostics/shift_decomposition.md), [cat/owl results](results_new/diagnostics/extra_vectors.log), [readout-only results](results_new/diagnostics/readout_random_baseline.md), [readout-only control code](check_readout_only.py))

5. **The J-part changes teacher number outputs more than the R-part.** Fingerprint KL is about 3.4× higher for J than R at L23 and about 69× higher at L26. A random-J control is also highly disruptive at L26, so this shows sensitivity to J-space steering rather than proving that the cat trait is uniquely carried by J. ([L23 results](results_new/diagnostics/fingerprint_L23.json), [L23 run log](results_new/diagnostics/fingerprint_L23.log), [L26 results](results_new/diagnostics/fingerprint.json), [L26 run log](results_new/diagnostics/fingerprint.log))

### Null: controls and limits

6. **The J-versus-R student transmission difference is not reliable in these small replications.** With data seed 42, J minus R is +3.12 percentage points in cat rate; with seed 43, the fresh matched pair gives −0.40 points. The contrast changes sign across the two seeds. ([seed-42 analysis](results_new/analysis_L23.txt), [seed-42/43 run records](RUNLOG.md), [seed-43 J run](results_new/run_catonlyJ_s43.log), [seed-43 R run](results_new/run_seed43_catonly_R.log))

7. **Student-to-student noise is large.** Neutral students alone range from 1.16% to 4.12% cat responses across seeds, so the single-seed J/R transmission differences are comparable to that spread or near its edge. ([neutral and student results](results_new/analysis_L23.txt), [matched-seed summary](results_new/diagnostics/round2_summary.md), [run records](RUNLOG.md))

8. **Blank’s full-vector student transmission did not reproduce at the tested α=3.** The L26 full-vector student produced 2.48% cat responses against 4.12% for its matched neutral. Blank’s released sufficiency data filename specifies α=6; we did not train a student on that released dataset or test its transmission. ([evaluation analysis](results_new/analysis.txt), [released-data inventory](results_new/diagnostics/hf_agu18dec_steering_vector_distillation_files.txt), [dataset check](results_new/diagnostics/blank_suff_dataset_check.md), [run records](RUNLOG.md))

9. **L26 J-part student steering also gives a generic cat drift, including the owl-vector control.** The vJ student reached 8.28% cat and the vJ-owl student 7.44%, against 4.12% for the seed-42 neutral. Because the owl control shows a similar increase, this is not evidence of cat-trait-specific transmission. ([evaluation analysis](results_new/analysis.txt), [L23/L26 comparison](results_new/analysis_L23.txt), [cat/owl vector comparison](results_new/diagnostics/extra_vectors.log), [evaluation samples](results_new/eval_results/student_t_vJ_owl_L26_a3_prompt_all_s42/eval_samples.jsonl))

## Reproducing and inspecting results

The `results/` files contain the original layer-sweep outputs. `results_new/diagnostics/` contains the later decomposition, fingerprint, and control summaries; `results_new/eval_results/` contains per-run summaries and samples; and `RUNLOG.md` records commands and run status. The analysis scripts live under the repository root and `transmission/`.

Model weights and the J-lens checkpoint are not included; see the scripts and `HANDOFF.md` for their expected locations and setup.
