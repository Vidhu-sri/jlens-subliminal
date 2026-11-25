# Repository cleanup — 2026-11-25

- Checked HF login: silvix9. Git identity retained.
- Removed 21 PID/launch-PID paths from tracking; local files retained and ignored.
- Removed the byte-identical duplicate handoff; moved the remaining historical handoff to docs/.
- Consolidated legacy evaluations, layer sweeps, and the root decomposition tensor under results_new/legacy/; adjusted analysis/output paths.
- Moved the full run log to logs/RUNLOG.md and updated logging hooks.
- Preserved original machine/account-bearing text locally before explicit public redactions.
- Added the model, upstream references, reproduction instructions, and a CPU-generated chart with source counts.
- Narrowed conclusions to the evidence and documented the historical readout-only solver limitation; no GPU jobs were launched or changed.

- HF archive verified: 141 files, 1,073,496,831 bytes, commit b93113beb45cf1117dc5c091f0eacd3dfe17d0b0. Remote byte sizes and LFS SHA-256 verified before untracking; all local files retained.
- Validation: README/support-page local links resolve; modified JSON files preserve all numeric values; Python compilation and shell syntax checks pass; CPU J/R analysis reproduces 5.88/3.76% J and 2.76/4.16% R. Figure visually inspected.
- Pending owner decisions: original-code license and authorization to rewrite Git history. No history rewrite or profile edit performed.
- Updated obsolete worker HF destination settings to silvix9 and removed remaining machine-specific shell paths. These historical workers were not executed.
