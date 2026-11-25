"""CPU-only README figure, derived directly from saved evaluation counts."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'results_new/eval_results'
OUT = ROOT / 'figures'

def result(run):
    p = EVAL / run / 'eval_results.json'
    d = json.loads(p.read_text())
    return 100 * d['target_hits'] / d['total_samples'], d['total_samples'], str(p.relative_to(ROOT))

neutrals = [result(f'student_t_neutral_L26_a0_prompt_all_s{s}') for s in (42, 43, 44)]
lo, hi = min(x[0] for x in neutrals), max(x[0] for x in neutrals)
rows = []
fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
conditions = [('full', 'Full'), ('cat_only', 'Cat − owl'), ('owl_only', 'Owl − cat'),
              ('catonly_J', 'J-part'), ('catonly_R', 'R-part'), ('rand', 'Random')]
for i, (cond, label) in enumerate(conditions):
    for seed, offset, color in [(42, -.19, '#246a9f'), (43, .19, '#d17a26')]:
        run = f'student_t_{cond}_L23_a3_prompt_all_s{seed}'
        if not (EVAL / run / 'eval_results.json').exists():
            continue
        rate, n, source = result(run)
        rows.append(dict(condition=cond, data_seed=seed, cat_percent=rate, samples=n, source=source))
        ax.bar(i + offset, rate, width=.36, color=color,
               label=f'Data seed {seed}' if (seed == 42 and i == 0) or (seed == 43 and cond == 'catonly_J') else None)
        ax.text(i + offset, rate + .09, f'{rate:.2f}', ha='center', fontsize=9)
ax.axhspan(lo, hi, color='#8e9b8c', alpha=.18, label=f'Neutral students: {lo:.2f}–{hi:.2f}%')
ax.set(xticks=range(len(conditions)), xticklabels=[x[1] for x in conditions],
       ylabel='Cat responses (%)', ylim=(0, 7.5), title='L23 student evaluations: the J/R contrast changes sign')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc='upper left', ncol=2)
fig.supxlabel('Unsteered student evaluation · α = 3 during teacher generation · 2,500 responses per student\nShading is the observed range of three neutral students, not a confidence interval.', fontsize=9)
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / 'cat_rates.png', dpi=180)
fig.savefig(OUT / 'cat_rates.svg')
with (OUT / 'cat_rates.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
print(json.dumps(rows, indent=2))
