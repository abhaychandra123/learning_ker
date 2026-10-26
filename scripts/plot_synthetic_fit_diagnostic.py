"""Plot the selected-case diagnostic; no new inference or parameter selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    data = json.loads((args.directory/'diagnostics.json').read_text())
    if 'diagnostic_wall_seconds' not in data:
        raise ValueError('Diagnostic is not yet complete')
    rows = data['reconstructions']
    def value(pair, domain, method):
        return next(r['relative_l2']*100 for r in rows
                    if r['pair_id']==pair and r['domain']==domain and r['method']==method)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    x = np.arange(3)
    pairs = [f'L1_C1_z{h:04.1f}_to_z00.5' for h in (1,6,10)]
    for offset, domain, color, label in [(-.19,'actual_comsol','#146b89','Actual COMSOL input'),
                                        (.19,'clean_synthetic','#e69d41','Clean synthetic input')]:
        values = [value(p,domain,'trained_16') for p in pairs]
        bars=axes[0].bar(x+offset, values, width=.36, color=color, label=label)
        axes[0].bar_label(bars,fmt='%.1f%%',padding=3,fontsize=9)
    axes[0].set_xticks(x,['1 → 0.5 µm','6 → 0.5 µm','10 → 0.5 µm'])
    axes[0].set_title('The trained model struggles even on clean inputs')
    methods=['initial_ista_16','trained_16','fista_16','fista_64','fista_256']
    labels=['ISTA\n16','Trained\n16 layers','FISTA\n16','FISTA\n64','FISTA\n256']
    for domain, color, label in [('actual_comsol','#146b89','Actual COMSOL input'),
                               ('clean_synthetic','#e69d41','Clean synthetic input')]:
        axes[1].plot(np.arange(5),[value(pairs[-1],domain,m) for m in methods],
                     marker='o',color=color,label=label)
    axes[1].set_xticks(np.arange(5),labels)
    axes[1].set_title('10 µm: more solver iterations recover more signal')
    for ax in axes:
        ax.set_ylabel('Relative L2 reconstruction error (%) ↓')
        ax.set_ylim(0,110); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower right',fontsize=9)
    fig.suptitle('L1_C1, Bz: full-resolution diagnostic of the completed synthetic model',fontsize=13)
    fig.text(.5,-.04,'All methods: analytical operator, pad 256. FISTA uses fixed iteration budgets; '
             'not a convergence claim. Selected cases, not the full evaluation.',ha='center',fontsize=9)
    fig.savefig(args.directory/'fit_diagnostic.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
