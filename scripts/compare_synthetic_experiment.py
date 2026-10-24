"""Compare completed, matched synthetic-control and augmented experiments."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from synthetic_data import digest, write_json


def compare_runs(control, augmented, output):
    roots = [Path(control).resolve(), Path(augmented).resolve()]
    setups = [json.loads((r/'setup.json').read_text()) for r in roots]
    statuses = [json.loads((r/'status.json').read_text()) for r in roots]
    if any(s['status'] != 'complete' for s in statuses):
        raise ValueError('Both experiments must complete before comparison')
    if statuses[0]['completed_epochs'] != statuses[1]['completed_epochs']:
        raise ValueError('Both arms must have the same optimizer-update budget')
    contracts = [json.loads(json.dumps(s['contract'])) for s in setups]
    subsets = [c['config'].pop('subset') for c in contracts]
    if subsets != ['matched-original', 'full'] or contracts[0] != contracts[1]:
        raise ValueError('Comparison requires matched data, code, seed and training settings')
    for key in ('scale_uT', 'model_config', 'pad', 'optimizer_updates_per_epoch'):
        if setups[0][key] != setups[1][key]:
            raise ValueError(f'Experimental control mismatch: {key}')
    if not set(setups[0]['sample_ids']) <= set(setups[1]['sample_ids']):
        raise ValueError('Control recipes must be a subset of augmented recipes')
    tables = []
    for root in roots:
        with (root/'actual_comsol.csv').open(newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if r['method'] == 'synthetic_final']
        table = {(r['pair_id'], r['component']): r for r in rows}
        if not table or len(table) != len(rows):
            raise ValueError('Missing or duplicate actual evaluation rows')
        tables.append(table)
    if set(tables[0]) != set(tables[1]):
        raise ValueError('Both arms must evaluate exactly the same actual pairs')
    pairs = []
    for key, a in tables[0].items():
        b = tables[1][key]
        row = dict(pair_id=key[0], component=key[1], geometry=a['geometry'])
        if any(a[k] != b[k] for k in ('geometry', 'sharp_z_um', 'blur_z_um')):
            raise ValueError('Actual row identities differ')
        for metric in ('nrmse_range', 'relative_l2', 'rmse_uT', 'amplitude_slope'):
            x, y = (float(r[metric]) if r[metric] else None for r in (a, b))
            if any(v is not None and not np.isfinite(v) for v in (x, y)):
                raise ValueError('Nonfinite evaluation metric')
            row['control_'+metric], row['augmented_'+metric] = x, y
            row['delta_'+metric] = y-x if x is not None and y is not None else None
        pairs.append(row)
    grouped = []
    for group in ['all']+sorted({r['geometry'] for r in pairs}):
        selected = [r for r in pairs if group == 'all' or r['geometry'] == group]
        row = dict(group=group, component_pairs=len(selected))
        for metric in ('nrmse_range', 'relative_l2', 'rmse_uT', 'amplitude_slope'):
            for arm in ('control', 'augmented', 'delta'):
                values = [r[arm+'_'+metric] for r in selected if r[arm+'_'+metric] is not None]
                row['mean_'+arm+'_'+metric] = float(np.mean(values)) if values else None
        valid = [r for r in selected if r['delta_nrmse_range'] is not None]
        row['nrmse_improved_pairs'] = sum(r['delta_nrmse_range'] < 0 for r in valid)
        grouped.append(row)
    all_rows = grouped[0]
    primary, secondary = all_rows['mean_delta_nrmse_range'], all_rows['mean_delta_relative_l2']
    verdict = ('Augmentation improves both predeclared mean error measures on these same-geometry actual COMSOL cases.'
               if primary is not None and secondary is not None and primary < 0 and secondary < 0 else
               'The predeclared requirement of improvement in both mean error measures was not met.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('paired_comparison.csv', pairs), ('summary.csv', grouped)]:
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_json(output/'results.json', dict(verdict=verdict, summary=grouped,
        epochs=statuses[0]['completed_epochs'], control_component_pairs=setups[0]['component_pairs'],
        augmented_component_pairs=setups[1]['component_pairs'],
        source_files={str(root/name): digest(root/name) for root in roots
                      for name in ('setup.json', 'status.json', 'actual_comsol.csv', 'final.pt')},
        limits='Single seed, shared source geometries/targets, augmentation changes height/rotation/noise coverage. '
               'No calibrated confidence or causal proof that limited data was the only problem.'))
    print(verdict, flush=True)
    return grouped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--augmented', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    compare_runs(args.control, args.augmented, args.output)


if __name__ == '__main__':
    main()
